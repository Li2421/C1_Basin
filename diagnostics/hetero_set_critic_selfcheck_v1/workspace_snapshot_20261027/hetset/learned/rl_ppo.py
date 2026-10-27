"""Multi-agent reinforcement learning planner: parameter-shared proximal policy optimization (decentralized actor and
critic on the same 72-dim local observation as the imitation planners; Long et al. 2018 / independent PPO style).
Training environment = the harness single-integrator physics WITHOUT safety projection: per-robot swept collision
tests (harness distance functions), a robot stops when it reaches its goal or collides.
Reward per robot and step: 2.5 x progress along the navigation-field distance, +15 on arrival, -15 on collision, -0.01.
Training scenes only (28 scenes, states 0-15).  usage: rl_ppo.py OUT ITERATIONS WORKERS"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from learned_obs import OBS_DIM

HID = 256
_W = {}


def init_params(rng):
    def lin(i, o, scale):
        return (rng.standard_normal((i, o)) * scale / np.sqrt(i)).astype(np.float32), np.zeros(o, np.float32)
    p = {}
    p['W1'], p['b1'] = lin(OBS_DIM, HID, np.sqrt(2)); p['W2'], p['b2'] = lin(HID, HID, np.sqrt(2)); p['W3'], p['b3'] = lin(HID, 2, 0.01)
    p['V1'], p['c1'] = lin(OBS_DIM, HID, np.sqrt(2)); p['V2'], p['c2'] = lin(HID, HID, np.sqrt(2)); p['V3'], p['c3'] = lin(HID, 1, 1.0)
    p['log_std'] = np.full(2, -0.7, np.float32)
    return p


def actor_np(p, x):
    h = np.tanh(x @ p['W1'] + p['b1']); h = np.tanh(h @ p['W2'] + p['b2'])
    return np.tanh(h @ p['W3'] + p['b3'])


def _init(norm):
    import run8
    run8._init(); _W.update(run8._W); _W['norm'] = norm
    import collect_expert
    _W['train'] = collect_expert.splits()[0]


def episode(args):
    """One training episode with stochastic actions; returns per-robot sequences."""
    import learned_obs as LO, runtime_new as RN, swap_geometry as SG
    p, seed = args
    rng = np.random.default_rng(seed)
    v = _W['train'][rng.integers(len(_W['train']))]; si = int(rng.integers(16))
    sc = _W['scenes'][v]; st = sc['states'][si]
    env = RN.SwapEnv(st, RN.LiteratureConfig(**sc['config'])); c = env.config
    walls = np.asarray(env.walls, float).reshape(-1, 2, 2); margin = env.instance.geometry.margin
    lo, hi = LO.bounds(walls, env.positions, env.goals); lo, hi = lo - 0.3, hi + 0.3
    fields = [LO.field_for(c.scene, walls, c.agent_radius, margin, g, lo, hi) for g in env.goals]
    geo = lambda i, q: float(fields[i].dist[fields[i].tree.query(q)[1]])
    P, V, G = env.positions.copy(), np.zeros_like(env.positions), env.goals.copy(); N = len(P)
    mean, std = _W['norm']; ls = p['log_std']
    done = np.zeros(N, bool); D = np.array([geo(i, P[i]) for i in range(N)])
    seq = [dict(obs=[], act=[], logp=[], rew=[]) for _ in range(N)]
    ii, jj = np.triu_indices(N, 1); end = {}
    for t in range(c.max_steps):
        o, _ = LO.observe(c.scene, P, V, G, walls, c.agent_radius, margin, c.max_speed, c.goal_tolerance, lo, hi)
        mu = actor_np(p, (o - mean) / std)
        a = mu + np.exp(ls) * rng.standard_normal(mu.shape)
        logp = (-0.5 * ((a - mu) / np.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        u = a * np.minimum(1.0, 1.0 / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-30)) * c.max_speed
        u[done] = 0
        before, after = P.copy(), P + c.dt * u
        hit = np.zeros(N, bool)
        if len(ii):
            pair = SG.point_segment_distance(np.zeros(2), before[ii] - before[jj], after[ii] - after[jj]) - 2 * c.agent_radius - c.agent_collision_margin
            bad = pair <= 0
            hit[ii[bad]] = True; hit[jj[bad]] = True
        if len(walls):
            wd = SG.segment_distance(before[:, None], after[:, None], walls[None, :, 0], walls[None, :, 1]).min(-1) - margin
            hit |= wd <= 0
        hit &= ~done
        P = after; V = u.copy()
        Dn = np.array([geo(i, P[i]) for i in range(N)])
        arrived = (np.linalg.norm(G - P, axis=-1) <= c.goal_tolerance) & ~done & ~hit
        r = 2.5 * (D - Dn) - 0.01 + 15.0 * arrived - 15.0 * hit
        for i in np.flatnonzero(~done):
            s = seq[i]; s['obs'].append(o[i]); s['act'].append(a[i]); s['logp'].append(logp[i]); s['rew'].append(r[i])
        for i in np.flatnonzero((arrived | hit) & ~done):
            end[i] = ('arrived' if arrived[i] else 'collision')
        done |= arrived | hit; D = Dn
        if done.all():
            break
    o_last, _ = LO.observe(c.scene, P, V, G, walls, c.agent_radius, margin, c.max_speed, c.goal_tolerance, lo, hi)
    out = []
    for i in range(N):
        s = seq[i]
        if not s['obs']:
            continue
        out.append(dict(obs=np.asarray(s['obs'], np.float32), act=np.asarray(s['act'], np.float32), logp=np.asarray(s['logp'], np.float32),
                        rew=np.asarray(s['rew'], np.float32), terminal=i in end, last=o_last[i].astype(np.float32), end=end.get(i, 'timeout')))
    return out


def main(out, iters, workers):
    import jax, jax.numpy as jnp, optax
    import glob
    os.makedirs(out, exist_ok=True)
    X = np.concatenate([np.load(f)['obs'].astype(np.float32) for f in sorted(glob.glob(os.path.join(HERE, 'data', 'r0', '*.npz')))])
    mean, std = X.mean(0), np.maximum(X.std(0), 1e-2); del X
    rng = np.random.default_rng(0); params = init_params(rng)
    jp = {k: jnp.asarray(v) for k, v in params.items()}
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4)); state = opt.init(jp)

    def actor(p, x):
        h = jnp.tanh(x @ p['W1'] + p['b1']); h = jnp.tanh(h @ p['W2'] + p['b2']); return jnp.tanh(h @ p['W3'] + p['b3'])

    def critic(p, x):
        h = jnp.tanh(x @ p['V1'] + p['c1']); h = jnp.tanh(h @ p['V2'] + p['c2']); return (h @ p['V3'] + p['c3'])[:, 0]

    def loss(p, x, a, lp0, adv, ret):
        mu = actor(p, x); ls = p['log_std']
        lp = (-0.5 * ((a - mu) / jnp.exp(ls)) ** 2 - ls - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
        ratio = jnp.exp(lp - lp0)
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv))
        vl = jnp.mean((critic(p, x) - ret) ** 2)
        ent = jnp.sum(ls)
        return pg + 0.5 * vl - 0.001 * ent, (pg, vl)

    @jax.jit
    def update(p, s, x, a, lp0, adv, ret):
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p, x, a, lp0, adv, ret)
        u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s, aux
    vf = jax.jit(critic)
    log = []; t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_init, initargs=((mean, std),)) as ex:
        for it in range(1, iters + 1):
            pn = {k: np.asarray(v) for k, v in jp.items()}
            seqs = [s for eps in ex.map(episode, [(pn, int(rng.integers(2 ** 31))) for _ in range(2 * workers)]) for s in eps]
            xs, acts, lps, advs, rets = [], [], [], [], []
            ends = {'arrived': 0, 'collision': 0, 'timeout': 0}
            for s in seqs:
                ends[s['end']] += 1
                x = (s['obs'] - mean) / std
                v = np.asarray(vf(jp, jnp.asarray(x)))
                vlast = 0.0 if s['terminal'] else float(np.asarray(vf(jp, jnp.asarray(((s['last'] - mean) / std)[None])))[0])
                vals = np.append(v, vlast); adv = np.zeros(len(v), np.float32); g = 0.0
                for k in reversed(range(len(v))):
                    delta = s['rew'][k] + 0.99 * vals[k + 1] - vals[k]; g = delta + 0.99 * 0.95 * g; adv[k] = g
                xs.append(x); acts.append(s['act']); lps.append(s['logp']); advs.append(adv); rets.append(adv + v)
            x, a, lp0, adv, ret = map(np.concatenate, (xs, acts, lps, advs, rets))
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            n = len(x); mb = 8192
            for _ in range(4):
                perm = rng.permutation(n)
                for j in range(0, n, mb):
                    b = perm[j:j + mb]
                    jp, state, aux = update(jp, state, jnp.asarray(x[b]), jnp.asarray(a[b]), jnp.asarray(lp0[b]), jnp.asarray(adv[b]), jnp.asarray(ret[b]))
            tot = sum(ends.values()); rec = dict(it=it, samples=int(n), arrived=ends['arrived'] / tot, collision=ends['collision'] / tot, timeout=ends['timeout'] / tot,
                                              std=float(np.exp(np.asarray(jp['log_std'])).mean()), seconds=round(time.time() - t0))
            log.append(rec); print(json.dumps(rec), flush=True)
            if it % 10 == 0 or it == iters:
                np.savez(os.path.join(out, 'params.npz'), **{k: np.asarray(v) for k, v in jp.items()})
                json.dump(dict(kind='ppo', mean=mean.tolist(), std=std.tolist(), iterations=it, log=log), open(os.path.join(out, 'meta.json'), 'w'))
    print('saved', out, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
