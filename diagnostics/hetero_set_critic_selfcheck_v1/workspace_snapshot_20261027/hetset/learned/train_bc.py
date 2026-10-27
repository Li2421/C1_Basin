"""Imitation training for one learned planner.  usage: train_bc.py KIND DATA_DIRS(comma) OUT SEED STEPS
States 14-15 of every training scene are held out for checkpoint selection (never validation/test scenes)."""
import os, sys, glob, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import jax, jax.numpy as jnp, optax
from flax import serialization
import learned_models as LM
from learned_obs import OBS_DIM, K


def load(dirs):
    tr, va = [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            si = int(os.path.basename(f).split('__')[1][:2])
            (va if si >= 14 else tr).append((z['obs'].astype(np.float32), z['act'].astype(np.float32)))
    cat = lambda L, i: np.concatenate([x[i] for x in L])
    return cat(tr, 0), cat(tr, 1), cat(va, 0), cat(va, 1)


def main(kind, dirs, out, seed, steps):
    os.makedirs(out, exist_ok=True)
    X, Y, Xv, Yv = load(dirs)
    init_ckpt = os.environ.get('INIT_CKPT')          # fine-tune: keep the previous round's normalization and weights
    if init_ckpt:
        m0 = json.load(open(f'{init_ckpt}/meta.json')); mean, std = np.asarray(m0['mean'], np.float32), np.asarray(m0['std'], np.float32)
    else:
        mean, std = X.mean(0), np.maximum(X.std(0), 1e-2)
    Vtr, Vva = LM.valid_of(X), LM.valid_of(Xv)
    Xn, Xvn = (X - mean) / std, (Xv - mean) / std
    rng = np.random.default_rng(seed)
    if len(Xvn) > 60000:
        sel = rng.choice(len(Xvn), 60000, replace=False); Xvn, Vva, Yv = Xvn[sel], Vva[sel], Yv[sel]
    model = LM.make(kind); f = LM.apply_fn(kind, model)
    dummy = (jnp.zeros((1, OBS_DIM)), jnp.zeros((1, K), bool), jnp.zeros((1, 2)), jnp.zeros((1,)))
    params = model.init(jax.random.PRNGKey(seed), *(dummy[:1] if kind == 'mlp' else dummy[:2] if kind == 'attention' else (dummy[0], dummy[2], dummy[3])))
    if init_ckpt:
        params = serialization.from_bytes(params, open(f'{init_ckpt}/params.msgpack', 'rb').read())
    sched = optax.cosine_decay_schedule(1e-3, steps, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)
    _, _, ab = LM.cosine_schedule()
    ab = jnp.asarray(ab)

    def loss_fn(p, xn, valid, y, key):
        if kind != 'diffusion':
            return jnp.mean(jnp.sum((f(p, xn, valid, None, None) - y) ** 2, -1))
        k1, k2 = jax.random.split(key)
        t = jax.random.randint(k1, (xn.shape[0],), 0, LM.T_DIFF)
        eps = jax.random.normal(k2, y.shape)
        a = jnp.sqrt(ab[t])[:, None] * y + jnp.sqrt(1 - ab[t])[:, None] * eps
        return jnp.mean(jnp.sum((f(p, xn, valid, a, t.astype(jnp.float32)) - eps) ** 2, -1))

    @jax.jit
    def step(p, s, xn, valid, y, key):
        l, g = jax.value_and_grad(loss_fn)(p, xn, valid, y, key)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    evalf = jax.jit(loss_fn)
    best, best_step, B, key, t0, hist = np.inf, -1, (2048 if kind == 'attention' else 4096), jax.random.PRNGKey(seed + 1), time.time(), []
    for it in range(1, steps + 1):
        idx = rng.integers(0, len(Xn), B); key, sub = jax.random.split(key)
        params, state, l = step(params, state, jnp.asarray(Xn[idx]), jnp.asarray(Vtr[idx]), jnp.asarray(Y[idx]), sub)
        if it % 1000 == 0 or it == steps:
            vk = jax.random.PRNGKey(12345); vl = []
            for j in range(0, len(Xvn), 8192):
                vk, s2 = jax.random.split(vk)
                vl.append(float(evalf(params, jnp.asarray(Xvn[j:j + 8192]), jnp.asarray(Vva[j:j + 8192]), jnp.asarray(Yv[j:j + 8192]), s2)) * len(Xvn[j:j + 8192]))
            vl = sum(vl) / len(Xvn); hist.append((it, float(l), vl))
            print(f'{kind} seed{seed} step {it} train {float(l):.4f} val {vl:.4f} {time.time() - t0:.0f}s', flush=True)
            if vl < best:
                best, best_step = vl, it
                open(f'{out}/params.msgpack', 'wb').write(serialization.to_bytes(params))
    json.dump(dict(kind=kind, seed=seed, steps=steps, best_step=best_step, val_loss=best, data=dirs, rows=int(len(X)), val_rows=int(len(Xvn)), init=init_ckpt,
                   mean=mean.tolist(), std=std.tolist(), history=hist), open(f'{out}/meta.json', 'w'))
    print('saved', out, 'best step', best_step, 'val', round(best, 4), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(','), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
