"""Project entity-encoder critic on the DB width family (PREREGISTRATION_ENTITY.md). Zero rollouts.

  python -m diagnostics.controller_geometry_identifiability_v1.entity_critic
Writes entity_results.json.
"""
from __future__ import annotations

import dataclasses, json, time
import numpy as np

from . import analysis as A

ARMS = ('ENT', 'ENT_w')
REGIMES = ('MANY', 'FEW', 'FEW_x2', 'EXTRAP_A', 'EXTRAP_B')
SEEDS = (17, 23, 41)
STEPS, BATCH, LR, WD = 6000, 120, 8e-4, 1e-4


def entity_table(D):
    """Entities for every (width, state); geometry only via obstacle entities."""
    from double_bottleneck.environment import Config
    from double_bottleneck.scenario import build_layout
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    p = A.read(A.HERE / 'protocol_main.json')
    arch = {s['state_uid']: s for s in A.read(A.ARCHIVE / 'protocol.json')['states']}
    base = A.read(A.ARCHIVE / 'protocol.json')['environments']['double_bottleneck']['config']['config']
    items = []
    for w in D['widths']:
        cfg = dataclasses.replace(Config(**base), bottleneck_width=float(w))
        walls = build_layout(cfg).walls
        obstacles = [dict(a=np.asarray(x[0]).tolist(), b=np.asarray(x[1]).tolist(), radius=float(cfg.wall_radius),
                          curvature=0., interior_sign=0.) for x in walls]
        for u in p['parent_states']:
            ph = arch[u]['physical']; n = len(ph['positions'])
            scene = dict(positions=ph['positions'], velocities=ph['velocities'], goals=ph['goals'],
                         flow=np.zeros((n, 2)).tolist(), radius=[cfg.agent_radius] * n, goal_tolerance=[cfg.goal_tolerance] * n,
                         obstacles=obstacles, policy_origin=[0., 0.], policy_axis=[1., 0.], policy_axis_required=True,
                         flow_committed=True, remaining_fraction=1., remaining_seconds=cfg.max_steps * cfg.dt,
                         dt=cfg.dt, max_speed=cfg.max_speed, max_seconds=cfg.max_steps * cfg.dt,
                         wall_margin=cfg.wall_collision_margin, agent_margin=cfg.agent_collision_margin,
                         monitor_active=bool(cfg.terminate_on_deadlock), progress_window_seconds=cfg.progress_window_seconds,
                         deadlock_hold_seconds=cfg.deadlock_hold_seconds, progress_epsilon=cfg.progress_epsilon,
                         speed_epsilon_fraction=cfg.speed_epsilon_fraction, monitor_elapsed=0., monitor_history_empty=True)
            items.append(rep.entities(scene))
    X = rep.batch(items)                                   # index = wi * S + si
    S = len(p['parent_states'])
    assert np.allclose(X['globals'][0], X['globals'][-1]) # globals carry no width information
    return X, S


def fit_predict(X, S, D, tr_idx, y, pred_idx, arm, seed):
    import jax, jax.numpy as jnp, optax
    from diagnostics.controller_q_diversity_v1.train import Critic
    Xj = {k: jnp.asarray(v, jnp.float32) for k, v in X.items()}
    if arm == 'ENT_w':
        wcol = np.repeat(np.asarray(D['widths'], np.float32), S)[:, None]
        Xj['globals'] = jnp.concatenate([Xj['globals'], jnp.asarray(wcol)], -1)
    onehot = jnp.eye(5, dtype=jnp.float32)
    eta = jnp.asarray(D['eta'], jnp.float32)
    n_agents = X['agents'].shape[1]
    net = Critic('state_field_compact')

    def inputs(ix):
        g = ix[:, 0] * S + ix[:, 1]
        x = {k: v[g] for k, v in Xj.items()}
        c = jnp.broadcast_to(onehot[ix[:, 2]][:, None, :], (len(g), n_agents, 5)) * x['agent_mask'][..., None]
        return x, c, eta[ix[:, 3]]

    tr = jnp.asarray(tr_idx, jnp.int32); yy = jnp.asarray(y, jnp.float32)
    params = net.init(jax.random.PRNGKey(seed), *inputs(tr[:2]))
    opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(LR, weight_decay=WD)); st = opt.init(params)

    @jax.jit
    def block(params, st, draws):
        def step(carry, b):
            pp, oo = carry
            def loss(q):
                return optax.sigmoid_binary_cross_entropy(net.apply(q, *inputs(tr[b])), yy[b]).mean()
            l, g = jax.value_and_grad(loss)(pp)
            u, oo = opt.update(g, oo, pp)
            return (optax.apply_updates(pp, u), oo), l
        return jax.lax.scan(step, (params, st), draws)

    rng = np.random.default_rng(seed)
    for _ in range(STEPS // 500):
        (params, st), _ = block(params, st, jnp.asarray(rng.integers(0, len(y), (500, BATCH)), jnp.int32))
    pr = jnp.asarray(pred_idx, jnp.int32)
    predict = jax.jit(lambda q, ix: jax.nn.sigmoid(net.apply(q, *inputs(ix))))
    return np.concatenate([np.asarray(predict(params, pr[k:k + 512])) for k in range(0, len(pred_idx), 512)])


def main():
    D = A.load(); assert D['missing'] == 0
    X, S = entity_table(D)
    W, g = D['widths'], D['grid']; wi = lambda w: W.index(w)
    test = [wi(w) for w in g['test']]; pool = [wi(w) for w in g['pool']]; few = [wi(w) for w in g['few']]
    allS = list(range(S)); t = sorted(g['test'])
    specs = {'MANY': [(p_, [(5 * i + k) % S for k in range(5)], [0, 1]) for i, p_ in enumerate(pool)],
             'FEW': [(f, allS, [0, 1]) for f in few], 'FEW_x2': [(f, allS, [0, 1, 2, 3]) for f in few],
             'EXTRAP_A': [(wi(w), allS, [0, 1]) for w in g['pool'] if w < t[1]],
             'EXTRAP_B': [(wi(w), allS, [0, 1]) for w in g['pool'] if w > t[2]]}
    targets = {k: test for k in specs}; targets['EXTRAP_A'] = [wi(t[2]), wi(t[3])]; targets['EXTRAP_B'] = [wi(t[0]), wi(t[1])]
    pidx, _, _ = A.features(D, test, allS)
    out, started = {}, time.monotonic()
    for arm in ARMS:
        for name in REGIMES:
            tr_idx, y = A.training_rows(D, specs[name])
            p = np.mean([fit_predict(X, S, D, tr_idx, y, pidx, arm, s) for s in SEEDS], 0).reshape(len(test), S, 5, 4)
            res = {}
            for tag, tw in (('own', targets[name]), ('at_A', targets['EXTRAP_A']), ('at_B', targets['EXTRAP_B'])):
                fn, _ = A.evaluate(D, np.stack([p[test.index(x)] for x in tw]), tw, tr_idx, y)
                res[tag] = fn(allS)
            out.setdefault(arm, {})[name] = res
            o = res['own']
            print(f"{arm:5s} {name:8s} B2 {o['B2']:+.3f} B1 {o['B1_skill']:+.3f} sel {o['sel_model']:.3f}/{o['sel_oracle']:.3f} "
                  f"eta-sel {o['eta_sel_model']:.3f}/{o['eta_sel_oracle']:.3f}  [{time.monotonic() - started:.0f}s]", flush=True)
    mlp = A.read(A.HERE / 'results.json')['test_B']['results']
    ext = lambda r, k: np.mean([r['EXTRAP_A']['own'][k] if 'own' in r['EXTRAP_A'] else r['EXTRAP_A']['own_targets'][k],
                                 r['EXTRAP_B']['own'][k] if 'own' in r['EXTRAP_B'] else r['EXTRAP_B']['own_targets'][k]])
    ent_ext, entw_ext = ext(out['ENT'], 'B2'), ext(out['ENT_w'], 'B2')
    mlp_ext = np.mean([mlp['EXTRAP_A']['own_targets']['B2'], mlp['EXTRAP_B']['own_targets']['B2']])
    verdict = dict(ENT_extrap_B2=ent_ext, ENT_w_extrap_B2=entw_ext, MLP_w_extrap_B2=mlp_ext,
                   ENT_FEW_B2=out['ENT']['FEW']['own']['B2'],
                   R1_representation_not_bottleneck=bool(abs(ent_ext - mlp_ext) <= 0.1 and out['ENT']['FEW']['own']['B2'] >= 0.6),
                   R2_entity_geometry_bottleneck=bool(ent_ext <= 0.3 and entw_ext >= 0.5),
                   R3_encoder_cannot_extrapolate=bool(ent_ext <= 0.3 and entw_ext <= 0.3))
    (A.HERE / 'entity_results.json').write_text(json.dumps(dict(results=out, verdict=verdict), indent=2) + '\n')
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
