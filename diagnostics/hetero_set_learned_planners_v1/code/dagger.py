"""DAgger round for one learned planner: roll it out (+ safety projection, eta = 0) on the 28 training scenes, states 0-15,
and relabel every 2nd visited state with a fresh prioritized A* plan computed from that state.
usage: dagger.py CKPT OUT_DIR"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import run as R, run8
import learned_actor as LA
from collect_expert import splits, H


def _init():
    run8._init(); LA.install()
    M = {m['controller']: m for m in json.load(open(H + '/CONTROLLER_MANIFEST.json'))}
    run8._W['expert'] = M['prioritized_astar0']


def job(t):
    import learned_obs as LO, runtime_new as RN
    from planning import Planner
    v, si, ckpt, out = t; W = run8._W; sc = W['scenes'][v]; st = sc['states'][si]
    c = W['make_new'](sc, LA.row('learner', ckpt), st, W['actions'][R.KEYS.index('safety')], 0)
    r, arrays = W['execute'](c)
    P, V, G = arrays['positions'], arrays['velocities'], arrays['goals']
    env0 = RN.SwapEnv(st, RN.LiteratureConfig(**sc['config'])); cfg = env0.config
    walls, lo, hi = LA.episode_bounds(env0); margin = env0.instance.geometry.margin
    obs, act, meta = [], [], []
    for k in range(0, len(P) - 1, 2):
        try:
            env = RN.SwapEnv(dict(st, positions=P[k].tolist(), velocities=V[k].tolist()), RN.LiteratureConfig(**sc['config']))
        except AssertionError:
            continue
        env.step_count = k
        a = Planner(W['expert']).action(env)
        o, arrived = LO.observe(cfg.scene, P[k], V[k], G, walls, cfg.agent_radius, margin, cfg.max_speed, cfg.goal_tolerance, lo, hi)
        keep = ~arrived
        obs.append(o[keep]); act.append((a[keep] / cfg.max_speed).astype(np.float32)); meta.append(np.stack([np.full(keep.sum(), k), np.flatnonzero(keep)], -1))
    if obs:
        np.savez_compressed(os.path.join(out, f'{v}__{si:02d}.npz'), obs=np.concatenate(obs).astype(np.float16), act=np.concatenate(act), meta=np.concatenate(meta).astype(np.int32))
    return v, si, r['outcome']


if __name__ == '__main__':
    ckpt, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); os.makedirs(out, exist_ok=True)
    TR, _, _ = splits()
    tasks = [(v, si, ckpt, out) for v in TR for si in range(16) if not os.path.exists(os.path.join(out, f'{v}__{si:02d}.npz'))]
    print(len(tasks), 'todo', flush=True); t0 = time.time(); res = []
    with ProcessPoolExecutor(int(os.environ.get('LWORKERS', '18')), mp_context=mp.get_context('spawn'), initializer=_init) as ex:
        for fu in as_completed([ex.submit(job, t) for t in tasks]):
            res.append(fu.result())
    import collections
    json.dump(res, open(os.path.join(out, 'learner_outcomes.json'), 'w'))
    print('done', round(time.time() - t0), 's learner outcomes (training scenes, safety only):', dict(collections.Counter(x[2] for x in res)), flush=True)
