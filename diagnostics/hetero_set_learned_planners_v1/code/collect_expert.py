"""Round-0 imitation data: prioritized A* planner (expert) + safety projection (eta = 0) on the 28 training scenes,
states 0-15.  Label = the planner's bounded nominal command; observation rebuilt from the saved trajectory.
Robots already at their goal are dropped (the runtime zeroes their command)."""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import run as R, run8
H = '/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'
OUT = os.path.join(HERE, 'data', 'r0')


def splits():
    S = [s['name'] for s in json.load(open(os.path.join(ROOT, 'scenes_all.json')))]
    Z = json.load(open(os.path.join(ROOT, 'FINAL_FROZEN.json')))
    return [s for s in S if s not in Z['val'] and not s.endswith('_te')], Z['val'], Z['test']


def build_rows(scene_name, sc, arrays, env_like):
    import learned_obs as LO
    P, V, U = arrays['positions'], arrays['velocities'], arrays['nominal']
    G = arrays['goals']; walls, radius, margin, cfg = env_like
    lo, hi = LO.bounds(walls, P.reshape(-1, 2), G)
    obs, act, meta = [], [], []
    for t in range(len(U)):
        o, arrived = LO.observe(scene_name, P[t], V[t], G, walls, radius, margin, cfg.max_speed, cfg.goal_tolerance, lo, hi)
        keep = ~arrived
        obs.append(o[keep]); act.append((U[t][keep] / cfg.max_speed).astype(np.float32))
        meta.append(np.stack([np.full(keep.sum(), t), np.flatnonzero(keep)], -1))
    return np.concatenate(obs), np.concatenate(act), np.concatenate(meta)


def env_info(sc, st):
    import runtime_new as RN
    env = RN.SwapEnv(st, RN.LiteratureConfig(**sc['config']))
    return np.asarray(env.walls, float).reshape(-1, 2, 2), env.config.agent_radius, env.instance.geometry.margin, env.config


def _init():
    run8._init()
    M = {m['controller']: m for m in json.load(open(H + '/CONTROLLER_MANIFEST.json'))}
    run8._W['expert'] = M['prioritized_astar0']


def job(t):
    v, si = t; W = run8._W; sc = W['scenes'][v]; st = sc['states'][si]
    c = W['make_new'](sc, W['expert'], st, W['actions'][R.KEYS.index('safety')], 0)
    r, arrays = W['execute'](c)
    obs, act, meta = build_rows(v, sc, arrays, env_info(sc, st))
    np.savez_compressed(os.path.join(OUT, f'{v}__{si:02d}.npz'), obs=obs.astype(np.float16), act=act, meta=meta.astype(np.int32))
    return v, si, r['outcome'], len(obs)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    TR, _, _ = splits()
    tasks = [(v, si) for v in TR for si in range(16) if not os.path.exists(os.path.join(OUT, f'{v}__{si:02d}.npz'))]
    print(len(tasks), 'todo', flush=True); t0 = time.time(); res = []
    with ProcessPoolExecutor(18, mp_context=mp.get_context('spawn'), initializer=_init) as ex:
        for fu in as_completed([ex.submit(job, t) for t in tasks]):
            res.append(fu.result())
    json.dump(res, open(os.path.join(OUT, 'outcomes.json'), 'w'))
    import collections
    print('done', round(time.time() - t0), 's', collections.Counter(x[2] for x in res), 'rows', sum(x[3] for x in res), flush=True)
