"""Navigation-only imitation data (round-40 style): every robot ALONE in the training scenes (states 0-15), expert =
prioritized A* planner + safety projection; the learner never sees a neighbour during training.
usage: solo.py collect OUT              (expert demonstrations)
       solo.py dagger CKPT OUT          (learner rolled out alone, every 2nd state relabelled by a fresh A* plan)"""
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


def solo_state(st, j):
    s = dict(st); s['N'] = 1
    for k in ('positions', 'goals', 'velocities'):
        s[k] = [st[k][j]]
    s['index'] = f"{st['index']}_solo{j}"
    return s


def job(t):
    import learned_obs as LO, runtime_new as RN
    from planning import Planner
    mode, v, si, ckpt, out = t; W = run8._W; sc = W['scenes'][v]; full = sc['states'][si]
    obs, act, outcomes = [], [], []
    for j in range(full['N']):
        st = solo_state(full, j)
        row = W['expert'] if mode == 'collect' else LA.row('learner', ckpt)
        c = W['make_new'](sc, row, st, W['actions'][R.KEYS.index('safety')], 0)
        r, arrays = W['execute'](c); outcomes.append(r['outcome'])
        P, V, G = arrays['positions'], arrays['velocities'], arrays['goals']
        env0 = RN.SwapEnv(st, RN.LiteratureConfig(**sc['config'])); cfg = env0.config
        walls, lo, hi = LA.episode_bounds(env0); margin = env0.instance.geometry.margin
        steps = range(len(arrays['nominal'])) if mode == 'collect' else range(0, len(P) - 1, 2)
        for k in steps:
            if mode == 'collect':
                a = arrays['nominal'][k]
            else:
                try:
                    env = RN.SwapEnv(dict(st, positions=P[k].tolist(), velocities=V[k].tolist()), RN.LiteratureConfig(**sc['config']))
                except AssertionError:
                    continue
                env.step_count = k; a = Planner(W['expert']).action(env)
            o, arrived = LO.observe(cfg.scene, P[k], V[k], G, walls, cfg.agent_radius, margin, cfg.max_speed, cfg.goal_tolerance, lo, hi)
            if not arrived[0]:
                obs.append(o[0]); act.append((np.asarray(a)[0] / cfg.max_speed).astype(np.float32))
    if obs:
        np.savez_compressed(os.path.join(out, f'{v}__{si:02d}.npz'), obs=np.asarray(obs, np.float16), act=np.asarray(act, np.float32))
    return v, si, outcomes


if __name__ == '__main__':
    mode = sys.argv[1]
    ckpt, out = (None, os.path.abspath(sys.argv[2])) if mode == 'collect' else (os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
    os.makedirs(out, exist_ok=True)
    TR, _, _ = splits()
    tasks = [(mode, v, si, ckpt, out) for v in TR for si in range(16) if not os.path.exists(os.path.join(out, f'{v}__{si:02d}.npz'))]
    print(len(tasks), 'todo', flush=True); t0 = time.time(); res = []
    with ProcessPoolExecutor(int(os.environ.get('LWORKERS', '18')), mp_context=mp.get_context('spawn'), initializer=_init) as ex:
        for fu in as_completed([ex.submit(job, t) for t in tasks]):
            res.append(fu.result())
    import collections
    json.dump(res, open(os.path.join(out, 'outcomes.json'), 'w'))
    print('done', round(time.time() - t0), 's solo outcomes:', dict(collections.Counter(o for _, _, os_ in res for o in os_)), flush=True)
