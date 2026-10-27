"""Gate and evaluation for learned planners on the frozen validation (8) and test (12) scenes, states 8-23.
Learned rows (LEARNED.json: [{name, ckpt}]) are appended after the 8 classical controllers (indices 8, 9, ...).
usage: lrun.py gate|eval"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import run as R, run8, rhr, nominal as NM
import learned_actor as LA
LEARNED = json.load(open(os.path.join(HERE, 'LEARNED.json')))
BASE = 8


def _init():
    run8._init(); LA.install()
    run8._W['controllers'] = list(run8._W['controllers'])[:BASE] + [LA.row(x['name'], os.path.join(HERE, x['ckpt'])) for x in LEARNED]


def _run(t):
    W = run8._W
    if t[0] == 'gate':
        _, v, ci, si, j = t; sc = W['scenes'][v]; st = dict(sc['states'][si]); st['N'] = 1
        for k in ('positions', 'goals', 'velocities'):
            st[k] = [st[k][j]]
        st['index'] = f'{si}_solo{j}'
        c = W['make_new'](sc, W['controllers'][ci], st, W['actions'][R.KEYS.index('safety')], 0); r, _ = W['execute'](c)
        return dict(scene=v, controller=ci, name=W['controllers'][ci]['controller'], state=si, robot=j, success=int(r['success']), outcome=r['outcome'])
    _, v, ci, si, key, eta = t; sc = W['scenes'][v]
    base = W['make_new'](sc, W['controllers'][ci], sc['states'][si], W['actions'][R.KEYS.index('safety')], 0)
    if key == 'nominal':
        r = NM.run_nominal(base)
    elif key in ('rhr', 'rhr_goal'):
        r = rhr.run_rhr(base, rhr.RULE['speed'], 'centre' if key == 'rhr' else 'goal')
    else:
        act = W['actions'][R.KEYS.index(key[5:])] if key.startswith('menu:') else dict(key='gen' + key, eta_a=list(eta), eta_b=list(eta), theta=None)
        r, _ = W['execute'](W['make_new'](sc, W['controllers'][ci], sc['states'][si], act, 0))
    return dict(scene=v, key=key, controller=ci, name=W['controllers'][ci]['controller'], state=si, success=int(r['success']), outcome=r['outcome'])


def main(mode):
    Z = json.load(open(os.path.join(ROOT, 'FINAL_FROZEN.json'))); V = Z['val'] + Z['test']
    S = {s['name']: s for s in json.load(open(os.path.join(ROOT, 'scenes_all.json')))}
    CI = [BASE + i for i, x in enumerate(LEARNED) if not os.environ.get('ONLY') or x['name'] in os.environ['ONLY'].split(',')]
    if mode == 'gate':
        out = os.path.join(HERE, 'gate.jsonl')
        tasks = [('gate', v, ci, si, j) for v in V for ci in CI for si in range(8, 24) for j in range(S[v]['states'][si]['N'])]
        ident = lambda d: ('gate', d['scene'], d['controller'], d['state'], d['robot'])
    else:
        out = os.path.join(HERE, 'eval.jsonl'); tasks = []
        for v in V:
            k = Z['choices'][v]['ens16']; e = Z['choices'][v]['proposals'][k]
            for ci in CI:
                for si in range(8, 24):
                    tasks += [('eval', v, ci, si, f'p{k}', e)] + [('eval', v, ci, si, key, None) for key in ('menu:sE22', 'rhr', 'rhr_goal', 'menu:safety', 'nominal')]
        ident = lambda d: ('eval', d['scene'], d['controller'], d['state'], d['key'])
    done = set()
    if os.path.exists(out):
        done = {ident(d) for d in map(json.loads, open(out))}
    todo = [t for t in tasks if t[:5] not in done]
    todo.sort(key=lambda t: 0 if t[1].startswith('hs_obst') else 1)
    print(len(todo), 'todo', flush=True); t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get('LWORKERS', '18')), mp_context=mp.get_context('spawn'), initializer=_init) as ex, open(out, 'a') as f:
        for fu in as_completed([ex.submit(_run, t) for t in todo]):
            f.write(json.dumps(fu.result()) + '\n'); f.flush()
    print('done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
