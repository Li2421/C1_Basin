"""Second frozen test batch (FROZEN_T2.json, 12 scenes): 7 classical + all learned planners x 6 methods (states 8-23),
plus the solo navigation gate for learned planners.  Run with HS_SCENES=scenes_all2.json."""
import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import lrun
if __name__ == '__main__':
    F = json.load(open(os.path.join(ROOT, 'FROZEN_T2.json'))); T2 = F['test2']
    S = {s['name']: s for s in json.load(open(os.path.join(ROOT, 'scenes_all2.json')))}
    LI = [8 + i for i in range(len(lrun.LEARNED))]
    tasks = []
    for v in T2:
        k = F['choices'][v]['ens16']; e = F['choices'][v]['proposals'][k]
        for ci in [0, 1, 2, 4, 5, 6, 7] + LI:
            for si in range(8, 24):
                tasks += [('eval', v, ci, si, f'p{k}', e)] + [('eval', v, ci, si, key, None) for key in ('menu:sE22', 'rhr', 'rhr_goal', 'menu:safety', 'nominal')]
        for ci in LI:
            for si in range(8, 24):
                tasks += [('gate', v, ci, si, j) for j in range(S[v]['states'][si]['N'])]
    out = os.path.join(HERE, 't2_eval.jsonl'); done = set()
    if os.path.exists(out):
        for d in map(json.loads, open(out)):
            done.add(('gate', d['scene'], d['controller'], d['state'], d['robot']) if 'robot' in d else ('eval', d['scene'], d['controller'], d['state'], d['key']))
    todo = [t for t in tasks if t[:5] not in done]; todo.sort(key=lambda t: 0 if t[1].startswith('hs_obst') else 1)
    print(len(todo), 'todo', flush=True); t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get('LWORKERS', '18')), mp_context=mp.get_context('spawn'), initializer=lrun._init) as ex, open(out, 'a') as f:
        for fu in as_completed([ex.submit(lrun._run, t) for t in todo]):
            f.write(json.dumps(fu.result()) + '\n'); f.flush()
    print('done', round(time.time() - t0), 's', flush=True)
