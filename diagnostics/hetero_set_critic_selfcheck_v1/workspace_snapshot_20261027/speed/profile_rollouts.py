import sys, os, json, time, cProfile, pstats, io
os.environ['JAX_PLATFORMS'] = 'cpu'
H = '/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'
sys.path.insert(0, H)
import bootstrap  # noqa
from common import actions
from design import controllers
from scenes import make_new
from execution import execute
import numpy as np
A = actions(); C = controllers(); keys = [a['key'] for a in A]
S = {s['name']: s for f in ['scenes_explore.json', 'scenes_explore2.json', 'scenes_explore3.json'] for s in json.load(open(f'/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/{f}'))}
which = sys.argv[1]; ci = int(sys.argv[2]); acts = sys.argv[3].split(',')
scene = S[which]
# warm-up (JIT compile) excluded from profile
c = make_new(scene, C[ci], scene['states'][0], A[keys.index('safety')], 0); t = time.time(); execute(c); print('warmup', round(time.time() - t, 2), 's', flush=True)
pr = cProfile.Profile(); rows = []; arrays_all = []
for si in range(2):
    for a in acts:
        c = make_new(scene, C[ci], scene['states'][si], A[keys.index(a)], 0)
        t = time.time(); pr.enable(); r, arr = execute(c); pr.disable(); dt = time.time() - t
        t2 = time.time(); buf = io.BytesIO(); np.savez_compressed(buf, **arr); dsave = time.time() - t2
        rows.append((si, a, r['outcome'], r['episode_length'], round(dt, 2), round(dsave, 2), round(buf.tell() / 1e6, 2)))
for r in rows: print('state %d eta %-6s outcome %-8s steps %4d run %.2fs trace-save %.2fs trace %.2fMB' % r)
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats('cumulative'); ps.print_stats(45); print(s.getvalue()[:9000])
