import sys, json, time
sys.path.insert(0,'/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness')
import bootstrap
from common import *
from execution import execute
T='/home/zhihan/research/Basin_C1/diagnostics/eta_diverse_scene_test_v1'
cases=json.load(open(f'{T}/main_CASES.json'))
picked={}
for c in cases:
    k=(c['state']['scene'],c['controller_payload']['nominal']['controller'])
    if c['state']['scene']=='xsc_div_size_r014' and k not in picked and c['state']['index'] in (1,2) and c['requested_action'] in ('sE01','sE29'):
        picked[k]=c
for k,c in picked.items():
    old=json.load(open(f"{T}/main/raw/{c['rollout_uid']}.json"))
    t=time.time(); r,a=execute(c)
    print(k, c['requested_action'], 'state',c['state']['index'], '| new', r['outcome'], r['success'], r['episode_length'], '| stored', old['outcome'], old['success'], old['episode_length'], '| same' if (r['outcome'],r['success'],r['episode_length'])==(old['outcome'],old['success'],old['episode_length']) else '| DIFFERENT', f'{time.time()-t:.1f}s')
