"""Correct the pilot's inappropriate live-B truncation under a new identity."""
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.planner import preflight

def main():
    original=json.loads((HERE/'rollout_protocol.json').read_text())
    p=json.loads((HERE/'protocol.json').read_text())
    jobs=[];requests=[]
    for j in original['jobs']:
        if j['kind']!='basis_feedback' or j['mode']!='live_B':continue
        q=dict(j);q['mode']='live_B_full';q['request_controller_uid']=q['request_controller_uid'].replace('live_B','live_B_full')
        jobs.append(q)
        requests.append(dict(state_uid=q['request_state_uid'],eta_uid=q['eta_uid'],
            controller_uid=q['request_controller_uid'],
            seed_keys=[json.dumps(dict(future_root=p['future_root'],seed=p['seed']),separators=(',',':'))]))
    (HERE/'live_full_protocol.json').write_text(json.dumps(dict(jobs=jobs,
        correction='Pilot live_B incorrectly inherited frozen-B reference censoring; this is a distinct, complete live-B run.'),indent=2)+'\n')
    path=HERE/'live_full_planned_rollouts.json';path.write_text(json.dumps(dict(requests=requests),indent=2)+'\n')
    out=preflight(path);(HERE/'live_full_cache_preflight.json').write_text(json.dumps(out,indent=2)+'\n')
    print(len(jobs),out['summary'])

if __name__=='__main__':main()
