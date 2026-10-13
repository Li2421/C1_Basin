"""Uniform narrow refinement of every predeclared Ring state-boundary slice."""
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight

def main():
    original=json.loads((HERE/'rollout_protocol.json').read_text())
    p=json.loads((HERE/'protocol.json').read_text())
    jobs=[];requests=[]
    for j in original['jobs']:
        if j['kind']!='state_margin' or j['scene']!='ring_exchange' or j['line_offset']!=0.:continue
        for offset in (-1e-5,1e-5):
            q=dict(j);q['line_offset']=offset
            eta=np.asarray(j['eta'],float)+offset*np.asarray(j['line_direction'],float)
            q['eta']=eta.tolist();q['eta_uid']=eta_identity(q['eta'])[0]
            q['request_controller_uid']=q['request_controller_uid']+'_narrow_root_v1'
            jobs.append(q)
            requests.append(dict(state_uid=q['request_state_uid'],eta_uid=q['eta_uid'],
                controller_uid=q['request_controller_uid'],
                seed_keys=[json.dumps(dict(future_root=p['future_root'],seed=p['seed']),separators=(',',':'))]))
    (HERE/'root_refine_protocol.json').write_text(json.dumps(dict(jobs=jobs,
        selection='All 2 Ring states × TT/FF × 7 state offsets; narrow ±1e-5 line fraction after coarse brackets'),indent=2)+'\n')
    path=HERE/'root_refine_planned_rollouts.json';path.write_text(json.dumps(dict(requests=requests),indent=2)+'\n')
    out=preflight(path);(HERE/'root_refine_cache_preflight.json').write_text(json.dumps(out,indent=2)+'\n')
    print(len(jobs),out['summary'])

if __name__=='__main__':main()
