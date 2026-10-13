"""Adaptive follow-up: action-family swaps on the remaining frozen Ring lines."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    lines=[x for x in json.loads((HERE/'h1_line_protocol.json').read_text())['lines']
           if x['available'] and x['uid']!='hard_confirm_ring_1_920001']
    jobs=[]
    for line in lines:
        a,b=np.asarray(line['endpoint_eta'],float)
        for index in range(17):
            t=index/16;eta=((1-t)*a+t*b).tolist()
            for mode in ('FF_base_TT_corr','TT_base_FF_corr'):
                jobs.append(dict(uid=line['uid'],mode=mode,fraction=t,index=index,eta=eta,
                                 eta_uid=eta_identity(eta)[0],seed=0,
                                 controller_uid='strong_hybrid_'+hashlib.sha256(mode.encode()).hexdigest()))
    protocol=dict(lines=lines,jobs=jobs,
        selection='adaptive follow-up on five remaining originally selected Ring states; includes nonmonotone boundary lines',
        formula='Pi_x^T[u_base(0)+u_donor(eta)-u_donor(0)]')
    (HERE/'hybrid_grid_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    path=HERE/'hybrid_grid_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=j['uid'],eta_uid=j['eta_uid'],
        controller_uid=j['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for j in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'hybrid_grid_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
