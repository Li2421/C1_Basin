"""Prespecify closed-loop action-family swapping on two prior Ring lines."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    segments={r['uid']:r for r in json.loads((PRIOR/'boundary_protocol.json').read_text())
              if r['scene']=='ring_exchange'}
    roots=json.loads((PRIOR/'boundary_normal_protocol.json').read_text())
    jobs=[];lines=[]
    for uid in sorted(segments):
        seg=segments[uid]
        root_t=[r['fraction'] for r in roots if r['uid']==uid]
        fractions=sorted(set([0.,.5,1.]+[float(np.clip(t+o,0,1)) for t in root_t for o in (-.015,0.,.015)]))
        lines.append(dict(uid=uid,endpoint_eta=seg['endpoint_eta'],fractions=fractions,
                          source='prior boundary_protocol and boundary_normal_protocol; fixed before hybrid outcomes'))
        a,b=np.asarray(seg['endpoint_eta'],float)
        for i,t in enumerate(fractions):
            eta=((1-t)*a+t*b).tolist()
            for mode in ('FF_base_TT_corr','TT_base_FF_corr'):
                uid_ctl='strong_hybrid_'+hashlib.sha256(mode.encode()).hexdigest()
                jobs.append(dict(uid=uid,mode=mode,fraction=t,index=i,eta=eta,
                                 eta_uid=eta_identity(eta)[0],controller_uid=uid_ctl,seed=0))
    protocol=dict(lines=lines,jobs=jobs,
        formula='Pi_x^T[u_base(x,xi,0)+u_donor(x,xi,eta)-u_donor(x,xi,0)]; base and donor in {TT,FF}',
        note='Finite action-family swap, not a pure M swap; terminal projection maintains safety and its distortion is measured.')
    (HERE/'hybrid_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    path=HERE/'hybrid_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=j['uid'],eta_uid=j['eta_uid'],
        controller_uid=j['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for j in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'hybrid_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
