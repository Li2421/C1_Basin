"""Outcome-blind full-panel QMC refinement after 64-point discrepancy check."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    p=json.loads((HERE/'protocol.json').read_text())
    lo=np.asarray(p['eta_box']['lower']);hi=np.asarray(p['eta_box']['upper'])
    eta=qmc.scale(qmc.Sobol(d=3,scramble=True,seed=20261009).random_base2(m=7),lo,hi)
    ctl={(c['uid'],c['chain']):c['controller_uid'] for c in p['cases']}
    cases=[]
    for st in p['states']:
        for i,x in enumerate(eta):
            for ch in ('TT','FF'):
                val=x.tolist();cases.append(dict(uid=st['uid'],scenario=st['scenario'],chain=ch,
                    kind='sobol128b',index=i,eta=val,eta_uid=eta_identity(val)[0],
                    controller_uid=ctl[(st['uid'],ch)],seed=0))
    q=dict(states=p['states'],cases=cases,future_root=p['future_root'],eta_box=p['eta_box'],
        selection='all 12 originally frozen states; independent scrambled Sobol128 seed 20261009 after coarse integration discrepancy',
        use='numerical quadrature sensitivity, not a new state selection')
    (HERE/'sobol128_protocol.json').write_text(json.dumps(q,indent=2)+'\n')
    path=HERE/'sobol128_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],
        controller_uid=x['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for x in cases]),indent=2)+'\n')
    pre=preflight(path);(HERE/'sobol128_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(cases),pre['summary'])


if __name__=='__main__':main()
