"""Outcome-blind check of controller identity in static P2′ features."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    p=json.loads((HERE/'P2PRIME_PROTOCOL.json').read_text())
    folder=HERE/'p2prime_features'
    data={name:np.load(folder/f'{name}.npz') for name in p['source_controllers']+p['held_controllers']}
    rows=[]
    for group in (p['source_controllers'],p['held_controllers']):
        for a,b in combinations(group,2):
            x=data[a];y=data[b]
            assert np.array_equal(x['state_uids'],y['state_uids'])
            d3=np.linalg.norm(x['R3']-y['R3'],axis=1)
            d0=np.linalg.norm(x['R4_action'][:,:8]-y['R4_action'][:,:8],axis=1)
            dj=np.linalg.norm(x['R4_action'][:,8:32]-y['R4_action'][:,8:32],axis=1)
            rows.append(dict(a=a,b=b,matched_states=len(d3),r3_mean=float(d3.mean()),r3_max=float(d3.max()),
                             projected_eta0_action_mean_distance=float(d0.mean()),
                             projected_jacobian_mean_distance=float(dj.mean())))
    out=dict(status='COMPLETED OUTCOME-BLIND',rows=rows,
             interpretation='R3 aliasing or R4 separation measures descriptor content, not prediction or closed-loop selection.')
    (HERE/'P2PRIME_STATIC_SEPARABILITY.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print([r for r in rows if (r['a'],r['b']) in (('FF','TT'),('TF','FT'))])


if __name__=='__main__':main()
