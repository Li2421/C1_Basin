"""Resolution follow-up on the first frozen Ring and Toy state."""
import json
import sys
from itertools import product
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    p=json.loads((HERE/'protocol.json').read_text())
    lo=np.asarray(p['eta_box']['lower']);hi=np.asarray(p['eta_box']['upper'])
    states=[next(x for x in p['states'] if x['scenario']==sc)
            for sc in ('ring_exchange','toy_give_way')]
    ctl={(c['uid'],c['chain']):c['controller_uid'] for c in p['cases']}
    jobs=[]
    for st in states:
        for ix in product(range(9),repeat=3):
            if all(j%2==0 for j in ix):continue
            eta=[float(lo[j]+ix[j]*(hi[j]-lo[j])/8) for j in range(3)]
            for ch in ('TT','FF'):
                jobs.append(dict(uid=st['uid'],scene=st['scenario'],index=list(ix),chain=ch,eta=eta,
                                 eta_uid=eta_identity(eta)[0],controller_uid=ctl[(st['uid'],ch)]))
    q=dict(states=states,jobs=jobs,box=p['eta_box'],
           selection='first frozen state per scenario; 9^3 nested follow-up, 5^3 even-index points reused',
           reason='check 3^3/5^3 cubical topology and volume resolution stability')
    (HERE/'grid9_protocol.json').write_text(json.dumps(q,indent=2)+'\n')
    path=HERE/'grid9_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],
        controller_uid=x['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'grid9_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
