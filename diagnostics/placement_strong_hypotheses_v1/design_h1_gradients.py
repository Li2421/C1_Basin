"""Paired margin/trajectory probes at both boundary locations of regular lines."""
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
    roots=json.loads((HERE/'h1_roots.json').read_text())
    lines={x['uid']:x for x in json.loads((HERE/'h1_line_protocol.json').read_text())['lines']}
    selected=('hard_confirm_ring_1_920004','hard_confirm_ring_1_920005')
    jobs=[];centers=[]
    for uid in selected:
        line=lines[uid];a,b=np.asarray(line['endpoint_eta'],float)
        for root_chain in ('TT','FF'):
            root=next(r for r in roots if r['uid']==uid and r['chain']==root_chain)
            t=root['first_root_fraction'];eta=((1-t)*a+t*b).tolist()
            centers.append(dict(uid=uid,root_chain=root_chain,fraction=t,eta=eta,
                                line_direction=(b-a).tolist(),bracket=root['bracket']))
            for eval_chain in ('TT','FF'):
                jobs.append(dict(uid=uid,root_chain=root_chain,eval_chain=eval_chain,kind='center',
                                 epsilon=0.,axis=None,sign=None,eta=eta,eta_uid=eta_identity(eta)[0],
                                 controller_uid=line['controllers'][eval_chain]))
                for eps in (.0005,.002):
                    for axis in range(3):
                        for sign in (-1,1):
                            point=np.asarray(eta).copy();point[axis]+=sign*eps
                            jobs.append(dict(uid=uid,root_chain=root_chain,eval_chain=eval_chain,kind='fd',
                                epsilon=eps,axis=axis,sign=sign,eta=point.tolist(),eta_uid=eta_identity(point.tolist())[0],
                                controller_uid=line['controllers'][eval_chain]))
    (HERE/'h1_gradient_protocol.json').write_text(json.dumps(dict(centers=centers,jobs=jobs,
        selection='regular single-crossing subset UID 920004/920005 from first six Ring states; post-hoc regularity filter'),indent=2)+'\n')
    path=HERE/'h1_gradient_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],
        controller_uid=x['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'h1_gradient_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
