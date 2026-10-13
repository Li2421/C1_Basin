"""Tangent-plane follow-up on two additional regular Ring boundary pairs."""
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
    g=json.loads((HERE/'h1_gradient_analysis.json').read_text())['rows']
    source=json.loads((HERE/'h1_gradient_protocol.json').read_text())
    lines={r['uid']:r for r in json.loads((HERE/'h1_line_protocol.json').read_text())['lines']}
    centers=[];jobs=[]
    radii=(.002,.005,.01,.02)
    offsets=((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    for c in source['centers']:
        uid,chain=c['uid'],c['root_chain']
        row=next(r for r in g if r['uid']==uid and r['root_chain']==chain and r['eval_chain']==chain)
        grad=np.asarray(row['estimates']['0.0005']['rho_gradient']);_,_,vh=np.linalg.svd(grad.reshape(1,3),full_matrices=True)
        tangent=vh[1:].T
        centers.append(dict(uid=uid,chain=chain,center_eta=c['eta'],center_rho=row['center_rho'],
                            gradient=grad.tolist(),gradient_stability=row['gradient_stability'],
                            tangent=tangent.tolist(),controller_uid=lines[uid]['controllers'][chain],
                            center_signature=row['center_signature']))
        for r in radii:
            for k,(a,b) in enumerate(offsets):
                eta=(np.asarray(c['eta'])+r*tangent@np.asarray([a,b])).tolist()
                jobs.append(dict(uid=uid,chain=chain,radius=r,offset_index=k,uv=[a,b],eta=eta,
                                 eta_uid=eta_identity(eta)[0],controller_uid=lines[uid]['controllers'][chain]))
    protocol=dict(centers=centers,jobs=jobs,radii=radii,offsets=offsets,
        selection='post-hoc regular single-crossing subset UID 920004/920005; both own placement roots')
    (HERE/'curvature_followup_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    path=HERE/'curvature_followup_planned_rollouts.json'
    path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],
        controller_uid=x['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs]),indent=2)+'\n')
    pre=preflight(path);(HERE/'curvature_followup_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(jobs),pre['summary'])


if __name__=='__main__':main()
