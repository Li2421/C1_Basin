"""Frozen tangent-plane radius sweep around prior Ring boundary estimates."""
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
    old=json.loads((PRIOR/'boundary_normals.json').read_text())
    jobs=[];centers=[]
    radius=(.002,.005,.01,.02)
    offsets=((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    for c in old:
        g=np.asarray(c['gradient_eps0005'],float)
        _,_,vh=np.linalg.svd(g.reshape(1,3),full_matrices=True)
        tangent=vh[1:,:].T
        centers.append(dict(uid=c['uid'],chain=c['chain'],center_eta=c['eta'],
            center_rho=c['baseline_rho'],gradient=g.tolist(),gradient_stability=c['gradient_stability'],
            tangent=tangent.tolist(),controller_uid=c['controller_uid'],
            old_goal_active_step=c['side_signatures']['0.0005'][0]['plus_goal_step']))
        for r in radius:
            for k,(a,b) in enumerate(offsets):
                eta=(np.asarray(c['eta'])+r*tangent@np.asarray([a,b])).tolist()
                jobs.append(dict(uid=c['uid'],chain=c['chain'],radius=r,offset_index=k,uv=[a,b],eta=eta,
                    eta_uid=eta_identity(eta)[0],controller_uid=c['controller_uid']))
    protocol=dict(centers=centers,jobs=jobs,radii=radius,offsets=offsets,
        selection='two prior preselected Ring states; both TT and FF own boundary centers, no angle selection',
        branch_requirement='same active margin type and active goal/safety time across each local fit')
    (HERE/'curvature_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    plan=dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],controller_uid=x['controller_uid'],
                             seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs])
    path=HERE/'curvature_planned_rollouts.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    result=preflight(path)
    (HERE/'curvature_cache_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(jobs),result['summary'])


if __name__=='__main__':main()
