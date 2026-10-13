"""Test first-order boundary-shift predictions at the other placement's root."""
import json
import sys
import numpy as np
from closed_loop_probe import HERE,HARD,make_runtime,simulate

sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    centers=json.loads((HERE/'boundary_normal_protocol.json').read_text())
    states={s['uid']:s for s in json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']}
    normals=json.loads((HERE/'boundary_normals.json').read_text())
    jobs=[];req=[]
    for uid in sorted({c['uid'] for c in centers}):
        cs={c['chain']:c for c in centers if c['uid']==uid}
        for target in ('TT','FF'):
            other='FF' if target=='TT' else 'TT';c=cs[other]
            eid=eta_identity(c['eta'])[0]
            jobs.append(dict(state_uid=uid,chain=target,source_boundary_chain=other,eta=c['eta'],eta_uid=eid))
            req.append(dict(state_uid=uid,eta_uid=eid,controller_uid=cs[target]['controller_uid'],
                seed_keys=['{"future_root":2026100601,"seed":0}']))
    p=HERE/'cross_boundary_planned_rollouts.json';p.write_text(json.dumps({'requests':req},indent=2)+'\n')
    cache=preflight(p);(HERE/'cross_boundary_cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous']:raise RuntimeError('ambiguous preflight')
    out=HERE/'cross_boundary_results';out.mkdir(exist_ok=True)
    rt=make_runtime('ring_exchange');root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    rows=[]
    for j in jobs:
        uid=j['state_uid'];chain=j['chain'];eid=j['eta_uid']
        f=out/f'{uid}_{chain}_{eid}.json';npz=out/f'{uid}_{chain}_{eid}.npz'
        if f.exists() and npz.exists():r=json.loads(f.read_text())
        else:
            arrays,r=simulate(rt,states[uid],chain,j['eta'],0,root,rt.config.max_steps)
            np.savez_compressed(npz,**arrays);f.write_text(json.dumps(r,indent=2)+'\n')
        c=next(x for x in centers if x['uid']==uid and x['chain']==chain)
        n=next(x for x in normals if x['uid']==uid and x['chain']==chain)
        seg=next(x for x in json.loads((HERE/'boundary_protocol.json').read_text()) if x['uid']==uid)
        tangent=np.asarray(seg['endpoint_eta'][1])-np.asarray(seg['endpoint_eta'][0])
        slope=float(np.asarray(n['gradient_eps0005'])@tangent)
        source_t=next(x for x in centers if x['uid']==uid and x['chain']==j['source_boundary_chain'])['fraction']
        predicted_t=source_t-r['rho']/slope if abs(slope)>1e-12 else None
        rows.append(dict(**j,other_placement_rho=r['rho'],other_placement_term=r['termination'],
            own_boundary_t=c['fraction'],source_boundary_t=source_t,
            own_local_slope=slope,predicted_root_from_other_point=predicted_t))
    (HERE/'cross_boundary_shift_predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
