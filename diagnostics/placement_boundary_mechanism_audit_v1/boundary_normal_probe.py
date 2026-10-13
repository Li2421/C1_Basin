"""Paired Ring boundary-normal probes with exact-eta shared-cache preflight."""
import json
import sys
from pathlib import Path
import numpy as np
from closed_loop_probe import HERE,HARD,make_runtime,simulate

sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def brackets(rows):
    pairs=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if (a['rho']>0)!=(b['rho']>0)]
    if len(pairs)!=1:raise RuntimeError('nonunique outcome bracket')
    return pairs[0]


def main():
    states=json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']
    jobs=[];centers=[]
    for st in states:
        if st['scenario']!='ring_exchange':continue
        d=json.loads((HERE/'boundary_results'/f"{st['uid']}_refined.json").read_text());seg=d['segment']
        for chain in ('TT','FF'):
            a,b=brackets(d['placements'][chain]);r1,r2=a['rho'],b['rho']
            t=(a['fraction']*abs(r2)+b['fraction']*abs(r1))/(abs(r1)+abs(r2))
            eta=((1-t)*np.asarray(seg['endpoint_eta'][0])+t*np.asarray(seg['endpoint_eta'][1])).tolist()
            centers.append(dict(uid=st['uid'],chain=chain,scene=st['scenario'],fraction=t,eta=eta,
                bracket=[a['fraction'],b['fraction']],bracket_rho=[r1,r2],controller_uid=seg['controller_uid'][chain]))
            jobs.append(dict(state=st,chain=chain,kind='center',epsilon=0.,j=None,sign=None,eta=eta))
            for eps in (.002,.0005):
                for j in range(3):
                    for sign in (-1,1):
                        x=np.asarray(eta).copy();x[j]+=sign*eps
                        jobs.append(dict(state=st,chain=chain,kind='fd',epsilon=eps,j=j,sign=sign,eta=x.tolist()))
    (HERE/'boundary_normal_protocol.json').write_text(json.dumps(centers,indent=2)+'\n')
    ctrl={(c['uid'],c['chain']):c['controller_uid'] for c in centers}
    plan={'requests':[dict(state_uid=x['state']['uid'],eta_uid=eta_identity(x['eta'])[0],
            controller_uid=ctrl[(x['state']['uid'],x['chain'])],seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs]}
    p=HERE/'boundary_normal_planned_rollouts.json';p.write_text(json.dumps(plan,indent=2)+'\n')
    cache=preflight(p)
    (HERE/'boundary_normal_cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous']:raise RuntimeError('ambiguous preflight')
    future_root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    out=HERE/'boundary_normal_results';out.mkdir(exist_ok=True)
    for sc in ('ring_exchange',):
        rt=make_runtime(sc)
        for x in [a for a in jobs if a['state']['scenario']==sc]:
            st=x['state'];chain=x['chain'];eid=eta_identity(x['eta'])[0]
            meta=out/f"{st['uid']}_{chain}_{eid}.json"
            npz=out/f"{st['uid']}_{chain}_{eid}.npz"
            if meta.exists() and npz.exists():continue
            arrays,r=simulate(rt,st,chain,x['eta'],0,future_root,rt.config.max_steps)
            np.savez_compressed(npz,**arrays)
            r.update(kind=x['kind'],epsilon=x['epsilon'],coordinate=x['j'],sign=x['sign'],eta_uid=eid)
            meta.write_text(json.dumps(r,indent=2)+'\n')
            print(st['uid'],chain,x['kind'],x['epsilon'],x['j'],x['sign'],r['rho'],r['termination'],flush=True)
    rows=[]
    for c in centers:
        uid,chain=c['uid'],c['chain']
        def record(eta):
            eid=eta_identity(eta)[0]
            return json.loads((out/f"{uid}_{chain}_{eid}.json").read_text())
        base=record(c['eta']);grads={};sigs={}
        for eps in (.002,.0005):
            g=[];sign=[]
            for j in range(3):
                m=np.asarray(c['eta']).copy();m[j]-=eps
                p=np.asarray(c['eta']).copy();p[j]+=eps
                rm,rp=record(m),record(p)
                g.append((rp['rho']-rm['rho'])/(2*eps))
                sign.append(dict(minus_term=rm['termination'],plus_term=rp['termination'],
                    minus_rho=rm['rho'],plus_rho=rp['rho'],
                    minus_goal_step=rm['goal_active_step'],plus_goal_step=rp['goal_active_step']))
            grads[eps]=np.asarray(g);sigs[eps]=sign
        stability=float(np.linalg.norm(grads[.002]-grads[.0005])/max(np.linalg.norm(grads[.0005]),1e-12))
        rows.append(dict(**c,baseline_rho=base['rho'],baseline_term=base['termination'],
            gradient_eps002=grads[.002].tolist(),gradient_eps0005=grads[.0005].tolist(),
            gradient_stability=stability,side_signatures=sigs))
    (HERE/'boundary_normals.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('uid','chain','fraction','baseline_rho','gradient_eps0005','gradient_stability')} for r in rows],indent=2))


if __name__=='__main__':main()
