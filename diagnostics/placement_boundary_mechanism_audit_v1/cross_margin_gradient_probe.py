"""Both placements' G, S and rho gradients at exactly the same Ring eta."""
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
    jobs=[];requests=[]
    for c in centers:
        target='FF' if c['chain']=='TT' else 'TT'
        ctrl=next(x for x in centers if x['uid']==c['uid'] and x['chain']==target)['controller_uid']
        for eps in (.002,.0005):
            for j in range(3):
                for sign in (-1,1):
                    eta=np.asarray(c['eta']).copy();eta[j]+=sign*eps
                    eta=eta.tolist();eid=eta_identity(eta)[0]
                    jobs.append(dict(uid=c['uid'],source_boundary_chain=c['chain'],target_chain=target,
                        epsilon=eps,coordinate=j,sign=sign,eta=eta,eta_uid=eid))
                    requests.append(dict(state_uid=c['uid'],eta_uid=eid,controller_uid=ctrl,
                        seed_keys=['{"future_root":2026100601,"seed":0}']))
    (HERE/'cross_margin_gradient_protocol.json').write_text(json.dumps(jobs,indent=2)+'\n')
    p=HERE/'cross_margin_gradient_planned_rollouts.json';p.write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(p)
    (HERE/'cross_margin_gradient_cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous']:raise RuntimeError('ambiguous preflight')
    rt=make_runtime('ring_exchange');root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    out=HERE/'cross_margin_gradient_results';out.mkdir(exist_ok=True)
    for j in jobs:
        uid=j['uid'];p=j['target_chain'];eid=j['eta_uid']
        f=out/f'{uid}_{p}_{eid}.json';npz=out/f'{uid}_{p}_{eid}.npz'
        if f.exists() and npz.exists():continue
        arrays,r=simulate(rt,states[uid],p,j['eta'],0,root,rt.config.max_steps)
        np.savez_compressed(npz,**arrays);f.write_text(json.dumps(r,indent=2)+'\n')
        print(uid,p,j['source_boundary_chain'],j['epsilon'],j['coordinate'],j['sign'],r['rho'],r['termination'],flush=True)
    summaries=[]
    for c in centers:
        uid=c['uid'];source=c['chain'];target='FF' if source=='TT' else 'TT'
        stats={}
        for p in (source,target):
            gd={};sd={};rg={};terms={}
            for eps in (.002,.0005):
                g=[];s=[];r=[];sides=[]
                for k in range(3):
                    lr=[]
                    for sign in (-1,1):
                        eta=np.asarray(c['eta']).copy();eta[k]+=sign*eps
                        eid=eta_identity(eta.tolist())[0]
                        folder=HERE/('boundary_normal_results' if p==source else 'cross_margin_gradient_results')
                        meta=json.loads((folder/f'{uid}_{p}_{eid}.json').read_text())
                        z=np.load(folder/f'{uid}_{p}_{eid}.npz')
                        lr.append((meta,z['actions'][0].ravel().copy(),z['positions'][80].ravel().copy()))
                        z.close()
                    minus,plus=lr
                    g.append((plus[1]-minus[1])/(2*eps));s.append((plus[2]-minus[2])/(2*eps))
                    r.append((plus[0]['rho']-minus[0]['rho'])/(2*eps))
                    sides.append([minus[0]['termination'],plus[0]['termination']])
                gd[eps]=np.stack(g,axis=1);sd[eps]=np.stack(s,axis=1);rg[eps]=np.asarray(r);terms[eps]=sides
            stats[p]=dict(G=gd[.0005].tolist(),S80=sd[.0005].tolist(),grad_rho=rg[.0005].tolist(),
                G_stability=float(np.linalg.norm(gd[.002]-gd[.0005])/max(np.linalg.norm(gd[.0005]),1e-12)),
                S80_stability=float(np.linalg.norm(sd[.002]-sd[.0005])/max(np.linalg.norm(sd[.0005]),1e-12)),
                rho_stability=float(np.linalg.norm(rg[.002]-rg[.0005])/max(np.linalg.norm(rg[.0005]),1e-12)),
                side_terminations=terms)
        gt=np.asarray(stats['TT']['grad_rho']);gf=np.asarray(stats['FF']['grad_rho'])
        angle=float(np.degrees(np.arccos(np.clip(gt@gf/max(np.linalg.norm(gt)*np.linalg.norm(gf),1e-12),-1,1))))
        summaries.append(dict(uid=uid,eta_source_boundary=source,eta=c['eta'],fraction=c['fraction'],
            TT=stats['TT'],FF=stats['FF'],rho_gradient_angle_deg=angle,
            relative_G_difference=float(np.linalg.norm(np.asarray(stats['FF']['G'])-np.asarray(stats['TT']['G']))/
                max(np.linalg.norm(stats['FF']['G']),1e-12)),
            relative_S80_difference=float(np.linalg.norm(np.asarray(stats['FF']['S80'])-np.asarray(stats['TT']['S80']))/
                max(np.linalg.norm(stats['FF']['S80']),1e-12))))
    (HERE/'same_eta_boundary_chain.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps([{k:x[k] for k in ('uid','eta_source_boundary','rho_gradient_angle_deg','relative_G_difference','relative_S80_difference')} for x in summaries],indent=2))


if __name__=='__main__':main()
