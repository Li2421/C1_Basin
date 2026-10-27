"""Paired bootstrap: types -> controllers -> states; two seeds stay clustered."""
from experiment import *
import numpy as np

def bootstrap(matrix,types,iterations=5000,seed=2026102003):
    # matrix: controller, state, method; state values already average both seeds.
    groups=[np.flatnonzero(types==t) for t in sorted(set(types))];rng=np.random.default_rng(seed);out=[]
    for _ in range(iterations):
        ts=[]
        for ti in rng.integers(len(groups),size=len(groups)):
            group=groups[ti];cs=[]
            for ci in rng.choice(group,len(group),replace=True):
                values=matrix[ci];available=np.flatnonzero(np.isfinite(values).all(-1))
                if len(available):cs.append(values[rng.choice(available,len(available),replace=True)].mean(0))
            if cs:ts.append(np.mean(cs,axis=0))
        out.append(np.mean(ts,axis=0))
    draws=np.asarray(out);point=np.mean([np.nanmean(matrix[g],axis=(0,1)) for g in groups],axis=0)
    return point,draws,np.quantile(draws,[.025,.975],axis=0)

METHOD_NAMES={'safety':'只加安全投影','fixed_eta10':'全程固定修正η10','fixed_eta13':'全程固定修正η13','human_progress':'人定双η·按进度切换','learned_scene':'场景内重新学习双η与进度阈值'}

def paired(rows,qualification,scene,methods):
    valid={r['controller'] for r in qualification if r['scene']==scene and r['qualified']}
    rows=[r for r in rows if r['case']['state']['scene']==scene and r['case']['controller_payload']['nominal']['controller'] in valid and r['case']['method'] in methods]
    lookup={};typ={}
    for r in rows:
        c=r['case'];name=c['controller_payload']['nominal']['controller'];typ[name]=c['controller_payload']['nominal']['type_code'];s=c['state'];lookup[(name,s['N'],s['index'],c['seed'],c['method'])]=r
    controllers=sorted(set(k[0] for k in lookup));states=sorted(set((k[1],k[2]) for k in lookup))
    M=np.full((len(controllers),len(states),len(methods)),np.nan);upper=M.copy();harm={m:dict(known=0,upper_including_unknown=0) for m in methods[1:]};counts={m:dict(episodes=0,success=0,collision=0,timeout=0,unknown=0) for m in methods}
    for ci,ctl in enumerate(controllers):
        for si,(n,state) in enumerate(states):
            rr=[[lookup.get((ctl,n,state,seed,m)) for seed in (0,1)] for m in methods]
            if any(x is None for arm in rr for x in arm):continue
            for mi,m in enumerate(methods):
                M[ci,si,mi]=np.mean([x['success'] for x in rr[mi]]);upper[ci,si,mi]=np.mean([min(1,x['success']+x['numerical_failure']) for x in rr[mi]])
                for x in rr[mi]:
                    counts[m]['episodes']+=1
                    for field in ('success','collision','timeout'):counts[m][field]+=x[field]
                    counts[m]['unknown']+=x['numerical_failure']
                if mi:
                    harm[m]['known']+=sum(a['success'] and not b['success'] and not b['numerical_failure'] for a,b in zip(rr[0],rr[mi]))
                    harm[m]['upper_including_unknown']+=sum(a['success'] and not b['success'] for a,b in zip(rr[0],rr[mi]))
    supported=np.isfinite(M).any((1,2));M=M[supported];upper=upper[supported];controllers=[n for n,v in zip(controllers,supported) if v];types=np.asarray([typ[n] for n in controllers]);result=[]
    if not len(controllers):return []
    for label in [*sorted(set(types)),'type_equal']:
        mask=np.ones(len(types),bool) if label=='type_equal' else types==label;point,draws,ci=bootstrap(M[mask],types[mask]);up,upperdraws,upci=bootstrap(upper[mask],types[mask])
        for i,m in enumerate(methods):
            delta=draws[:,i]-draws[:,0];row=dict(scene=scene,type=label,method=m,success_rate=point[i],ci95=ci[:,i].tolist(),success_upper=up[i],upper_ci95=upci[:,i].tolist(),delta_vs_safety=point[i]-point[0],delta_ci95=np.quantile(delta,[.025,.975]).tolist(),controllers=int(mask.sum()),evidence='DEVELOPMENT')
            row['identified_delta_bounds']=[float(point[i]-up[0]),float(up[i]-point[0])]
            row['unknown_sensitive_delta_ci95']=[float(np.quantile(draws[:,i]-upperdraws[:,0],.025)),float(np.quantile(upperdraws[:,i]-draws[:,0],.975))]
            if m=='learned_scene':
                row['delta_vs_fixed']={}
                for fixedname in ('fixed_eta10','fixed_eta13','human_progress'):
                    if fixedname in methods:
                        fixed=methods.index(fixedname);row['delta_vs_fixed'][fixedname]=dict(delta=float(point[i]-point[fixed]),ci95=np.quantile(draws[:,i]-draws[:,fixed],[.025,.975]).tolist())
            result.append(row)
    return dict(statistics=result,pooled_counts=counts,harms_vs_safety=harm,methods=methods,paired_states=states,qualified_controllers=controllers)
