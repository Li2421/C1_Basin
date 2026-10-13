"""Frozen-method DEVELOPMENT endpoints, paired unknown bounds and eligibility."""
from util import *
from scipy.optimize import minimize
from scipy.stats import spearmanr

POLICIES=['argmax','pessimistic1','pessimistic0.5','conservative']
def choose(logits,policy,src):
    if policy=='argmax':return logits.argmax(-1)
    q=expit(logits);mu=q.mean(0);sd=q.std(0)
    if policy.startswith('pessimistic'):
        k=float(policy[len('pessimistic'):]);a=(mu-k*sd).argmax(-1)
    else:
        winners=q.argmax(-1);candidate=mu.argmax(-1)
        agreement=(winners==candidate[None]).all(0)
        gain=np.take_along_axis(mu,candidate[...,None],-1)[...,0]-mu[...,src]
        a=np.where(agreement&(gain>=.10),candidate,src)
    return np.tile(a[None],(3,1,1))
def picked(x,a):return np.take_along_axis(x,a[...,None],-1)[...,0]
def contrast(Y,U,a,b):
    ya=picked(Y,a);yb=picked(Y,b);ua=picked(U,a);ub=picked(U,b);different=a!=b
    return ya-yb-ub*different,ya+ua*different-yb
def bootstrap(lo,hi,draws):
    lo=np.asarray(lo);hi=np.asarray(hi);bl=lo[draws].mean(1);bh=hi[draws].mean(1)
    return [float(np.quantile(bl,.025)),float(np.quantile(bh,.975))]
def simultaneous(lo,hi,draws):
    lower=np.array(lo).T;upper=np.array(hi).T;M=np.c_[lower,upper];obs=M.mean(0);boot=M[draws].mean(1);se=boot.std(0,ddof=1)
    t=np.abs((boot-obs)/np.where(se>1e-12,se,1));crit=float(np.quantile(t.max(1),.95));n=len(lo)
    return np.c_[obs[:n]-crit*se[:n],obs[n:]+crit*se[n:]],crit
def hull(points,target):
    points=np.asarray(points,float);target=np.asarray(target,float);delta=points-target;N=len(points)
    def fn(w):d=w@delta;return .5*d@d,delta@d
    res=minimize(fn,np.full(N,1/N),jac=True,bounds=[(0,1)]*N,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(N)},method='SLSQP',options={'maxiter':5000,'ftol':1e-10})
    valid=res.success and abs(res.x.sum()-1)<1e-6 and res.x.min()>-1e-7
    return dict(nearest=float(np.linalg.norm(delta,axis=1).min()),convex_hull=float(np.linalg.norm(res.x@delta)) if valid else None,solver_success=bool(valid),solver_message=str(res.message),iterations=int(res.nit),weights=res.x.tolist())
def distance(data,arm,n,folder):
    if arm=='B0':return dict(nearest=0.,convex_hull=0.,solver_success=True,meaning='no controller coordinates')
    if arm in ('B1','B2'):
        if data.bed=='blend':
            f=OldFeatures(data.p5,'R3' if arm=='B1' else 'R_oracle')
            co={c:f.descriptor(c,data.common_states).mean(0) for c in data.sources+[n]}
            norm=np.load(folder/'normalization.npz');mean=norm['mean'][23:];scale=norm['scale'][23:]
        else:
            co={c:np.r_[data.means[c],np.zeros(64)] for c in data.sources+[n]};norm=np.load(folder/'normalization.npz');mean=norm['mean'][23:];scale=norm['scale'][23:]
    else:
        kind='F1_PCA2' if arm=='F4' else arm
        co={c:data.coordinates(c,kind) for c in data.sources+[n]};size=len(co[n])
        if arm=='F4':
            norm=np.load(folder/'classifier.npz');mean=norm['mean'];scale=norm['scale']
        else:
            norm=np.load(folder/'normalization.npz');mean=norm['mean'][23:23+size];scale=norm['scale'][23:23+size]
    return hull(np.array([(co[c]-mean)/scale for c in data.sources]),(co[n]-mean)/scale)
def method_predictions(t):
    result={}
    for arm,old in [('B0','R0'),('B1','R3'),('B2','R_oracle')]:
        if t['bed']=='real' and arm=='B2':continue
        folder=(R5/'stage_2/fits'/f"LOPO_fold{t['fold']}_{t['coverage']}_{old}") if t['bed']=='blend' else R4/'fits'/f"fold{t['fold']}_{old}"
        logits=np.load(folder/'predictions.npz')['held']
        if t['bed']=='real':logits=logits[:,None]
        result[arm]=(logits,None,folder)
    for arm in NEW_ARMS:
        folder=HERE/'fits'/task_name(dict(t,arm=arm))
        if (folder/'complete.json').exists():result[arm]=(np.load(folder/'predictions.npz')['held'],None,folder)
    folder=HERE/'logistic'/task_name(dict(t,arm='F4'))
    if (folder/'complete.json').exists():result['F4']=(None,np.load(folder/'predictions.npz')['choices'],folder)
    return result
def endpoints(bed):
    if bed=='blend':
        p=read(R5/'stage_2/protocol.json');out=np.load(R5/'stage_2/TEST_ENDPOINT_LABELS.npz')['outcomes'];disc=np.load(R5/'stage_2/TEST_DISCOVERY_LABELS.npz')['outcomes'];names=p['controller_order'];gates=read(R5/'stage_2/GATE_RESULTS.json')['rows']
    else:
        p=read(R4/'protocol.json');lab=np.load(R4/'LABELS.npz')['outcomes'];out=lab[:,128:,:,4:8];disc=lab[:,128:,:,:4];names=p['controller_order'];gates=read(R4/'GATE_RESULTS.json')['rows']
    fixed={n:int(Data.curve(disc[i]).argmax()) for i,n in enumerate(names)}
    return (out==1).mean(-1),(out==-1).mean(-1),names,fixed,gates
def main(bed='blend',gate_only=False):
    Y,U,names,fixed,gates=endpoints(bed);NS=Y.shape[1];draws=np.random.default_rng(20261013061 if bed=='blend' else 20261013062).integers(NS,size=(5000,NS))
    rows=[];vectors={};all_head=[];distances=[];distance_cache={}
    for t in task_list(bed):
        data=Data(t);methods=method_predictions(t);r0=methods['B0'][0].argmax(-1)
        for arm,(logits,direct,folder) in methods.items():
            policies=['argmax'] if arm in ('B0','F4') else POLICIES
            for policy in policies:
                choice=direct if direct is not None else choose(logits,policy,data.src);method=arm+'__'+policy
                key=(t['coverage'],method)
                for j,n in enumerate(t['held']):
                    idx=names.index(n);y=Y[idx][None];u=U[idx][None];a=choice[:,j];b0=r0[:,j];src=np.full_like(a,data.src);target=np.full_like(a,fixed[n])
                    success=picked(y,a).mean(0);unknown=picked(u,a).mean(0);lo0,hi0=contrast(y,u,a,b0);los,his=contrast(y,u,a,src);lot,hit=contrast(y,u,target,src)
                    necessary=next(g['necessary'] for g in gates if g['controller']==n and (bed=='real' or (g['split']=='LOPO' and g['coverage']==t['coverage'])))
                    vectors.setdefault(key,{})[n]=dict(e1lo=lo0.mean(0),e1hi=hi0.mean(0),srclo=los.mean(0),srchi=his.mean(0),gaplo=lot.mean(0),gaphi=hit.mean(0),selected=success,unknown=unknown,necessary=necessary,source_success=Y[idx,:,data.src],target_success=Y[idx,:,fixed[n]],choice=a)
                    if logits is not None:
                        q=expit(logits[:,j]);predicted=picked(q,a).mean(0)
                        valid_actual=picked(np.divide(y,1-u,out=np.full_like(y,np.nan),where=u<1),a)
                        optimism=dict(mean_predicted=float(predicted.mean()),observed_valid_Q=float(np.nanmean(valid_actual)),overestimate_lower=float((predicted-success-unknown).mean()),overestimate_upper=float((predicted-success).mean()),ci95_conservative=bootstrap(predicted-success-unknown,predicted-success,draws))
                    else:optimism=dict(status='NOT APPLICABLE: regime probability is not predicted success Q')
                    hist=np.bincount(a.ravel(),minlength=16)/a.size;heads=[]
                    for eta in range(16):
                        labels=np.concatenate([data.lab[c][:,eta].ravel() for c in data.sources]);request=labels!=-2;valid=(labels==0)|(labels==1)
                        heads.append(dict(eta=eta,selection_probability=float(hist[eta]),requested=int(request.sum()),valid=int(valid.sum()),unknown=int((labels==-1).sum()),source_success=float((labels==1).sum()/valid.sum()) if valid.sum() else None,equal_controller_source_Q=float(np.mean([data.curves[c][eta] for c in data.sources])),controllers_with_valid_labels=int(sum(np.any(data.lab[c][:,eta]>=0) for c in data.sources))))
                    # Distance is policy-independent and computed once per arm.
                    geometry_key=(t['coverage'],t['fold'],arm,n)
                    if geometry_key not in distance_cache:distance_cache[geometry_key]=distance(data,arm,n,folder)
                    dd=distance_cache[geometry_key];distances.append(dict(bed=bed,coverage=t['coverage'],fold=t['fold'],arm=arm,method=method,policy=policy,controller=n,relative_R0_loss_lower=float(-hi0.mean()),relative_R0_loss_upper=float(-lo0.mean()),**dd))
                    all_head.append(dict(coverage=t['coverage'],controller=n,method=method,heads=heads))
                    rows.append(dict(coverage=t['coverage'],fold=t['fold'],controller=n,arm=arm,policy=policy,method=method,necessary=necessary,source_prior_eta=data.src,target_fixed_eta=fixed[n],success_lower=float(success.mean()),success_upper=float((success+unknown).mean()),gain_R0_lower=float(lo0.mean()),gain_R0_upper=float(hi0.mean()),gain_src_lower=float(los.mean()),gain_src_upper=float(his.mean()),choice_histogram=hist.tolist(),D2=optimism,catastrophe_certain=bool(his.mean()<=-.10+1e-12),catastrophe_possible=bool(los.mean()<=-.10+1e-12),catastrophe_point_lower=bool((success.mean()-Y[idx,:,data.src].mean())<=-.10+1e-12)))
    summaries=[];e1lo=[];e1hi=[]
    for (coverage,method),vv in vectors.items():
        values=list(vv.values());lo=np.mean([v['e1lo'] for v in values],0);hi=np.mean([v['e1hi'] for v in values],0);e1lo.append(lo);e1hi.append(hi)
        rr=[r for r in rows if r['coverage']==coverage and r['method']==method];needed=[v for v in values if v['necessary']]
        def effect(label):
            l=np.mean([v[label+'lo'] for v in needed],0);h=np.mean([v[label+'hi'] for v in needed],0)
            return dict(lower=float(l.mean()),upper=float(h.mean()),ci95_descriptive=bootstrap(l,h,draws))
        e2src=effect('src');e2r0=effect('e1');num=np.mean([v['srclo'] for v in needed],0);den=np.mean([v['gaplo'] for v in needed],0)
        nb=num[draws].mean(1);db=den[draws].mean(1);ratio=float(num.mean()/den.mean()) if den.mean()>0 else None
        unstable=int((db<=0).sum());ratio_ci=np.quantile(nb/db,[.025,.975]).tolist() if ratio is not None and not unstable else None
        summaries.append(dict(coverage=coverage,method=method,arm=rr[0]['arm'],policy=rr[0]['policy'],controller_count=len(rr),success_lower=float(np.mean([r['success_lower'] for r in rr])),success_upper=float(np.mean([r['success_upper'] for r in rr])),E1=dict(mean_lower=float(lo.mean()),mean_upper=float(hi.mean()),margin=-.02),E2=dict(necessary_count=len(needed),vs_src=e2src,vs_R0=e2r0,gap_closed_ratio_lower_point=ratio,gap_closed_ratio_ci95_descriptive=ratio_ci,ratio_bootstraps_nonpositive_denominator=unstable,target_fixed_gap_lower=float(den.mean())),E3=dict(certain_count=sum(r['catastrophe_certain'] for r in rr),possible_count=sum(r['catastrophe_possible'] for r in rr),point_count=sum(r['catastrophe_point_lower'] for r in rr),possible_rate=sum(r['catastrophe_possible'] for r in rr)/len(rr))))
    ci,crit=simultaneous(e1lo,e1hi,draws)
    for s,interval in zip(summaries,ci):
        s['E1'].update(ci95_simultaneous=interval.tolist(),passed=bool(interval[0]>-.02))
        s['eligible']=s['E1']['passed'] and s['E3']['possible_count']==0
        s['positive_transfer_candidate']=s['eligible'] and s['E2']['vs_src']['ci95_descriptive'][0]>0 and s['E2']['vs_R0']['ci95_descriptive'][0]>0
    repair=[s for s in summaries if s['arm'] not in ('B0','B1','B2') or s['policy']!='argmax']
    if bed=='real':
        eligible=read(HERE/'STAGE2_ELIGIBILITY.json')['eligible_methods']
        for s in summaries:s['planned_real_primary']=s['arm'] in ('B0','B1') or s['policy'] in eligible.get(s['arm'],[])
        repair=[s for s in repair if s['planned_real_primary']]
    baseline_disaster=any(s['E3']['certain_count']>0 for s in summaries if s['arm'] in ('B1','B2') and s['policy']=='argmax')
    safe=[s for s in repair if s['eligible']];positive=[s for s in safe if s['positive_transfer_candidate']]
    verdict='SUPPORTED: operational repair eliminates harm' if baseline_disaster and safe else 'REJECTED AS SUFFICIENT REPAIR UNDER THIS DESIGN' if baseline_disaster else 'INCONCLUSIVE: baseline harm not established'
    correlations=[]
    for coverage in sorted({r['coverage'] for r in distances}):
        for method in sorted({r['method'] for r in distances}):
            rr=[r for r in distances if r['coverage']==coverage and r['method']==method];loss=np.array([r['relative_R0_loss_upper'] for r in rr])
            for metric in ('nearest','convex_hull'):
                pairs=[(r[metric],r['relative_R0_loss_upper']) for r in rr if r.get(metric) is not None];x=np.array([r[0] for r in pairs]);y=np.array([r[1] for r in pairs])
                valid=len(x)>2 and np.std(x)>1e-9 and np.std(y)>1e-9
                correlations.append(dict(coverage=coverage,method=method,metric=metric,n=len(x),pearson=float(np.corrcoef(x,y)[0,1]) if valid else None,spearman=float(spearmanr(x,y).statistic) if valid else None,scope='descriptive; target controllers/fold-scalings and states are dependent'))
    result=dict(status='COMPLETE',bed=bed,evidence_level='DEVELOPMENT',endpoint_seeds=[2,3] if bed=='blend' else [4,5,6,7],summaries=summaries,rows=rows,D1=distances,D1_correlations=correlations,D3=all_head,H_method=verdict,positive_transfer_candidates=[dict(coverage=s['coverage'],method=s['method']) for s in positive],safe_repairs=[dict(coverage=s['coverage'],method=s['method']) for s in safe],bootstrap_draws=5000,bootstrap_unit='same physical state resampled jointly across all controllers, fits averaged',E1_maxT_critical=crit,E1_family='all methods and coverage settings evaluated in this testbed',new_rollouts=0,shared_db_writes=0)
    write(HERE/(bed+'_results.json'),result)
    if bed=='blend':
        eligible={}
        for s in safe:eligible.setdefault(s['arm'],[]).append(s['policy'])
        eligible={k:sorted(set(v)) for k,v in eligible.items()};real_arms=[a for a in eligible if a in NEW_ARMS and 'oracle_low' not in a]
        realtasks=[dict(t,arm=a) for t in task_list('real') for a in real_arms];write(HERE/'FIT_TASKS_REAL.json',realtasks)
        freeze=dict(utc=utc(),stage1_result_sha256=sha(HERE/'blend_results.json'),eligible_methods=eligible,fit_architectures=real_arms,real_fit_tasks=len(realtasks),oracle_low_status='INAPPLICABLE: target real Flow regime is not available without target labels',F3_B1_real_unconditional=True,regime_logistic_real_execute='F4' in eligible,selection_not_target_tuning=True)
        if (HERE/'STAGE2_ELIGIBILITY.json').exists():
            prev=read(HERE/'STAGE2_ELIGIBILITY.json');assert prev['eligible_methods']==eligible and prev['fit_architectures']==real_arms
        else:write(HERE/'STAGE2_ELIGIBILITY.json',freeze)
    print(json.dumps(dict(bed=bed,H_method=verdict,safe=len(safe),positive=len(positive),methods=len(summaries))))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'blend')
