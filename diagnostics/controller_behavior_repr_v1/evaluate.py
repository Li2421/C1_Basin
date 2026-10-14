"""Frozen endpoint evaluation; cannot train or change any selection rule."""
from core import *
from data import Data,name
from scipy.special import expit

PRIMARY=['pessimistic1','conservative','F4']
def bootstrap(lo,hi,draws):return [float(np.quantile(lo[draws].mean(1),.025)),float(np.quantile(hi[draws].mean(1),.975))]
def simultaneous(lo,hi,draws):
    lower=np.array(lo).T;upper=np.array(hi).T;M=np.c_[lower,upper];obs=M.mean(0);boot=M[draws].mean(1);se=boot.std(0,ddof=1);t=np.abs((boot-obs)/np.where(se>1e-12,se,1));critical=float(np.quantile(t.max(1),.95));n=len(lo)
    return np.c_[obs[:n]-critical*se[:n],obs[n:]+critical*se[n:]],critical

def endpoints(bed):
    if bed=='real':
        p=read(R4/'protocol.json');l=np.load(R4/'LABELS.npz')['outcomes'][:,128:,:,4:8];g=read(R4/'GATE_RESULTS.json')['rows']
    else:
        p=read(R5/'stage_2/protocol.json');l=np.load(R5/'stage_2/TEST_ENDPOINT_LABELS.npz')['outcomes'];g=read(R5/'stage_2/GATE_RESULTS.json')['rows']
    return (l==1).mean(-1),(l==-1).mean(-1),p['controller_order'],g

def main(bed,sens=False):
    gate();ts=[t for t in read(HERE/'TASKS.json') if t['bed']==bed and t['sensitivity']==sens];assert ts
    assert all((HERE/'fits'/name(t)/'complete.json').exists() for t in ts)
    Y,U,names,gates=endpoints(bed);ns=Y.shape[1];seed=20261013072 if bed=='real' else 20261013073;draws=np.random.default_rng(seed).integers(ns,size=(5000,ns));vectors={};rows=[];seen=set();baseline_seen=set()
    def add(t,rep,policy,choice,logits,d,r0,primary=True):
        coverage=t['coverage'];key=f'{coverage}|{rep}|{policy}';vv=vectors.setdefault(key,{})
        for j,n in enumerate(t['held']):
            assert n not in vv,(key,n)
            idx=names.index(n);y=Y[idx][None];u=U[idx][None];a=choice[:,j];b=r0[:,j];src=np.full_like(a,d.src);lo0,hi0=contrast(y,u,a,b);los,his=contrast(y,u,a,src);success=picked(y,a).mean(0);unknown=picked(u,a).mean(0)
            necessary=next(g['necessary'] for g in gates if g['controller']==n and (bed=='real' or (g['split']=='LOPO' and g['coverage']==coverage)))
            vv[n]=dict(e1lo=lo0.mean(0),e1hi=hi0.mean(0),srclo=los.mean(0),srchi=his.mean(0),selected=success,unknown=unknown,necessary=necessary,choice=a,source_eta=d.src)
            qmetrics=None
            if logits is not None:
                z=logits[:,j];q=expit(z);obs=np.divide(y,1-u,out=np.zeros_like(y),where=u<1);valid=u<1
                mse=float(np.sum((q-obs)**2*valid)/(3*valid.sum()));nll=float(np.sum(y*np.logaddexp(0,-z)+(1-u-y)*np.logaddexp(0,z))/(3*(1-u).sum()));pred=picked(q,a).mean(0)
                qmetrics=dict(valid_head_MSE=mse,valid_seed_NLL=nll,selected_mean_predicted_Q=float(pred.mean()),selected_optimism_lower=float((pred-success-unknown).mean()),selected_optimism_upper=float((pred-success).mean()))
            rows.append(dict(key=key,bed=bed,coverage=coverage,representation=rep,policy=policy,primary=primary,fold=t['fold'],controller=n,necessary=necessary,source_prior_eta=d.src,success_lower=float(success.mean()),success_upper=float((success+unknown).mean()),gain_R0_lower=float(lo0.mean()),gain_R0_upper=float(hi0.mean()),gain_src_lower=float(los.mean()),gain_src_upper=float(his.mean()),gain_src_ci95_descriptive=bootstrap(los.mean(0),his.mean(0),draws),gain_R0_ci95_descriptive=bootstrap(lo0.mean(0),hi0.mean(0),draws),choice_histogram=(np.bincount(a.ravel(),minlength=16)/a.size).tolist(),catastrophe_possible=bool(los.mean()<=-.10+1e-12),catastrophe_certain=bool(his.mean()<=-.10+1e-12),Q_prediction=qmetrics))
    for t in ts:
        d=Data(t);pred=np.load(HERE/'fits'/name(t)/'predictions.npz')['held'];b=np.load(r0_folder(dict(t,coverage='LOCO' if bed=='real' else t['coverage']))/'predictions.npz')['held'];b=b[:,None] if bed=='real' else b;r0=b.argmax(-1)
        for policy in ['pessimistic1','conservative','argmax','pessimistic0.5']:add(t,t['representation'],policy,choose(pred,policy,d.src),pred,d,r0,policy in PRIMARY)
        c=np.load(HERE/'classifiers'/name(t)/'predictions.npz')['choices'];add(t,t['representation'],'F4',c,None,d,r0)
        special=(t['coverage'],t['fold'])
        if special not in seen:
            seen.add(special);p=np.load(HERE/'classifiers'/name(t)/'P_rule.npz')['choices'];add(t,'P_rule','P_rule',p,None,d,r0)
            add(t,'B0','argmax',r0,b,d,r0,False);add(t,'source_prior','fixed',np.full_like(r0,d.src),None,d,r0,False)
    # P_rule is the same fixed action under both coverages when prior is unchanged.
    pkeys=[k for k in vectors if '|P_rule|' in k];p_equivalent=len(pkeys)==2 and all(np.array_equal(vectors[pkeys[0]][n]['choice'],vectors[pkeys[1]][n]['choice']) for n in names)
    p_contrast_equivalent=p_equivalent and all(np.array_equal(vectors[pkeys[0]][n][k],vectors[pkeys[1]][n][k]) for n in names for k in ('e1lo','e1hi'))
    summaries=[];store_arrays={};key_to_index={}
    for key,vv in vectors.items():
        vals=[vv[n] for n in names];needed=[v for v in vals if v['necessary']];rr=[r for r in rows if r['key']==key];assert len(rr)==len(names)
        e1lo=np.mean([v['e1lo'] for v in vals],0);e1hi=np.mean([v['e1hi'] for v in vals],0);store_arrays[key]=dict(e1lo=e1lo,e1hi=e1hi)
        e2={}
        for label,base in [('src','vs_src'),('e1','vs_R0')]:
            lo=np.mean([v[label+'lo'] for v in needed],0);hi=np.mean([v[label+'hi'] for v in needed],0);store_arrays[key][base]=(lo,hi);e2[base]=dict(lower=float(lo.mean()),upper=float(hi.mean()),ci95_descriptive=bootstrap(lo,hi,draws))
        s=dict(key=key,coverage=rr[0]['coverage'],representation=rr[0]['representation'],policy=rr[0]['policy'],primary=rr[0]['primary'],controller_count=len(rr),success_lower=float(np.mean([r['success_lower'] for r in rr])),success_upper=float(np.mean([r['success_upper'] for r in rr])),E1=dict(mean_lower=float(e1lo.mean()),mean_upper=float(e1hi.mean()),ci95_descriptive=bootstrap(e1lo,e1hi,draws),margin=-.02),E2=dict(necessary_count=len(needed),**e2),E3=dict(possible_count=sum(r['catastrophe_possible'] for r in rr),certain_count=sum(r['catastrophe_certain'] for r in rr),possible_rate=sum(r['catastrophe_possible'] for r in rr)/len(rr)),Q_prediction=dict(mean_valid_head_MSE=float(np.mean([r['Q_prediction']['valid_head_MSE'] for r in rr])),mean_valid_seed_NLL=float(np.mean([r['Q_prediction']['valid_seed_NLL'] for r in rr]))) if rr[0]['Q_prediction'] else None)
        summaries.append(s);key_to_index[key]=len(summaries)-1
    primary=[s for s in summaries if s['primary']];los=[store_arrays[s['key']]['e1lo'] for s in primary];his=[store_arrays[s['key']]['e1hi'] for s in primary]
    ci,critical=simultaneous(los,his,draws)
    for s,c in zip(primary,ci):s['E1'].update(ci95_simultaneous=c.tolist(),passed=bool(c[0]>-.02))
    family_records=[]
    for cov in dict.fromkeys(s['coverage'] for s in primary):
        for family in ('R3','R5','P_rule'):
            members=[s for s in primary if s['coverage']==cov and (s['representation'].startswith(family) if family!='P_rule' else s['representation']=='P_rule')]
            if not members:continue
            pairs=[(s,base) for s in members for base in ('vs_src','vs_R0')];lo=[store_arrays[s['key']][base][0] for s,base in pairs];hi=[store_arrays[s['key']][base][1] for s,base in pairs];cis,crit=simultaneous(lo,hi,draws)
            for (s,base),c in zip(pairs,cis):s['E2'][base]['ci95_family_simultaneous']=c.tolist()
            family_records.append(dict(coverage=cov,family=family,keys=[s['key'] for s in members],comparisons=['vs_src','vs_R0'],maxT_critical=crit))
    for s in summaries:
        if not s['primary']:s.update(SPT=None,CT=None);continue
        s['SPT']=bool(s['E1']['passed'] and s['E3']['possible_count']==0 and all(s['E2'][base]['ci95_family_simultaneous'][0]>0 for base in ('vs_src','vs_R0')))
        rr=[r for r in rows if r['key']==s['key'] and r['necessary']];s['CT']=bool(s['SPT'] and all(r['gain_src_lower']>0 and r['gain_R0_lower']>0 for r in rr)) if bed=='real' else None
    cells=[]
    for fam in family_records:
        if fam['family']=='P_rule':continue
        ss=[s for s in primary if s['key'] in fam['keys']];cells.append(dict(coverage=fam['coverage'],representation_family=fam['family'],passed=any(s['SPT'] for s in ss),SPT_arms=[s['key'] for s in ss if s['SPT']],CT_arms=[s['key'] for s in ss if s['CT']]))
    artifact=HERE/(bed+('_imputed' if sens else '')+'_results.json')
    write(artifact,dict(status='COMPLETE',utc=utc(),evidence_level='DEVELOPMENT',bed=bed,sensitivity=sens,endpoint_seeds=[4,5,6,7] if bed=='real' else [2,3],summaries=summaries,rows=rows,cells=cells,E1_maxT_critical=critical,E1_family_keys=[s['key'] for s in primary],E1_effective_family_count=len(primary)-(1 if p_contrast_equivalent else 0),P_rule_coverage_choices_identical=p_equivalent,E2_families=family_records,bootstrap_draws=5000,bootstrap_seed=seed,bootstrap_unit='physical state jointly across all controllers; fits averaged',unknown_rule='paired conservative bounds; same eta cancels',new_rollouts=0,shared_db_writes=0))
    np.savez_compressed(HERE/(bed+('_imputed' if sens else '')+'_state_vectors.npz'),**{key.replace('|','__')+'__'+n+'__'+k:v for key,vv in vectors.items() for n,vals in vv.items() for k,v in vals.items() if k in ('e1lo','e1hi','srclo','srchi','choice','selected','unknown')})
    print(json.dumps(dict(bed=bed,sensitivity=sens,cells=cells,SPT=[s['key'] for s in primary if s['SPT']])),flush=True)

if __name__=='__main__':main(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='imputed')
