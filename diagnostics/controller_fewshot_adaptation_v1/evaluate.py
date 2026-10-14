"""Exact crossed state/repetition bootstrap of immutable adaptation choices."""
from core import *
import time

def setup():
    gate();files=[]
    for target in range(14):
        f=HERE/'simulations'/f'{target}.npz';m=read(f.with_suffix('.json'));assert m['status']=='COMPLETE' and m['unable_count']==0 and sha(f)==m['simulation_sha256'];files.append(dict(path=str(f),sha256=sha(f)))
    for target in range(7):
        f=HERE/'predictions'/f'{target}.npz';files.append(dict(path=str(f),sha256=sha(f)))
    assert not (HERE/'ADAPTATION_FREEZE.json').exists();write(HERE/'ADAPTATION_FREEZE.json',dict(utc=utc(),before_new_endpoint_evaluation=True,files=files,preregistration_sha256=sha(HERE/'PREREGISTRATION.md')))
    for bed,ns,seed in [('main',64,202610130883),('stageB',24,202610130884)]:
        rng=np.random.default_rng(seed);sw=rng.multinomial(ns,np.full(ns,1/ns),size=5000).astype(np.float64)/ns;rw=rng.multinomial(200,np.full(200,1/200),size=(7,5000)).astype(np.float64)/200
        np.savez_compressed(HERE/f'bootstrap_{bed}.npz',state_weights=sw,repeat_weights=rw)
    print('Adaptation frozen before endpoint evaluation')

def descriptors(methods):
    cols=[dict(kind='method',strategy='U',method='P0',key='P0')]
    cols += [dict(kind='method',strategy=s,method=m,key=f'{s}|{m}') for s in STRATEGIES for m in methods if m!='P0']
    pairs=[('H','E'),('H-safe','E-safe')]
    if len(methods)>5:
        pairs += [(a,b) for backbone in ('R0','F2') for a,b in [(f'C-{backbone}','E'),(f'C-{backbone}-safe','E-safe'),(f'C-ctl-{backbone}','E'),(f'C-ctl-{backbone}-safe','E-safe'),(f'C-{backbone}',f'C-ctl-{backbone}'),(f'C-{backbone}-safe',f'C-ctl-{backbone}-safe')]]
    cols += [dict(kind='pair',strategy=s,method=a,reference=b,key=f'{s}|{a}-{b}') for s in STRATEGIES for a,b in pairs]
    nm=sum(c['kind']=='method' for c in cols);cols += [dict(c,kind='oracle',key=c['key']+'|vs_oracle') for c in cols[:nm]]
    return cols,nm

def cross_boot(vectors,rw,sw):
    """Vectors (contrast, bound, repeat, state); resample both independently."""
    nc,nb,nr,ns=vectors.shape;matrix=vectors.transpose(2,0,1,3).reshape(nr,-1).astype(np.float64);out=np.empty((5000,nc,nb),float)
    for start in range(0,5000,250):
        stop=min(start+250,5000);repeat_mean=(rw[start:stop]@matrix).reshape(stop-start,nc,nb,ns)
        out[start:stop]=np.einsum('bcns,bs->bcn',repeat_mean,sw[start:stop],optimize=False)
    return out

def main(bi,bed='main'):
    gate();assert read(HERE/'ADAPTATION_FREEZE.json')['before_new_endpoint_evaluation'];start=time.time();B=int(BUDGETS[bi]);ids=list(range(7)) if bed=='main' else list(range(7,14));sims=[np.load(HERE/'simulations'/f'{i}.npz') for i in ids];methods=sims[0]['methods'].tolist();cols,nm=descriptors(methods);nc=len(cols)
    if bed=='main':
        all_lab=np.load(R4/'LABELS.npz')['outcomes'];lab=all_lab[:,128:,:,4:8];disc=all_lab[:,128:,:,:4]
    else:
        v=np.load(P1/'P1_LABELS.npz');names=read(HERE/'PROTOCOL.json')['controller_order'];vn=v['controller_names'].tolist();lab=np.array([np.where(v['U'][vn.index(names[i])]==1,-1,v['Y'][vn.index(names[i])]).astype(np.int8) for i in ids]);disc=lab
    Y=(lab==1).mean(-1);U=(lab==-1).mean(-1);ns=Y.shape[1];boot=np.load(HERE/f'bootstrap_{bed}.npz');rw=boot['repeat_weights'];sw=boot['state_weights'];vec=np.zeros((7,nc,2,200,ns));bs=np.zeros((7,5000,nc,2));rows=[];oracle=[]
    for target,(sim,y,u) in enumerate(zip(sims,Y,U)):
        d=disc[target];succ=(d==1).sum(-1);valid=((d==1)|(d==0)).sum(-1);curve=np.nanmean(np.divide(succ,valid,out=np.full_like(succ,np.nan,dtype=float),where=valid>0),axis=0);eta_oracle=int(curve.argmax());prior=int(sim['source_prior_eta']);oracle.append(dict(target=ids[target],eta=eta_oracle,prior=prior,curve=curve,success_lower=float(y[:,eta_oracle].mean()),success_upper=float((y[:,eta_oracle]+u[:,eta_oracle]).mean())))
        for ci,c in enumerate(cols):
            st=STRATEGIES.index(c['strategy']);mi=methods.index(c['method']);a=sim['choices'][st,bi,mi]
            b=sim['choices'][st,bi,methods.index(c['reference'])] if c['kind']=='pair' else np.full_like(a,eta_oracle if c['kind']=='oracle' else prior)
            lo,hi=contrast(y[None,None],u[None,None],a,b);v=np.stack([lo.mean(1),hi.mean(1)]);vec[target,ci]=v
            if c['kind']=='method':
                success=picked(y[None,None],a).mean(1);unknown=picked(u[None,None],a).mean(1);possible=(v[0].mean(-1)<=-.10+1e-12);certain=(v[1].mean(-1)<=-.10+1e-12)
                rows.append(dict(key=c['key'],strategy=c['strategy'],method=c['method'],target=ids[target],controller=read(HERE/'PROTOCOL.json')['controller_order'][ids[target]],budget=B,source_prior_eta=prior,success_lower=float(success.mean()),success_upper=float((success+unknown).mean()),gain_lower=float(v[0].mean()),gain_upper=float(v[1].mean()),deployment_gain_quantiles=np.quantile(v[0].mean(-1),[.025,.5,.975]),possible_catastrophe_rate=float(possible.mean()),certain_catastrophe_rate=float(certain.mean()),any_possible_catastrophe=bool(possible.any()),any_certain_catastrophe=bool(certain.any()),eta_histogram=(np.bincount(a.ravel(),minlength=16)/a.size).tolist(),sampling_unknown_mean=float((sim['sampled_outcomes'][st,:,:B]==-1).sum(1).mean()),unable=int(sim['unable_to_collect'][st,:,bi].sum())))
        bs[target]=cross_boot(vec[target],rw[target],sw)
    # Simultaneous inference family within budget includes all policies, comparisons,
    # necessary individual/combined gains, and per-target oracle gaps.
    specs=[];obs=[];samples=[]
    def add(key,label,col,targets):
        specs.append(dict(key=key,metric=label,column=col,targets=targets));obs.append(vec[targets,col].mean((0,2,3)));samples.append(bs[targets,:,col,:].mean(0))
    for ci,c in enumerate(cols):
        if c['kind']=='method':
            add(c['key'],'E1',ci,list(range(7)))
            if bed=='main':
                add(c['key'],'E2_combined',ci,[1,4]);add(c['key'],'E2_base_u',ci,[1]);add(c['key'],'E2_v10_local',ci,[4])
        elif c['kind']=='pair':
            add(c['key'],'paired_all',ci,list(range(7)))
            if bed=='main':add(c['key'],'paired_necessary',ci,[1,4])
        else:
            for target in range(7):add(c['key'],'oracle_target'+str(ids[target]),ci,[target])
    obs=np.array(obs);samples=np.stack(samples,axis=1);se=samples.std(0,ddof=1);t=np.abs((samples-obs[None])/np.where(se>1e-12,se,1)[None]);critical=float(np.quantile(t.max((1,2)),.95));interval=np.c_[obs[:,0]-critical*se[:,0],obs[:,1]+critical*se[:,1]];inferences=[dict(spec,lower=float(o[0]),upper=float(o[1]),ci95_simultaneous=c.tolist()) for spec,o,c in zip(specs,obs,interval)]
    summaries=[]
    for ci,c in enumerate(cols[:nm]):
        rr=[r for r in rows if r['key']==c['key']];e1=next(x for x in inferences if x['key']==c['key'] and x['metric']=='E1');e2=next((x for x in inferences if x['key']==c['key'] and x['metric']=='E2_combined'),None)
        catastrophe=float(np.mean([r['possible_catastrophe_rate'] for r in rr]));certain=float(np.mean([r['certain_catastrophe_rate'] for r in rr]));anyrate=float(np.mean([r['any_possible_catastrophe'] for r in rr]));spt=bool(e1['ci95_simultaneous'][0]>-.02 and catastrophe==0 and e2 is not None and e2['ci95_simultaneous'][0]>0)
        ct=bool(spt and all(r['gain_lower']>0 for r in rr if r['target'] in (1,4))) if bed=='main' else None
        summaries.append(dict(key=c['key'],method=c['method'],strategy=c['strategy'],budget=B,success_lower=float(np.mean([r['success_lower'] for r in rr])),success_upper=float(np.mean([r['success_upper'] for r in rr])),E1=e1,E2=e2,E3=dict(possible_rate=catastrophe,certain_rate=certain,any_sample_target_possible_rate=anyrate,any_sample_target_certain_rate=float(np.mean([r['any_certain_catastrophe'] for r in rr]))),SPT=spt if bed=='main' else None,CT=ct))
    for row in rows:
        target=ids.index(row['target']);oi=nm+sum(c['kind']=='pair' for c in cols)+next(i for i,c in enumerate(cols[:nm]) if c['key']==row['key']);glo,ghi=vec[target,oi].mean((1,2));oc=next(x for x in inferences if x['key']==row['key']+'|vs_oracle' and x['metric']=='oracle_target'+str(row['target']));p0=next(r for r in rows if r['key']=='P0' and r['target']==row['target']);den=oracle[target]['success_lower']-p0['success_upper'] if oracle[target]['eta']!=oracle[target]['prior'] else 0.
        row.update(G=dict(gap_lower=float(-ghi),gap_upper=float(-glo),gain_vs_oracle_ci95=oc['ci95_simultaneous'],gap_closed_lower_point=float(row['gain_lower']/den) if den>0 else None,gap_closed_denominator_lower=float(den),within_2pp_point=bool(glo>=-.02),within_2pp_certified=bool(oc['ci95_simultaneous'][0]>=-.02)))
    result=dict(utc=utc(),status='COMPLETE',bed=bed,budget=B,budget_index=bi,evidence_level='DEVELOPMENT',summaries=summaries,rows=rows,inferences=inferences,oracle=oracle,maxT_critical=critical,family_size=len(specs),bootstrap_draws=5000,bootstrap_unit='same physical state across targets; 200 repetitions resampled per target; shared draws across all methods/strategies/budgets',elapsed_seconds=time.time()-start,endpoint_seeds=[4,5,6,7] if bed=='main' else [0,1,2,3],new_rollouts=0,shared_db_writes=0)
    folder=HERE/'endpoints';folder.mkdir(exist_ok=True);write(folder/f'{bed}_B{B}.json',result)
    # Needed for bootstrap sample-efficiency and budget-contrast calculations.
    np.savez_compressed(folder/f'{bed}_B{B}_bootstrap.npz',target_bootstrap=bs,target_point=vec.mean((3,4)),spec_bootstrap=samples,spec_point=obs,columns_json=np.array(json.dumps(cols)),specs_json=np.array(json.dumps(specs)),main_method_columns=nm)
    print(json.dumps(dict(bed=bed,B=B,seconds=time.time()-start,critical=critical,SPT=[s['key'] for s in summaries if s['SPT']])),flush=True)
if __name__=='__main__':setup() if sys.argv[1]=='setup' else main(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'main')
