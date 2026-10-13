"""Discovery gates first; independent endpoint paired state max-T next."""
from common import *
import numpy as np
from scipy.special import expit,logit
from train import Features

def curve(x):
    s=(x==1).sum(-1);v=((x==1)|(x==0)).sum(-1)
    q=np.divide(s,v,out=np.full_like(s,np.nan,dtype=float),where=v>0)
    return np.nanmean(q,axis=0)
def rates(x):return (x==1).mean(-1),(x==-1).mean(-1)
def select(x,choices):return np.take_along_axis(x,choices[...,None],axis=-1)[...,0]
def contrast(y,u,a,b):
    ya=select(y,a);yb=select(y,b);ua=select(u,a);ub=select(u,b);different=a!=b
    return ya-yb-ub*different,ya+ua*different-yb
def max_t(matrix,seed=2026101310):
    matrix=np.asarray(matrix);assert matrix.shape[0]==32
    draws=np.random.default_rng(seed).integers(32,size=(5000,32));obs=matrix.mean(0);boot=matrix[draws].mean(1);se=boot.std(0,ddof=1)
    critical=float(np.quantile(np.max(np.abs((boot-obs)/np.where(se>1e-12,se,1)),axis=1),.95))
    return obs,np.stack((obs-critical*se,obs+critical*se),axis=-1),critical
def source_curves(p):
    a=read(PREVIOUS/'protocol.json');b=read(PREVIOUS/'stage_b/protocol.json');orig=np.load(PREVIOUS/'LABELS.npz')['outcomes'][:,:128,:,:4];extra=np.load(PREVIOUS/'stage_b/LABELS.npz')['outcomes'];blend=np.load(HERE/'stage_2/TRAIN_LABELS.npz')['outcomes']
    return {**{n:curve(l) for n,l in zip(a['controller_order'],orig)},**{n:curve(l) for n,l in zip(b['controller_order'],extra)},**{n:curve(l) for n,l in zip(p['controller_order'],blend)}}

def gates():
    d=HERE/'stage_2';p=read(d/'protocol.json');tasks=read(d/'FIT_TASKS.json');curves=source_curves(p);discovery=np.load(d/'TEST_DISCOVERY_LABELS.npz')['outcomes'];rows=[]
    for task in tasks:
        if task['arm']!='R0':continue
        src=int(np.mean([curves[n] for n in task['sources']],axis=0).argmax())
        for held in task['held']:
            x=discovery[p['controller_order'].index(held)];vectors=[];upper=[];chosen=[];oracle=[]
            for first,second in ((0,1),(1,0)):
                valid=(x[:,:,first]>=0);s=(x[:,:,first]==1).sum(0);v=valid.sum(0);q=np.divide(s,v,out=np.full(16,-1.),where=v>0);eta=int(q.argmax());chosen.append(eta)
                y,u=rates(x[:,:,second:second+1]);a=np.full(32,eta);b=np.full(32,src);lo,hi=contrast(y,u,a,b);vectors.append(lo);upper.append(hi)
                c=np.where(valid,x[:,:,first],-1).argmax(-1);oc,_=contrast(y,u,c,b);oracle.append(oc)
            lo=np.mean(vectors,axis=0);hi=np.mean(upper,axis=0);boot=lo[np.random.default_rng(2026101309).integers(32,size=(5000,32))].mean(1);interval=np.quantile(boot,[.025,.975])
            rows.append(dict(split=task['split'],fold=task['fold'],coverage=task['coverage'],controller=held,source_prior_eta=src,target_cf_fixed_eta=chosen,
                L_lower=float(lo.mean()),L_upper=float(hi.mean()),ci95_conservative=interval.tolist(),necessary=bool(lo.mean()>=.05 and interval[0]>0),state_oracle_cf_gain_lower=float(np.mean(oracle))))
    frozen=dict(status='COMPLETE DISCOVERY ONLY',rows=rows,discovery_seeds=[0,1],endpoint_seeds_unread=[2,3],bootstrap_draws=5000,bootstrap_seed=2026101309,
        counts={s:{c:sum(r['necessary'] and r['split']==s and r['coverage']==c for r in rows) for c in ('DENSE','SPARSE')} for s in ('LOPO','LOAO')},source_curves={n:v.tolist() for n,v in curves.items()},gate_code_sha256=sha(__file__))
    freeze(d/'GATE_RESULTS.json',frozen);return frozen

def main():
    d=HERE/'stage_2';p=read(d/'protocol.json');gate=gates();tasks=read(d/'FIT_TASKS.json');curves={n:np.asarray(v) for n,v in gate['source_curves'].items()}
    outcomes=np.load(d/'TEST_ENDPOINT_LABELS.npz')['outcomes'];train=np.load(d/'TRAIN_LABELS.npz')['outcomes'];disc=np.load(d/'TEST_DISCOVERY_LABELS.npz')['outcomes'];Y,U=rates(outcomes)
    vectors={};rows=[];task_status=[];arms=p['arms']
    # Compute nearest baselines only using source-normalized R3 and common train states.
    features=Features(p,'R3');initial=p['states'][:p['train_states']]
    cache_predictions={}
    for task in tasks:
        name=f"{task['split']}_fold{task['fold']}_{task['coverage']}_{task['arm']}";f=d/'fits'/name;meta=read(f/'complete.json');task_status.append(meta)
        if meta['status']=='COMPLETE':cache_predictions[name]=np.load(f/'predictions.npz')
    for task in tasks:
        arm=task['arm'];prefix=f"{task['split']}_fold{task['fold']}_{task['coverage']}";name=prefix+'_'+arm
        if name not in cache_predictions:continue
        z=cache_predictions[name];r0=cache_predictions[prefix+'_R0']['held'];src=int(np.mean([curves[n] for n in task['sources']],axis=0).argmax())
        sf=np.array([features.static(n,initial,'R3') for n in task['sources']],float);mu=sf.mean((0,1));std=sf.std((0,1));scale=np.where(std>1e-6,std,1.);centers=((sf-mu)/scale).mean(1)
        for i,held in enumerate(task['held']):
            idx=p['controller_order'].index(held);g=next(r for r in gate['rows'] if (r['split'],r['fold'],r['coverage'],r['controller'])==(task['split'],task['fold'],task['coverage'],held))
            hm=((features.static(held,initial,'R3')-mu)/scale).mean(0);dd=np.linalg.norm(centers-hm,axis=1);nearest=task['sources'][int(dd.argmin())];near=int(curves[nearest].argmax())
            own=curve(disc[idx]);target=int(own.argmax());choice=z['held'][:,i].argmax(-1);r0choice=r0[:,i].argmax(-1);wrong=z['mismatched'][:,i].argmax(-1)
            y=Y[idx][None];u=U[idx][None];sy=select(y,choice);su=select(u,choice)
            comps={};upper_comps={}
            for label,b in [('src',np.full_like(choice,src)),('near',np.full_like(choice,near)),('R0',r0choice),('target_fixed',np.full_like(choice,target))]:
                lo,hi=contrast(y,u,choice,b);comps[label]=lo.mean(0);upper_comps[label]=hi.mean(0)
            lo,hi=contrast(y[:,None],u[:,None],choice[:,None],wrong);comps['mismatch']=lo.mean((0,1));upper_comps['mismatch']=hi.mean((0,1))
            prior_choice=expit(z['controller_prior'][:,i]).mean(1).argmax(-1);lo,hi=contrast(y,u,choice,prior_choice[:,None]);comps['state_residual']=lo.mean(0);upper_comps['state_residual']=hi.mean(0)
            calibration={}
            for k in (1,2,4):
                small=train[idx,:k];succ=(small==1).sum((0,2));valid=((small==0)|(small==1)).sum((0,2));measured=np.divide(succ,valid,out=np.full(16,np.nan),where=valid>0)
                priorcurve=np.mean([curves[n] for n in task['sources']],axis=0);caleta=int(np.where(np.isfinite(measured),measured,priorcurve).argmax())
                predicted=expit(z['target_calibration'][:,i,:k]).mean((0,1));offset=np.zeros(16);mask=valid>0
                offset[mask]=logit(np.clip(measured[mask],1e-3,1-1e-3))-logit(np.clip(predicted[mask],1e-3,1-1e-3));cc=(z['held'][:,i]+offset).argmax(-1)
                gain,_=contrast(y,u,choice,np.full_like(choice,caleta));calkey=f'calibration_fixed_{k}';comps[calkey]=gain.mean(0)
                calibration[str(k)]=dict(uses_target_labels=True,target_state_indices=list(range(k)),rollouts=k*32,observed_eta_heads=int(mask.sum()),fixed_eta=caleta,fixed_success_lower=float(Y[idx,:,caleta].mean()),fixed_success_upper=float((Y[idx,:,caleta]+U[idx,:,caleta]).mean()),offset_success_lower=float(select(y,cc).mean()))
            q=np.divide(Y[idx],1-U[idx],out=np.full_like(Y[idx],np.nan),where=U[idx]<1);pred=expit(z['held'][:,i]);mse=float(np.nanmean((pred-q[None])**2))
            pattern=np.nanmean(q,0)-np.mean([curves[n] for n in task['sources']],0);pm=pred.mean((0,1))-expit(r0[:,i]).mean((0,1));valid=np.isfinite(pattern)
            corr=float(np.corrcoef(pattern[valid],pm[valid])[0,1]) if np.std(pm[valid])>1e-10 and valid.sum()>2 else None
            vector_key=(task['split'],task['coverage'],arm,held);vectors[vector_key]=dict(contrasts=comps,selected=sy.mean(0),unknown=su.mean(0),choices=choice)
            rows.append(dict(split=task['split'],fold=task['fold'],coverage=task['coverage'],arm=arm,held_controller=held,necessary=g['necessary'],source_prior_eta=src,
                nearest_source=nearest,nearest_eta=near,nearest_distances=dict(zip(task['sources'],dd.tolist())),target_discovery_fixed_eta=target,
                success_lower=float(sy.mean()),success_upper=float((sy+su).mean()),success_by_fit=sy.mean(1).tolist(),unknown_selected=float(su.mean()),
                gain_lower={k:float(v.mean()) for k,v in comps.items()},gain_upper={k:float(v.mean()) for k,v in upper_comps.items()},
                baselines_success_lower=dict(src=float(Y[idx,:,src].mean()),near=float(Y[idx,:,near].mean()),R0=float(select(y,r0choice).mean()),target_fixed=float(Y[idx,:,target].mean())),
                learned_controller_prior_eta=prior_choice.tolist(),controller_prior_success_lower=float(select(y,prior_choice[:,None]).mean()),
                hindsight_discrete_state_upper_success=float((Y[idx]+U[idx]).max(-1).mean()),hindsight_discrete_state_regret_lower=float(Y[idx].max(-1).mean()-(sy+su).mean()),
                eta_histograms=[np.bincount(c,minlength=16).tolist() for c in choice],q_mse_observed_Q2=mse,controller_eta_pattern_correlation=corr,calibration=calibration))
    primary={};cols=['src','near','R0','mismatch']
    for coverage in ('DENSE','SPARSE'):
        necessary=[r['controller'] for r in gate['rows'] if r['split']=='LOPO' and r['coverage']==coverage and r['necessary']];evaluations={}
        for arm in arms[1:]:
            available=[n for n in necessary if ('LOPO',coverage,arm,n) in vectors]
            if len(necessary)<4 or available!=necessary:
                evaluations[arm]=dict(status='INCONCLUSIVE — TOO FEW NECESSARY CONTROLLERS' if len(necessary)<4 else 'NOT EXECUTED — INCOMPLETE INPUTS',passed=False);continue
            mat=np.stack([np.mean([vectors['LOPO',coverage,arm,n]['contrasts'][c] for n in necessary],0) for c in cols],-1)
            obs,interval,critical=max_t(mat);evaluations[arm]=dict(status='PASS' if np.all(interval[:,0]>0) else 'NO JOINT SELECTION PASS',passed=bool(np.all(interval[:,0]>0)),max_t_critical=critical,
                comparisons={c:dict(mean=float(obs[i]),ci95_simultaneous=interval[i].tolist()) for i,c in enumerate(cols)})
        primary[coverage]=dict(necessary_controllers=necessary,necessary_count=len(necessary),representations=evaluations)
    dense=set(primary['DENSE']['necessary_controllers']);sparse=set(primary['SPARSE']['necessary_controllers']);coverage_results={}
    for tag,targets in [('union',sorted(dense|sparse)),('intersection',sorted(dense&sparse)),('all',p['controller_order'])]:
        validarms=[a for a in arms if targets and all(('LOPO',c,a,n) in vectors for n in targets for c in ('DENSE','SPARSE'))]
        if not validarms:coverage_results[tag]=dict(targets=targets,status='EMPTY TARGETS OR UNAVAILABLE INPUTS');continue
        mat=[]
        for arm in validarms:
            vv=[]
            for n in targets:
                a=vectors['LOPO','DENSE',arm,n];b=vectors['LOPO','SPARSE',arm,n];idx=p['controller_order'].index(n)
                low,_=contrast(Y[idx][None],U[idx][None],a['choices'],b['choices']);vv.append(low.mean(0))
            mat.append(np.mean(vv,0))
        obs,interval,critical=max_t(np.stack(mat,-1),seed=2026101311)
        coverage_results[tag]=dict(targets=targets,status='COMPLETE',max_t_critical=critical,effects={a:dict(mean=float(obs[i]),ci95_simultaneous=interval[i].tolist()) for i,a in enumerate(validarms)})
    dp=primary['DENSE']['representations'];sp=primary['SPARSE']['representations'];rootcause='INCONCLUSIVE'
    if min(primary[c]['necessary_count'] for c in ('DENSE','SPARSE'))>=4:
        supported=[]
        for arm in ('R3','R4_action','R5_behavior'):
            effect=coverage_results.get('union',{}).get('effects',{}).get(arm)
            if dp[arm]['passed'] and not sp[arm]['passed'] and effect and effect['ci95_simultaneous'][0]>0:supported.append(arm)
        if supported:rootcause='SUPPORTED: COVERAGE IMPROVES SELECTION IN THIS SYNTHETIC FAMILY; EXTRA LABEL VOLUME CONFOUNDED'
        elif dp['R_oracle']['passed'] and not any(dp[a]['passed'] for a in ('R3','R4_action')):
            rootcause='SUPPORTED: STATIC REPRESENTATION INSUFFICIENCY IN THIS FAMILY'+('; R5 BEHAVIOR PROXY SUCCEEDS' if dp['R5_behavior']['passed'] else '')
        elif not dp['R_oracle']['passed']:rootcause='METHOD/PROTOCOL DEFECT CANDIDATE: PRIVILEGED ORACLE DOES NOT PASS JOINT SELECTION GATE; CHECK CI, COVERAGE AND NEAREST-BASELINE CEILING'
        else:rootcause='PARTIALLY SUPPORTED: AT LEAST ONE DENSE REPRESENTATION TRANSFERS, BUT COVERAGE/REPRESENTATION ROOTCAUSE NOT SEPARATED'
    result=dict(status='COMPLETE',preregistered=True,scope='held-out synthetic blend controllers and new natural states; old train states DEVELOPMENT',rows=rows,primary=primary,coverage_effect=coverage_results,rootcause_verdict=rootcause,
        bootstrap_draws=5000,bootstrap_unit='same 32 physical state indices resampled jointly for all controllers/fits; fit seeds averaged',max_t_family='four comparisons within each representation, not across all representations',
        gates=gate,task_execution=task_status,independent_optimization_fits=3*sum(t.get('optimization_performed',False) for t in task_status),checkpoint_references=3*sum(t['status']=='COMPLETE' for t in task_status),
        R5_deployment='32 unlabeled simulator probes/controller; 200 steps each, not static',q_prediction='secondary, observed Q2 label noise retained',
        unknown_policy='paired conservative interval; same eta/seed exact outcomes cancel; missing and numerical unknown never zero failure labels')
    write(d/'results.json',result);print(json.dumps(dict(primary=primary,rootcause=rootcause)))
if __name__=='__main__':main()
