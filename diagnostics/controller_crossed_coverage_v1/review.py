"""Predeclared source-support contrast; no held-out scene model selection."""
import csv,json
import numpy as np
from .study import HERE,BASE,ROOT,SEEDS,read,write,freeze,sha
from diagnostics.controller_ff_factorial_repair_v1.review import (
    select,selection_difference,mse_gain,intervals,derangements,reversal,corr)
from diagnostics.controller_flow_endpoint_probe_v1.analyze import centered
from diagnostics.controller_q_diversity_v1.evaluate import donor_bootstrap


def pack(l,u,bl,bu):
    return dict(identification_bounds=[float(l),float(u)],censor_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bu,.975))])


def run():
    assert sha(__file__)==read(HERE/'analysis_frozen.json')['analysis_sha256']
    d=dict(np.load(HERE/'dataset.npz'));old=dict(np.load(BASE/'dataset.npz'));pred={};fits=[]
    for kind,arm in [('state','union_native_slots'),('eta','eta_only')]:
        now=[];before=[]
        for seed in SEEDS:
            p=HERE/'fits/joint'/arm/f'seed{seed}';z=read(p/'complete.json');b=BASE/'fits/joint'/arm/f'seed{seed}';bz=read(b/'complete.json')
            assert sha(p/'checkpoint.msgpack')==z['checkpoint_sha256']
            assert read(p/'normalization.json')==read(b/'normalization.json')
            assert read(p/'sampling_audit.json')['sampling_weights']==read(b/'sampling_audit.json')['sampling_weights']
            q=np.load(p/'predictions.npz')['probability'];v=np.load(p/'pre_fit_predictions.npz')['probability']
            cached=np.load(b/'predictions.npz')['probability']
            diff=float(np.max(abs(v[:len(cached)]-cached)));assert diff<1e-5,diff
            now.append(q);before.append(v)
            fits.append(dict(kind=kind,seed=seed,new=z['selected'],old=bz['selected'],prefit_replay_max_abs=diff,
                new_cpu=read(p/'sampling_audit.json')['cpu_affinity'],old_cpu=read(b/'sampling_audit.json')['cpu_affinity']))
        pred['N128_'+kind]=np.asarray(now);pred['N32_'+kind]=np.asarray(before)
    matrix=ROOT/'controller_ff_factorial_repair_v1';md=dict(np.load(matrix/'dataset.npz'));mr=dict(np.load(matrix/'raw_labels.npz'))
    lookup={(str(s),str(c)):i for i,(s,c) in enumerate(zip(md['state_uid'],md['controller']))}
    newraw=dict(np.load(HERE/'raw_labels.npz'));newlookup={(str(s),str(c)):i for i,(s,c) in enumerate(zip(newraw['state_uid'],newraw['controller']))}
    panels=[]
    for sc in sorted(set(d['scene'])):
        for ref in ('flow','goal'):
            ids=np.flatnonzero((d['scene']==sc)&(d['split']=='validation')&(d['source']=='matrix_goal_and_flow32')&np.char.startswith(d['controller'],ref+'_'))
            panels.append((str(sc)+'__'+ref,'SOURCE_VALIDATION',ids,mr,lookup))
    panels.append(('ring_exchange__new96','NEW_COMBINATIONS_BEFORE_FIT_TRAIN_AFTER_FIT',np.flatnonzero(d['source']=='crossed_ring128'),newraw,newlookup))
    arrays={};saved={};rows=[];paired=[];individual=[];failures=[]
    for panel,scope,ids,raw,ri in panels:
        ids=np.array(sorted(ids,key=lambda i:(d['state_uid'][i],d['controller'][i])));n=len(np.unique(d['state_uid'][ids]));assert len(ids)==n*5
        shape=(n,5,4);js=[0,5,10,15];rids=[ri[str(d['state_uid'][i]),str(d['controller'][i])] for i in ids]
        ss,uu,obs=[raw[k][rids][:,js].reshape(shape+(4,)) for k in ('success','unknown','observed')];assert obs.all()
        lo,hi=ss.mean(-1),(ss+uu).mean(-1);mid=(lo+hi)/2
        halves_lo=ss.reshape(shape+(2,2)).mean(-1);halves_hi=(ss+uu).reshape(shape+(2,2)).mean(-1)
        boot=np.random.default_rng(2026102601+n).integers(n,size=(20000,n));arrays[panel+'__bootstrap_indices']=boot
        perms=derangements(n);saved[panel+'__permutations']=perms;saved[panel+'__states']=d['state_uid'][ids][::5]
        saved[panel+'__truth_lower']=lo;saved[panel+'__truth_upper']=hi
        pp={name:v[:,ids][:,:,js].mean(0).reshape(shape) for name,v in pred.items()}
        for name,q in pp.items():
            key=panel+'__'+name;choice=q.argmax(-1);freq=np.bincount(choice.ravel(),minlength=4)/choice.size
            row=dict(panel=panel,scope=scope,model=name,N=n,Q_MSE_midpoint=float(np.mean((q-mid)**2)),
                state_eta_MSE_midpoint=float(np.mean((centered(q,[0,2])-centered(mid,[0,2]))**2)),
                state_eta_corr_midpoint=corr(centered(q,[0,2]),centered(mid,[0,2])) if name.endswith('state') else None,
                triple_corr_midpoint=corr(centered(q,[0,1,2]),centered(mid,[0,1,2])) if name.endswith('state') else None,
                success_bounds=[float(select(lo,choice).mean()),float(select(hi,choice).mean())],
                selected_unknown=int(select(uu.sum(-1),choice).sum()),
                entropy_bits=float(-np.sum(freq[freq>0]*np.log2(freq[freq>0]))),selection_frequency=freq.tolist())
            for event in ('collision','deadlock','timeout'):
                a=raw[event][rids][:,js].reshape(shape+(4,)).sum(-1);row['selected_'+event]=int(select(a,choice).sum())
            saved[key+'__prediction']=q;saved[key+'__choice']=choice;saved[key+'__mismatched_choices']=choice[perms]
            ml=np.zeros((n,n));mu=ml.copy()
            for donor in range(n):
                a,b=selection_difference(lo,hi,choice,np.broadcast_to(choice[donor],choice.shape));ml[:,donor],mu[:,donor]=a.mean(1),b.mean(1)
            pl,al,ok=donor_bootstrap(ml,boot);pu,au,ok2=donor_bootstrap(mu,boot);good=ok&ok2
            row['matched_minus_mismatch']=pack(pl,pu,al[good],au[good]);arrays[key+'__mismatch_lower']=al;arrays[key+'__mismatch_upper']=au
            dl,du=selection_difference(lo[...,None,:],hi[...,None,:],choice[...,None],np.arange(4)[None,None,:]);bl=[];bu=[]
            for start in range(0,len(boot),2000):
                ib=boot[start:start+2000];bl.append(dl[ib].mean(1).min(-1).mean(-1));bu.append(du[ib].mean(1).min(-1).mean(-1))
            bl,bu=np.concatenate(bl),np.concatenate(bu)
            row['minus_reselected_best_per_C_fixed_K4']=pack(dl.mean(0).min(-1).mean(),du.mean(0).min(-1).mean(),bl,bu)
            arrays[key+'__fixed_lower']=bl;arrays[key+'__fixed_upper']=bu
            a,b=selection_difference(lo,hi,choice,pp['N128_eta' if name.startswith('N128') else 'N32_eta'].argmax(-1))
            row['minus_matched_data_eta_only']=intervals(a.mean(1),b.mean(1),boot,key+'__eta',arrays)
            row['reversal']=reversal(halves_lo,halves_hi,q,boot)
            row['oracle_bounds']=[float(lo.max(-1).mean()),float(hi.max(-1).mean())]
            row['regret_bounds']=[float(np.maximum(lo.max(-1)-select(hi,choice),0).mean()),float((hi.max(-1)-select(lo,choice)).mean())]
            rows.append(row)
        qa,qb=pp['N128_state'],pp['N32_state'];a,b=mse_gain(qb,qa,lo,hi)
        entry=dict(panel=panel,scope=scope,N=n,Q_MSE_gain=intervals(a.mean((1,2)),b.mean((1,2)),boot,panel+'__Qgain',arrays))
        a,b=selection_difference(lo,hi,qa.argmax(-1),qb.argmax(-1));entry['selection_gain']=intervals(a.mean(1),b.mean(1),boot,panel+'__selection',arrays)
        def bounds(a,b,l,u,axes):
            a,b=centered(a,axes),centered(b,axes);x,y=2*(a-b)*l-a*a+b*b,2*(a-b)*u-a*a+b*b
            dims=tuple(range(a.ndim-3,a.ndim));return np.minimum(x,y).mean(dims),np.maximum(x,y).mean(dims)
        for label,axes in [('state_eta',[0,2]),('state_controller_eta',[0,1,2])]:
            pl,pu=bounds(qa,qb,lo,hi,axes);al=[];au=[]
            for start in range(0,len(boot),2000):
                ib=boot[start:start+2000];l,u=bounds(qa[ib],qb[ib],lo[ib],hi[ib],[a+1 for a in axes]);al.append(l);au.append(u)
            al,au=np.concatenate(al),np.concatenate(au);arrays[panel+'__'+label+'_gain_lower']=al;arrays[panel+'__'+label+'_gain_upper']=au
            entry[label+'_MSE_gain']=pack(pl,pu,al,au)
        paired.append(entry)
        for j,seed in enumerate(SEEDS):
            for name,v in pred.items():
                q=v[j,ids][:,js].reshape(shape)
                individual.append(dict(panel=panel,model=name,seed=seed,Q_MSE=float(np.mean((q-mid)**2)),state_eta_MSE=float(np.mean((centered(q,[0,2])-centered(mid,[0,2]))**2))))
        for i,s in enumerate(d['state_uid'][ids][::5]):
            failures.append(dict(panel=panel,state_uid=str(s),model_regret_upper=float((hi[i].max(-1)-select(lo[i],qa[i].argmax(-1))).mean()),
                numerical_unknown=int(uu[i].sum()),matched_choice=qa[i].argmax(-1).tolist(),discovery_preferred=halves_lo[i,...,0].argmax(-1).tolist()))
    target=[r for r in individual if r['panel']=='ring_exchange__flow' and r['model'].endswith('state')]
    means={m:{k:float(np.mean([r[k] for r in target if r['model']==m])) for k in ('Q_MSE','state_eta_MSE')} for m in ('N32_state','N128_state')}
    ratios={k:means['N128_state'][k]/means['N32_state'][k] for k in means['N32_state']}
    wins=[]
    for s in SEEDS:
        a=next(r for r in target if r['seed']==s and r['model']=='N128_state');b=next(r for r in target if r['seed']==s and r['model']=='N32_state')
        wins.append(all(a[k]<b[k] for k in ('Q_MSE','state_eta_MSE')))
    source_regrets={k:float(np.mean([z[k]['regret'] for z in fits if z['kind']=='state'])) for k in ('new','old')}
    gate=bool(all(x<=.85 for x in ratios.values()) and sum(wins)>=2 and source_regrets['new']<=source_regrets['old']+.01)
    result=dict(rows=rows,paired=paired,individual_fits=individual,source_fits=fits,source_coverage_gate_passed=gate,
        gate_details=dict(mean_individual_fit_errors=means,error_ratios=ratios,seed_joint_improvement=wins,global_source_validation_regret=source_regrets),
        boundary='SOURCE-selected N8 validation diagnostic only. Added96states are oldnativeTRAIN; theircrossedlabels unseenbyfrozenN32 but TRAINforN128. NoLOSO/K16/B15/untouchedconfirmationclaim. AllfixedcontrolrestrictedtotheobservedK4panel. Unknownboundsretained.')
    write(HERE/'results.json',result);np.savez_compressed(HERE/'bootstrap.npz',**arrays);np.savez_compressed(HERE/'predictions_and_controls.npz',**saved)
    write(HERE/'failure_cases.json',sorted(failures,key=lambda x:x['model_regret_upper'],reverse=True))
    for name,values in [('summary.csv',rows),('individual_fit_metrics.csv',individual)]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in values)
    lines=['CROSSED SOURCE COVERAGE N32->N128 AT FIXED Q4',f'SOURCE_COVERAGE_GATE_PASSED={gate}',json.dumps(result['gate_details']),result['boundary'],
        'Physical/nominal inputs and original4032rows are cache-exact. Frozen normalization, input, optimizer,6000steps,sourceVALselection,native20mass. Fairneweta-only receivesallnewlabels.',
        '7680missingrollouts;96existingnaturalTRAINstatesx5fixedFFFlow-referenceoperatorsx4unchangedetasx4seeds. No newglobalphysicalstates. No outcome filtering.',
        '20kpairedphysical-statebootstrap;200derangements plusdonor-recipientbootstrap;allK4fixedperC reselectedinsidebootstrap;allstrongreversalpairsdiscovery2/eval2.',
        '', 'panel model Q_MSE state_eta_corr triple_corr success_bounds']
    for r in rows:lines.append(f"{r['panel']} {r['model']} {r['Q_MSE_midpoint']} {r['state_eta_corr_midpoint']} {r['triple_corr_midpoint']} {r['success_bounds']}")
    lines+=['','RingSOURCEpaired: '+json.dumps(next(r for r in paired if r['panel']=='ring_exchange__flow')),
        'Positivegate requiresindependentheld-physicalstateconfirmation. Negativegate prunes thisfourfoldcoveragechangeas sufficientrepair,notallpossiblecoverageeffects. Noautomatic512/Q8/hyperparametersweep.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_SOURCE_COVERAGE_DIAGNOSTIC',gate_passed=gate,new_fits=6,new_rollouts=7680,root_cause_closed=False,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(gate=gate,details=result['gate_details']),flush=True)


if __name__=='__main__':run()
