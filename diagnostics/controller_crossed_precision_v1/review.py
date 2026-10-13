"""Source-only Q2/Q4 contrast; every source scene and reference retained."""
import csv,json
import numpy as np
from .study import HERE,BASE,ROOT,SEEDS,read,write,sha


def run():
    from diagnostics.controller_ff_factorial_repair_v1.review import mse_gain,selection_difference,intervals,corr,select,derangements
    from diagnostics.controller_q_diversity_v1.evaluate import donor_bootstrap
    from diagnostics.controller_flow_endpoint_probe_v1.analyze import centered
    d=dict(np.load(HERE/'dataset.npz'));pred={};source=[]
    for name,folder,arm in [('q4_state',HERE,'union_native_slots'),('q4_eta',HERE,'eta_only'),('q2_state',BASE,'union_native_slots'),('q2_eta',BASE,'eta_only')]:
        aa=[]
        for seed in SEEDS:
            path=folder/'fits/joint'/arm/f'seed{seed}';z=read(path/'complete.json')
            assert sha(path/'checkpoint.msgpack')==z['checkpoint_sha256']
            if folder==HERE:
                assert read(path/'normalization.json')==read(BASE/'fits/joint'/arm/f'seed{seed}/normalization.json')
                assert read(path/'sampling_audit.json')['sampling_weights']==read(BASE/'fits/joint'/arm/f'seed{seed}/sampling_audit.json')['sampling_weights']
            aa.append(np.load(path/'predictions.npz')['probability'])
            source.append(dict(name=name,seed=seed,selected=z['selected'],train=z['source_train_metrics'],
                cpu_affinity=read(path/'sampling_audit.json')['cpu_affinity']))
        pred[name]=np.asarray(aa)
    rows=[];contrasts=[];arrays={};saved={}
    for sc in sorted(set(d['scene'])):
        for ref in ('flow','goal'):
            ids=np.flatnonzero((d['scene']==sc)&(d['split']=='validation')&(d['source']=='matrix_goal_and_flow32')&np.char.startswith(d['controller'],ref+'_'))
            ids=np.array(sorted(ids,key=lambda i:(d['state_uid'][i],d['controller'][i])));n=len(np.unique(d['state_uid'][ids]));assert n==8 and len(ids)==40
            shape=(n,5,4);ss,ff,uu=[d[k][ids][:,[0,5,10,15]].reshape(shape) for k in ('success','failure','unknown')]
            count=ss+ff+uu;assert (count==4).all();lo=ss/count;hi=(ss+uu)/count;mid=(lo+hi)/2
            pp={a:z[:,ids][:,:,[0,5,10,15]].mean(0).reshape(shape) for a,z in pred.items()}
            boot=np.random.default_rng(2026102501+n).integers(n,size=(20000,n));prefix=str(sc)+'__'+ref
            arrays[prefix+'__bootstrap_indices']=boot;saved[prefix+'__states']=d['state_uid'][ids][::5];saved[prefix+'__truth_lower']=lo;saved[prefix+'__truth_upper']=hi
            saved[prefix+'__permutations']=derangements(n)
            for name,q in pp.items():
                choose=q.argmax(-1);key=prefix+'__'+name;freq=np.bincount(choose.ravel(),minlength=4)/choose.size
                row=dict(scene=str(sc),reference=ref,model=name,N=n,controllers=5,Q_MSE_midpoint=float(np.mean((q-mid)**2)),
                    state_eta_corr_midpoint=corr(centered(q,[0,2]),centered(mid,[0,2])) if name.endswith('state') else None,
                    triple_corr_midpoint=corr(centered(q,[0,1,2]),centered(mid,[0,1,2])) if name.endswith('state') else None,
                    success_bounds=[float(select(lo,choose).mean()),float(select(hi,choose).mean())],
                    entropy_bits=float(-np.sum(freq[freq>0]*np.log2(freq[freq>0]))),selected_unknown=int(select(uu,choose).sum()))
                saved[key+'__prediction']=q;saved[key+'__choice']=choose
                ml=np.zeros((n,n));mu=ml.copy()
                for donor in range(n):
                    a,b=selection_difference(lo,hi,choose,np.broadcast_to(choose[donor],choose.shape));ml[:,donor],mu[:,donor]=a.mean(1),b.mean(1)
                pl,al,ok=donor_bootstrap(ml,boot);pu,au,ok2=donor_bootstrap(mu,boot);good=ok&ok2
                row['matched_minus_mismatch']=dict(identification_bounds=[pl,pu],censor_CI95=[float(np.quantile(al[good],.025)),float(np.quantile(au[good],.975))])
                arrays[key+'__mismatch_lower']=al;arrays[key+'__mismatch_upper']=au
                rows.append(row)
            qa,qb=pp['q4_state'],pp['q2_state'];a,b=mse_gain(qb,qa,lo,hi)
            entry=dict(scene=str(sc),reference=ref,N=n,Q_MSE_gain=intervals(a.mean((1,2)),b.mean((1,2)),boot,prefix+'__Qgain',arrays))
            a,b=selection_difference(lo,hi,qa.argmax(-1),qb.argmax(-1));entry['selection_gain']=intervals(a.mean(1),b.mean(1),boot,prefix+'__selection',arrays)
            for label,axes in [('state_eta',[0,2]),('state_controller_eta',[0,1,2])]:
                def bounds(a,b,l,u,ax):
                    a,b=centered(a,ax),centered(b,ax);x,y=2*(a-b)*l-a*a+b*b,2*(a-b)*u-a*a+b*b
                    dims=tuple(range(a.ndim-3,a.ndim));return np.minimum(x,y).mean(dims),np.maximum(x,y).mean(dims)
                pl,pu=bounds(qa,qb,lo,hi,axes);al=[];au=[]
                for start in range(0,len(boot),500):
                    ib=boot[start:start+500];l,u=bounds(qa[ib],qb[ib],lo[ib],hi[ib],[a+1 for a in axes]);al.append(l);au.append(u)
                al,au=np.concatenate(al),np.concatenate(au);arrays[prefix+'__'+label+'_gain_lower']=al;arrays[prefix+'__'+label+'_gain_upper']=au
                entry[label+'_MSE_gain']=dict(identification_bounds=[float(pl),float(pu)],censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(au,.975))])
            entry['individual_fit_Q_gain']=[];entry['individual_fit_state_eta_gain']=[]
            for j,seed in enumerate(SEEDS):
                a,b=[pred[z][j,ids][:,[0,5,10,15]].reshape(shape) for z in ('q4_state','q2_state')]
                l,h=mse_gain(b,a,lo,hi);entry['individual_fit_Q_gain'].append(dict(seed=seed,bounds=[float(l.mean()),float(h.mean())]))
                l,h=bounds(a,b,lo,hi,[0,2]);entry['individual_fit_state_eta_gain'].append(dict(seed=seed,bounds=[float(l),float(h)]))
            contrasts.append(entry)
    target=next(r for r in contrasts if r['scene']=='ring_exchange' and r['reference']=='flow')
    gate=target['Q_MSE_gain']['censor_CI95'][0]>0 and target['state_eta_MSE_gain']['censor_CI95'][0]>0
    gate=bool(gate and sum(x['bounds'][0]>0 for x in target['individual_fit_Q_gain'])>=2 and sum(x['bounds'][0]>0 for x in target['individual_fit_state_eta_gain'])>=2)
    write(HERE/'results.json',dict(rows=rows,paired=contrasts,source=source,source_precision_gate_passed=gate,
        boundary='Checkpoint-selectedSOURCEVAL,8physicalstates/scene. Diagnosticonly;no freshconfirmationorLOSOrepair. Stage24newseedsarenowTRAINandnotcalledTEST. No extrarollouts.'))
    np.savez_compressed(HERE/'bootstrap.npz',**arrays);np.savez_compressed(HERE/'predictions_and_controls.npz',**saved)
    lines=['SAME SUPPORT / Q2->Q4 PRECISION ONLY',f'SOURCE_PRECISION_GATE_PASSED={gate}',
        'Sixfits,all32Ringstates and5Flow-referencecontrollers unchanged. Allnonlabelarrays,eligiblecells,samplerweights,drawRNG,optimizer,6000steps,sourceVALrule unchanged;sourceTRAINnormalization asserted identical. Eta-only receives exactlysameQ4labels.',
        'Reused1280newcachedindependentseedoutcomes. No rolloutgenerated here. Stage24frozenmodelreplicationremainsvalidforoldmodels;thoseoutcomesnowTRAINfornewmodels.',
        'All4SOURCEscenesandbothgoal/Flowbranchesreported.20kphysical-statepairedbootstrap,censorbounds;midpointcorrelationssensitivityonly. SOURCEVALselectedandN8,so a positivegateisnotindependentconfirmation.',
        'RingFlowpairedcontrast: '+json.dumps(target),'',
        'scene reference model Q_MSE state_eta_corr triple_corr selected_success']
    for r in rows:lines.append(f"{r['scene']} {r['reference']} {r['model']} {r['Q_MSE_midpoint']} {r['state_eta_corr_midpoint']} {r['triple_corr_midpoint']} {r['success_bounds']}")
    lines+=['','A failureprunesQ2->Q4precisionas a sufficientrepair,notallpossiblenoiseeffects. NoautomaticQ8/16sweep. Input-sufficiencyandstatecoverageare notsettledbythiscontrast.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    with (HERE/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    write(HERE/'completion.json',dict(status='COMPLETE_SOURCE_PRECISION_DIAGNOSTIC',gate_passed=gate,new_fits=6,new_rollouts=0,report_sha256=sha(HERE/'REPORT.txt'),root_cause_closed=False))
    print(dict(gate=gate,Ring_Qgain=target['Q_MSE_gain'],Ring_interaction_gain=target['state_eta_MSE_gain']),flush=True)


if __name__=='__main__':run()
