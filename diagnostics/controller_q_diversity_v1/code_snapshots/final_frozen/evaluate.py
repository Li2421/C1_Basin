"""Paired, state-clustered controller/state falsification on sealed confirmation."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
import numpy as np
from .experiment import HERE,read,write,sha,connect
from .train import KINDS


def corr(a,b,mask=None):
    a=np.asarray(a);b=np.asarray(b)
    if mask is None:mask=np.ones(a.shape,bool)
    x=a[mask];y=b[mask]
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(np.corrcoef(x,y)[0,1])


def csvwrite(path,rows):
    with open(path,'w') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)


def interval(samples):
    return [float(x) for x in np.quantile(samples,[.025,.975])]


def masked_mean(value,mask,axis):
    count=np.sum(mask,axis=axis)
    return np.sum(np.where(mask,value,0.),axis=axis)/np.maximum(count,1)


def center_known(value,mask):
    return value-masked_mean(value,mask,axis=0)[None,...]


def donor_bootstrap(matrix,boot):
    """Resample both recipient states and the mismatched donor distribution.

    Each row is a recipient and each column a donor. A mismatch must always
    use a different physical state, including when bootstrap IDs repeat.
    """
    n=matrix.shape[0];weights=np.zeros((len(boot),n),float)
    np.add.at(weights,(np.repeat(np.arange(len(boot)),n),boot.ravel()),1.)
    denominator=n-weights
    numerator=weights@matrix.T-weights*np.diag(matrix)[None,:]
    values=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    draws=(weights*values).sum(1)/n
    valid=np.all((denominator>0)|(weights==0),axis=1)
    estimate=(matrix.sum()-np.trace(matrix))/(n*(n-1))
    return float(estimate),draws,valid


def square_gain_bounds(baseline,model,lo,hi):
    # (baseline-Q)^2 - (model-Q)^2 is affine in Q: extrema at interval ends.
    x=(baseline-lo)**2-(model-lo)**2;y=(baseline-hi)**2-(model-hi)**2
    return np.minimum(x,y),np.maximum(x,y)


def interaction_skill_bounds(predicted_controller_difference,truth_lo,truth_hi,remove_eta_mean=False):
    """Full-support skill bounds without assuming numerical censoring is random.

    Center the predicted controller contrast over physical states, and also
    over eta for the three-way interaction. Orthogonality makes squared-error
    skill affine in the uncentered truth: 2<Y,P prediction>-||P prediction||².
    Thus unknown outcomes can be bounded exactly at interval endpoints even
    when there are no fully observed physical-state panels.
    """
    value=predicted_controller_difference-predicted_controller_difference.mean(-2,keepdims=True)
    if remove_eta_mean:value=value-value.mean(-1,keepdims=True)
    a=2*truth_lo*value-value**2;b=2*truth_hi*value-value**2
    return np.minimum(a,b).mean((-2,-1)),np.maximum(a,b).mean((-2,-1))


def reversal(pred,lo,hi,evlo,evhi,bootstrap):
    n,c,k=pred.shape;a,b=np.triu_indices(k,1);scores=[];counts=[];records=[];switch_lo=[];switch_hi=[]
    state_correct=np.zeros(n);state_total=np.zeros(n)
    eval_correct=np.zeros(n);eval_total=np.zeros(n)
    for ci in range(c):
        direction=np.where(lo[:,ci,a]-hi[:,ci,b]>=.375,1,
                           np.where(lo[:,ci,b]-hi[:,ci,a]>=.375,-1,0))
        strong=(np.sum(direction==1,axis=0)>=3)&(np.sum(direction==-1,axis=0)>=3)
        pp=np.sign(pred[:,ci,a]-pred[:,ci,b])
        ee=np.where(evlo[:,ci,a]>evhi[:,ci,b],1,np.where(evlo[:,ci,b]>evhi[:,ci,a],-1,0))
        for j in np.flatnonzero(strong):
            mask=direction[:,j]!=0;correct=(pp[:,j]==direction[:,j]).astype(float)+.5*(pp[:,j]==0)
            state_correct+=correct*mask;state_total+=mask
            resolved=mask&(ee[:,j]!=0)
            independent_correct=(pp[:,j]==ee[:,j]).astype(float)+.5*(pp[:,j]==0)
            eval_correct+=independent_correct*resolved;eval_total+=resolved
            positive=direction[:,j]==1;negative=direction[:,j]==-1
            scores.append(.5*(correct[positive].mean()+correct[negative].mean()))
            counts.append(float(correct[positive].mean()*correct[negative].mean()))
            pn=positive[bootstrap].sum(1);nn=negative[bootstrap].sum(1)
            pa=(correct*positive)[bootstrap].sum(1)/np.maximum(pn,1)
            na=(correct*negative)[bootstrap].sum(1)/np.maximum(nn,1)
            switch_lo.append(pa*na)
            switch_hi.append(np.where(pn>0,pa,1.)*np.where(nn>0,na,1.))
            records.append(dict(controller_index=ci,eta_a=int(a[j]),eta_b=int(b[j]),
                 positive_states=int(positive.sum()),negative_states=int(negative.sum()),
                 balanced_direction_accuracy=scores[-1],both_sides_switch_accuracy=counts[-1],
                 evaluation_reversal_verified_fraction=float(np.mean(ee[mask,j]==direction[mask,j])),
                 evaluation_ambiguous_fraction=float(np.mean(ee[mask,j]==0)),
                 independent_evaluation_resolved_cases=int(resolved.sum()),
                 independent_evaluation_direction_accuracy=float(independent_correct[resolved].mean()) if resolved.any() else None))
    bs=state_correct[bootstrap].sum(1)/np.maximum(state_total[bootstrap].sum(1),1)
    eval_denominator=eval_total[bootstrap].sum(1)
    eval_bs=eval_correct[bootstrap].sum(1)/np.maximum(eval_denominator,1)
    summary=dict(reversal_pairs=len(records),state_pair_cases=int(state_total.sum()),
         physical_states=int(np.sum(state_total>0)),balanced_direction_accuracy=float(np.mean(scores)) if scores else None,
         switch_accuracy=float(np.mean(counts)) if counts else None,
         switch_95CI=[float(np.quantile(np.mean(switch_lo,0),.025)),float(np.quantile(np.mean(switch_hi,0),.975))] if counts else None,
         micro_direction_accuracy=float(state_correct.sum()/state_total.sum()) if state_total.sum() else None,
         micro_accuracy_95CI=interval(bs) if state_total.sum() else None,
         independent_evaluation_resolved_cases=int(eval_total.sum()),
         independent_evaluation_direction_accuracy=float(eval_correct.sum()/eval_total.sum()) if eval_total.sum() else None,
         independent_evaluation_direction_95CI=interval(eval_bs[eval_denominator>0]) if eval_total.sum() else None)
    return summary,records


def run():
    frozen=read(HERE/'models_frozen.json');guard=read(HERE/'confirmation_predictions_frozen.json')
    assert guard['predictions_sha256']==sha(HERE/'confirmation_predictions.npz')
    d=np.load(HERE/'data/final/dataset.npz');p=np.load(HERE/'confirmation_predictions.npz')
    for k in ('state_uid','controller','scene'):np.testing.assert_array_equal(d[k],p[k])
    assert np.all(d['split']=='final')
    source=np.load(HERE/'data'/frozen.get('data_variant',f"source{frozen['train_limit']}")/'dataset.npz')
    training=source['split']=='train'
    source_q=source['success'][training]/(source['success'][training]+source['failure'][training]+source['unknown'][training])
    source_groups=[];source_weights=[]
    for scene in sorted(set(source['scene'][training])):
        controllers=sorted(set(source['controller'][training]))
        for controller in controllers:
            mask=(source['scene'][training]==scene)&(source['controller'][training]==controller)
            if mask.any():
                source_groups.append(source_q[mask].mean(0))
                source_weights.append((.2 if controller=='flow' else .8/(len(controllers)-1))
                                      if 'flow' in controllers else 1./len(controllers))
    source_fixed=int(np.argmax(np.average(source_groups,axis=0,weights=source_weights)))
    safety={}
    with connect(True) as db:
        for r in read(HERE/'batches/safety_confirmation/cases.json'):
            value=db.execute('SELECT success,numerical_failure,collision FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            assert value is not None,'Safety baseline incomplete'
            key=(r['state_uid'],r['controller_name'].removesuffix('_safety'))
            if key not in safety:safety[key]=np.zeros((16,3),int)
            safety[key][r['seed']]=list(value)
    tables=[];comparisons=[];mechanisms=[];decisions=[];reversals=[];fixed_tables=[];boot_outputs={};seed_results=[]
    subset_rows=[];subset_comparisons=[];preference_rows=[]
    for scene in sorted(set(d['scene'])):
        ids=np.flatnonzero(d['scene']==scene);states=sorted(set(d['state_uid'][ids]));n=len(states)
        assert len(ids)==n*2
        ss=d['seed_success'][ids].reshape(n,2,16,16)
        uu=d['seed_unknown'][ids].reshape(n,2,16,16)
        cc=d['seed_collision'][ids].reshape(n,2,16,16)
        lo=ss.mean(-1);hi=(ss+uu).mean(-1)
        dl=ss[...,:8].mean(-1);dh=(ss+uu)[...,:8].mean(-1)
        el=ss[...,8:].mean(-1);eh=(ss+uu)[...,8:].mean(-1)
        known=16-uu.sum(-1);y=ss.sum(-1)/np.maximum(known,1)
        rng=np.random.default_rng(2026100921);boot=rng.integers(n,size=(20000,n));boot_outputs[scene+'_indices']=boot
        sa=np.stack([safety[(d['state_uid'][i],d['controller'][i])] for i in ids]).reshape(n,2,16,3)
        preds={kind:p[kind][:,ids].mean(0).reshape(n,2,16) for kind in KINDS}
        primary=frozen['primary'];basepred=preds[primary]
        preds['controller_mismatch']=p[primary+'_controller_mismatch'][:,ids].mean(0).reshape(n,2,16)
        choices={k:v.argmax(-1) for k,v in preds.items()}
        choices['TRAIN_fixed']=np.full((n,2),source_fixed,int)
        discfixed=np.argmax(dl.mean(0),axis=-1)
        choices['discovery_fixed']=np.tile(discfixed,(n,1))
        choices['evaluation_posthoc_fixed']=np.tile(np.argmax(el.mean(0),axis=-1),(n,1))
        choices['Oracle']=el.argmax(-1)
        row=np.arange(n)[:,None];ct=np.arange(2)[None,:]
        for kind in KINDS:
            for seed_index,seed in enumerate([17,23,41]):
                v=p[kind][seed_index,ids].reshape(n,2,16);chosen=v.argmax(-1)
                seed_results.append(dict(scene=scene,model=kind,seed=seed,physical_states=n,
                   Q_MAE=float(np.mean(np.abs(v-y)[known>0])),
                   evaluation8_success=float(el[row,ct,chosen].mean()),
                   B15_controller_state_cases=int((ss.sum(-1)[row,ct,chosen]>=15).sum())))
        measures={};selected_intervals={}
        for method,choice in choices.items():
            ev=el[row,ct,choice];full=lo[row,ct,choice];b15=ss.sum(-1)[row,ct,choice]>=15
            upper_b15=(ss+uu).sum(-1)[row,ct,choice]>=15
            if method=='Oracle':
                # Each upper bound independently optimizes its own metric.
                full=lo.max(-1);b15=(ss.sum(-1)>=15).any(-1);upper_b15=((ss+uu).sum(-1)>=15).any(-1)
            col=cc.sum(-1)[row,ct,choice];unknown=uu.sum(-1)[row,ct,choice]
            measures[method]=dict(eval_success=ev.mean(1),full_success=full.mean(1),B15=b15.mean(1),
                                   regret=(el.max(-1)-ev).mean(1))
            selected_intervals[method]=dict(eval_lower=ev,eval_upper=eh[row,ct,choice],
                B15_lower=b15.astype(float),B15_upper=upper_b15.astype(float))
            for ci,controller in enumerate(['analytic6','analytic7']):
                qmae=float(np.mean(np.abs(preds[method][:,ci]-y[:,ci])[known[:,ci]>0])) if method in preds else None
                qmse=float(np.mean((preds[method][:,ci]-y[:,ci])[known[:,ci]>0]**2)) if method in preds else None
                probability=np.clip(preds[method][:,ci],1e-6,1-1e-6) if method in preds else None
                qnll=float(np.sum(-ss[:,ci].sum(-1)*np.log(probability)-(known[:,ci]-ss[:,ci].sum(-1))*np.log1p(-probability))/known[:,ci].sum()) if method in preds else None
                tables.append(dict(scene=scene,controller=controller,method=method,physical_states=n,
                    B15_count=int(b15[:,ci].sum()),B15_upper_with_numerical_unknown=int(upper_b15[:,ci].sum()),
                    B15_rate=float(b15[:,ci].mean()),evaluation8_success=float(ev[:,ci].mean()),
                    all16_success=float(full[:,ci].mean()),oracle_regret=float(np.mean(el[:,ci].max(-1)-ev[:,ci])),
                    Q_MAE_known_outcomes=qmae,Q_MSE_known_outcomes=qmse,NLL_known_outcomes=qnll,selected_collision_count=int(col[:,ci].sum()),
                    selected_collision_rate=float(col[:,ci].sum()/(n*16)),selected_numerical_count=int(unknown[:,ci].sum()),
                    caveat='Discovery-selected fixed B15 is descriptive; independent comparison uses evaluation8' if method=='discovery_fixed' else 'Evaluation-posthoc upper-bound comparator' if method=='evaluation_posthoc_fixed' else 'Metric-specific oracle upper bounds; evaluation8 choice may differ from all16/B15 choice; collision/numerical fields refer to evaluation8 choice' if method=='Oracle' else ''))
                for j,sid in enumerate(states):
                    decisions.append(dict(scene=scene,state_uid=sid,controller=controller,method=method,
                       eta_index=int(choice[j,ci]),evaluation_success=float(ev[j,ci]),
                       B15=int(ss.sum(-1)[j,ci,choice[j,ci]]>=15),
                       metric_specific_B15_upper_bound=int(b15[j,ci]) if method=='Oracle' else None,
                       collisions=int(col[j,ci]),numerical=int(unknown[j,ci])))
        measures['Safety']=dict(eval_success=sa[:,:,8:,0].mean((1,2)),full_success=sa[:,:,:,0].mean((1,2)),
                                 B15=(sa[:,:,:,0].sum(-1)>=15).mean(1),regret=el.max(-1).mean(1)-sa[:,:,8:,0].mean((1,2)))
        selected_intervals['Safety']=dict(eval_lower=sa[:,:,8:,0].mean(-1),
            eval_upper=(sa[:,:,8:,0]+sa[:,:,8:,1]).mean(-1),
            B15_lower=(sa[:,:,:,0].sum(-1)>=15).astype(float),
            B15_upper=((sa[:,:,:,0]+sa[:,:,:,1]).sum(-1)>=15).astype(float))
        for ci,cname in enumerate(['analytic6','analytic7']):
            v=sa[:,ci];tables.append(dict(scene=scene,controller=cname,method='Safety',physical_states=n,
                B15_count=int((v[:,:,0].sum(-1)>=15).sum()),B15_upper_with_numerical_unknown=int(((v[:,:,0]+v[:,:,1]).sum(-1)>=15).sum()),
                B15_rate=float((v[:,:,0].sum(-1)>=15).mean()),evaluation8_success=float(v[:,8:,0].mean()),all16_success=float(v[:,:,0].mean()),
                oracle_regret=float(el[:,ci].max(-1).mean()-v[:,8:,0].mean()),Q_MAE_known_outcomes=None,
                selected_collision_count=int(v[:,:,2].sum()),selected_collision_rate=float(v[:,:,2].mean()),
                selected_numerical_count=int(v[:,:,1].sum()),caveat='Frozen terminal safety projection without eta correction'))
            for j,sid in enumerate(states):
                decisions.append(dict(scene=scene,state_uid=sid,controller=cname,method='Safety',
                    eta_index=None,evaluation_success=float(v[j,8:,0].mean()),B15=int(v[j,:,0].sum()>=15),
                    metric_specific_B15_upper_bound=None,collisions=int(v[j,:,2].sum()),numerical=int(v[j,:,1].sum())))
        # The physical state is permuted jointly in its encoding and all its
        # controller queries. Keep controller identity and scene unchanged.
        global_lookup={v:i for i,v in enumerate(ids)};matched=p[primary].mean(0)
        permutation_q=[];permutation_b=[];permutation_error=[];permutation_collision=[];permutation_unknown=[]
        permutation_qu=[];permutation_bu=[];permutation_error_bound=[]
        for perm in p['state_permutations']:
            pp=matched[perm[ids]].reshape(n,2,16);choice=pp.argmax(-1)
            permutation_q.append(el[row,ct,choice]);permutation_b.append(ss.sum(-1)[row,ct,choice]>=15)
            permutation_error.append(masked_mean((pp-y)**2,known>0,axis=(1,2)))
            permutation_collision.append(cc.sum(-1)[row,ct,choice]);permutation_unknown.append(uu.sum(-1)[row,ct,choice])
            permutation_qu.append(eh[row,ct,choice]);permutation_bu.append((ss+uu).sum(-1)[row,ct,choice]>=15)
            bound,_=square_gain_bounds(pp,basepred,lo,hi)
            permutation_error_bound.append(bound.mean((1,2)))
        pmq=np.mean(permutation_q,0);pmb=np.mean(permutation_b,0)
        measures['state_mismatch']=dict(eval_success=pmq.mean(1),B15=pmb.mean(1),
            regret=(el.max(-1)-pmq).mean(1))
        selected_intervals['state_mismatch']=dict(eval_lower=pmq,eval_upper=np.mean(permutation_qu,0),
            B15_lower=pmb,B15_upper=np.mean(permutation_bu,0))
        groups={}
        fixed_disc_upper=dh[row,ct,np.tile(discfixed,(n,1))]
        for margin in (2,3,4):
            groups[f'adaptive_demand_{margin}of8']=dl.max(-1)-fixed_disc_upper>=margin/8
        aa,bb=np.triu_indices(16,1);strong_states=np.zeros((n,2),bool)
        for ci in range(2):
            direction=np.where(dl[:,ci,aa]-dh[:,ci,bb]>=.375,1,
                np.where(dl[:,ci,bb]-dh[:,ci,aa]>=.375,-1,0))
            eligible=(np.sum(direction==1,axis=0)>=3)&(np.sum(direction==-1,axis=0)>=3)
            strong_states[:,ci]=np.any((direction!=0)&eligible[None,:],axis=1)
            preferred=dl[:,ci].argmax(-1);selected=choices[primary][:,ci]
            for j in range(16):
                mask=preferred==j
                preference_rows.append(dict(scene=scene,controller=f'analytic{ci+6}',discovery_preferred_eta=j,
                    physical_states=int(mask.sum()),discovery_best_fixed_eta=int(discfixed[ci]),
                    adaptive_demand_states=int((groups['adaptive_demand_3of8'][:,ci]&mask).sum()),
                    selected_eta_counts=np.bincount(selected[mask],minlength=16).tolist(),
                    tie_break='lowest candidate index; zero or tied preference is not counted as strict demand'))
        groups['strong_reversal']=strong_states
        for group,mask in groups.items():
            total=int(mask.sum());count=int(mask.any(1).sum())
            if not total:
                subset_rows.append(dict(scene=scene,group=group,method=None,physical_states=0,
                    controller_state_cases=0,evaluation8_success=None,B15_descriptive=None,regret=None))
                continue
            for method,values in selected_intervals.items():
                subset_rows.append(dict(scene=scene,group=group,method=method,physical_states=count,
                    controller_state_cases=total,evaluation8_success=float(values['eval_lower'][mask].mean()),
                    B15_descriptive=float(values['B15_lower'][mask].mean()),
                    regret=float((el.max(-1)-values['eval_lower'])[mask].mean())))
            weights=mask.sum(1);den=weights[boot].sum(1);valid_draw=den>0
            for baseline in ('eta_only','state_only','TRAIN_fixed','discovery_fixed','controller_mismatch','state_mismatch','Safety'):
                value=(selected_intervals[primary]['eval_lower']-selected_intervals[baseline]['eval_lower'])*mask
                samples=value.sum(1)[boot].sum(1)[valid_draw]/den[valid_draw]
                ci=interval(samples)
                subset_comparisons.append(dict(scene=scene,group=group,comparison=f'{primary}-{baseline}',
                    physical_states=count,controller_state_cases=total,mean_difference=float(value.sum()/total),
                    CI95_low=ci[0],CI95_high=ci[1],empty_bootstrap_fraction=float(np.mean(~valid_draw))))
                boot_outputs['_'.join((scene,group,baseline,'eval8'))]=samples
        for ci,cname in enumerate(['analytic6','analytic7']):
            tables.append(dict(scene=scene,controller=cname,method='state_mismatch_100',physical_states=n,
                B15_count=float(pmb[:,ci].sum()),B15_upper_with_numerical_unknown=None,B15_rate=float(pmb[:,ci].mean()),
                evaluation8_success=float(pmq[:,ci].mean()),all16_success=None,oracle_regret=float((el[:,ci].max(-1)-pmq[:,ci]).mean()),
                Q_MAE_known_outcomes=None,selected_collision_count=float(np.mean(permutation_collision,0)[:,ci].sum()),
                selected_collision_rate=float(np.mean(permutation_collision,0)[:,ci].sum()/(n*16)),
                selected_numerical_count=float(np.mean(permutation_unknown,0)[:,ci].sum()),
                caveat='Mean of100 fixed same-scene, same-controller state derangements; fractional robust counts'))
        def compare(label,values):
            samples=np.asarray(values)[boot].mean(1)
            comparisons.append(dict(scene=scene,comparison=label,physical_states=n,mean_difference=float(np.mean(values)),CI95_low=interval(samples)[0],CI95_high=interval(samples)[1]))
            boot_outputs[scene+'_'+label]=samples
        # For two equally weighted controllers, a predictor that receives no
        # controller information has minimum MSE (Q_c0-Q_c1)^2/4 even with an
        # otherwise perfect state/eta representation. Independent seed halves
        # avoid the upward Monte Carlo bias of squaring a noisy Q difference.
        delta_disc=dl[:,0]-dl[:,1];delta_eval=el[:,0]-el[:,1]
        compare('no_controller_Q_MSE_floor_cross_half_verified_success',
                (delta_disc*delta_eval/4).mean(1))
        lower_disc=dl[:,0]-dh[:,1];upper_disc=dh[:,0]-dl[:,1]
        lower_eval=el[:,0]-eh[:,1];upper_eval=eh[:,0]-el[:,1]
        products=np.stack([lower_disc*lower_eval,lower_disc*upper_eval,
                           upper_disc*lower_eval,upper_disc*upper_eval])
        compare('no_controller_Q_MSE_floor_cross_half_censor_worst',(products.min(0)/4).mean(1))
        compare('no_controller_Q_MSE_floor_cross_half_censor_best',(products.max(0)/4).mean(1))
        for baseline in ['eta_only','state_only','Safety','TRAIN_fixed','discovery_fixed','evaluation_posthoc_fixed','controller_mismatch','state_mismatch']:
            for metric in ['eval_success','B15']:
                if baseline=='discovery_fixed' and metric=='B15':continue
                compare(f'{primary}-{baseline}_{metric}',measures[primary][metric]-measures[baseline][metric])
                key='eval' if metric=='eval_success' else 'B15'
                bound=selected_intervals[primary][key+'_lower']-selected_intervals[baseline][key+'_upper']
                compare(f'{primary}-{baseline}_{metric}_censor_worst',bound.mean(1))
        for baseline in ['eta_only','state_only','controller_mismatch']:
            gain=(preds[baseline]-y)**2-(basepred-y)**2
            compare(f'{baseline}-matched_Q_MSE',masked_mean(gain,known>0,axis=(1,2)))
            lower,upper=square_gain_bounds(preds[baseline],basepred,lo,hi)
            compare(f'{baseline}-matched_Q_MSE_censor_worst',lower.mean((1,2)))
            operational_gain=(preds[baseline]-lo)**2-(basepred-lo)**2
            compare(f'{baseline}-matched_verified_success_Q_MSE',operational_gain.mean((1,2)))
        compare('state_mismatch-matched_Q_MSE',np.mean(permutation_error,0)-masked_mean(np.square(basepred-y),known>0,axis=(1,2)))
        compare('state_mismatch-matched_Q_MSE_censor_worst',np.mean(permutation_error_bound,0))
        # Exact expected mismatch across all n-1 other physical states. The
        # joint bootstrap resamples donor predictions as well as recipient
        # outcomes, avoiding conditional-on-fixed-donors confidence intervals.
        donors=choices[primary]
        value_grid=np.mean(np.stack([el[:,ci,donors[:,ci]] for ci in range(2)],axis=-1),axis=-1)
        robust_values=(ss.sum(-1)>=15).astype(float)
        robust_grid=np.mean(np.stack([robust_values[:,ci,donors[:,ci]] for ci in range(2)],axis=-1),axis=-1)
        error_grid=masked_mean((basepred[None,:,:,:]-y[:,None,:,:])**2,
            np.broadcast_to(known[:,None,:,:]>0,(n,n,2,16)),axis=(2,3))
        worst_grid,_=square_gain_bounds(basepred[None,:,:,:],basepred[:,None,:,:],lo[:,None,:,:],hi[:,None,:,:])
        joint_metrics={
            'matched-exact_state_mismatch_eval8':(value_grid,measures[primary]['eval_success'],1.),
            'matched-exact_state_mismatch_B15':(robust_grid,measures[primary]['B15'],1.),
            'exact_state_mismatch-matched_Q_MSE':(error_grid,masked_mean((basepred-y)**2,known>0,axis=(1,2)),-1.),
            'exact_state_mismatch-matched_Q_MSE_censor_worst':(worst_grid.mean((2,3)),np.zeros(n),-1.),
        }
        for name,(matrix,matched_value,sign) in joint_metrics.items():
            estimate,samples,valid_draws=donor_bootstrap(matrix,boot)
            gain=sign*(matched_value[boot].mean(1)-samples);ci=interval(gain[valid_draws])
            comparisons.append(dict(scene=scene,comparison=name+'_joint_state_bootstrap',physical_states=n,
                mean_difference=float(sign*(matched_value.mean()-estimate)),CI95_low=ci[0],CI95_high=ci[1]))
            boot_outputs[scene+'_'+name+'_joint']=gain[valid_draws]
        for ci,cname in enumerate(['analytic6','analytic7']):
            vm=el[:,ci,donors[:,ci]];bm=robust_values[:,ci,donors[:,ci]]
            wrong_value=(vm.sum(1)-np.diag(vm))/(n-1);wrong_robust=(bm.sum(1)-np.diag(bm))/(n-1)
            tables.append(dict(scene=scene,controller=cname,method='state_mismatch_exact',physical_states=n,
                B15_count=float(wrong_robust.sum()),B15_upper_with_numerical_unknown=None,B15_rate=float(wrong_robust.mean()),
                evaluation8_success=float(wrong_value.mean()),all16_success=None,
                oracle_regret=float((el[:,ci].max(-1)-wrong_value).mean()),Q_MAE_known_outcomes=None,
                selected_collision_count=None,selected_collision_rate=None,selected_numerical_count=None,
                caveat='Exact expected mismatch across all other physical states; paired CI resamples recipients and donors jointly'))
        delta_truth=y[:,0]-y[:,1];delta_model=basepred[:,0]-basepred[:,1]
        valid_delta=(known[:,0]>0)&(known[:,1]>0)
        compare('controller_contrast_MSE_skill',masked_mean(delta_truth**2-(delta_model-delta_truth)**2,valid_delta,axis=1))
        worst,_=square_gain_bounds(np.zeros_like(delta_model),delta_model,lo[:,0]-hi[:,1],hi[:,0]-lo[:,1])
        compare('controller_contrast_MSE_skill_censor_worst',worst.mean(1))
        contrast_lo=lo[:,0]-hi[:,1];contrast_hi=hi[:,0]-lo[:,1]
        for label,remove_eta in [('state_controller_interaction_skill_full_support',False),
                                 ('state_controller_eta_interaction_skill_full_support',True)]:
            lower,upper=interaction_skill_bounds(delta_model,contrast_lo,contrast_hi,remove_eta)
            lower_bs,upper_bs=interaction_skill_bounds(delta_model[boot],contrast_lo[boot],contrast_hi[boot],remove_eta)
            comparisons.append(dict(scene=scene,comparison=label+'_censor_worst',physical_states=n,
                mean_difference=float(lower),CI95_low=interval(lower_bs)[0],CI95_high=interval(lower_bs)[1]))
            comparisons.append(dict(scene=scene,comparison=label+'_censor_best',physical_states=n,
                mean_difference=float(upper),CI95_low=interval(upper_bs)[0],CI95_high=interval(upper_bs)[1]))
            boot_outputs[scene+'_'+label+'_lower']=lower_bs;boot_outputs[scene+'_'+label+'_upper']=upper_bs
        tt=delta_truth[boot];mm=delta_model[boot];vm=valid_delta[boot]
        tt-=masked_mean(tt,vm,axis=1)[:,None,:];mm-=masked_mean(mm,vm,axis=1)[:,None,:]
        skill=masked_mean(tt**2-(mm-tt)**2,vm,axis=(1,2));dt=center_known(delta_truth,valid_delta);dm=center_known(delta_model,valid_delta)
        comparisons.append(dict(scene=scene,comparison='state_centered_controller_contrast_skill',physical_states=n,
             mean_difference=float(masked_mean(dt**2-(dm-dt)**2,valid_delta,axis=(0,1))),CI95_low=interval(skill)[0],CI95_high=interval(skill)[1]))
        boot_outputs[scene+'_state_centered_controller_contrast_skill']=skill
        # Remove controller-dependent eta-independent difficulty. A model that
        # only knows which controller is globally stronger cannot pass this.
        eta_true=delta_truth-masked_mean(delta_truth,valid_delta,axis=1)[:,None]
        eta_model=delta_model-masked_mean(delta_model,valid_delta,axis=1)[:,None]
        eta_skill=masked_mean(eta_true**2-(eta_model-eta_true)**2,valid_delta,axis=1)
        compare('controller_eta_interaction_MSE_skill',eta_skill)
        # Full state x controller x eta interaction, removing both state and
        # eta marginals on the common complete-observation panel.
        complete=(known==16).all(axis=1).all(axis=1)
        if complete.sum()>=3:
            tt=delta_truth[complete];mm=delta_model[complete]
            tt=tt-tt.mean(0)-tt.mean(1,keepdims=True)+tt.mean()
            mm=mm-mm.mean(0)-mm.mean(1,keepdims=True)+mm.mean()
            ii=np.random.default_rng(2026100940).integers(len(tt),size=(20000,len(tt)))
            bt=tt[ii];bm=mm[ii]
            bt-=bt.mean(1,keepdims=True);bm-=bm.mean(1,keepdims=True)
            bs=(bt**2-(bm-bt)**2).mean((1,2));ci=interval(bs)
            comparisons.append(dict(scene=scene,comparison='complete_panel_state_controller_eta_interaction_skill',
                physical_states=int(complete.sum()),mean_difference=float(np.mean(tt**2-(mm-tt)**2)),
                CI95_low=ci[0],CI95_high=ci[1]))
            boot_outputs[scene+'_complete_panel_three_way_skill']=bs
        # Strongest fixed-policy adversary: choose an eta independently for each
        # held controller, and reselect it inside every bootstrap replicate.
        observed=measures[primary]['eval_success'];resampled=el[boot].mean(1)
        gains=observed[boot].mean(1)-resampled.max(-1).mean(-1)
        comparisons.append(dict(scene=scene,comparison='matched-bootstrap_reselected_per_controller_fixed_eval8',physical_states=n,
            mean_difference=float(observed.mean()-el.mean(0).max(-1).mean()),CI95_low=interval(gains)[0],CI95_high=interval(gains)[1]))
        boot_outputs[scene+'_best_fixed_reselected']=gains
        robust=(ss.sum(-1)>=15).astype(float)
        gains=measures[primary]['B15'][boot].mean(1)-robust[boot].mean(1).max(-1).mean(-1)
        comparisons.append(dict(scene=scene,comparison='matched-bootstrap_reselected_per_controller_fixed_B15',physical_states=n,
             mean_difference=float(measures[primary]['B15'].mean()-robust.mean(0).max(-1).mean()),
             CI95_low=interval(gains)[0],CI95_high=interval(gains)[1]))
        boot_outputs[scene+'_best_fixed_B15_reselected']=gains
        for ci in range(2):
            for j in range(16):fixed_tables.append(dict(scene=scene,controller=f'analytic{6+ci}',eta_index=j,physical_states=n,
                 evaluation8_success=float(el[:,ci,j].mean()),B15_count=int((ss[:,ci,j].sum(-1)>=15).sum())))
        centered_truth=center_known(y,known>0)
        for kind,pred in preds.items():
            centered=center_known(pred,known>0);selection=pred.argmax(-1)
            hist=np.bincount(selection.ravel(),minlength=16)/selection.size
            stats,pairs=reversal(pred,dl,dh,el,eh,boot)
            stats['reversal_physical_states']=stats.pop('physical_states')
            reversals.extend([dict(scene=scene,model=kind,**r) for r in pairs])
            delta_true=y[:,0]-y[:,1];delta_pred=pred[:,0]-pred[:,1]
            mechanism=dict(scene=scene,model=kind,physical_states=n,
                state_effect_correlation=corr(centered,centered_truth,known>0),
                state_effect_half_reliability=corr(dl-dl.mean(0),el-el.mean(0)),
                controller_effect_correlation=corr(delta_pred,delta_true,valid_delta),
                state_centered_controller_effect_correlation=corr(center_known(delta_pred,valid_delta),center_known(delta_true,valid_delta),valid_delta),
                controller_contrast_MAE=float(np.mean(np.abs(delta_pred-delta_true)[valid_delta])),
                controller_eta_interaction_correlation=corr(
                    delta_pred-masked_mean(delta_pred,valid_delta,axis=1)[:,None],
                    delta_true-masked_mean(delta_true,valid_delta,axis=1)[:,None],valid_delta),
                selection_entropy_bits=float(-np.sum(hist[hist>0]*np.log2(hist[hist>0]))),
                controller_selection_change_rate=float(np.mean(selection[:,0]!=selection[:,1])),
                state_selection_change_rate=float(np.mean([1-sum(k*(k-1) for k in np.bincount(selection[:,ci],minlength=16))/(n*(n-1)) for ci in range(2)])),
                selection_frequency=json.dumps(hist.tolist()),**stats)
            mechanisms.append(mechanism)
        # Controller-induced eta reversals at exactly the same physical state.
        a,b=np.triu_indices(16,1)
        direction=np.where(dl[:,:,a]-dh[:,:,b]>=.375,1,np.where(dl[:,:,b]-dh[:,:,a]>=.375,-1,0))
        reversal_cases=direction[:,0]*direction[:,1]==-1
        eval_direction=np.where(el[:,:,a]>eh[:,:,b],1,np.where(el[:,:,b]>eh[:,:,a],-1,0))
        for kind,pred in preds.items():
            sign=np.sign(pred[:,:,a]-pred[:,:,b]);correct=np.all(sign==direction,axis=1)&reversal_cases
            denominator=reversal_cases.sum(1);numerator=correct.sum(1)
            samples=numerator[boot].sum(1)/np.maximum(denominator[boot].sum(1),1)
            mechanisms.append(dict(scene=scene,model=kind,physical_states=n,
               controller_induced_reversal_cases=int(denominator.sum()),
               both_controller_rankings_correct=float(numerator.sum()/denominator.sum()) if denominator.sum() else None,
               bootstrap_CI=interval(samples) if denominator.sum() else None,
               independent_evaluation_both_directions_verified=float(np.mean(np.all(eval_direction==direction,axis=1)[reversal_cases])) if denominator.sum() else None))
    csvwrite(HERE/'selection_tables.csv',tables);csvwrite(HERE/'paired_comparisons.csv',comparisons)
    csvwrite(HERE/'per_state_decisions.csv',decisions);csvwrite(HERE/'all_fixed_eta_policies.csv',fixed_tables)
    write(HERE/'mechanism_metrics.json',mechanisms);write(HERE/'all_strong_reversal_cases.json',reversals)
    write(HERE/'training_seed_confirmation_metrics.json',seed_results)
    csvwrite(HERE/'secondary_subset_tables.csv',subset_rows)
    if subset_comparisons:csvwrite(HERE/'secondary_subset_comparisons.csv',subset_comparisons)
    write(HERE/'natural_preference_groups.json',preference_rows)
    np.savez_compressed(HERE/'bootstrap_outputs.npz',**boot_outputs)
    write(HERE/'evaluation_complete.json',dict(primary=frozen['primary'],physical_states=len(set(d['state_uid'])),
        scene_counts={s:len(set(d['state_uid'][d['scene']==s])) for s in sorted(set(d['scene']))},
        candidate_rollouts=int(d['success'].sum()+d['failure'].sum()+d['unknown'].sum()),
        candidate_numerical=int(d['unknown'].sum()),candidate_collisions=int(d['seed_collision'].sum()),
        safety_rollouts=sum(len(v) for v in safety.values()),
        prediction_hash=sha(HERE/'confirmation_predictions.npz'),data_hash=sha(HERE/'data/final/dataset.npz'),
        test_retuning=False,scope='Two scenes jointly trained; unseen analytic controller parameters and fresh physical states. Not four-scene LOSO or unseen native Flow confirmation.'))
    print('State-clustered evaluation and all fixed/permutation controls complete',flush=True)


if __name__=='__main__':run()
