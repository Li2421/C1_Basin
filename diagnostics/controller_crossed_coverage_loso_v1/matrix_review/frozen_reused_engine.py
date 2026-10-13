from diagnostics.controller_q_diversity_v1.evaluate import donor_bootstrap
"""All-scene falsification on frozen DEVELOPMENT labels; no model selection."""
import itertools
import numpy as np
from .learn import FOLDS, CONDITIONS, SEEDS, SCENES, INDICES, centered
from .study import HERE, read, write, sha


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or min(a.std(), b.std()) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def select(a, chosen):
    return np.take_along_axis(a, chosen[..., None], -1)[..., 0]


def selection_difference(lo, hi, first, second):
    equal = first == second
    lower = np.where(equal, 0., select(lo, first) - select(hi, second))
    upper = np.where(equal, 0., select(hi, first) - select(lo, second))
    return lower, upper


def mse_gain(base, model, lo, hi):
    # (base-y)^2-(model-y)^2 is affine in y: exact censor bounds.
    a = (base-lo)**2 - (model-lo)**2
    b = (base-hi)**2 - (model-hi)**2
    return np.minimum(a, b), np.maximum(a, b)


def intervals(a, b, boot, key, arrays):
    a, b = np.asarray(a), np.asarray(b)
    assert a.ndim == b.ndim == 1
    al, bu = a[boot].mean(1), b[boot].mean(1)
    arrays[key+'_lower'], arrays[key+'_upper'] = al, bu
    return dict(identification_bounds=[float(a.mean()), float(b.mean())],
                censor_CI95=[float(np.quantile(al,.025)), float(np.quantile(bu,.975))])


def derangements(n):
    rng = np.random.default_rng(2026101402 + n)
    answer = []
    while len(answer) < 200:
        a = rng.permutation(n)
        if (a != np.arange(n)).all():
            answer.append(a)
    return np.asarray(answer)


def centered_axes(a, axes):
    for axis in axes:
        a = a - a.mean(axis, keepdims=True)
    return a


def reversal(lo, hi, q, boot):
    # Last axis is independent rollout half. Threshold is fixed, symmetric,
    # censor-safe and discovery-only. All qualifying C/eta pairs are included.
    n, nc, ne, _ = lo.shape
    numerator, denominator = np.zeros(n), np.zeros(n)
    evaluation_numerator, evaluation_denominator = np.zeros(n), np.zeros(n)
    pair_adv_lo, pair_adv_hi, pair_count = np.zeros(n), np.zeros(n), np.zeros(n)
    cases, both = [], []
    for c in range(nc):
        for a, b in itertools.combinations(range(ne), 2):
            positive = lo[:,c,a,0]-hi[:,c,b,0] >= .5
            negative = lo[:,c,b,0]-hi[:,c,a,0] >= .5
            if min(positive.sum(),negative.sum()) < 2:
                continue
            keep = positive | negative
            diff = q[:,c,a]-q[:,c,b]
            score = np.where(diff == 0, .5, np.where(positive, diff > 0, diff < 0))
            numerator[keep] += score[keep]
            denominator[keep] += 1
            ed = (lo[:,c,a,1]+hi[:,c,a,1]-lo[:,c,b,1]-hi[:,c,b,1])/2
            eg = keep & ((lo[:,c,a,1] > hi[:,c,b,1]) | (lo[:,c,b,1] > hi[:,c,a,1]))
            evaluation_numerator[eg] += np.where(diff[eg] == 0, .5, np.sign(diff[eg]) == np.sign(ed[eg]))
            evaluation_denominator[eg] += 1
            chosen = np.where(diff >= 0, a, b)
            opposite = np.where(diff >= 0, b, a)
            al, au = selection_difference(lo[:,c,:,1],hi[:,c,:,1],chosen,opposite)
            pair_adv_lo[keep] += al[keep]
            pair_adv_hi[keep] += au[keep]
            pair_count[keep] += 1
            a_truth = (lo[positive,c,a,1]-hi[positive,c,b,1]).mean()
            b_truth = (lo[negative,c,b,1]-hi[negative,c,a,1]).mean()
            both.append(float(score[positive].mean()*score[negative].mean()))
            cases.append(dict(controller_index=c, eta_indices=[INDICES[a],INDICES[b]],
                positive_states=int(positive.sum()), negative_states=int(negative.sum()),
                evaluation_positive_lower=float(a_truth),evaluation_negative_lower=float(b_truth),
                both_directions_confirmed=bool(a_truth>0 and b_truth>0),
                positive_model_accuracy=float(score[positive].mean()),negative_model_accuracy=float(score[negative].mean())))
    def ratio_interval(a,b):
        if not b.sum():
            return dict(value=None,CI95=None,cases=0,states=0)
        den=b[boot].sum(1);keep=den>0
        bs=a[boot].sum(1)[keep]/den[keep]
        return dict(value=float(a.sum()/b.sum()),CI95=np.quantile(bs,[.025,.975]).tolist(),
                    cases=int(b.sum()),states=int((b>0).sum()))
    return dict(qualifying_pairs=len(cases),independently_confirmed_pairs=sum(z['both_directions_confirmed'] for z in cases),
        discovery_switch_accuracy=ratio_interval(numerator,denominator),
        evaluation_agreement_on_discovery_cases=ratio_interval(evaluation_numerator,evaluation_denominator),
        both_sides_accuracy=float(np.mean(both)) if both else None,
        selected_pair_vs_opposite_evaluation_lower=ratio_interval(pair_adv_lo,pair_count),
        selected_pair_vs_opposite_evaluation_upper=ratio_interval(pair_adv_hi,pair_count),cases=cases)


def run():
    d=dict(np.load(HERE/'dataset.npz'));raw=dict(np.load(HERE/'raw_labels.npz'))
    rows, comparisons, mechanisms, failures, fixed_rows = [], [], [], [], []
    arrays, saved, training = {}, {}, []
    prediction={}
    for fold in FOLDS:
        prediction[fold]={}
        for condition in CONDITIONS:
            zs=[]
            for seed in SEEDS:
                path=HERE/'fits'/fold/condition/f'seed{seed}'
                complete=read(path/'complete.json')
                a=np.load(path/'predictions.npz')
                np.testing.assert_array_equal(a['eta_indices'],INDICES)
                zs.append(a['probability'])
                training.append(dict(fold=fold,condition=condition,seed=seed,**complete['selected']))
            prediction[fold][condition]=np.mean(zs,0)
    for i,j,k in np.argwhere(raw['collision']):
        failures.append(dict(type='collision',state_uid=str(d['state_uid'][i]),scene=str(d['scene'][i]),
            controller=str(d['controller'][i]),eta_index=int(j),seed=int(k),rollout_uid=str(raw['rollout_uid'][i,j,k])))
    for fold,held in FOLDS.items():
        scenes=SCENES if held is None else (held,)
        for sc in scenes:
            panels=('validation',) if held is None else ('all_held','validation','held_train_stratum')
            for panel in panels:
                keep=d['scene']==sc
                if panel=='validation':keep&=d['split']=='validation'
                if panel=='held_train_stratum':keep&=d['split']=='train'
                # Reference strata are prespecified, never selected by results.
                for reference in ('all','flow','goal'):
                    use=keep.copy()
                    if reference!='all':use&=d['kind']==reference
                    ix=np.flatnonzero(use);ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
                    sids=np.unique(d['state_uid'][ix]);controllers=np.unique(d['controller'][ix]);n,nc=len(sids),len(controllers)
                    assert len(ix)==n*nc
                    expected=np.repeat(sids,nc);np.testing.assert_array_equal(d['state_uid'][ix],expected)
                    shape=(n,nc,4)
                    rs,ru,seen=(raw[k][ix][:,list(INDICES)].reshape(shape+(4,)) for k in ('success','unknown','observed'))
                    counts=seen.sum(-1);lo=rs.sum(-1)/counts;hi=(rs+ru).sum(-1)/counts;mid=(lo+hi)/2
                    # Independent halves respect each row's frozen Q2/Q4 plan.
                    hlo=np.zeros(shape+(2,));hhi=hlo.copy()
                    for si in range(n):
                        for ci in range(nc):
                            nt=int(counts[si,ci,0]);assert nt in (2,4) and (counts[si,ci]==nt).all()
                            assert seen[si,ci,:,:nt].all() and not seen[si,ci,:,nt:].any()
                            for half,ids in enumerate((range(nt//2),range(nt//2,nt))):
                                ss=np.take(rs[si,ci],list(ids),axis=-1)
                                uu=np.take(ru[si,ci],list(ids),axis=-1)
                                hlo[si,ci,...,half]=ss.sum(-1)/(nt//2)
                                hhi[si,ci,...,half]=(ss+uu).sum(-1)/(nt//2)
                    boot=np.random.default_rng(2026101401+n).integers(n,size=(20000,n))
                    permutations=derangements(n)
                    prefix=f'{fold}__{sc}__{panel}__{reference}'
                    saved[prefix+'__state_uid']=sids;saved[prefix+'__controllers']=controllers
                    saved[prefix+'__mismatch_permutations']=permutations
                    saved[prefix+'__success']=rs;saved[prefix+'__unknown']=ru;saved[prefix+'__seen']=seen
                    pp={c:p[ix].reshape(shape) for c,p in prediction[fold].items()}
                    base=pp['eta_only_same_all_labels'];bsel=base.argmax(-1)
                    chosen_map={c:q.argmax(-1) for c,q in pp.items()}
                    meta=dict(fold=fold,scene=sc,panel=panel,reference=reference,N=n,controllers=nc)
                    for c,q in pp.items():
                        name=prefix+'__'+c;chosen=chosen_map[c]
                        saved[name+'__probability']=q
                        selected_lo,selected_hi=select(lo,chosen),select(hi,chosen)
                        success_bounds=[float(selected_lo.mean()),float(selected_hi.mean())]
                        frequency=np.bincount(chosen.ravel(),minlength=4)/chosen.size
                        entropy=float(-np.sum(frequency[frequency>0]*np.log2(frequency[frequency>0])))
                        errlo=np.where(q<lo,(q-lo)**2,np.where(q>hi,(q-hi)**2,0))
                        errhi=np.maximum((q-lo)**2,(q-hi)**2)
                        statecorr=corr(centered_axes(q,[0]),centered_axes(mid,[0]))
                        interactioncorr=corr(centered_axes(q,[0,2]),centered_axes(mid,[0,2]))
                        ca,cb=np.triu_indices(nc,1)
                        ccontrast=corr((q[:,ca]-q[:,cb]),(mid[:,ca]-mid[:,cb]))
                        regret_lo=np.maximum(0,lo.max(-1)-selected_hi)
                        other_hi=hi.copy();np.put_along_axis(other_hi,chosen[...,None],selected_lo[...,None],-1)
                        regret_hi=np.maximum(0,other_hi.max(-1)-selected_lo)
                        rows.append(dict(**meta,condition=c,selected_success_bounds=success_bounds,
                            Q_MSE_identification_bounds=[float(errlo.mean()),float(errhi.mean())],
                            Q_MSE_midpoint_sensitivity=float(np.mean((q-mid)**2)),
                            oracle_regret_bounds=[float(regret_lo.mean()),float(regret_hi.mean())],
                            oracle_success_bounds=[float(lo.max(-1).mean()),float(hi.max(-1).mean())],
                            state_effect_corr_midpoint_sensitivity=statecorr,state_eta_corr_midpoint_sensitivity=interactioncorr,
                            controller_contrast_corr_midpoint_sensitivity=ccontrast,selection_frequency=frequency.tolist(),selection_entropy_bits=entropy,
                            numerical_unknown=int(ru.sum()),rollouts=int(seen.sum()),
                            selected_collision=int(select(raw['collision'][ix][:,list(INDICES)].sum(-1).reshape(shape),chosen).sum()),
                            selected_numerical_unknown=int(select(ru.sum(-1),chosen).sum()),
                            selected_timeout=int(select(raw['timeout'][ix][:,list(INDICES)].sum(-1).reshape(shape),chosen).sum()),
                            selected_deadlock=int(select(raw['deadlock'][ix][:,list(INDICES)].sum(-1).reshape(shape),chosen).sum())))
                        dl,du=selection_difference(lo,hi,chosen,bsel)
                        comparisons.append(dict(**meta,condition=c,comparison='selection_minus_eta_only',
                            **intervals(dl.mean(1),du.mean(1),boot,name+'__vs_eta',arrays)))
                        # Every possible fixed eta, including per-controller fixed
                        # policies, with baseline re-selection in each bootstrap.
                        difference_lo=np.empty(shape);difference_hi=np.empty(shape)
                        for j in range(4):
                            fj=np.full(chosen.shape,j);a,b=selection_difference(lo,hi,chosen,fj)
                            difference_lo[...,j],difference_hi[...,j]=a,b
                            fixed_rows.append(dict(**meta,condition=c,eta_index=INDICES[j],
                                fixed_success_bounds=[float(lo[...,j].mean()),float(hi[...,j].mean())],
                                **intervals(a.mean(1),b.mean(1),boot,name+f'__vs_fixed{j}',arrays)))
                        # Averaging controller first enforces one global eta;
                        # minimizing before controller average gives each C its own eta.
                        for policy in ('global','per_controller'):
                            al,bu=[],[]
                            for start in range(0,len(boot),1000):
                                ii=boot[start:start+1000]
                                a=difference_lo[ii].mean(1);b=difference_hi[ii].mean(1)
                                if policy=='global':
                                    al.append(a.mean(1).min(-1));bu.append(b.mean(1).min(-1))
                                else:
                                    al.append(a.min(-1).mean(-1));bu.append(b.min(-1).mean(-1))
                            al,bu=np.concatenate(al),np.concatenate(bu)
                            a,b=difference_lo.mean(0),difference_hi.mean(0)
                            point=[float(a.mean(0).min()),float(b.mean(0).min())] if policy=='global' else [float(a.min(-1).mean()),float(b.min(-1).mean())]
                            arrays[name+'__fixed_'+policy+'_lower'],arrays[name+'__fixed_'+policy+'_upper']=al,bu
                            comparisons.append(dict(**meta,condition=c,comparison='selection_minus_posthoc_best_fixed_'+policy,
                                identification_bounds=point,censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(bu,.975))]))
                        fixed_disc=hlo[...,0].mean(0).argmax(-1)
                        dl,du=selection_difference(hlo[...,1],hhi[...,1],chosen,np.broadcast_to(fixed_disc,chosen.shape))
                        comparisons.append(dict(**meta,condition=c,comparison='evaluation_selection_minus_discovery_per_controller_fixed',
                            **intervals(dl.mean(1),du.mean(1),boot,name+'__vs_discovery_fixed',arrays)))
                        # Exact expectation over every wrong state donor. This
                        # equals the expectation of uniform random derangements.
                        ml,mu,ql,qu=(np.zeros((n,n)) for _ in range(4))
                        rate=np.zeros(n)
                        for donor in range(n):
                            wrong=np.broadcast_to(chosen[donor],chosen.shape)
                            a,b=selection_difference(lo,hi,chosen,wrong)
                            ml[:,donor]=a.mean(1);mu[:,donor]=b.mean(1)
                            a,b=mse_gain(q[donor],q,lo,hi)
                            ql[:,donor]=a.mean((1,2));qu[:,donor]=b.mean((1,2))
                            rate+=(wrong!=chosen).mean(1)/(n-1)
                        perm_selected=chosen[permutations];saved[name+'__mismatch_selected']=perm_selected
                        for label,a,b,suffix in [('matched_minus_same_scene_wrong_state',ml,mu,'__mismatch'),('wrong_state_Q_MSE_minus_matched',ql,qu,'__mismatch_Q')]:
                            pl,al,okl=donor_bootstrap(a,boot);pu,au,oku=donor_bootstrap(b,boot);good=okl&oku
                            arrays[name+suffix+'_lower']=al;arrays[name+suffix+'_upper']=au
                            comparisons.append(dict(**meta,condition=c,comparison=label,
                                identification_bounds=[pl,pu],censor_CI95=[float(np.quantile(al[good],.025)),float(np.quantile(au[good],.975))]))
                        mechanisms.append(dict(**meta,condition=c,state_selection_change_rate=float(rate.mean()),
                            reversal=reversal(hlo,hhi,q,boot),independent_half_state_eta_corr=corr(
                                centered_axes((hlo[...,0]+hhi[...,0])/2,[0,2]),
                                centered_axes((hlo[...,1]+hhi[...,1])/2,[0,2]))))
                        for donor_policy in ('all','same_reference'):
                            ql,qu=np.zeros(n),np.zeros(n)
                            for j in range(nc):
                                candidates=[k for k in range(nc) if k!=j and (donor_policy=='all' or
                                    controllers[k].split('_')[0]==controllers[j].split('_')[0])]
                                for k in candidates:
                                    a,b=mse_gain(q[:,k],q[:,j],lo[:,j],hi[:,j])
                                    ql+=a.mean(-1)/len(candidates)/nc;qu+=b.mean(-1)/len(candidates)/nc
                            comparisons.append(dict(**meta,condition=c,comparison='wrong_controller_'+donor_policy+'_Q_MSE_minus_matched',
                                **intervals(ql,qu,boot,name+'__wrong_controller_'+donor_policy,arrays)))
                    # Factorial controls: predictions are never used to choose
                    # target rows, controller families, candidates or thresholds.
                    main=pp['encounter_same_all_labels']
                    for control in ('field_same_all_labels','encounter_flow_eta_prior'):
                        a,b=mse_gain(pp[control],main,lo,hi)
                        comparisons.append(dict(**meta,condition='encounter_same_all_labels',comparison='Q_MSE_improvement_over_'+control,
                            **intervals(a.mean((1,2)),b.mean((1,2)),boot,prefix+'__Q_vs_'+control,arrays)))
                        a,b=selection_difference(lo,hi,chosen_map['encounter_same_all_labels'],chosen_map[control])
                        comparisons.append(dict(**meta,condition='encounter_same_all_labels',comparison='selection_improvement_over_'+control,
                            **intervals(a.mean(1),b.mean(1),boot,prefix+'__selection_vs_'+control,arrays)))
    np.savez_compressed(HERE/'development_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/'development_predictions.npz',**saved)
    write(HERE/'development_results.json',dict(rows=rows,comparisons=comparisons,fixed_policies=fixed_rows,
        mechanisms=mechanisms,collision_cases=failures,checkpoint_choices=training,
        protocol_sha256=sha(HERE/'evaluation_protocol.json'),
        scope='DEVELOPMENT only:40held/8within-scene states, K4 diagnostic, Q2/Q4 labels. Midpoint correlations are censor sensitivity values, not identified true-Q statistics. No formal fresh/K16/B15 claim.'))
    for r in rows:
        if r['panel']=='all_held' and r['reference']=='flow' and r['condition'] in ('eta_only_same_all_labels','encounter_same_all_labels'):
            print({k:r[k] for k in ('scene','condition','selected_success_bounds','state_eta_corr_midpoint_sensitivity')},flush=True)


if __name__=='__main__':
    run()
