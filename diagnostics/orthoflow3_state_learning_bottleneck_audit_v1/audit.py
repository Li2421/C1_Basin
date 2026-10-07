"""Zero-rollout source-only state-learning adjudication; never reads target labels."""
from pathlib import Path
from itertools import combinations, permutations
import csv, hashlib, json, sqlite3
import numpy as np
from scipy.special import expit
from scipy.stats import binom, pearsonr, spearmanr

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SRC=OUT.parent/'orthoflow3_controller_state_residual_factorial_v1'
SEEDS=(17,23,41)

def read(p):return json.loads(Path(p).read_text())
def write(name,v):(OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load():
    d=dict(np.load(SRC/'variants/large_20/dataset.npz'))
    rows=read(SRC/'pairs.json');states=read(SRC/'states.json')
    ix=np.stack([np.flatnonzero(d['state_index']==j) for j in range(len(states))])
    assert ix.shape==(56,16)
    tr=np.array([j for j in range(56) if d['split'][ix[j,0]]=='train'])
    va=np.array([j for j in range(56) if d['split'][ix[j,0]]=='validation'])
    assert len(tr)==46 and len(va)==7
    assert not {states[j]['source_group'] for j in tr}&{states[j]['source_group'] for j in va}
    ys=np.full((3,56,16,16),np.nan)
    profiles=read(SRC/'protocol.json')['profiles']
    with sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True) as db:
        for c,profile in enumerate(profiles):
            for j in range(56):
                for e,i in enumerate(ix[j]):
                    rs=db.execute('''SELECT seed_key,success,numerical_failure FROM rollout
                        WHERE state_uid=? AND eta_uid=? AND controller_uid=?
                        AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0''',
                        (rows[i]['state_uid'],rows[i]['eta_uid'],profile['controller_uid'])).fetchall()
                    for sk,y,bad in rs:
                        k=json.loads(sk).get('future_index',-1)
                        if 0<=k<16 and not bad:ys[c,j,e,k]=y
    s=np.nansum(ys,-1);n=np.isfinite(ys).sum(-1)
    np.testing.assert_array_equal(s,d['standard_success'][:,ix])
    np.testing.assert_array_equal(n-s,d['standard_failure'][:,ix])
    active=np.r_[tr,va]
    write('data_integrity.json',{'canonical_seed_aggregation_matches_frozen_dataset':True,
        'database_read_only':True,'target_labels_opened':False,'new_rollouts':0,
        'TRAIN_families':len(tr),'VAL_families':len(va),'TRAIN_pairs':int(len(tr)*16*3),
        'VAL_pairs':int(len(va)*16*3),'source_group_overlap':0,
        'TRAIN_observed_count_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique((d['success']+d['failure'])[:,ix[tr]],return_counts=True))},
        'TRAIN_complete_standard_Q16_pairs':int((n[:,tr]==16).sum()),
        'VAL_complete_standard_Q16_pairs':int((n[:,va]==16).sum()),
        'VAL_numerical_or_unobserved_seed_count':int((16-n[:,va]).sum()),
        'dataset_sha256':digest(SRC/'variants/large_20/dataset.npz')})
    return d,rows,states,ix,tr,va,ys,s,n

def strong_reversals(s,n,denom=16):
    lo=s/denom;hi=(denom-(n-s))/denom
    result=[]
    for a,b in combinations(range(s.shape[1]),2):
        for c in range(3):
            for i,j in combinations(range(16),2):
                da=(lo[c,a,i]-hi[c,a,j],hi[c,a,i]-lo[c,a,j])
                db=(lo[c,b,i]-hi[c,b,j],hi[c,b,i]-lo[c,b,j])
                sa=1 if da[0]>=.25 else -1 if da[1]<=-.25 else 0
                sb=1 if db[0]>=.25 else -1 if db[1]<=-.25 else 0
                if sa*sb<0:result.append((c,a,b,i,j,sa))
    return np.asarray(result,int).reshape(-1,6)

def nll(p,s,n):
    p=np.clip(p,1e-7,1-1e-7)
    return float(np.mean((-(s*np.log(p)+(n-s)*np.log1p(-p))).sum((1,2))/n.sum((1,2))))

def selection(p,s,n):
    j=p.argmax(-1);ss=np.take_along_axis(s,j[...,None],-1)[...,0]
    ff=np.take_along_axis(n-s,j[...,None],-1)[...,0]
    eligible=(s>=15).any(-1)
    good=ss>=15;unknown=(ss<15)&(ff<2)
    return {'certified_B15':int((good&eligible).sum()),'unresolved':int((unknown&eligible).sum()),
        'known_failure':int((~good&~unknown&eligible).sum()),'eligible':int(eligible.sum()),
        'B15_lower':int((good&eligible).sum()),'B15_upper':int(((good|unknown)&eligible).sum())}

def benchmark(s,n):
    good=s>=15;possible=n-s<=1;eligible=good.any(-1)
    fixed=good.sum(1).max(1)
    result={'families':7,'controller_state_cases':21,'oracle_B15_lower':int(eligible.sum()),
        'oracle_B15_upper':int(possible.any(-1).sum()),'best_global_eta_lower':int(good.sum((0,1)).max()),
        'best_controller_fixed_lower':int(fixed.sum()),
        'best_controller_fixed_upper':int(possible.sum(1).max(1).sum()),
        'controller_oracle':eligible.sum(1).tolist(),'controller_fixed':fixed.tolist(),
        'maximum_certified_adaptive_B15_headroom':int(eligible.sum()-fixed.sum()),
        'note':'Outcome-based fixed choices are descriptive ceilings, not deployable trained models.'}
    r=strong_reversals(s,n)
    robust_vs_bad=0;mutual=0
    for c,a,b,i,j,sign in r:
        if sign<0:i,j=j,i
        robust_vs_bad+=int(s[c,a,i]>=15 and s[c,b,j]>=15 and (16-(n-s)[c,a,j])<=8 and (16-(n-s)[c,b,i])<=8)
        mutual+=int(s[c,a,i]>=15 and s[c,b,j]>=15 and (n-s)[c,a,j]>=2 and (n-s)[c,b,i]>=2)
    result.update(strong_reversal_quadruples=len(r),robust_opposite_nonB15_reversals=mutual,
                  robust_opposite_clear_failure_reversals=robust_vs_bad,
                  unique_involved_state_pairs=len(set((int(x[1]),int(x[2])) for x in r)),
                  unique_controller_eta_pairs=len(set((int(x[0]),int(x[3]),int(x[4])) for x in r)))
    write('benchmark_headroom.json',result)
    return r

def permutation_audit(d,ix,va,s,n,r):
    perms=np.asarray(list(permutations(range(len(va)))),int)
    report=[]
    for variant,kind in [('large_20','physical_context'),('large_20','context_only'),
                         ('large_80','physical_context'),('large_20','eta_only')]:
        for seed in SEEDS:
            path=SRC/'variants'/variant/'models'/kind/f'seed{seed}'
            summary=read(path/'summary.json')
            z=np.load(path/'validation_predictions.npz')['correct'][:,ix[va]]
            p=np.clip(expit(z.astype(float)),1e-7,1-1e-7);pp=p[:,perms,:].transpose(1,0,2,3)
            loss=(-(s[None]*np.log(np.clip(pp,1e-12,1))+(n-s)[None]*np.log1p(-np.clip(pp,0,1-1e-12))))
            null_nll=np.mean(loss.sum((2,3))/n.sum((1,2))[None],axis=1)
            j=pp.argmax(-1)
            ns=np.take_along_axis(np.broadcast_to(s,pp.shape),j[...,None],-1)[...,0]
            nf=np.take_along_axis(np.broadcast_to(n-s,pp.shape),j[...,None],-1)[...,0]
            eligible=(s>=15).any(-1)
            hits=((ns>=15)&eligible).sum((1,2));upper=(((ns>=15)|(nf<2))&eligible).sum((1,2))
            c,a,b,ei,ej,sg=r.T
            order=(np.sign(pp[:,c,a,ei]-pp[:,c,a,ej])==sg)&(np.sign(pp[:,c,b,ei]-pp[:,c,b,ej])==-sg)
            nr=order.sum(1)
            identity=np.flatnonzero(np.all(perms==np.arange(7),axis=1))[0]
            observed=nll(p,s,n)
            assert abs(observed-null_nll[identity])<1e-10
            fixed_profile=np.broadcast_to(p.mean(1,keepdims=True),p.shape)
            row={'variant':variant,'kind':kind,'seed':seed,'observed_NLL':observed,
                'selection_bounds':selection(p,s,n),'state_reversals_correct':int(nr[identity]),
                'state_reversals_total':len(r),'whole_state_profile_permutation':{
                  'all_permutations':len(perms),'NLL_null_mean':float(null_nll.mean()),
                  'NLL_null_minus_observed':float(null_nll.mean()-observed),
                  'descriptive_NLL_tail_fraction':float((null_nll<=observed+1e-12).mean()),
                  'reversal_correct_null_mean':float(nr.mean()),
                  'descriptive_reversal_tail_fraction':float((nr>=nr[identity]).mean()),
                  'B15_lower_null_mean':float(hits.mean()),'B15_upper_null_mean':float(upper.mean()),
                  'descriptive_B15_tail_fraction':float((hits>=hits[identity]).mean())},
                'state_collapsed_prediction_NLL':nll(fixed_profile,s,n),
                'state_collapsed_selection':selection(fixed_profile,s,n),
                'caution':'Post-hoc source-VAL diagnostic after VAL checkpoint selection, not independent test significance.',
                'single_shuffle_audits':{k:{'NLL':summary[k]['NLL'],'B15':summary[k]['selected_B15'],
                    'unresolved':summary[k]['selected_unknown'],
                    'state_reversal_correct':summary[k]['state_reversals']['correct']}
                    for k in ('state_shuffle','context_shuffle','state_response_shuffle','wrong_controller','eta_shuffle')}}
            report.append(row)
    write('state_assignment_permutations.json',report)

def reversal_uncertainty(ys,s,n,r):
    probabilities=[];details=[]
    for c,a,b,i,j,sg in r:
        ps=[];dis=[]
        for state,direction in ((a,sg),(b,-sg)):
            ai,aj=ys[c,state,i],ys[c,state,j]
            valid=np.isfinite(ai)&np.isfinite(aj)
            delta=direction*(ai[valid]-aj[valid]);plus=int((delta>0).sum());minus=int((delta<0).sum())
            prob=float(binom.sf(plus-1,plus+minus,.5)) if plus+minus else 1.
            ps.append(prob);dis.append({'paired_valid_seeds':int(valid.sum()),'favored_discordant':plus,'opposite_discordant':minus,'one_sided_p':prob})
        p=min(1.,2*max(ps));probabilities.append(p)
        details.append({'controller':int(c),'state_A':int(a),'state_B':int(b),'eta_i':int(i),'eta_j':int(j),
                        'Q_order_A_sign':int(sg),'contrasts':dis,'two_direction_intersection_union_p':p})
    # Multiplicity includes all state-pair x eta-pair x controller tests,
    # not only the 76 empirically selected reversals.
    total_tests=3*(7*6//2)*(16*15//2)
    order=np.argsort(probabilities);adj=np.ones(len(order));pv=np.asarray(probabilities)[order]
    if len(pv):
        qq=np.minimum.accumulate((pv*total_tests/np.arange(1,len(pv)+1))[::-1])[::-1]
        adj[order]=np.minimum(qq,1.)
    for row,q in zip(details,adj):row['BH_q_over_all_tested_quadruples']=float(q)
    halves=[]
    for part,other in ((slice(0,8),slice(8,16)),(slice(8,16),slice(0,8))):
        sa=np.nansum(ys[...,part],-1);na=np.isfinite(ys[...,part]).sum(-1)
        sb=np.nansum(ys[...,other],-1);nb=np.isfinite(ys[...,other]).sum(-1)
        a=strong_reversals(sa,na,8);b=strong_reversals(sb,nb,8)
        known={tuple(row) for row in b}
        halves.append({'discovery_seed_slice':[part.start,part.stop],'discovery_reversals':len(a),
                       'same_direction_gap_ge025_in_other_half':sum(tuple(row) in known for row in a)})
    result={'raw_empirical_Q16_reversals':len(r),'nominal_two_direction_p_le005':int((np.asarray(probabilities)<=.05).sum()),
            'BH_q_le005_all_quadruples':int((adj<=.05).sum()),'total_quadruples_tested':total_tests,
            'split_seed_replication':halves,'details':details,
            'note':'Bounds for numerical missing outcomes are not binomial confidence intervals. These paired tests quantify sampling uncertainty separately.'}
    write('reversal_uncertainty.json',result)

def centered(a,interaction):
    # Mean removal within controller/eta, optionally also state main effects.
    col=np.nanmean(a,axis=1,keepdims=True)
    if interaction:return a-col-np.nanmean(a,axis=2,keepdims=True)+np.nanmean(a,axis=(1,2),keepdims=True)
    return a-col

def half_stats(y):
    valid=np.isfinite(y[...,:4]).all(-1)
    a=np.where(valid,np.mean(y[...,:2],-1),np.nan)
    b=np.where(valid,np.mean(y[...,2:4],-1),np.nan)
    results={}
    for interaction in (False,True):
        ra,rb=centered(a,interaction),centered(b,interaction)
        ok=np.isfinite(ra)&np.isfinite(rb);aa,bb=ra[ok],rb[ok]
        cov=float(np.mean((aa-aa.mean())*(bb-bb.mean())))
        corr=float(np.corrcoef(aa,bb)[0,1])
        fullvar=float(np.var((aa+bb)/2))
        results['interaction' if interaction else 'state_given_controller_eta']={
            'complete_Q4_cells':int(valid.sum()),'split_half_residual_correlation':corr,
            'cross_half_covariance':cov,'observed_Q4_residual_variance':fullvar,
            'estimated_reliable_fraction_of_Q4_residual_variance':cov/max(fullvar,1e-12)}
    return results

def noise_audit(ys):
    result=half_stats(ys)
    rng=np.random.default_rng(20261004);values={k:[] for k in result}
    for _ in range(2000):
        draw=rng.integers(0,ys.shape[1],ys.shape[1]);ss=half_stats(ys[:,draw])
        for k in values:values[k].append(ss[k]['cross_half_covariance'])
    for k in values:result[k]['source_family_bootstrap_covariance_95CI']=np.quantile(values[k],[.025,.975]).tolist()
    result['caveat']='Two+two independent future seed halves; covariance estimates real state variation, not predictability from h. Seed reuse across eta induces within-family dependence, hence family bootstrap.'
    write('TRAIN_label_reliability.json',result)

def trajectory_audit(d,ix,tr):
    ss=d['success'][:,ix[tr]];ff=d['failure'][:,ix[tr]];nn=ss+ff
    q=ss/np.maximum(nn,1);entropy=-(ss*np.log(np.clip(q,1e-12,1))+ff*np.log(np.clip(1-q,1e-12,1)))
    lower=float((entropy.sum((1,2))/nn.sum((1,2))).mean())
    records=[]
    for variant,kind in [('large_20','physical_context'),('large_20','context_only'),('large_80','physical_context')]:
        for seed in SEEDS:
            path=SRC/'variants'/variant/'models'/kind/f'seed{seed}'
            r=read(path/'summary.json');h=read(path/'history.json')
            loss=np.asarray([row['NLL'] for row in h]);rev=np.asarray([row['state_reversals']['correct'] for row in h]);hit=np.asarray([row['selected_B15'] for row in h])
            records.append({'variant':variant,'kind':kind,'seed':seed,'selected_step':r['best_step'],
                'TRAIN_NLL':r['training']['NLL'],'VAL_NLL':r['correct']['NLL'],
                'TRAIN_empirical_per_pair_entropy_lower_bound':lower,
                'VAL_NLL_last_step':float(loss[-1]),'selected_reversals':r['correct']['state_reversals']['correct'],
                'max_VAL_reversals_any_checkpoint_diagnostic_only':int(rev.max()),
                'selected_B15':r['correct']['selected_B15'],'max_VAL_B15_any_checkpoint_diagnostic_only':int(hit.max()),
                'trajectory_Spearman_NLL_vs_reversal_count':float(spearmanr(loss,rev).statistic) if len(set(rev))>1 else None})
    write('loss_and_fit_audit.json',{'records':records,'loss_definition':'controller-balanced observed Bernoulli likelihood, gradients softplus(z)-q; no rank term',
        'minibatch':'32 common state-eta indices replicated across all three controllers; within each controller normalized by observed trials',
        'count_target_semantics':'actual successes/failures only; numerical outcomes removed',
        'caveat':'Empirical entropy is an optimistic in-sample fit bound for noisy Q4 labels, not Bayes risk. Looking at ranking-best VAL checkpoints here does not select new models.'})

def input_and_residual_audit(d,ix,tr,va,s,n):
    active=d['split']=='train'
    ee=d['eta'][active];cc=d['context'][:,active][d['valid'][:,active]]
    expected={'eta_center':ee.mean(0),'eta_scale':np.maximum(ee.std(0),.1),
              'context_center':cc.mean(0),'context_scale':np.maximum(cc.std(0),.05)}
    q=s[:,va]/np.maximum(n[:,va],1)
    rq=q-q.mean(1,keepdims=True)
    result=[]
    for kind in ('physical_context','context_only','eta_only'):
        for seed in SEEDS:
            path=SRC/'variants/large_20/models'/kind/f'seed{seed}'
            norm=read(path/'normalization.json')
            for key in expected:np.testing.assert_allclose(norm[key],expected[key],rtol=1e-6,atol=1e-7)
            summary=read(path/'summary.json')
            p=expit(np.load(path/'validation_predictions.npz')['correct'][:,ix[va]].astype(float))
            rp=p-p.mean(1,keepdims=True)
            result.append({'kind':kind,'seed':seed,'normalization_matches_frozen_TRAIN_statistics':True,
                'backend_consistency':summary['backend_consistency'],'parameter_audit':summary['parameter_audit'],
                'within_controller_eta_state_residual_correlation':float(np.corrcoef(rp.ravel(),rq.ravel())[0,1]) if np.var(rp)>1e-12 else None,
                'prediction_to_empirical_state_residual_variance_ratio':float(np.var(rp)/np.var(rq)),
                'state_residual_MAE':float(np.abs(rp-rq).mean()),
                'caution':'Empirical residual variance includes finite-Q16 uncertainty. Small variance does not alone prove underfitting or missing information.'})
    write('input_and_state_residual_audit.json',{'models':result,
        'controller_path_audit':read(SRC/'controller_path_audit.json'),
        'context_is_state_conditioned':True,
        'context_definition':'Two deterministic short physical response probes from the actual state, nominal eta=0 and the candidate eta; aggregate progress/clearance/safety/correction',
        'state_ablation_warning':'Removing the explicit h branch does not remove state information from C(h,eta,controller).',
        'H80_not_superset_of_H20':'Same 24 summary channels recomputed over a longer horizon; H20 statistics are not retained as a separate block.'})

def main():
    d,rows,states,ix,tr,va,ys,s,n=load()
    sv,nv=s[:,va],n[:,va]
    reversals=benchmark(sv,nv)
    permutation_audit(d,ix,va,sv,nv,reversals)
    reversal_uncertainty(ys[:,va],sv,nv,reversals)
    noise_audit(ys[:,tr])
    trajectory_audit(d,ix,tr)
    input_and_residual_audit(d,ix,tr,va,s,n)
    print(json.dumps({'headroom':read(OUT/'benchmark_headroom.json'),
        'label_reliability':read(OUT/'TRAIN_label_reliability.json'),
        'new_rollouts':0,'target_labels_opened':False},indent=2))

if __name__=='__main__':main()
