"""Why does useful controller context not reliably improve top-1?

Frozen source models only; no model selection, retraining, rollout, or target
labels. Top-three candidate subsets are defined by TRAIN counts, not VAL wins.
"""
from pathlib import Path
import hashlib, json, os, sqlite3, sys
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent/'context_value_followup'
SRC=OUT.parent.parent/'orthoflow3_controller_state_residual_factorial_v1'
SEEDS=(17,23,41)

def read(p):return json.loads(Path(p).read_text())
def write(name,x):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def losses(p,s,n):
    p=np.clip(p,1e-7,1-1e-7)
    return -(s*np.log(p)+(n-s)*np.log1p(-p))/(3*n.sum((1,2))[:,None,None])

def selection(p,s,n):
    eligible=(s>=15).any(-1)
    pick=p.argmax(-1)
    success=np.take_along_axis(s,pick[...,None],-1)[...,0]
    failure=np.take_along_axis(n-s,pick[...,None],-1)[...,0]
    good=success>=15;unknown=(success<15)&(failure<2)
    return {'certified_B15':int((good&eligible).sum()),'unresolved':int((unknown&eligible).sum()),
        'known_failure':int((~good&~unknown&eligible).sum()),'oracle_eligible':int(eligible.sum()),
        'selected_eta_indices':pick.tolist()}

def main():
    variant=SRC/'variants/large_20'
    d=dict(np.load(variant/'dataset.npz'))
    tr=np.unique(d['state_index'][d['split']=='train'])
    va=np.unique(d['state_index'][d['split']=='validation'])
    ix=np.array([np.flatnonzero(d['state_index']==i) for i in range(56)])
    s=d['success'][:,ix[va]].astype(float)
    n=s+d['failure'][:,ix[va]]
    ss=d['success'][:,ix[tr]].astype(float)
    nn=ss+d['failure'][:,ix[tr]]
    prior=(ss.sum(1)+.5)/(nn.sum(1)+1)
    top=np.argsort(prior,axis=-1)[:,-3:]
    mask=np.zeros((3,16),bool);np.put_along_axis(mask,top,True,axis=-1)
    q=s/n
    cases=[];models=[]
    for seed in SEEDS:
        folder=variant/'models'
        pe=expit(np.load(folder/f'eta_only/seed{seed}/validation_predictions.npz')['correct'][:,ix[va]].astype(float))
        pf=expit(np.load(folder/f'physical_context/seed{seed}/validation_predictions.npz')['correct'][:,ix[va]].astype(float))
        diff=losses(pe,s,n)-losses(pf,s,n)
        gain=float(diff.sum());outside=float(np.where(~mask[:,None],diff,0).sum())
        mean_p=np.broadcast_to(pf.mean(1,keepdims=True),pf.shape)
        models.append({'seed':seed,'eta_only_NLL':float(losses(pe,s,n).sum()),
            'full_NLL':float(losses(pf,s,n).sum()),'controller_eta_TRAIN_prior_NLL':float(losses(prior[:,None],s,n).sum()),
            'state_collapsed_model_prediction_NLL':float(losses(mean_p,s,n).sum()),
            'total_NLL_gain':gain,'TRAIN_top3_gain':gain-outside,'other13_gain':outside,
            'other13_share_of_NLL_gain':outside/gain,
            'top3_pair_share':float(mask.mean()),'NLL_gain_by_eta_index':diff.sum((0,1)).tolist(),
            'selected_eta_indices':pf.argmax(-1).tolist(),
            'third_controller_eta1_minus_eta3_predicted':(pf[2,:,1]-pf[2,:,3]).tolist()})
        eligible=(s>=15).any(-1)
        for c,j in zip(*np.where(eligible)):
            pick=int(pf[c,j].argmax())
            if s[c,j,pick]>=15:continue
            good=np.flatnonzero(s[c,j]>=15)
            best=int(good[np.argmax(pf[c,j,good])])
            cases.append({'seed':seed,'controller':int(c),'state_index':int(va[j]),'selected_eta_index':pick,
                'selected_success':int(s[c,j,pick]),'selected_observed_trials':int(n[c,j,pick]),
                'selected_probability':float(pf[c,j,pick]),
                'status':'NUMERICALLY_UNRESOLVED' if n[c,j,pick]-s[c,j,pick]<2 else 'CERTIFIED_NON_B15',
                'highest_scored_B15_eta_index':best,'B15_success':int(s[c,j,best]),
                'B15_observed_trials':int(n[c,j,best]),'B15_predicted_probability':float(pf[c,j,best])})
    write('probability_gain_decomposition.json',{'scope':'Known source controllers; already-used source VAL; post-hoc only',
        'TRAIN_top3_by_controller':top.tolist(),'models':models,'selection_failures_or_unknowns':cases,
        'new_rollouts':0,'target_labels_opened':False,
        'interpretation':'Gain decomposition describes where probability error improved, not a causal allocation or validation of a replacement model.'})

    # Re-read only source-TRAIN matched records relevant to the observed top-1
    # contrast. This is a diagnosed contrast, not a preselected confirmatory test.
    rows=read(SRC/'pairs.json');profile=read(SRC/'protocol.json')['profiles'][2]
    y=np.full((len(tr),2,4),np.nan)
    with sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True) as db:
        for j,st in enumerate(tr):
            for k,e in enumerate((1,3)):
                row=rows[ix[st,e]]
                for sk,success,bad in db.execute('''SELECT seed_key,success,numerical_failure FROM rollout
                    WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0
                    AND compatibility_quality='EXACT_REUSE' ''',(row['state_uid'],row['eta_uid'],profile['controller_uid'])):
                    seed_index=json.loads(sk).get('future_index',-1)
                    if 0<=seed_index<4 and not bad:y[j,k,seed_index]=success
    assert np.isfinite(y).all()
    for k,e in enumerate((1,3)):
        np.testing.assert_array_equal(y[:,k].sum(-1),ss[2,:,e])
        np.testing.assert_array_equal(np.isfinite(y[:,k]).sum(-1),nn[2,:,e])
    delta=y[:,0]-y[:,1];effect=delta.mean(1)
    first,second=delta[:,:2].mean(1),delta[:,2:].mean(1)
    rng=np.random.default_rng(17);draw=rng.integers(0,len(tr),(10000,len(tr)))
    aa,bb=first[draw],second[draw]
    covariance=np.mean((aa-aa.mean(1,keepdims=True))*(bb-bb.mean(1,keepdims=True)),axis=1)
    plus=int((delta>0).sum());minus=int((delta<0).sum())
    write('decision_contrast_supervision.json',{
        'controller':'second','eta_indices':[1,3],'eta_values':d['eta'][ix[tr[0],[1,3]]].tolist(),
        'TRAIN_families':len(tr),'seed_count_per_family_eta':4,'total_trials_per_eta':len(tr)*4,
        'success_counts':[int(y[:,0].sum()),int(y[:,1].sum())],
        'eta1_only_success':plus,'eta3_only_success':minus,
        'pooled_paired_sign_p_exploratory':float(binomtest(plus,plus+minus,.5).pvalue),
        'mean_eta1_minus_eta3_Q':float(effect.mean()),
        'family_bootstrap_95CI_for_mean_difference':np.quantile(effect[draw].mean(1),[.025,.975]).tolist(),
        'state_counts_prefer_eta1_prefer_eta3_tied':[int((effect>0).sum()),int((effect<0).sum()),int((effect==0).sum())],
        'split_half_contrast_correlation':float(np.corrcoef(first,second)[0,1]),
        'split_half_contrast_covariance':float(np.mean((first-first.mean())*(second-second.mean()))),
        'family_bootstrap_95CI_for_contrast_covariance':np.quantile(covariance,[.025,.975]).tolist(),
        'source_VAL_state_indices':va.tolist(),
        'source_VAL_observed_Q_difference':(q[2,:,1]-q[2,:,3]).tolist(),
        'source_VAL_pair_has_complete_Q16':((n[2,:,1]==16)&(n[2,:,3]==16)).tolist(),
        'caveat':'Low contrast reliability does not prove no true state effect. The contrast was identified post-hoc; source VAL numerical unknowns remain unknown.'})

    # Frozen checkpoint inference only, not another training branch. Limit this
    # tiny CPU diagnostic to two available cores, preserving shared resources.
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    jax.config.update('jax_enable_x64',False)
    assert jax.default_backend()=='cpu'
    x=dict(np.load(variant/'entities.npz'));model=Critic(True,True,False)
    predictor=jax.jit(lambda pars,xx,ee,cc:model.apply(pars,xx,ee,cc,jnp.zeros((len(ee),3))))
    result=[];calibration=[]
    for seed in SEEDS:
        folder=variant/f'models/physical_context/seed{seed}'
        norm={k:np.asarray(v,np.float32) for k,v in read(folder/'normalization.json').items()}
        eta=((d['eta']-norm['eta_center'])/norm['eta_scale']).astype(np.float32)
        context=((d['context']-norm['context_center'])/norm['context_scale']).astype(np.float32)
        params=serialization.msgpack_restore((folder/'checkpoint.msgpack').read_bytes())
        all_indices=np.concatenate((ix[tr][:,[1,3]].ravel(),ix[va].ravel()))
        z=np.asarray(predictor(params,gather(x,d['state_index'][all_indices]),jnp.asarray(eta[all_indices]),jnp.asarray(context[2,all_indices])))
        np.testing.assert_allclose(z[2*len(tr):],np.load(folder/'validation_predictions.npz')['correct'][2,ix[va].ravel()],rtol=1e-5,atol=1e-5)
        train_p=expit(z[:2*len(tr)].reshape(len(tr),2).astype(float));pdiff=train_p[:,0]-train_p[:,1]
        val_p=expit(z[2*len(tr):].reshape(len(va),16).astype(float));vdiff=val_p[:,1]-val_p[:,3]
        complete=(n[2,:,1]==16)&(n[2,:,3]==16)
        result.append({'seed':seed,'TRAIN_predicted_Q_means':train_p.mean(0).tolist(),
            'TRAIN_mean_predicted_contrast':float(pdiff.mean()),
            'TRAIN_contrast_corr_with_empirical_Q4':float(np.corrcoef(pdiff,effect)[0,1]),
            'TRAIN_positive_negative_tied_predicted_contrasts':[int((pdiff>0).sum()),int((pdiff<0).sum()),int((pdiff==0).sum())],
            'VAL_complete_Q16_states':int(complete.sum()),
            'VAL_complete_contrast_correlation':float(np.corrcoef(vdiff[complete],(q[2,:,1]-q[2,:,3])[complete])[0,1]),
            'VAL_predicted_contrast':vdiff.tolist(),
            'checkpoint_sha256':hashlib.sha256((folder/'checkpoint.msgpack').read_bytes()).hexdigest(),
            'frozen_VAL_logits_reproduced':True})
        # Privileged known-controller/exact-eta diagnostic. Uniformly fit all
        # 48 intercepts by actual TRAIN count NLL, not only the diagnosed pair.
        # It is NOT a new continuous critic, and cannot score unseen controllers.
        both=np.r_[ix[tr].ravel(),ix[va].ravel()]
        train_z=[];val_z=[]
        for c in range(3):
            zz=np.asarray(predictor(params,gather(x,d['state_index'][both]),jnp.asarray(eta[both]),jnp.asarray(context[c,both])))
            train_z.append(zz[:len(tr)*16].reshape(len(tr),16))
            val_z.append(zz[len(tr)*16:].reshape(len(va),16))
        train_z=np.array(train_z,dtype=float);val_z=np.array(val_z,dtype=float)
        np.testing.assert_allclose(val_z,np.load(folder/'validation_predictions.npz')['correct'][:,ix[va]],rtol=1e-5,atol=1e-5)
        offset=np.zeros((3,16));before=0.;after=0.
        for c in range(3):
            for e in range(16):
                zz=train_z[c,:,e];success=ss[c,:,e];failure=nn[c,:,e]-success
                def objective(b):
                    return float((success*np.logaddexp(0,-zz-b)+failure*np.logaddexp(0,zz+b)).sum())
                fit=minimize_scalar(objective,bounds=(-8.,8.),method='bounded',options={'xatol':1e-8})
                assert fit.success and fit.fun<=objective(0)+1e-6
                offset[c,e]=fit.x
        corrected=expit(val_z+offset[:,None,:])
        original=expit(val_z)
        calibration.append({'seed':seed,'offsets':offset.tolist(),
            'TRAIN_NLL_original':float(losses(expit(train_z),ss,nn).sum()),
            'TRAIN_NLL_calibrated':float(losses(expit(train_z+offset[:,None,:]),ss,nn).sum()),
            'VAL_NLL_original':float(losses(original,s,n).sum()),
            'VAL_NLL_calibrated':float(losses(corrected,s,n).sum()),
            'VAL_original_selection':selection(original,s,n),
            'VAL_calibrated_selection':selection(corrected,s,n),
            'eta1_eta3_controller2_offsets':offset[2,[1,3]].tolist()})
    write('frozen_decision_contrast_fit.json',{'models':result,'new_training':False,'new_rollouts':0,
        'warning':'Neither five complete source-VAL states nor chosen post-hoc contrasts establish new generalization; this localizes an existing failure.'})
    write('TRAIN_only_intercept_diagnostic.json',{'models':calibration,
        'objective':'Observed-count Bernoulli NLL only, independently bounded [-8,8] logit intercept for each known controller/exact eta',
        'target_labels_used':False,'VAL_used_for_intercept_fit_or_selection':False,
        'not_deployable_zero_shot_model':True,'not_new_continuous_critic':True,
        'caution':'Privileged controller identity and exact-eta lookup permitted solely to diagnose average preference misfit. It neither adds state information nor proves transfer, and may overfit low-count TRAIN groups.'})
    print(json.dumps({'NLL_gain_share_outside_TRAIN_top3':[r['other13_share_of_NLL_gain'] for r in models],
        'TRAIN_success_counts':[int(y[:,0].sum()),int(y[:,1].sum())],
        'contrast_models':result,'intercept_diagnostic':calibration,'new_rollouts':0},indent=2),flush=True)

if __name__=='__main__':main()
