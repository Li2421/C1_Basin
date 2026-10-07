"""Read-only adjudication after frozen evaluation; no training or rollout."""
import json,os
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.stats import spearmanr
from .build_source import ROOT,OUT,load,dump,sha,SOURCES
from .evaluate import physical_inputs,CONF
from .train import data

def audit():
    ps,ss,x,flats=physical_inputs()
    # Independently reproduce archived Phase-A supervised scores to validate
    # current reference-Flow/observation reconstruction without using outcomes.
    from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as old
    norm=load(ROOT/'diagnostics/orthoflow3_generator_critic_v1/normalization.json')
    model=old.Critic();dims={sc:len(norm['scenarios'][sc]['h_mean']) for sc in old.SCENARIOS}
    init=old.merge_initialized(model,dims,len(norm['environment_keys'])+3,critic=True)
    manifest=load(CONF.parent/'critic_frozen.json');p=serialization.from_bytes(init,open(manifest['checkpoint'],'rb').read())
    # Match archived environment context implementation exactly.
    from diagnostics.orthoflow3_ring_revision_v1.run_fresh import context_vector
    from new_benchmark_common.basin_dataset_v1 import _environment_descriptor
    from types import SimpleNamespace
    cfg=load(ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config']
    descriptor=_environment_descriptor(SimpleNamespace(name='ring_exchange',config=SimpleNamespace(**cfg),kind='ring'))
    context=context_vector(descriptor,norm,'ring_exchange');hn=norm['scenarios']['ring_exchange']
    hh=np.repeat((flats-np.asarray(hn['h_mean'],np.float32))/np.asarray(hn['h_std'],np.float32),17,axis=0)
    eta=np.asarray([r['etas'] for r in ps],np.float32).reshape(-1,3)
    scores=expit(np.asarray(model.apply(p,jnp.asarray(hh),jnp.asarray(np.repeat(context[None],len(hh),axis=0)),jnp.asarray((eta-old.CENTER)/old.RADIUS),method=model.ring))).reshape(-1,17)
    stored=np.asarray([r['scores'] for r in ps]);error=float(np.max(abs(scores-stored)))
    assert error<2e-4,('reference inference mismatch',error)
    # Verify exact cache aggregation agrees with the prior frozen evidence.
    prior=load(CONF/'seed_evidence.json');by={(r['state_uid'],r['candidate']):r for r in prior['rows']}
    truth=load(OUT/'target_truth.json');errors=[]
    for t in truth:
        for j in range(16):
            r=by[(t['state_uid'],j+1)]
            assert r['robust']==t['robust'][j],(t['state_uid'],j,'robust')
            if r['Q16'] is not None and t['q'][j] is not None:errors.append(abs(r['Q16']-t['q'][j]))
            assert abs(r['Q_lower']-t['lower'][j])<1e-12 and abs(r['Q_upper']-t['upper'][j])<1e-12
    rows,sx,_,_,_=data();source_states=load(OUT/'source_states.json')
    train_state_indices=sorted({r['state_index'] for r in rows if r['split']=='train'})
    train=sx['obstacles'][train_state_indices];target=x['obstacles']
    # Curvature is physical channel 5; source has segment boundaries only.
    source_curve=np.unique(train[...,5]);target_curve=np.unique(target[...,5])
    pred=load(OUT/'target_predictions.json')['rows'];valid=[]
    for p,t in zip(pred,truth):
        for j,q in enumerate(t['q']):
            if q is not None:valid.append((p['eta'][j][1],q,p['scores']['shared'][j]))
    v=np.asarray(valid)
    slopes={'Ring_eta2_vs_empiricalQ_spearman':float(spearmanr(v[:,0],v[:,1]).statistic),'Ring_eta2_vs_shared_score_spearman':float(spearmanr(v[:,0],v[:,2]).statistic)}
    result={'archived_reference_score_max_abs_error':error,'cache_vs_frozen_outcome_max_error':max(errors),
        'candidate_indexing':'original stochastic indices1..16; all960 tuples matched','source_curvature_features':source_curve.tolist(),
        'target_curvature_features':target_curve.tolist(),'source_target_state_uid_overlap':len({r['state_uid'] for r in source_states}&{r['state_uid'] for r in pred}),
        'source_policy_axis_applicability':np.unique(sx['agents'][train_state_indices,:,10][sx['agent_mask'][train_state_indices]>0]).tolist(),
        'target_policy_axis_applicability':np.unique(x['agents'][:,:,10][x['agent_mask']>0]).tolist(),
        'eta2_diagnostic':slopes,'remaining_time_present':True,'target_training_reads':0,'new_rollout':0,
        'interpretation':'Observed adverse eta ordering and unsupported target curvature; data support and state-conditioned extrapolation are confounded. No evidence that representation is intrinsically insufficient.'}
    dump('post_evaluation_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
