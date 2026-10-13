"""Frozen-model root-cause probe on all128 cached natural Q4 screen states.

No training, tuning, new rollout, or outcome-selected state subset.
These historical states are DEVELOPMENT evidence, not untouched confirmation.
"""
from __future__ import annotations
import argparse
import gzip
import json
import sqlite3
import sys
from collections import defaultdict
import numpy as np
from scipy.special import expit
from .study import HERE, OLD, FIELD, read, write, freeze, sha

OUT=HERE/'extended128'


def prepare():
    manifest=read(OLD/'decision_pair_challenge_v3/screen_manifest.json')
    protocol=read(HERE/'protocol.json');frozen=read(HERE/'models_frozen.json')
    assert manifest['controller_uids']==protocol['controller_uids']
    np.testing.assert_array_equal(manifest['etas'],protocol['eta'])
    states=manifest['states'];assert len(states)==128
    source_physical={json.dumps(s['physical'],sort_keys=True) for s in protocol['states']}
    assert not any(json.dumps(s['physical'],sort_keys=True) in source_physical for s in states)
    frozen_protocol=dict(scope='Full natural128 pre-screen pool, Q4; DEVELOPMENT only. Not the outcome-selected16 Q16 challenge subset.',
        manifest=str(OLD/'decision_pair_challenge_v3/screen_manifest.json'),manifest_sha256=sha(OLD/'decision_pair_challenge_v3/screen_manifest.json'),
        controller_uids=manifest['controller_uids'],states=states,eta=manifest['etas'],seed_count=4,
        models_sha256=sha(HERE/'models_frozen.json'),primary=frozen['primary'],
        models=read(HERE/'root_cause_scope_amendment.json')['models'],
        fitting='None: reuse all source80 TRAIN/source28 VAL frozen checkpoints and normalization without changes',
        purpose='Check whether controller-by-state and controller-by-state-by-eta relations generalize across states; not new-controller transfer.',
        bootstrap='20000 physical-state clusters; retain both controller conditions and all eta',
        missing_outcomes='Preserve numerical unknown; selected success reports lower/upper bounds',
        new_rollouts=0,incremental_method_tuning=False)
    freeze(OUT/'protocol.json',frozen_protocol)
    sid={s['uid']:i for i,s in enumerate(states)}
    eta={np.asarray(e,np.float64).tobytes():i for i,e in enumerate(manifest['etas'])}
    ctl={manifest['controller_uids'][c]:i for i,c in enumerate(('base','held'))}
    conn=sqlite3.connect(f'file:{OLD}/field_rollout.sqlite?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    values=np.full((128,2,16,4),-2,np.int8);collision=np.zeros_like(values,bool);raw=[]
    for r in conn.execute('SELECT * FROM continuation WHERE phase=?',('decision_pair_challenge_v3_screen',)):
        r=dict(r)
        if r['state_uid'] not in sid or r['controller_uid'] not in ctl:continue
        j=eta[np.asarray(json.loads(r['eta_json']),np.float64).tobytes()]
        key=(sid[r['state_uid']],ctl[r['controller_uid']],j,r['seed'])
        assert values[key]==-2
        values[key]=-1 if r['numerical_unresolved'] else r['success'];collision[key]=bool(r['collision'])
        raw.append(r)
    conn.close();assert not (values==-2).any(),dict(observed=int((values!=-2).sum()),expected=values.size)
    with gzip.open(OUT/'raw_seed_evidence.jsonl.gz','wt') as f:
        for row in raw:f.write(json.dumps(row,sort_keys=True)+'\n')
    np.savez_compressed(OUT/'labels.npz',outcome=values,collision=collision,
        success=(values==1).sum(-1),failure=(values==0).sum(-1),unknown=(values==-1).sum(-1),
        state_uid=np.array([s['uid'] for s in states]))
    write(OUT/'cache_audit.json',dict(seed_outcomes=len(raw),new_rollouts=0,overlap_with_original124=0,
        states=128,unknown=int((values==-1).sum()),collision=int(collision.sum()),labels_sha256=sha(OUT/'labels.npz')))


def features():
    import jax
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1.ring_t0_source import runtime
    from field_pipeline_v1.held import AlternateFourRingRuntime
    from diagnostics.physical_state_critic_formal_v1.physical_entities import entities
    from diagnostics.controller_q_diversity_v1.runtime import controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    p=read(OUT/'protocol.json');bases=[runtime(),AlternateFourRingRuntime('ring_exchange')]
    jax.config.update('jax_enable_x64',False)
    def forbidden(*args,**kwargs):raise RuntimeError('No simulator step permitted')
    for rt in bases:rt.env_cls.step=forbidden
    data=defaultdict(list)
    for i,state in enumerate(p['states']):
        for k,v in entities('ring_exchange',state['physical']).items():data['entity_'+k].append(v)
        raw=np.stack([np.concatenate((controller_context(rt,state),field_context(rt,state)),-1) for rt in bases])
        data['branch_compact'].append(np.asarray(compact_field(raw)))
        if (i+1)%32==0:print('Static held-state measurements',i+1,flush=True)
    np.savez_compressed(OUT/'features.npz',**{k:np.stack(v) for k,v in data.items()})
    write(OUT/'feature_audit.json',dict(states=128,simulator_steps=0,eta_queries=0,
        success_labels_used=False,features_sha256=sha(OUT/'features.npz')))


def evaluate():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic
    from .evaluate import correlate,center_eta,center_both
    from .learn import metrics
    p=read(OUT/'protocol.json');assert p['models_sha256']==sha(HERE/'models_frozen.json')
    d=np.load(OUT/'labels.npz');features=np.load(OUT/'features.npz')
    x={k[7:]:jnp.asarray(np.repeat(features[k],2,axis=0),jnp.float32) for k in features.files if k.startswith('entity_')}
    mask=np.asarray(x['agent_mask'],bool);compact=features['branch_compact'];base=compact[:,0]
    future=compact;current=np.repeat(base[:,None],2,axis=1);e=jnp.asarray(p['eta'],jnp.float32)
    s=d['success'];f=d['failure'];u=d['unknown'];truth=s/np.maximum(s+f,1);lower=s/4;upper=(s+u)/4
    n=len(s);rng=np.random.default_rng(2026100923);bootstrap=rng.integers(n,size=(20000,n))
    permutations=[]
    while len(permutations)<100:
        perm=rng.permutation(n)
        if np.all(perm!=np.arange(n)):permutations.append(perm)
    permutations=np.stack(permutations)
    rows=np.arange(n)[:,None];cs=np.arange(2)[None,:]
    def selected(array,choice):return array[rows,cs,choice]
    def ci(a):return np.quantile(a,[.025,.975]).tolist()
    def contrast(a,b):
        delta=np.asarray(a,float)-np.asarray(b,float)
        if delta.ndim>1:delta=delta.mean(tuple(range(1,delta.ndim)))
        return dict(mean=float(delta.mean()),ci95=ci(delta[bootstrap].mean(-1)),physical_states=n)
    results={};predictions={};seed_results={};boot_values={}
    delta_true=truth[:,1]-truth[:,0]
    for kind in p['models']:
        if kind in ('eta_only','state_only'):raw=np.zeros(current.shape[:-1]+(14,),np.float32)
        elif kind=='current_field':raw=current
        elif kind=='branch_nominal':raw=np.concatenate((current[...,:14],future[...,:14]),-1)
        else:raw=np.concatenate((current,future),-1)
        raw=raw.reshape((-1,)+raw.shape[2:]);zseeds=[];seed_results[kind]=[]
        model=Critic(kind if kind in ('eta_only','state_only') else 'state_field')
        for seed in (17,23,41):
            folder=HERE/'fits'/kind/f'seed{seed}';norm=read(folder/'normalization.json')
            c=jnp.asarray(np.clip((raw-np.asarray(norm['mean']))/np.asarray(norm['scale']),-10,10)*mask[...,None],jnp.float32)
            params=model.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
            params=serialization.from_bytes(params,(folder/'checkpoint.msgpack').read_bytes())
            @jax.jit
            def infer(ii,jj):return model.apply(params,{k:v[ii] for k,v in x.items()},c[ii],e[jj])
            ii=np.repeat(np.arange(n*2),16);jj=np.tile(np.arange(16),n*2)
            z=np.concatenate([np.asarray(infer(jnp.asarray(ii[k:k+256]),jnp.asarray(jj[k:k+256]))) for k in range(0,len(ii),256)]).reshape(n,2,16)
            zseeds.append(z);dq=expit(z[:,1])-expit(z[:,0])
            seed_results[kind].append(dict(seed=seed,**metrics(z,s,f),
                centered_delta_correlation=correlate(center_eta(dq),center_eta(delta_true))))
        q=expit(np.stack(zseeds)).mean(0);predictions[kind]=q;delta=q[:,1]-q[:,0]
        choice=q.argmax(-1);value=selected(lower,choice)
        mismatched=q[permutations];mvalue=np.stack([selected(lower,z.argmax(-1)) for z in mismatched]).mean(0)
        wrong_value=selected(lower,q[:,::-1].argmax(-1))
        per_state_mse=((q-truth)**2).mean((1,2));wrong_mse=((q[:,::-1]-truth)**2).mean((1,2))
        matched_samples=value.mean(-1)[bootstrap].mean(-1)
        best_fixed=lower[bootstrap].mean((1,2)).max(-1)
        best_fixed_by_c=lower[bootstrap].mean(1).max(-1).mean(-1)
        pred_interaction=center_both(delta);true_interaction=center_both(delta_true)
        # Target-noise squared term cancels in this gain over zero interaction.
        interaction_gain=(2*pred_interaction*true_interaction-pred_interaction**2).mean(-1)
        results[kind]=dict(mean_success_lower=float(value.mean()),mean_success_upper=float(selected(upper,choice).mean()),
            mean_success_by_controller=value.mean(0).tolist(),wrong_controller_success=float(wrong_value.mean()),
            mismatched_state_success=float(mvalue.mean()),matched_minus_mismatch=contrast(value,mvalue),
            matched_minus_wrong_controller=contrast(value,wrong_value),
            q_mse_improvement_from_correct_controller=contrast(wrong_mse,per_state_mse),
            q_mse=float(per_state_mse.mean()),controller_delta_correlation=correlate(delta,delta_true),
            eta_centered_controller_delta_correlation=correlate(center_eta(delta),center_eta(delta_true)),
            state_eta_centered_controller_delta_correlation=correlate(pred_interaction,true_interaction),
            state_controller_eta_interaction_MSE_gain_over_zero=contrast(interaction_gain,np.zeros(n)),
            bootstrap_reselected_global_fixed_ci95=ci(matched_samples-best_fixed),
            bootstrap_reselected_controller_fixed_ci95=ci(matched_samples-best_fixed_by_c),
            selected_oracle_regret=float((lower.max(-1)-value).mean()),
            selection_counts=[np.bincount(choice[:,c],minlength=16).tolist() for c in range(2)])
        boot_values[kind+'_minus_reselected_fixed']=matched_samples-best_fixed
        np.savez_compressed(OUT/f'predictions_{kind}.npz',matched=q,wrong_controller=q[:,::-1],
            mismatched=mismatched,permutations=permutations,per_seed=expit(np.stack(zseeds)))
    eta_choice=predictions['eta_only'].argmax(-1)
    for kind,q in predictions.items():
        results[kind]['minus_eta_only']=contrast(selected(lower,q.argmax(-1)),selected(lower,eta_choice))
    for kind in ('eta_only','state_only','current_field'):
        np.testing.assert_allclose(predictions[kind][:,0],predictions[kind][:,1],atol=1e-7,rtol=0)
    write(OUT/'results.json',dict(scope=p['scope'],primary=p['primary'],states=n,seeds_per_cell=4,
        models=results,seed_results=seed_results,
        all_fixed_success=lower.mean((0,1)).tolist(),controller_fixed_success=lower.mean(0).tolist(),
        oracle_success_lower=float(lower.max(-1).mean()),numerical_unknown=int(u.sum()),
        collision_count=int(d['collision'].sum()),training_used_these_states=False,
        new_rollouts=0,robust_B15='Not estimable from Q4',
        limitation='Same two controller programs appear in training. Previously inspected historical development pool. No new-controller or final-confirmation claim.'))
    np.savez_compressed(OUT/'bootstrap.npz',indices=bootstrap,**boot_values)
    print(json.dumps({k:{m:v[m] for m in ['q_mse','controller_delta_correlation','eta_centered_controller_delta_correlation','mean_success_lower','matched_minus_mismatch']} for k,v in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','evaluate','run'])
    a=p.parse_args()
    if a.stage=='run':prepare();features();evaluate()
    else:globals()[a.stage]()
