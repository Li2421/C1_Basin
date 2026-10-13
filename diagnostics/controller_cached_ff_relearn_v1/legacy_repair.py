"""Single-factor diagnosis of the legacy controller-cache defect.

Old labels, old architecture, old optimizer, old checkpoint rule. Only repair
the controller attribution of the nominal H20 channels. Never a final method.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from .study import HERE, OLD, FIELD, read, write, freeze, sha

OUT=HERE/'legacy_repair'
ARMS=('original_cache','controller_correct_cache')


def prepare():
    states=read(HERE/'protocol.json')['states']
    freeze(OUT/'protocol.json',dict(
        scope='Root-cause isolation of a verified controller attribution bug; not H20 deployment or optimization tuning',
        states=states,arms=list(ARMS),seeds=[17,23,41],
        labels='Exact original source108 selected-Q16 records, 80 TRAIN /28 VAL; no additional Q4 cells',
        test='Original16 held physical states, now DEVELOPMENT',
        architecture_optimizer_selection='Exact original learn.train_critic full: 1600 maximum steps,8e-4 AdamW,1e-4 decay,batch32,source VAL NLL checkpoint/early stop',
        intervention='Replace held-controller nominal prefix0:10 by its correctly namespaced baseline. Recompute progress and clearance differences21,23 by adding old nominal minus corrected nominal.',
        channel22='Final-position distance cannot be reconstructed from aggregate context. Drop this ONE channel in BOTH arms. This restriction is fixed before repaired-input training.',
        controller_features='H20 allowed only for this legacy defect diagnosis. The new final candidate continues to use static policy queries, not rollout context.',
        new_Q_labels=0,feature_preflight='Existing corrected-eta summaries are reused; only held nominal prefix is recomputed. Cached held nominal is controller-ambiguous and cannot be exactly reused.',
        controller_code_sha256=sha(OLD/'held.py'),context_code_sha256=sha(OLD/'learn.py'),
        no_architecture_loss_or_hyperparameter_search=True))


def features(index,shards):
    import jax
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1.held import HeldRuntime
    from field_pipeline_v1 import learn
    p=read(OUT/'protocol.json');rt=HeldRuntime('ring_exchange')
    jax.config.update('jax_enable_x64',False)
    done=0;steps=[0];old_step=rt.env_cls.step
    def step(self,*args,**kwargs):steps[0]+=1;return old_step(self,*args,**kwargs)
    rt.env_cls.step=step
    try:
        for i,state in enumerate(p['states']):
            if i%shards!=index:continue
            dest=OUT/'prefixes'/(state['uid']+'.json')
            if dest.exists():continue
            learn.NOMINAL_CACHE.clear()
            cx=learn.h20_context(rt,state,np.zeros(3))
            write(dest,dict(state_uid=state['uid'],held_nominal_prefix=cx[:10].tolist(),
                cache_cleared_before_controller_query=True,controller_code_sha256=p['controller_code_sha256']))
            done+=1
            if done%16==0:print('Repaired static artifact count',done,flush=True)
    finally:rt.env_cls.step=old_step;learn.NOMINAL_CACHE.clear()
    write(OUT/f'features_complete{index}.json',dict(index=index,shards=shards,states=done,
        physical_steps_for_input_repair=steps[0],new_Q_labels=0))


def raw_context(rows,arm):
    result=np.asarray([r['context'] for r in rows],np.float32).copy()
    if arm=='controller_correct_cache':
        for i,r in enumerate(rows):
            if r.get('controller_condition',r.get('condition'))!='held':continue
            corrected=np.asarray(read(OUT/'prefixes'/(r['state_uid']+'.json'))['held_nominal_prefix'],np.float32)
            original=result[i,:10].copy()
            result[i,:10]=corrected
            result[i,21]+=original[0]-corrected[0]
            result[i,23]+=original[3]-corrected[3]
    result[:,22]=0
    return result


def fit(arm,seed):
    assert arm in ARMS
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1 import learn
    learn.BASE=OLD/'decision_pair_critic_v3';learn.MODEL=OUT/arm/'models';learn.SCENES=('ring_exchange',)
    original_data=learn.data
    def patched_data(include_missing_context=False):
        rec,x,si,eta,_,tr,va,_=original_data(include_missing_context)
        cx=raw_context(rec,arm);center=cx[tr].mean(0);scale=np.maximum(cx[tr].std(0),.05)
        return rec,x,si,eta,(cx-center)/scale,tr,va,dict(center=center.tolist(),scale=scale.tolist())
    learn.data=patched_data
    learn.train_critic('full',seed)
    write(OUT/arm/f'fit{seed}.json',dict(arm=arm,seed=seed,protocol_sha256=sha(OUT/'protocol.json'),
        original_training_code_sha256=sha(OLD/'learn.py'),data_source_sha256=sha(OLD/'decision_pair_critic_v3/source_pairs.json'),
        sole_controlled_difference='Controller attribution of H20 baseline; channel22 removed in both arms'))


def evaluate():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from scipy.special import expit
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1.learn import Critic,CENTER,RADIUS,gather
    from .evaluate import correlate,center_eta
    source=read(OLD/'decision_pair_critic_v3/source_pairs.json')['records']
    train=np.array([r['split']=='train' for r in source])
    manifest=read(OLD/'decision_pair_test_v3/manifest.json')
    pairs=read(OLD/'decision_pair_test_v3/pairs.json')['pairs']
    by={(r['state_uid'],r['condition'],r['eta_index']):r for r in pairs}
    rows=[by[s['uid'],c,j] for s in manifest['states'] for c in ('base','held') for j in range(16)]
    x=dict(np.load(OLD/'decision_pair_test_v3/state_features.npz'))
    si=np.repeat(np.arange(16),32);eta=(np.asarray([r['eta'] for r in rows],np.float32)-CENTER)/RADIUS
    truth=np.array([r['success']/max(r['success']+r['fail'],1) for r in rows]).reshape(16,2,16)
    qtrue_delta=truth[:,1]-truth[:,0]
    rng=np.random.default_rng(2026100924);boot=rng.integers(16,size=(20000,16));predictions={};results={}
    for arm in ARMS:
        cx=raw_context(source,arm);center=cx[train].mean(0);scale=np.maximum(cx[train].std(0),.05)
        context=(raw_context(rows,arm)-center)/scale;qs=[];details=[]
        for seed in (17,23,41):
            folder=OUT/arm/'models/critic_full'/f'seed{seed}'
            model=Critic(True,True,False)
            params=model.init(jax.random.PRNGKey(seed),gather(x,[0]),jnp.zeros((1,3)),jnp.zeros((1,24)),jnp.zeros((1,3)))
            params=serialization.from_bytes(params,(folder/'checkpoint.msgpack').read_bytes())
            z=np.asarray(model.apply(params,gather(x,si),jnp.asarray(eta),jnp.asarray(context),jnp.zeros((len(si),3))))
            q=expit(z).reshape(16,2,16);qs.append(q);delta=q[:,1]-q[:,0]
            details.append(dict(seed=seed,selected=read(folder/'summary.json'),
                q_mse=float(np.mean((q-truth)**2)),delta_correlation=correlate(delta,qtrue_delta),
                eta_centered_delta_correlation=correlate(center_eta(delta),center_eta(qtrue_delta))))
        q=np.mean(qs,0);predictions[arm]=q;delta=q[:,1]-q[:,0]
        results[arm]=dict(seed_results=details,q_mse=float(np.mean((q-truth)**2)),
            controller_delta_mse=float(np.mean((delta-qtrue_delta)**2)),
            delta_correlation=correlate(delta,qtrue_delta),eta_centered_delta_correlation=correlate(center_eta(delta),center_eta(qtrue_delta)))
    original=predictions[ARMS[0]];repaired=predictions[ARMS[1]]
    gain=((original-truth)**2-(repaired-truth)**2).mean((1,2))
    delta_gain=(((original[:,1]-original[:,0])-qtrue_delta)**2-((repaired[:,1]-repaired[:,0])-qtrue_delta)**2).mean(-1)
    write(OUT/'results.json',dict(arms=results,
        probability_mse_improvement=dict(mean=float(gain.mean()),ci95=np.quantile(gain[boot].mean(-1),[.025,.975]).tolist()),
        controller_delta_mse_improvement=dict(mean=float(delta_gain.mean()),ci95=np.quantile(delta_gain[boot].mean(-1),[.025,.975]).tolist()),
        physical_states=16,new_Q_labels=0,
        boundary='Input defect is confirmed independently. This small retrospective development ablation estimates its effect under the unchanged legacy trainer; it is not final confirmation or a new H20 method.'))
    np.savez_compressed(OUT/'predictions_bootstrap.npz',**predictions,truth=truth,bootstrap_indices=boot)
    print(read(OUT/'results.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','fit','evaluate']);p.add_argument('--index',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='features':features(a.index,a.shards)
    elif a.stage=='fit':fit(a.arm,a.seed)
    else:evaluate()
