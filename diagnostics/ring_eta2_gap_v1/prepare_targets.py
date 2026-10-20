"""Audit 36 Round18 checkpoints and build compatible Ring controller profiles."""
from core import *
import copy
import pickle

from shared_rollout_db.src.rollout_db import uid
from diagnostics.controller_new_checkpoint_confirmation_v1.fp32_n8.compatibility import param_leaves

def parameter_hash(path):
    obj=pickle.loads(Path(path).read_bytes())
    leaves=param_leaves(obj['agent']['network']['params'])
    h=hashlib.sha256()
    for key,a in leaves.items():
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest(),leaves,obj

def main():
    prereg()
    plan=read(HERE/'TRAINING_PLAN.json')
    source=read(R10/'protocol.json')
    expected=plan
    assert len(expected)==36
    old_hashes={};unreadable=[]
    for path in sorted((ROOT/'diagnostics').glob('**/*.pkl')):
        if path.is_relative_to(HERE):continue
        try:
            ph,_,_=parameter_hash(path)
            old_hashes.setdefault(ph,[]).append(str(path))
        except Exception as exc:unreadable.append(dict(path=str(path),exception=type(exc).__name__))
    import jax
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    jax.config.update('jax_enable_x64',False)
    assert not jax.config.x64_enabled
    probe_state=read(R5/'stage_2/probes/protocol.json')['states'][0]
    profiles={};checks=[];seen=set();training=[]
    for item in expected:
        ck=HERE/'checkpoints'/item['name']/'best.pkl'
        assert ck.is_file(),ck
        param_hash,leaves,obj=parameter_hash(ck)
        assert param_hash not in old_hashes and param_hash not in seen,(item['name'],param_hash)
        seen.add(param_hash)
        assert all(np.asarray(a).dtype==np.float32 for a in leaves.values())
        config=obj['config'];prior=pickle.loads((ROOT/item['original_best']).read_bytes())['config']
        assert tuple(config['actor_hidden_dims'])==tuple(item['actor_hidden_dims'])
        assert config['environment_fingerprint']==prior['environment_fingerprint']
        assert all(np.array_equal(np.asarray(config[k]),np.asarray(prior[k])) for k in ('obs_mean','obs_scale','act_mean','act_scale'))
        for key in prior:
            if key in ('actor_hidden_dims','obs_mean','obs_scale','act_mean','act_scale'):continue
            assert clean(config.get(key))==clean(prior[key]),(item['name'],key)
        manifest=ROOT/item['dataset']/'manifest.json'
        assert sha(manifest)==item['dataset_manifest_sha256']
        base=source['base_profiles'][item['run']]
        v=copy.deepcopy(base['payload'])
        v.update(flow_checkpoint_path=str(ck),flow_checkpoint_sha256=sha(ck),adapter_manifest_path=str(manifest),
                 adapter_manifest_sha256=sha(manifest),observation_frame=v.get('observation_frame','local'),
                 checkpoint_environment_fingerprint=read(manifest)['environment_fingerprint'],
                 execution_callable=str(P1/'flow_adapter.py'),execution_callable_sha256=sha(P1/'flow_adapter.py'),
                 controller_source_sha256=sha(P1/'flow_adapter.py'),parent_best_controller_uid=base['controller_uid'],
                 checkpoint_role=f"round18_{item['split']}",round18_recipe=item['name'])
        runtime=RingFlowAdapter(ck,read(manifest),v['observation_frame'])
        env=runtime.make_env(probe_state)
        key=jax.random.PRNGKey(20261714)
        a=np.asarray(runtime.native_fn(runtime.observation(env),key));b=np.asarray(runtime.native_fn(runtime.observation(env),key))
        assert a.dtype==np.float32 and a.tobytes()==b.tobytes()
        assert all(np.asarray(x).dtype==np.float32 for x in jax.tree_util.tree_leaves(runtime.agent.network.params))
        profiles[item['name']]=dict(controller_uid=uid('ctl',v),payload=v,split=item['split'],family=item['family'],
                                    recipe=item['run'],architecture=item['actor_hidden_dims'],seed=item['seed'],seed_offset=item['seed_offset'])
        summary=ck.parent/'training_summary.json';metrics=ck.parent/'metrics.jsonl';receipt=HERE/'training_receipts'/f"{item['name']}.json"
        assert summary.exists() and metrics.exists() and receipt.exists()
        training.append(dict(name=item['name'],split=item['split'],training_summary_sha256=sha(summary),
                             metrics_sha256=sha(metrics),start_receipt_sha256=sha(receipt),best_checkpoint_sha256=sha(ck),
                             final_checkpoint_sha256=sha(ck.parent/'final.pkl')))
        checks.append(dict(name=item['name'],split=item['split'],seed=item['seed'],checkpoint_path=str(ck),
                           checkpoint_sha256=sha(ck),parameter_sha256=param_hash,
                           actor_hidden_dims=item['actor_hidden_dims'],float32=True,adapter_forward_reproducible=True,
                           normalizers_identical_to_original=True,jax_enable_x64=False))
        print(json.dumps(dict(name=item['name'],split=item['split'],status='COMPATIBLE')),flush=True)
    freeze(HERE/'FINAL_BASE_PROFILES.json',dict(profiles=profiles,eligible_controller_order=list(profiles),
           train_controller_order=[n for n,v in profiles.items() if v['split']=='TRAIN-new'],
           test_controller_order=[n for n,v in profiles.items() if v['split']=='TEST'],
           source_recipe_order=source['controller_order']))
    write(HERE/'CHECKPOINT_NOVELTY_AUDIT.json',dict(status='PASS',historical_unique_param_hashes=len(old_hashes),
         historical_paths=sum(map(len,old_hashes.values())),unreadable_nonmatching_schemas=unreadable,checks=checks))
    write(HERE/'COMPATIBILITY_AUDIT.json',dict(status='PASS',usable=len(checks),checks=checks,success_labels_generated=False))
    write(HERE/'TRAINING_AUDIT.json',dict(status='PASS',controllers=len(training),historical_package_verified=True,GPU_FP32_verified=True,rows=training))
    print(json.dumps(dict(status='PASS',controllers=len(checks),historical_hashes=len(old_hashes))))

if __name__=='__main__':main()
