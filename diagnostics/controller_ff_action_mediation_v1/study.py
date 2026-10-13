"""Locate a failure between controller->current action and long-horizon Q.

Queries execute the existing FF action function at the CURRENT state with
independent, prespecified random keys. Environment.step is forbidden. This is
an explicitly privileged diagnostic, not a declared final critic interface.
"""
import argparse,json
from pathlib import Path
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import HERE as FOUR,read,write,freeze,sha
from diagnostics.controller_ff_crossed_cache_full_v1.study import HERE as FULL

HERE=Path(__file__).resolve().parent
INDICES=(0,5,10,15)

def prepare():
    freeze(HERE/'protocol.json',dict(schema='ff_current_action_mediation_v1',new_physical_rollouts=0,
        hypothesis='Does supplying the exact current physical FF action repair transfer, beyond a static field descriptor? A positive result isolates failure in the learned controller/eta/safety composition; a negative result leaves future state/controller interactions unresolved.',
        status='DEVELOPMENT diagnostic; not final model or fresh confirmation',
        conditions=['operator_only','operator_plus_current_action'],
        physical_input='Existing masked entities with known physical execution constants; no scene or controller ID.',
        architecture='Same per-agent fused physical/controller encoder and hidden widths; five appended channels set to zero in operator_only. Compare these two arms causally; do not attribute architecture differences from the old bilinear network to the action input.',
        action_input='Goal-frame mean2, variance2 and solved-query fraction1, four independent fixed-key evaluations of exact frozen FF action at current physical state. No environment step, no later-state query, no success-label features.',
        keys=dict(root=2026101204,fold_in=[0,1,2,3],independent_of_label_seed_roots=True),
        eta_indices=list(INDICES),states='All160 existing natural32TRAIN+8VAL states across4scenes; same5matrixcontrollers; exact old Q2/Q4 labels.',
        optimization=dict(steps=6000,batch_size=120,learning_rate=.0008,weight_decay=.0001,seeds=[17,23,41],validation_every=200),
        checkpoint='Source-validation Q probability MSE+0.5controller contrast MSE; regret then NLL tie break.',
        isolation='All gradients, normalization and checkpoint selection source-only; held-scene labels diagnostic only.',
        missing_static_action='Preserve query failures with solved fraction; no numerical reruns; no fabricated action for a failed query. Zero mean/variance only if all queries fail, with explicit zero fraction.',
        literature=['https://arxiv.org/abs/2204.02877','https://arxiv.org/abs/2010.09536'],
        literature_boundary='Policy/environment factorization motivates checking the action mediator; these publications do not guarantee zero-shot value transfer on the present maps.'))


def features(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField,FAMILIES
    p=read(FOUR/'protocol.json');bases={};runtimes={};errors=[]
    for si in range(index,len(p['states']),workers):
        state=p['states'][si];sc=state['scenario'];sid=state['state_uid']
        if sc not in bases:bases[sc]=native_runtime(sc)
        base=bases[sc];jax.config.update('jax_enable_x64',False)
        def forbidden_step(*a,**kw):raise RuntimeError('No physical rollout allowed in action mediation')
        base.env_cls.step=forbidden_step
        for name in FAMILIES:
            path=HERE/'features'/sid/(name+'.npz')
            if path.exists():continue
            if (sc,name) not in runtimes:runtimes[sc,name]=EndpointMatchedField(base,name)
            rt=runtimes[sc,name];env=rt.make_env(state);n=len(env.positions)
            e=env.goals-env.positions;e/=np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-12)
            lat=np.stack((-e[:,1],e[:,0]),-1);values=np.zeros((16,n,5),np.float32);raw=np.full((16,4,n,2),np.nan)
            success=np.zeros((16,4),bool);fail=[]
            for j in INDICES:
                for k in range(4):
                    key=jax.random.fold_in(jax.random.PRNGKey(2026101204),k)
                    try:
                        action,_,_=rt.action(env,key,np.asarray(p['eta'][j]),'field')
                        raw[j,k]=action;success[j,k]=True
                    except Exception as ex:
                        fail.append(dict(eta_index=j,key_index=k,type=type(ex).__name__,message=str(ex)[:400]))
                if success[j].any():
                    a=raw[j,success[j]];local=np.stack(((a*e).sum(-1),(a*lat).sum(-1)),-1)/rt.config.max_speed
                    values[j,:,:2]=local.mean(0);values[j,:,2:4]=local.var(0)
                values[j,:,4]=success[j].mean()
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,action_features=values,raw_actions=raw,query_success=success,physical_steps=0)
            if fail:errors.append(dict(state_uid=sid,controller=name,failures=fail))
    write(HERE/f'features_worker{index}.json',dict(complete=True,physical_steps=0,query_errors=errors))


def assemble():
    old=np.load(FULL/'dataset.npz');keep=old['kind']=='matrix';data={}
    for k in old.files:data[k]=old[k][keep]
    # Feature branches are identical for these stationary matrix programs.
    np.testing.assert_array_equal(data['feature'][...,:64],data['feature'][...,64:])
    values=[];failed=0;queries=0
    for sid,name in zip(data['state_uid'],data['controller']):
        f=np.load(HERE/'features'/str(sid)/(str(name)+'.npz'))
        assert int(f['physical_steps'])==0
        a=f['action_features'];values.append(np.pad(a,((0,0),(0,4-a.shape[1]),(0,0))))
        success=f['query_success'][list(INDICES)];failed+=int((~success).sum());queries+=success.size
    data['action_feature']=np.asarray(values);data['feature']=data['feature'][...,:64]
    np.savez_compressed(HERE/'dataset.npz',**data)
    freeze(HERE/'dataset_audit.json',dict(dataset_sha256=sha(HERE/'dataset.npz'),rows=len(values),static_queries=queries,
        query_failures=failed,physical_steps=0,new_rollouts=0,old_success_labels_unchanged=True))
    print(json.dumps(read(HERE/'dataset_audit.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','assemble']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    if a.stage=='features':features(a.index,a.workers)
    else:globals()[a.stage]()
