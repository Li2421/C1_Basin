"""Cached-label upper-bound inputs for the known six-parameter policy family."""
import argparse,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import read,write,freeze,sha
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'controller_q_diversity_v1'
VARIANTS=('factorial_toy_constants','factorial_ring_constants')
ARMS=('exact_parameters','exact_parameters_initializer','field_initializer')
SEEDS=(17,23,41)
JOBS=[dict(variant=v,arm=a,seed=s) for v in VARIANTS for a in ARMS for s in SEEDS]
RESERVED=[3,4,5,6,10,11,12,13]


def prepare():
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    p=read(OLD/'protocol.json');states={s['state_uid']:s for s in p['states']}
    runtimes={sc:native_runtime(sc) for sc in p['controllers']}
    freeze(HERE/'protocol.json',dict(
        prior_reuse_audit=['controller_q_diversity_v1/input_identifiability.json','controller_q_diversity_v1/factorial_development_results.json',
            'controller_cross_scene_root_v2/crossed_support','controller_q_diversity_v1/training_plan.json'],
        previously_answered='Six analyticparameters recoverable fromcachedfieldqueries ontheoldtestedcases; bilinearfieldheads/scaling/coveragecontrolsalreadytested. Do not repeat these.',
        remaining_gap='No direct exact-program-parameter or explicit nativeFFinitialization-distribution input experiment found. Existing source/target constants audit explicitly noted different native initializers.',
        inputs='C: exact known6 analyticparameters, optionally8per-agent initialization fields; or existingcompact64field+same8initializerfields. S: existingphysicalentities andrestoredexecutionconstants. eta: exactoriginal16pool.',
        initialization_fields=['physical_mean_goal_forward/vmax','physical_mean_goal_lateral/vmax','physical_cov_ff/vmax^2','physical_cov_fl/vmax^2','physical_cov_ll/vmax^2','native_chart_axis0_goal_forward','native_chart_axis0_goal_lateral','native_component_clip_bound/vmax'],
        architecture_control='Identical oldnetwork/parameter shapes/init/optimizer/draws/checkpointbudget.8knowninitializerfields occupy8otherwisezero physical channels locally; sourcefiles unchanged. Parameterfeatures paddedtoexisting64Cchannels.',
        checkpoint='Reuse originalsource-only held-controller4/5 Q MSE+0.5Ccontrast MSE; regret thenNLL thenearliest ties. Not NLL-only. No targetlabelsforfitting/normalization/checkpoint.',
        baselines='Reuse exact oldeta-only,compactfield andcompactfield-bilinear checkpoints, sameanalytic0..3TRAINlabels. No refits.',
        training_rule_sha256=sha(OLD/'training_plan.json'),old_train_code_sha256=sha(OLD/'train.py'),wrapper_sha256=sha(__file__),
        jobs=JOBS,new_rollouts=0,new_policy_function_queries=0,
        evaluation='OpenedDEVELOPMENT only. Source128TRAIN/16VAL andopposite144states. Also32oldFINALphysicalstates withunseenanalytic6/7 asalready-opened independentstate/controller diagnostic; notfreshconfirmation.',
        scope='Two-scene analytic-program diagnostic, NOT four-sceneLOSO norcomplete description of arbitrary neuralFlow. Full knownfamily+initialization input tests whether approximate controller encoding alone explainsfailure.',
        literature=['https://arxiv.org/abs/2002.11833','https://arxiv.org/abs/2010.09536','https://proceedings.mlr.press/v119/raileanu20a.html']))
    for variant in VARIANTS+('final',):
        d=np.load(OLD/'data'/variant/'dataset.npz');theta=[];initializers=[]
        for sid,name,sc in zip(d['state_uid'],d['controller'],d['scene']):
            assert name.startswith('analytic')
            rt=runtimes[str(sc)];physical=states[str(sid)]['physical'];env=SimpleNamespace(**{k:np.asarray(v,float) for k,v in physical.items() if k in ('positions','velocities','goals')})
            n=len(env.positions);e=env.goals-env.positions;e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12);lat=np.stack((-e[:,1],e[:,0]),-1)
            local=lambda w:np.stack(((w*e).sum(-1),(w*lat).sum(-1)),-1)
            mu=local(rt.to_world(rt.mean,env))/rt.config.max_speed
            b0=rt.to_world(np.column_stack((np.ones(n),np.zeros(n))),env)
            b1=rt.to_world(np.column_stack((np.zeros(n),np.ones(n))),env)
            l0=local(b0*rt.scale[:,0,None])/rt.config.max_speed;l1=local(b1*rt.scale[:,1,None])/rt.config.max_speed
            cov=l0[:,:,None]*l0[:,None,:]+l1[:,:,None]*l1[:,None,:]
            z=np.column_stack((mu,cov[:,0,0],cov[:,0,1],cov[:,1,1],local(b0),np.full(n,1/rt.config.max_speed)))
            assert z.shape==(n,8)
            theta.append(p['controllers'][str(sc)][str(name)]['payload']['controller_parameters'])
            initializers.append(np.pad(z,((0,4-n),(0,0))))
        np.savez_compressed(HERE/f'program_features_{variant}.npz',parameters=np.asarray(theta,np.float32),initializer=np.asarray(initializers,np.float32),state_uid=d['state_uid'],controller=d['controller'])
    for arm in ARMS:
        root=HERE/arm;root.mkdir(exist_ok=True)
        target=root/'training_plan.json'
        if not target.exists():target.symlink_to(OLD/'training_plan.json')
        for variant in VARIANTS:
            dest=root/'data'/variant;dest.mkdir(parents=True,exist_ok=True)
            for name in ('dataset.npz','manifest.json'):
                target=dest/name
                if not target.exists():target.symlink_to(OLD/'data'/variant/name)
    shutil.copy2(OLD/'train.py',HERE/'frozen_parent_train.py')
    write(HERE/'input_artifact_audit.json',dict(files={str(f.name):sha(f) for f in HERE.glob('program_features_*.npz')},
        source_datasets={v:sha(OLD/'data'/v/'dataset.npz') for v in VARIANTS+('final',)},new_rollouts=0,policy_function_queries=0,
        checkpoint_provenance={sc:dict(path=str(rt.checkpoint),sha256=sha(rt.checkpoint),mean=np.asarray(rt.mean).tolist(),scale=np.asarray(rt.scale).tolist()) for sc,rt in runtimes.items()}))


def install(arm,variant):
    from diagnostics.controller_q_diversity_v1 import train
    import jax.numpy as jnp
    original_norm,original_inputs=train.normalization,train.inputs
    def features(d):
        f=dict(np.load(HERE/f'program_features_{variant}.npz'))
        np.testing.assert_array_equal(f['state_uid'],d['state_uid']);np.testing.assert_array_equal(f['controller'],d['controller'])
        return f
    def norm(d,kind):
        if arm=='field_initializer':return original_norm(d,kind)
        f=features(d);a=f['parameters'][d['split']=='train']
        return dict(mean=a.mean(0).tolist(),scale=np.maximum(a.std(0),.05).tolist(),source_train_only=True,representation='exact_analytic_parameters',clip=10.)
    def inputs(d,n):
        f=features(d)
        if arm=='field_initializer':x,c,e=original_inputs(d,n)
        else:
            x={k[7:]:jnp.asarray(d[k],jnp.float32) for k in d.files if k.startswith('entity_')};e=jnp.asarray(d['eta'],jnp.float32)
            a=np.clip((f['parameters']-np.asarray(n['mean']))/np.asarray(n['scale']),-10,10)
            c=np.zeros((len(a),4,64),np.float32);c[:,:,:6]=a[:,None];c*=d['entity_agent_mask'][...,None];c=jnp.asarray(c)
        if arm.endswith('initializer'):
            assert np.all(np.asarray(x['agents'])[...,RESERVED]==0)
            x=dict(x);x['agents']=x['agents'].at[...,jnp.asarray(RESERVED)].set(jnp.asarray(f['initializer']))
        return x,c,e
    train.HERE=HERE/arm;train.normalization=norm;train.inputs=inputs
    return train


def fit(index):
    j=JOBS[index];p=read(HERE/'protocol.json');assert sha(OLD/'train.py')==p['old_train_code_sha256']
    train=install(j['arm'],j['variant']);train.fit('state_field_compact',j['seed'],128,j['variant'])
    out=HERE/j['arm']/'fits'/j['variant']/'state_field_compact'/f"seed{j['seed']}"
    write(out/'intervention_audit.json',dict(job=j,wrapper_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),
        source_dataset_sha256=sha(OLD/'data'/j['variant']/'dataset.npz'),new_rollouts=0,source_train_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int);a=p.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
