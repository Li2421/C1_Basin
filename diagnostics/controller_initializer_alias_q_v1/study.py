"""Does a proven static controller-input alias change terminal Q? Four TRAIN.

Original exact labels are reused; only the predeclared altered initializer is
simulated. This is an information-sufficiency diagnostic, not a LOSO repair.
"""
import argparse, copy, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_diversity_v1.runtime import FIELD
from shared_rollout_db.src.rollout_db import connect, transaction, uid, canonical, eta_identity
from shared_rollout_db.src.cache_writer import append_journal
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_four_scene_mechanism_v1'
EXPERIMENT='exp_controller_initializer_alias_four_Q_v1'
SCENE='four_way_intersection'
FAMILIES=('isotropic','anisotropic','rotate_positive','rotate_negative')
read,write,freeze,sha=old.read,old.write,old.freeze,old.sha


def prepare():
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    assert read(ROOT/'controller_initializer_alias_v1/results.json')['static_input_alias_verified']
    parent=read(BASE/'protocol.json')
    states=[s for s in parent['states'] if s['scenario']==SCENE and s['study_split']=='train']
    assert len(states)==32
    base=native_runtime(SCENE);jax.config.update('jax_enable_x64',False)
    z=np.asarray(jax.random.normal(jax.random.PRNGKey(2026100908),(1,base.agent.config['joint_act_dim']))).reshape(-1,2)
    scale=2*base.scale;mean=base.mean+(base.scale-scale)*z
    profiles={}
    for name in FAMILIES:
        original=parent['controllers'][SCENE][name];p=copy.deepcopy(original['payload'])
        p['base_controller']='endpoint_matched_response_initializer_alias_v1'
        p['initializer_override']=dict(mean=mean.tolist(),scale=scale.tolist(),alpha=2.,
            descriptor_key=2026100908,definition='mean_prime=mean+(scale-scale_prime)*z_descriptor; scale_prime=2*scale')
        p['initializer_wrapper_sha256']=sha(__file__)
        profiles[name]=dict(controller_uid=uid('ctl',p),payload=p,old_controller_uid=original['controller_uid'])
    p=copy.deepcopy(parent)
    p.update(schema='four_train_initializer_alias_Q_v1',states=states,controllers={SCENE:profiles},
        train_seeds=[0,1],validation_seeds=[],eta_indices=[0,5,10,15])
    p['cause_test']=dict(
        hypothesis='A single-random-key compact64 field descriptor aliases Gaussian initializer laws that differ in long-Q state x eta preference.',
        history='Old nominal endpoint alias already proved; M recovery and exact initializer augmentation already tested. NEW question is precise CURRENT compact64 alias with same M/reference and different Gaussian law. Static Stage32 proves analytic alias and action differences. No repeated input-augmentation fit.',
        sampling='ALL32 original natural Four TRAIN states. All4 uncoupled M, fixed K4 indices0/5/10/15, original paired seeds0/1. No state/candidate/success selection.',
        scene_choice='Four shared goal program has independently verified long-Q interactions, low censoring, and the long-Q noise explanation was rejected. This is not four-scene confirmation.',
        intervention='Only wrapper initializer mean/scale. Same frozen nominal endpoint, M, Flow checkpoint, safety, FF, eta, environment, horizon, evaluator and RNG streams. Base Flow properties unchanged.',
        primary='Independent-seed cross product of new-minus-old Q differences, both raw and eta-centered; isolate state x eta by two-way centering across state and eta. Physical-state bootstrap20000, conservative bounds for every unknown. All families aggregate and individually; no winner selection.',
        interpretation='Positive raw difference energy proves current compact64 is not universally sufficient for this extended controller family. Positive state_eta energy additionally proves missing information matters to state-conditioned preference. Neither proves that existing native-scene LOSO failure is caused by these absent initializer crossings.',
        continuation='No automatic training or rollout expansion. If long-Q alias absent/uncertain, retain limitation and do not call root cause. If present, old initializer-augmentation negative still requires separate support/composition causal contrast.',
        new_rollouts=1024,baseline_rollouts_reused=1024,new_physical_states=0,new_fits=0,
        final_confirmation=False,literature=read(ROOT/'controller_initializer_alias_v1/protocol.json')['literature'])
    p['source_code_hashes']=read(ROOT/'controller_q_noise_coupling_v1/protocol.json')['source_code_hashes']
    assert all(sha(f)==h for f,h in p['source_code_hashes'].items())
    p['worker_code_sha256']=sha(__file__)
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Four TRAIN exact controller descriptor alias Q test',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=24,pure_Q=True,new_rollouts_max=1024))))
        for name,z in profiles.items():
            v=z['payload']
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (z['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    cases=[];baseline=[]
    for s in states:
        for name,z in profiles.items():
            for j in p['eta_indices']:
                for k in range(2):
                    sk=canonical(dict(future_index=k,future_root=2026100403,rng_namespace=s['rng_namespace']))
                    for group,cid in ((cases,z['controller_uid']),(baseline,z['old_controller_uid'])):
                        key=dict(state=s['state_uid'],eta=eta_identity(p['eta'][j])[0],controller=cid,seed=sk)
                        group.append(dict(state_uid=key['state'],eta_uid=key['eta'],controller_uid=cid,seed_key=sk,rollout_uid=uid('roll',key),
                            scene=SCENE,controller_name=name,split='train',state_index=s['study_index'],eta_index=j,eta=p['eta'][j],seed=k))
    counts=Counter();missing=[];groups=defaultdict(list)
    with connect(True) as db:
        for group,label in ((cases,'altered'),(baseline,'original')):
            for r in group:
                z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
                if z is None:
                    assert label=='altered','Missing baseline label; do not replace experiment silently'
                    missing.append(r)
                else:
                    assert not z['conflict_quarantined']
                    assert all(z[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                    counts[label+'_unknown' if z['numerical_failure'] else label+'_exact_reuse']+=1
                if label=='altered':groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing)
    freeze(HERE/'cases.json',cases);freeze(HERE/'baseline_cases.json',baseline);freeze(HERE/'missing_cases.json',missing)
    freeze(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    freeze(HERE/'cache_history_audit.json',dict(full_inventory_path=str(ROOT/'controller_initializer_alias_v1/cache_history_audit.json'),
        inventory_sha256=sha(ROOT/'controller_initializer_alias_v1/cache_history_audit.json'),profiles_scanned=248,counts=dict(counts),
        reason='Original full-label cache exact-reused. No altered-law profile/labels existed in full inventory; only1024 new altered-controller trials. State/encoding change does not invalidate original labels.'))
    print(dict(counts))


def worker(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json');assert sha(__file__)==p['worker_code_sha256']
    assert all(sha(f)==h for f,h in p['source_code_hashes'].items())
    pre=read(HERE/'cache_preflight.json')['summary'];exact=read(HERE/'exact_identity_preflight.json')
    assert pre['ambiguous']==0 and pre['genuinely_missing']==exact['genuinely_missing']+exact.get('altered_unknown',0)
    cases=read(HERE/'missing_cases.json')[index::workers];states={s['state_uid']:s for s in p['states']}
    base=native_runtime(SCENE);jax.config.update('jax_enable_x64',False)
    original_mean,original_scale=base.mean.copy(),base.scale.copy();runtimes={}
    for name in FAMILIES:
        rt=EndpointMatchedField(base,name);v=p['controllers'][SCENE][name]['payload']['initializer_override']
        rt.mean=np.asarray(v['mean']);rt.scale=np.asarray(v['scale']);runtimes[name]=rt
    np.testing.assert_array_equal(base.mean,original_mean);np.testing.assert_array_equal(base.scale,original_scale)
    buffer=[];start=time.monotonic()
    for r in cases:
        dest=HERE/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():buffer.append(read(dest));continue
        rt=runtimes[r['controller_name']];env=rt.make_env(states[r['state_uid']])
        term='timeout';err=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for t in range(rt.config.max_steps):
            try:
                action,_,n=rt.action(env,key_for(states[r['state_uid']],r['seed'],t),np.asarray(r['eta']),'field');calls+=n;steps+=1
                if first is None:first=np.asarray(action).tolist()
                _,_,done,info=env.step(action);term=info['termination']
                if done:break
            except Exception as ex:
                err=dict(step=t,type=type(ex).__name__,message=str(ex)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        result=dict(schema='controller_response_root_seed_v1',experiment_uid=EXPERIMENT,protocol_sha256=sha(HERE/'protocol.json'),case=r,
            **{k:r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
            success=int(summary['collision_free_success'] and term=='success' and err is None),deadlock=int(term=='deadlock'),
            timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(err is not None),episode_length=steps,
            outcome=term,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=err,summary=summary,
            execution_semantics_version='scene_precision_guard_v1_initializer_alias',jax_enable_x64=bool(jax.config.x64_enabled),
            execution_code_sha256=sha(__file__),host_backend_sha256=sha(old.HERE/'numpy_backend.py'))
        assert not(result['success'] and (result['collision'] or result['numerical_failure']))
        write(dest,result);buffer.append(result)
        if len(buffer)>=4:append_journal(buffer,EXPERIMENT,f'four_{index}');buffer=[]
    if buffer:append_journal(buffer,EXPERIMENT,f'four_{index}')
    write(HERE/f'worker{index}.json',dict(cases=len(cases),seconds=time.monotonic()-start,completed=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','worker']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=36)
    a=ap.parse_args();prepare() if a.stage=='prepare' else worker(a.index,a.workers)
