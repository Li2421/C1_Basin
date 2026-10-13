"""Cross identical FF operators with four real environment mechanisms.

This is an actual field-intervention diagnostic. Historical four-scene scripts
named FF but executing terminal projection are not used as compatible Q labels.
"""
import argparse,copy,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_response_families_v1 import study as parent
from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField,FAMILIES,REFERENCE
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,FIELD
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical

HERE=Path(__file__).resolve().parent
REPAIR=HERE.parent/'controller_operator_repair_v1'
RESPONSE=HERE.parent/'controller_response_families_v1'
AUDIT=HERE.parent/'controller_cross_scene_root_v2'
EXPERIMENT='exp_controller_four_scene_ff_mechanism_v1'
read,write,freeze,sha=old.read,old.write,old.freeze,old.sha

def prepare():
    previous=read(REPAIR/'protocol.json');states=copy.deepcopy(previous['states'])
    profiles=copy.deepcopy(previous['controllers']);environments=copy.deepcopy(previous['environments'])
    nativeaudit=read(AUDIT/'four_scene_runtime_audit.json')
    for row in nativeaudit['rows']:
        sc=row['scene'];assert row['map_adapter_matches'] and row['radius_matches'] and row['goal_tolerance_matches']
        config=row['config'];envfp=dict(config=config,environment_code_sha256=sha(FIELD/'source'/sc/'environment.py'),reset='native true-t0 reset, empty termination monitor')
        sid=uid('scenario',envfp);environments[sc]=dict(scenario_uid=sid,config=envfp,code_config_fingerprint=uid('env',envfp))
        code={f:sha(FIELD/f) for f in ('poc/field_controller.py','validation_v2/adapters.py','validation_stage2/factorial_controller.py','source/shared_control/basis_families.py')}
        native=dict(scenario_uid=sid,flow_checkpoint_sha256=row['checkpoint_sha256'],flow_checkpoint_path=row['checkpoint'],orthoflow3_sha256=code['source/shared_control/basis_families.py'],
            safety_projection_sha256=sha(FIELD/'source/shared_control/hard_projection.py'),cbf=row['cbf'],environment_sha256=uid('env',envfp),horizon=config['max_steps'],dt=config['dt'],
            success_semantics='native collision_free_success AND termination=success AND no numerical failure',
            conditioning='true_t0;native Flow resampled each step;no committed first action;unchanged physical eta basis recomputed each step',
            rng=previous['controllers']['ring_exchange']['flow']['payload']['rng'],jax_enable_x64=False,pseudo_steps=row['FF_steps'],action_code_hashes=code,
            agent_slot_semantics='unchanged native checkpoint/runtime slots; unified encoder masks only',chain='FF',
            execution_callable='frozen native Runtime.action(chain=field)',execution_callable_sha256=sha(FIELD/'poc/field_controller.py'))
        profiles[sc]={'flow':dict(controller_uid=uid('ctl',native),payload=native)}
        for family in FAMILIES:
            payload=copy.deepcopy(native)
            payload.update(base_controller='endpoint_matched_response_family_v1',response_family=family,controller_parameters=REFERENCE,
                controller_source_sha256=sha(RESPONSE/'runtime.py'),parent_runtime_sha256=sha(old.HERE/'runtime.py'),numpy_backend_sha256=sha(old.HERE/'numpy_backend.py'),
                execution_callable=str(RESPONSE/'runtime.py'),execution_callable_sha256=sha(RESPONSE/'runtime.py'),conditioning='true-t0; same unprojected nominal endpoint; varying FF response operator')
            profiles[sc][family]=dict(controller_uid=uid('ctl',payload),payload=payload)
        chosen=read(AUDIT/f'candidate_states_{sc}.json')['states'];counts=Counter()
        for src in chosen:
            s=copy.deepcopy(src);split=s.pop('split');i=counts[split];counts[split]+=1
            content=uid('content',dict(physical=s['physical'],reset='empty monitor true-t0'))
            s.update(uid=s['alias'],scenario=sc,scenario_uid=sid,content_hash=content,state_uid=uid('state',dict(scenario=sid,content=content)),
                family=s['alias'],study_split=split,study_index=i)
            s['metadata'].update(split=split,source_group=s['alias'],outcome_blind=True)
            states.append(s)
    p=copy.deepcopy(previous);p.update(schema='four_scene_ff_operator_mechanism_v1',states=states,controllers=profiles,environments=environments,
        scene_sampling='Original natural global state pools; DB/Four use frozen outcome-blind uniform32TRAIN+8VAL selection from512/32existing physical states.',
        scope='Root diagnosis and repair; four-scene source-only LOSO on actual frozen FF execution, not historical terminal-intervention labels.',
        no_final_confirmation_yet=True)
    freeze(HERE/'protocol.json',p)
    for name in ('runtime.py','static_operator_audit.json'):
        if not (HERE/name).exists():(HERE/name).symlink_to(RESPONSE/name)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Four-scene actualFF controller mechanism coverage',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(max_cpu_when_idle=24,pure_Q=True))))
        for sc,e in environments.items():
            db.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',(e['scenario_uid'],sc,e['code_config_fingerprint'],canonical(e['config'])))
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(environments[s['scenario']]['config']),canonical(dict(source_alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXPERIMENT))
        for sc,controls in profiles.items():
            for name,ctl in controls.items():
                v=ctl['payload'];prior=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert prior is None or prior[0]==canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    cases=[old.case(p,s,c,j,k) for s in states for c in FAMILIES for j in p['eta_indices'] for k in (p['train_seeds'] if s['study_split']=='train' else p['validation_seeds'])]
    cases.sort(key=lambda r:(r['split'],r['state_index'],r['controller_name'],r['eta_index'],r['seed'],r['scene']))
    freeze(HERE/'cases.json',cases);groups=defaultdict(list);missing=[];counts=Counter()
    with connect(True) as db:
        for r in cases:
            x=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if x is None:missing.append(r)
            else:
                assert not x['conflict_quarantined'] and all(x[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if x['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing)
    write(HERE/'missing_cases.json',missing);write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    freeze(HERE/'mechanism_protocol.json',dict(
        cause_test='Does crossing the same functionally-distinct operators with multiple source geometries repair the source-scene controller prior reversal?',
        train_states_per_scene=32,validation_states_per_scene=8,controllers=list(FAMILIES),
        eta_indices=[0,5,10,15],rollouts_per_train_cell=2,rollouts_per_validation_cell=4,
        groups=['joint four-scene','four source-only leave-one-scene-out folds'],
        source_only=['fitting','normalization','checkpoint selection'],
        methods=['eta_only','nominal14_padded64','static_field_compact64'],
        frozen_network='Same existing masked physical entities + bilinear head',
        checkpoint='Source-validation probability MSE+0.5controller-contrast MSE; ranking/regret tracked. No target label selection.',
        budget='6000steps,120validcells/step,three optimizationseeds17/23/41; no loss/architecture sweep.',
        new_Q_outcomes=3840,reused_Q_outcomes=3840,new_physical_rollouts_reason='No compatible DB/Four matrix-response-controller outcomes exist. Old terminal-intervention results are not actualFF labels.',
        limitation='DEVELOPMENT causal support experiment. More source geometries also adds independent states; no exclusive-cause claim from this contrast alone. New fresh confirmation required after method freezing.',
        literature=['https://proceedings.mlr.press/v119/raileanu20a.html','https://proceedings.mlr.press/v119/zhang20t.html']))
    print(json.dumps(dict(counts)),flush=True)

def audit():
    import jax
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json');rows=[]
    for scene in ('double_bottleneck','four_way_intersection'):
        base=native_runtime(scene);jax.config.update('jax_enable_x64',False)
        from poc.field_controller import Runtime
        for s in [s for s in p['states'] if s['scenario']==scene][:2]:
            for name in ('isotropic','rotate_positive'):
                rt=EndpointMatchedField(base,name);env=rt.make_env(s);key=key_for(s,0,0)
                for j in (0,15):
                    expected=[];errors=[]
                    for mode in ('numpy','original'):
                        try:
                            obs,safety,flow,safe,delta,_=rt.prepare(env,key,np.asarray(p['eta'][j]))
                            a,_=(rt.integrate(env,obs,key,delta,safety) if mode=='numpy' else Runtime.integrate(rt,env,obs,key,delta,safety))
                            expected.append(a);errors.append(None)
                        except Exception as ex:expected.append(None);errors.append((type(ex).__name__,str(ex)))
                    assert errors[0]==errors[1]
                    if errors[0] is None:np.testing.assert_array_equal(expected[0],expected[1])
                    rows.append(dict(scene=scene,state_uid=s['state_uid'],family=name,eta=j,exact_equal=True,error=errors[0]))
    write(HERE/'native_backend_audit.json',dict(rows=rows,all_equal=True,physical_steps=0))

def features(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.physical_state_critic_formal_v1 import physical_entities as phys
    p=read(HERE/'protocol.json');bases={};runtimes={}
    for s in p['states'][index::workers]:
        sc=s['scenario']
        for name in FAMILIES:
            dest=HERE/'features'/s['state_uid']/(name+'.npz')
            if dest.exists():continue
            prior=REPAIR/'features'/s['state_uid']/(name+'.npz')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if prior.exists():dest.symlink_to(prior);continue
            if sc not in bases:bases[sc]=native_runtime(sc)
            jax.config.update('jax_enable_x64',bool(p['controllers'][sc]['flow']['payload']['jax_enable_x64']))
            if (sc,name) not in runtimes:runtimes[sc,name]=EndpointMatchedField(bases[sc],name)
            rt=runtimes[sc,name]
            class Interface:
                def __getattr__(self,k):return getattr(rt,k)
            nominal=controller_context(rt,s);field=field_context(Interface(),s);entity=phys.entities(sc,s['physical'])
            np.savez_compressed(dest,nominal=nominal,field=field,**{'entity_'+k:v for k,v in entity.items()})
    write(HERE/'features'/f'worker{index}.json',dict(complete=True,simulator_steps=0,labels_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','features','worker','merge']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=96);a=p.parse_args()
    if a.action in ('prepare','audit'):globals()[a.action]()
    elif a.action=='features':features(a.index,a.workers)
    else:
        parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT
        if a.action=='worker':
            assert read(HERE/'native_backend_audit.json')['all_equal']
            parent.worker(a.index,a.workers)
        else:parent.merge()
