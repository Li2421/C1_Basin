"""Endpoint-alias falsification using frozen FF and natural cached states."""
import argparse, copy, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,FIELD,controller_context
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from shared_rollout_db.src.cache_writer import append_journal
from .runtime import EndpointMatchedField,FAMILIES,REFERENCE

HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_controller_response_families_ff_root_v1'
read,write,freeze,sha=old.read,old.write,old.freeze,old.sha

def prepare():
    source=read(old.HERE/'protocol.json')
    states=[s for s in source['states'] if s['study_split']=='train' and s['study_index']<8]
    # The existing comprehensive database audit is reused. All new controller
    # identities encode the exact new source and the unchanged execution hashes.
    profiles={}
    for scene in sorted({s['scenario'] for s in states}):
        native=source['controllers'][scene]['flow'];profiles[scene]={'flow':copy.deepcopy(native)}
        for family in FAMILIES:
            payload=copy.deepcopy(native['payload'])
            payload.update(base_controller='endpoint_matched_response_family_v1',response_family=family,
                controller_parameters=REFERENCE,controller_source_sha256=sha(HERE/'runtime.py'),
                parent_runtime_sha256=sha(old.HERE/'runtime.py'),numpy_backend_sha256=sha(old.HERE/'numpy_backend.py'),
                execution_callable=str(HERE/'runtime.py'),execution_callable_sha256=sha(HERE/'runtime.py'),
                conditioning='true-t0; same unprojected nominal endpoint; varying FF response operator')
            payload.pop('legacy_controller_uid_verified_by_original_function',None)
            profiles[scene][family]=dict(controller_uid=uid('ctl',payload),payload=payload)
    p=dict(schema='endpoint_alias_root_probe_v1',states=states,controllers=profiles,eta=source['eta'],
           eta_indices=[0,5,10,15],seeds=[0,1],environments=source['environments'],
           runtime_sha256=sha(HERE/'runtime.py'),FF_operator_sha256=source['FF_operator_sha256'],
           parent_protocol_sha256=sha(old.HERE/'protocol.json'),database_audit_sha256=sha(old.HERE/'cache_inventory.json'),
           selected_before_new_labels=True,scope='DEVELOPMENT causal mechanism probe, not final generalization or B15',
           execution='Identical original FF/safety/horizon/evaluator; only base controller changes',
           unknown='Preserve numerical unknown; never rerun to overwrite',bootstrap_unit='physical state')
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Endpoint matched FF response root falsification',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(max_cpu=24))))
        for scene,controls in profiles.items():
            for name,ctl in controls.items():
                v=ctl['payload'];previous=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert previous is None or previous[0]==canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    cases=[old.case(p,s,c,j,k) for s in states for c in [*FAMILIES,'flow'] for j in p['eta_indices'] for k in p['seeds']]
    cases.sort(key=lambda r:(r['state_index'],r['controller_name'],r['eta_index'],r['seed'],r['scene']))
    freeze(HERE/'cases.json',cases);groups=defaultdict(list);missing=[];count=Counter()
    with connect(True) as db:
        for r in cases:
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if a is not None:
                assert not a['conflict_quarantined']
                assert all(a[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                count['cached_unknown' if a['numerical_failure'] else 'exact_reuse']+=1
            else:missing.append(r)
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    count['genuinely_missing']=len(missing)
    write(HERE/'missing_cases.json',missing)
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    write(HERE/'exact_identity_preflight.json',dict(count,cases_sha256=sha(HERE/'cases.json')))
    print(json.dumps(dict(count)),flush=True)

def audit():
    import jax
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.contact_features import measure
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json');rows=[];vectors={}
    for scene in sorted(p['controllers']):
        base=native_runtime(scene)
        jax.config.update('jax_enable_x64',p['controllers'][scene]['flow']['payload']['jax_enable_x64'])
        for s in [v for v in p['states'] if v['scenario']==scene][:2]:
            descriptors=[];fields=[];actions=[]
            for family in FAMILIES:
                rt=EndpointMatchedField(base,family)
                nominal=controller_context(rt,s);contact,stats=measure(rt,s)
                descriptors.append(np.concatenate([nominal,contact],-1))
                # Expose the generic controller-query interface; the legacy
                # feature helper's scalar-relaxation assertion applies only
                # to its original isotropic family, not these matrix fields.
                class QueryInterface:
                    def __getattr__(self,key):return getattr(rt,key)
                fields.append(field_context(QueryInterface(),s))
                env=rt.make_env(s);key=key_for(s,0,0)
                _,u,path=rt.nominal(env,key,True)
                np.testing.assert_allclose(path[-1],u,atol=2e-12,rtol=2e-12)
                # Static field derivative fully distinguishes these matrices.
                response=rt.unprojected_response()
                per=[]
                for j in p['eta_indices']:
                    try:a,_,_=rt.action(env,key,np.asarray(p['eta'][j]),'field');per.append(a)
                    except Exception:per.append(np.full_like(env.positions,np.nan))
                actions.append(per)
                vectors[f"{s['state_uid']}_{family}_response"]=response
            desc=np.asarray(descriptors);field=np.asarray(fields);a=np.asarray(actions)
            np.testing.assert_allclose(desc,desc[:1].repeat(len(FAMILIES),axis=0),rtol=0,atol=1e-7)
            rows.append(dict(scene=scene,state_uid=s['state_uid'],nominal_contact_max_difference=float(abs(desc-desc[0]).max()),
                static_field_max_difference=float(abs(field-field[0]).max()),
                FF_first_action_max_difference=float(np.nanmax(abs(a-a[:1]))),
                physical_rollout_steps=0))
            vectors[s['state_uid']+'_descriptor']=desc;vectors[s['state_uid']+'_field']=field;vectors[s['state_uid']+'_actions']=a
    np.savez_compressed(HERE/'static_operator_audit.npz',**vectors)
    write(HERE/'static_operator_audit.json',dict(rows=rows,all_endpoint_alias_verified=True,
        code_sha256=sha(HERE/'runtime.py'),labels_used=False,physical_rollout_steps=0,
        scope='Primary nominal14+contact18 input aliases these FF programs exactly. Q disagreement still requires independent rollout evidence.'))
    print(json.dumps(rows),flush=True)

def worker(index,workers):
    import jax
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json');assert p['runtime_sha256']==sha(HERE/'runtime.py')
    assert p['FF_operator_sha256']==sha(FIELD/'poc/field_controller.py')
    proof=read(HERE/'static_operator_audit.json');assert proof['all_endpoint_alias_verified'] and proof['code_sha256']==p['runtime_sha256']
    pre=read(HERE/'cache_preflight.json')['summary'];assert pre['ambiguous']==0
    exact=read(HERE/'exact_identity_preflight.json');assert pre['genuinely_missing']==exact['genuinely_missing']+exact.get('cached_unknown',0)
    cases=read(HERE/'missing_cases.json');cases=cases[index::workers]
    states={s['state_uid']:s for s in p['states']};bases={};runtimes={};buffer=[];started=time.monotonic()
    for r in cases:
        dest=HERE/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():buffer.append(read(dest));continue
        s=states[r['state_uid']];scene=r['scene'];name=r['controller_name']
        if scene not in bases:bases[scene]=native_runtime(scene)
        if (scene,name) not in runtimes:runtimes[scene,name]=bases[scene] if name=='flow' else EndpointMatchedField(bases[scene],name)
        jax.config.update('jax_enable_x64',bool(p['controllers'][scene][name]['payload']['jax_enable_x64']))
        rt=runtimes[scene,name];env=rt.make_env(s);term='timeout';err=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for t in range(rt.config.max_steps):
            try:
                a,_,n=rt.action(env,key_for(s,r['seed'],t),np.asarray(r['eta']),'field');calls+=n;steps+=1
                if first is None:first=np.asarray(a).tolist()
                _,_,done,info=env.step(a);term=info['termination']
                if done:break
            except Exception as ex:err=dict(step=t,type=type(ex).__name__,message=str(ex)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','obstacle_collision','outer_collision','agent_collision'])
        row=dict(schema='controller_response_root_seed_v1',experiment_uid=EXPERIMENT,protocol_sha256=sha(HERE/'protocol.json'),case=r,
            **{k:r[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']},
            success=int(summary['collision_free_success'] and term=='success' and err is None),deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(err is not None),
            episode_length=steps,outcome=term,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=err,summary=summary,
            execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=sha(__file__),host_backend_sha256=sha(old.HERE/'numpy_backend.py'))
        assert not(row['success'] and (row['collision'] or row['numerical_failure']))
        write(dest,row);buffer.append(row)
        if len(buffer)>=4:append_journal(buffer,EXPERIMENT,f'root_{index}');buffer=[]
    if buffer:append_journal(buffer,EXPERIMENT,f'root_{index}')
    write(HERE/f'worker{index}.json',dict(cases=len(cases),seconds=time.monotonic()-started,completed=True))

def merge():
    old.HERE=HERE;old.EXPERIMENT=EXPERIMENT;old.merge()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','audit','worker','merge']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=24);args=ap.parse_args()
    if args.action=='worker':worker(args.index,args.workers)
    else:globals()[args.action]()
