"""Small common-program data-quality pilot; exact cache identities, no training."""
import argparse,copy,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import HERE as FOUR,read,write,freeze,sha
from diagnostics.controller_response_families_v1 import study as parent
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,FIELD,controller_context
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from .runtime import NavigationEndpointField,FAMILIES

HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_controller_shared_navigation_probe_v1'


def prepare():
    source=read(FOUR/'protocol.json')
    geometry=read(HERE/f"geometry_audit_{sha(HERE/'runtime.py')[:12]}.json")
    assert all(r['unroutable']==r['path_errors']==0 for r in geometry['rows'])
    states=[copy.deepcopy(s) for s in source['states'] if s['study_split']=='train' and s['study_index']<8]
    assert len(states)==32
    profiles={}
    for sc in sorted({s['scenario'] for s in states}):
        profiles[sc]={}
        for family in FAMILIES:
            payload=copy.deepcopy(source['controllers'][sc]['flow']['payload'])
            payload.update(base_controller='shared_visibility_navigation_affine_ff_v1',response_family=family,
                controller_source_sha256=sha(HERE/'runtime.py'),parent_matrix_source_sha256=sha(parent.HERE/'runtime.py'),
                parent_runtime_sha256=sha(old.HERE/'runtime.py'),numpy_backend_sha256=sha(old.HERE/'numpy_backend.py'),
                geometry_adapter_sha256=sha(HERE.parent/'physical_state_critic_formal_v1/physical_entities.py'),
                execution_callable=str(HERE/'runtime.py'),execution_callable_sha256=sha(HERE/'runtime.py'),
                target_provider='Same visibility-graph navigation program and parameters in all scenes, using only map/positions/goals. No native Flow query.',
                active_Flow_weights=False,physical_noise='mean0, isotropic scale=max_speed, Cartesian in every scene',
                agent_slot_semantics='Shared permutation-equivariant analytic program; native API order only serializes physical agent records',
                conditioning='true-t0; shared map-aware nominal endpoint across three affine FF operators')
            # The adapter still loads the native runtime but its learned weights
            # are never called; retain the inherited hash explicitly as provenance.
            payload['inactive_runtime_checkpoint_sha256']=payload.pop('flow_checkpoint_sha256')
            payload['flow_checkpoint_sha256']='NONE_SHARED_ANALYTIC_PROGRAM'
            payload.pop('legacy_controller_uid_verified_by_original_function',None)
            profiles[sc]['navigation_'+family]=dict(controller_uid=uid('ctl',payload),payload=payload)
    code=[HERE/'runtime.py',parent.HERE/'runtime.py',old.HERE/'runtime.py',old.HERE/'numpy_backend.py',
          HERE.parent/'physical_state_critic_formal_v1/physical_entities.py']
    p=dict(schema='shared_program_navigation_data_quality_v1',states=states,controllers=profiles,eta=source['eta'],
        eta_indices=[0,5,10,15],seeds=[0,1],environments=source['environments'],
        runtime_sha256=sha(HERE/'runtime.py'),FF_operator_sha256=source['FF_operator_sha256'],
        parent_protocol_sha256=sha(FOUR/'protocol.json'),source_code_hashes={str(f):sha(f) for f in code},
        state_selection='Same first8 outcome-blind TRAIN indices per scene as prior Flow-endpoint pilot. No newstates or outcome-selected subsets.',
        purpose='Separate simultaneous map/Flow-program shift from map transfer, using a shared task-aware program. Pilot only establishes whether this crossed data has useful Q/state/eta signal.',
        prior_evidence_audit='shared_controller_cache_audit.json, existing scene-specific experts, original analytic goal panel, Flow-endpoint pilot all audited; no exact navigation-program labels exist.',
        scope='Opened DEVELOPMENT data-quality pilot, K4/Q2, not B15 or final generalization.',
        expected_new_outcomes=768,expected_existing_goal_and_Flow_comparison_outcomes=1536,
        additional_rollouts_reason='Exact new shared controller program, removes learned scene-specific policy and input normalization confound; oldlabels cannot be relabeled under different execution.',
        decision='Finishall4scenes; compare cached matched goal/Flow panels and independent halves before any training expansion. Noautomaticlargefollow-onrun.',
        unchanged=['physicalstates','eta16pool withsame4diagnosticindices','FFimplementation','safety','horizon','collision/deadlock/success evaluator','seedkeys'],
        numerical='Preserve unknown outcomes; no retries to turnunknown into success',bootstrap_unit='physicalstate',
        literature=['https://arxiv.org/abs/2204.02877','https://proceedings.mlr.press/v119/raileanu20a.html'])
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Shared navigation program FF diagnostic',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=24))))
        for sc,controls in profiles.items():
            for name,ctl in controls.items():
                v=ctl['payload']
                previous=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert previous is None or previous[0]==canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    cases=[old.case(p,s,c,j,k) for s in states for c in profiles[s['scenario']] for j in p['eta_indices'] for k in p['seeds']]
    cases.sort(key=lambda r:(r['state_index'],r['controller_name'],r['eta_index'],r['seed'],r['scene']))
    freeze(HERE/'cases.json',cases);missing=[];counts=Counter();groups=defaultdict(list)
    with connect(True) as db:
        for r in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if row is None:missing.append(r)
            else:
                assert not row['conflict_quarantined'] and all(row[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if row['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing)
    write(HERE/'missing_cases.json',missing);write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    print(dict(counts),flush=True)


def audit():
    import jax
    p=read(HERE/'protocol.json');rows=[]
    for sc in sorted(p['controllers']):
        s=next(s for s in p['states'] if s['scenario']==sc);base=native_runtime(sc)
        jax.config.update('jax_enable_x64',bool(next(iter(p['controllers'][sc].values()))['payload']['jax_enable_x64']))
        def forbidden(*args,**kw):raise RuntimeError('Native Flow or simulator step called by common-program input audit')
        base.native_fn=forbidden;base.trace_fn=forbidden;base.velocity_fn=forbidden;base.env_cls.step=forbidden
        endpoint=[];actions=[]
        for family in FAMILIES:
            rt=NavigationEndpointField(base,family);env=rt.make_env(s)
            endpoint.append(rt.nominal(env,jax.random.PRNGKey(2026101008))[1])
            a,_,_=rt.action(env,jax.random.PRNGKey(2026101009),np.asarray(p['eta'][0]),'field');actions.append(a)
        assert all(np.array_equal(endpoint[0],a) for a in endpoint[1:])
        rows.append(dict(scene=sc,native_Flow_calls=0,physical_steps=0,identical_nominal_endpoint=True,
            max_operator_action_difference=float(np.max(abs(np.array(actions)-actions[0])))))
    write(HERE/'static_operator_audit.json',dict(all_endpoint_alias_verified=True,code_sha256=p['runtime_sha256'],rows=rows,physical_steps=0))
    print(rows,flush=True)


def worker(index,workers):
    p=read(HERE/'protocol.json');assert all(sha(Path(f))==h for f,h in p['source_code_hashes'].items())
    parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT
    parent.EndpointMatchedField=lambda base,name:NavigationEndpointField(base,name.removeprefix('navigation_'))
    parent.worker(index,workers)


def merge():
    parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT;parent.merge()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit','worker','merge']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=48);a=p.parse_args()
    if a.stage=='worker':worker(a.index,a.workers)
    else:globals()[a.stage]()
