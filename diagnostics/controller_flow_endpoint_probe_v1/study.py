"""Small crossed DATA-QUALITY probe; no model fitting or final claim."""
import argparse,copy,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import HERE as FOUR,read,write,freeze,sha
from diagnostics.controller_response_families_v1 import study as parent
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,FIELD,controller_context
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from .runtime import FlowEndpointMatchedField,FAMILIES

HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_controller_flow_endpoint_data_probe_v1'


def prepare():
    source=read(FOUR/'protocol.json');states=[copy.deepcopy(s) for s in source['states'] if s['study_split']=='train' and s['study_index']<8]
    profiles={}
    for sc in sorted({s['scenario'] for s in states}):
        profiles[sc]={}
        for family in FAMILIES:
            profiles[sc]['goal_'+family]=copy.deepcopy(source['controllers'][sc][family])
            payload=copy.deepcopy(source['controllers'][sc]['flow']['payload'])
            payload.update(base_controller='flow_endpoint_matched_affine_ff_family_v1',response_family=family,
                controller_source_sha256=sha(HERE/'runtime.py'),parent_matrix_source_sha256=sha(parent.HERE/'runtime.py'),
                parent_runtime_sha256=sha(old.HERE/'runtime.py'),numpy_backend_sha256=sha(old.HERE/'numpy_backend.py'),
                execution_callable=str(HERE/'runtime.py'),execution_callable_sha256=sha(HERE/'runtime.py'),
                target_provider='Exact frozen native Flow nominal at each current state/key; no physics step',
                conditioning='true-t0; same native Flow nominal endpoint across five families; different affine FF response paths')
            payload.pop('legacy_controller_uid_verified_by_original_function',None)
            profiles[sc]['flow_'+family]=dict(controller_uid=uid('ctl',payload),payload=payload)
    p=dict(schema='flow_endpoint_data_quality_probe_v1',states=states,controllers=profiles,eta=source['eta'],eta_indices=[0,5,10,15],seeds=[0,1],
        environments=source['environments'],runtime_sha256=sha(HERE/'runtime.py'),FF_operator_sha256=source['FF_operator_sha256'],
        parent_protocol_sha256=sha(FOUR/'protocol.json'),database_audit_sha256=sha(old.HERE/'cache_inventory.json'),
        source_code_hashes={str(f):sha(f) for f in [HERE/'runtime.py',parent.HERE/'runtime.py',old.HERE/'runtime.py',old.HERE/'numpy_backend.py']},
        state_selection='First8 previously outcome-blind TRAIN indices per scene; all four scenes retained. No fresh final states or outcome-based hard-state sampling.',
        purpose='Test whether simple goal-reference controllers erase useful state/eta supervision. Hold all five affine response operators, original physical states, eta and seeds fixed; substitute the frozen competent Flow nominal plan.',
        scope='DEVELOPMENT ground-truth data-quality pilot, not B15/K16 selection or generalization confirmation.',
        cache='All1280 goal-reference outcomes must be reused. Only genuinely missing Flow-endpoint-controller outcomes generated; never reuse nativeFlow outcomes as labels for a different vector field.',
        additional_rollouts_reason='New affine controllers preserve the frozen Flow endpoint but alter its vector field; no compatible labels for these exact programs in the audited databases.',
        expected_new_outcomes=1280,expected_reuse_outcomes=1280,
        unchanged=['Flow weights','physical initial states','FF operator','safety projection','eta pool','evaluator','horizon','seed semantics'],
        decision='Finish every scene. Assess independent-seed controller/state/eta signal and failure modes before any training expansion; no automatic large follow-on run.',
        numerical='Preserve all unknown outcomes, no rerun/overwrite',bootstrap_unit='physical state',
        literature=['https://arxiv.org/abs/2204.02877','https://proceedings.mlr.press/v119/raileanu20a.html'])
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Flow-endpoint crossed mechanism data probe',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap_when_idle=24))))
        for sc,cs in profiles.items():
            for name,ctl in cs.items():
                v=ctl['payload'];previous=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert previous is None or previous[0]==canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    cases=[old.case(p,s,c,j,k) for s in states for c in profiles[s['scenario']] for j in p['eta_indices'] for k in p['seeds']]
    cases.sort(key=lambda r:(r['state_index'],r['controller_name'],r['eta_index'],r['seed'],r['scene']))
    freeze(HERE/'cases.json',cases);missing=[];counts=Counter();groups=defaultdict(list)
    with connect(True) as db:
        for r in cases:
            a=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if a is None:missing.append(r)
            else:
                assert not a['conflict_quarantined'] and all(a[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if a['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    assert all(r['controller_name'].startswith('flow_') for r in missing),'Goal-reference cached evidence unexpectedly absent'
    counts['genuinely_missing']=len(missing);write(HERE/'missing_cases.json',missing)
    write(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    print(json.dumps(dict(counts)),flush=True)


def audit():
    import jax
    p=read(HERE/'protocol.json');rows=[]
    for sc in sorted(p['controllers']):
        s=next(s for s in p['states'] if s['scenario']==sc);base=native_runtime(sc);jax.config.update('jax_enable_x64',False)
        env=base.make_env(s);nom=[];actions=[]
        for family in FAMILIES:
            rt=FlowEndpointMatchedField(base,family)
            for seed in (0,1):
                key=jax.random.PRNGKey(2026101210+seed);expected=base.nominal(env,key)[1];actual=rt.nominal(env,key)[1]
                np.testing.assert_array_equal(actual,expected)
            nom.append(controller_context(rt,s))
            a,_,_=rt.action(env,jax.random.PRNGKey(2026101211),np.asarray(p['eta'][0]),'field');actions.append(a)
        assert all(np.array_equal(nom[0],n) for n in nom[1:])
        rows.append(dict(scene=sc,nominal_input_equal=True,first_FF_action_max_difference=float(np.max(np.abs(np.asarray(actions)-actions[0])))))
    write(HERE/'static_operator_audit.json',dict(all_endpoint_alias_verified=True,code_sha256=p['runtime_sha256'],physical_steps=0,rows=rows))
    print(json.dumps(rows),flush=True)


def worker(index,workers):
    p=read(HERE/'protocol.json')
    assert all(sha(Path(f))==h for f,h in p['source_code_hashes'].items())
    parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT
    parent.EndpointMatchedField=lambda base,name:FlowEndpointMatchedField(base,name.removeprefix('flow_'))
    parent.worker(index,workers)

def merge():
    parent.HERE=HERE;parent.EXPERIMENT=EXPERIMENT;parent.merge()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit','worker','merge']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=64);a=p.parse_args()
    if a.stage=='worker':worker(a.index,a.workers)
    else:globals()[a.stage]()
