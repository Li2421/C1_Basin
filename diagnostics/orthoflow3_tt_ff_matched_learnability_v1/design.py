"""Outcome-blind matched state/candidate design; no task rollout execution."""
from __future__ import annotations
import argparse
import collections
import dataclasses
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parent
MAIN=ROOT.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
FP=FIELD/'field_pipeline_v1'
sys.path.insert(0,str(MAIN))
from shared_rollout_db.src.rollout_db import canonical,uid,eta_identity,connect,transaction
sys.path.insert(0,str(FIELD))
import numpy as np
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SCENES=('toy_give_way','ring_exchange')
COUNTS={'train':32,'validation':8,'test':16}
FUTURE_ROOT=2026100403
EXPERIMENT='exp_orthoflow3_tt_ff_matched_learnability_v1'


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')


def freeze(p,x):
    if Path(p).exists():assert read(p)==x, ('Frozen artifact differs',str(p))
    else:write(p,x)


def cases(protocol,split=None):
    for s in protocol['states']:
        if split is not None and s['metadata']['split']!=split:continue
        for j,e in enumerate(protocol['eta']):
            for chain in ('TT','FF'):
                n=4 if s['metadata']['split']=='train' else 16
                for seed in range(n):
                    sk=canonical(dict(future_index=seed,future_root=FUTURE_ROOT,rng_namespace=s['rng_namespace']))
                    yield dict(state_uid=s['state_uid'],eta_uid=eta_identity(e)[0],controller_uid=protocol['controllers'][s['scenario']][chain]['controller_uid'],
                        seed_key=sk,seed=seed,chain=chain,scene=s['scenario'],source_alias=s['uid'],eta_index=j,eta=e,split=s['metadata']['split'])


def main():
    assert not (ROOT/'protocol.json').exists(),'Design already frozen'
    from validation_stage2.run_rollouts import make_runtime
    from field_pipeline_v1.external_toy_runner import controller_uid as toy_ff_uid
    from field_pipeline_v1.ring_t0_source import controller_uid as ring_ff_uid
    from ring_exchange.environment import sample_instance,LocalFrameConfig
    import jax
    oldfamily=read(FIELD/'family_study_v2/protocol.json')
    for rel,expected in oldfamily['controller_code_hashes'].items():assert sha(FIELD/rel)==expected,(rel,'Frozen operator changed')
    eta=oldfamily['eta'][:16];assert len(eta)==16
    screens={'toy_give_way':read(FP/'toy_source_screen.json'),'ring_exchange':read(FP/'ring_t0_screen.json')}
    for sc,m in screens.items():assert m['etas']==eta
    cfg_ring=read(FIELD/'assets/diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json')['scenario_config']
    states=[];provenance=[];profiles={};environments={}
    for scene_index,sc in enumerate(SCENES):
        rt=make_runtime(sc);env=rt.make_env(screens[sc]['states'][0])
        config=dataclasses.asdict(rt.config)
        sf=FIELD/('source/single_integrator/environment.py' if sc=='toy_give_way' else 'source/ring_exchange/environment.py')
        envfp=dict(config=config,environment_code_sha256=sha(sf),reset='native true-t0 reset, empty termination monitor')
        sid=uid('scenario',envfp);environments[sc]=dict(scenario_uid=sid,config=envfp,code_config_fingerprint=uid('env',envfp))
        code={p:sha(FIELD/p) for p in ('poc/field_controller.py','validation_v2/adapters.py','validation_stage2/factorial_controller.py','source/shared_control/basis_families.py')}
        common=dict(scenario_uid=sid,flow_checkpoint_sha256=sha(rt.checkpoint),flow_checkpoint_path=str(rt.checkpoint),orthoflow3_sha256=code['source/shared_control/basis_families.py'],
            safety_projection_sha256=sha(FIELD/('source/single_integrator/cbf.py' if sc=='toy_give_way' else 'source/shared_control/hard_projection.py')),
            cbf=rt.cbf.to_dict(),environment_sha256=uid('env',envfp),horizon=int(rt.config.max_steps),dt=float(rt.config.dt),
            success_semantics='native collision_free_success AND termination=success AND no numerical failure',
            conditioning='true_t0;native Flow resampled each step;no committed first action;unchanged physical eta basis recomputed each step',
            rng=dict(future_root=FUTURE_ROOT,key='fold_in(fold_in(fold_in(PRNGKey(root),SHA256(source_alias)[:8]),future_index),physical_step)',
                     seed_key_includes_namespace=True),jax_enable_x64=bool(jax.config.jax_enable_x64),pseudo_steps=10,
            action_code_hashes=code,agent_slot_semantics='unchanged native checkpoint/runtime slots; unified encoder masks only')
        ffid=toy_ff_uid() if sc=='toy_give_way' else ring_ff_uid()
        ff={**common,'chain':'FF','execution_callable':str(FP/('external_toy_runner.py' if sc=='toy_give_way' else 'ring_t0_source.py')),
            'execution_callable_sha256':sha(FP/('external_toy_runner.py' if sc=='toy_give_way' else 'ring_t0_source.py')),'legacy_controller_uid_verified_by_original_function':ffid}
        tt={**common,'chain':'TT','execution_callable':'frozen Runtime.action(mode=old);same field-study native RNG/reset contract','study_runtime_semantics_version':'matched_tt_v1',
            'execution_callable_sha256':sha(ROOT/'execution.py')}
        profiles[sc]={'FF':dict(controller_uid=ffid,payload=ff),'TT':dict(controller_uid=uid('ctl',tt),payload=tt)}
        oldstates=screens[sc]['states']
        for split_index,(split,total) in enumerate(COUNTS.items()):
            existing=[s for s in oldstates if s['metadata']['split']==split]
            assert len(existing)<=total
            chosen=[json.loads(canonical(s)) for s in existing]
            for j in range(total-len(chosen)):
                seed=945510000+scene_index*100000+split_index*10000+j
                if sc=='toy_give_way':
                    rng=np.random.default_rng(seed);x=rng.uniform(.55,1.05,2);y=rng.uniform(-.025,.025,2)
                    pos=np.array([[-x[0],y[0]],[x[1],y[1]]],np.float32).astype(float).tolist()
                    physical=dict(positions=pos,velocities=[[0.,0.],[0.,0.]],goals=[[1.09,0.],[-1.09,0.]],timestep=0)
                else:
                    ins=sample_instance(seed,'development',LocalFrameConfig(**cfg_ring))
                    physical=dict(positions=ins.positions.tolist(),velocities=ins.velocities.tolist(),goals=ins.goals.tolist(),timestep=0)
                chosen.append(dict(uid=f'tt_ff_v1_{sc}_{split}_{seed}',scenario=sc,physical=physical,
                    metadata=dict(split=split,source_group=f'tt_ff_v1:{sc}:{seed}',draw_seed=seed,outcome_blind=True,
                        source='fresh WIDE-IC uniform' if sc=='toy_give_way' else 'fresh native development IC')))
            for s in chosen:
                p=s['physical'];assert p['timestep']==0
                env=rt.make_env(s)
                for a in ('positions','velocities','goals'):np.testing.assert_allclose(getattr(env,a),p[a],atol=1e-12,rtol=0)
                # Distinct initial states are retained regardless of future Q.
                s['scenario_uid']=sid;s['content_hash']=uid('content',dict(physical=p,reset='empty monitor true-t0'))
                s['state_uid']=uid('state',dict(scenario=sid,content=s['content_hash']))
                s['rng_namespace']=int(hashlib.sha256(s['uid'].encode()).hexdigest()[:8],16)
                s['family']=s['metadata'].get('source_group',s['uid'])
                assert s['metadata']['split']==split
                states.append(s)
    assert len(states)==112 and len({s['state_uid'] for s in states})==112 and len({s['family'] for s in states})==112
    def physical_key(s):
        return (s['scenario'],canonical({k:s['physical'][k] for k in ('positions','velocities','goals','timestep')}))
    previous=oldfamily['states']+oldfamily.get('sensitivity_states',[])
    assert not ({physical_key(s) for s in states}&{physical_key(s) for s in previous}), 'Reused previously opened physical state'
    assert len({physical_key(s) for s in states})==len(states),'Physical family overlap'
    for split in COUNTS:
        assert len({s['family'] for s in states if s['metadata']['split']==split})==2*COUNTS[split]
    frozen=dict(schema='tt_ff_matched_learnability_v1',states=states,eta=eta,controllers=profiles,environments=environments,
        plan_sha256=sha(ROOT/'plan.json'),source_screens={sc:sha(FP/('toy_source_screen.json' if sc=='toy_give_way' else 'ring_t0_screen.json')) for sc in SCENES},
        paired_seed_policy=dict(indices=list(range(16)),future_root=FUTURE_ROOT,train_prefix=4,rng_namespace='SHA256 frozen source_alias first8hex'),
        new_state_rules=read(ROOT/'plan.json')['fresh_state_rules'],source_outcomes_used_in_state_or_eta_design=False,
        previously_opened_family_v2_states_excluded=True,generator_modified=False,eta_lifting_safety_unchanged=True,
        script_sha256=sha(__file__),execution_sha256=sha(ROOT/'execution.py'),fresh_test_open_only_after_model_freeze=True)
    # No success labels have been loaded above. Freeze every future test family
    # before source collection, training, or model comparisons.
    freeze(ROOT/'protocol.json',frozen)
    allcases=list(cases(frozen));assert len(allcases)==32768
    freeze(ROOT/'cases.json',allcases)
    # Semantic registration only; outcome import/execution has a separate
    # journal/merger path and must be followed by global cache preflight.
    with connect() as con,transaction(con):
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'TT/FF matched learnability',str(ROOT),sha(ROOT/'protocol.json'),sha(__file__),canonical(dict(new_rollouts=0,paired=True))))
        for sc,e in environments.items():
            con.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',
                (e['scenario_uid'],sc,e['code_config_fingerprint'],canonical(e['config'])))
        for s in states:
            con.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(environments[s['scenario']]['config']),canonical(dict(source_alias=s['uid'],metadata=s['metadata'])), 'CONTENT_EXACT'))
            con.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXPERIMENT))
        for e in eta:
            eid,v,hx=eta_identity(e);con.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        for sc,pp in profiles.items():
            for chain,ctl in pp.items():
                p=ctl['payload']; prior=con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert prior is None or prior[0]==canonical(p), ('Existing different controller semantics',ctl['controller_uid'])
                con.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],p['scenario_uid'],p['flow_checkpoint_sha256'],p['orthoflow3_sha256'],uid('safe',dict(code=p['safety_projection_sha256'],cbf=p['cbf'])),str(p['horizon']),str(p['dt']),p['success_semantics'],p['conditioning'],canonical(p['rng']),canonical(p),'EXACT_PROFILE'))
    requests=[]
    group=collections.defaultdict(list)
    for r in allcases:group[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    for (s,e,c),ss in group.items():requests.append(dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ss))
    freeze(ROOT/'planned_rollouts.json',dict(requests=requests))
    write(ROOT/'working_state.json',dict(phase='design_frozen_cache_import_required',new_rollouts=0,requested=32768,source_states=80,test_states=32,
        counts_per_scene=COUNTS,models_not_trained=True,test_outcomes_not_opened=True))
    print(json.dumps(dict(states=len(states),source_states=80,independent_test_states=32,eta=16,requested=len(allcases),new_rollouts=0)))


if __name__=='__main__':main()
