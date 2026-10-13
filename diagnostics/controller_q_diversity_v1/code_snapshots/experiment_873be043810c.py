"""Frozen, cache-first FF controller diversity experiment orchestration."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,ROOT as DBROOT
from shared_rollout_db.src.cache_writer import append_journal

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[1]
EXPERIMENT='exp_controller_q_diversity_ff_20261009_v1'
SCENES=('toy_give_way','ring_exchange')
FLAGS=('success','deadlock','timeout','collision','numerical_failure')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol():
    p=read(HERE/'protocol.json')
    if (HERE/'safety_profiles.json').exists():
        extra=read(HERE/'safety_profiles.json')
        for scene,profiles in extra.items():p['controllers'][scene].update(profiles)
    return p


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def freeze(p,x):
    if Path(p).exists():assert read(p)==x,('Immutable artifact',str(p))
    else:write(p,x)


def design():
    assert (HERE/'cache_inventory.json').exists(),'Audit all databases first'
    if (HERE/'protocol.json').exists():return
    from .runtime import native_runtime,FIELD
    from scipy.stats import qmc
    previous=read(MAIN/'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/protocol.json')
    expanded=read(MAIN/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    raw=qmc.LatinHypercube(6,seed=2026100911).random(8)
    lo=np.array([.55,.2,-.85,0.,0.,1.5]);hi=np.array([1.1,.9,.85,.6,.8,4.5])
    parameters=(lo+raw*(hi-lo)).tolist()
    roles=['train']*4+['validation']*2+['held_controller']*2
    profiles={};states=[]
    for scene_id,scene in enumerate(SCENES):
        native=previous['controllers'][scene]['FF']
        profiles[scene]={'flow':copy.deepcopy(native)}
        for ci,values in enumerate(parameters):
            payload=copy.deepcopy(native['payload'])
            payload.update(base_controller='analytic_relaxation_goal_avoidance_v1',
                controller_parameters=values,controller_source_sha256=sha(HERE/'runtime.py'),
                parent_flow_noise_normalization=payload['flow_checkpoint_sha256'],
                conditioning='native true-t0; analytic base vector field; frozen FF operator and eta basis',
                execution_callable=str(HERE/'runtime.py'),execution_callable_sha256=sha(HERE/'runtime.py'))
            payload.pop('legacy_controller_uid_verified_by_original_function',None)
            profiles[scene][f'analytic{ci}']=dict(controller_uid=uid('ctl',payload),payload=payload,role=roles[ci])
        old=[s for s in previous['states'] if s['scenario']==scene and s['metadata']['split']=='train']
        if scene=='ring_exchange':
            old=[s for s in expanded['states'] if s['metadata']['split']=='train']
        for split,count in [('train',128),('validation',16),('final',32)]:
            existing=old[:count] if split=='train' else []
            chosen=copy.deepcopy(existing)
            for j in range(len(chosen),count):
                seed=1026091000+scene_id*10000+{'train':0,'validation':1000,'final':2000}[split]+j
                if scene=='toy_give_way':
                    rng=np.random.default_rng(seed);x=rng.uniform(.55,1.05,2);y=rng.uniform(-.025,.025,2)
                    pos=np.array([[-x[0],y[0]],[x[1],y[1]]],np.float32).astype(float)
                    physical=dict(positions=pos.tolist(),velocities=np.zeros((2,2)).tolist(),goals=[[1.09,0.],[-1.09,0.]],timestep=0)
                else:
                    from ring_exchange.environment import sample_instance,LocalFrameConfig
                    cfg=previous['environments'][scene]['config']['config']
                    ins=sample_instance(seed,'development',LocalFrameConfig(**cfg))
                    physical=dict(positions=ins.positions.tolist(),velocities=ins.velocities.tolist(),goals=ins.goals.tolist(),timestep=0)
                alias=f'controller_q_diversity_v1_{scene}_{split}_{seed}'
                content=uid('content',dict(physical=physical,reset='empty monitor true-t0'))
                sid=previous['environments'][scene]['scenario_uid']
                chosen.append(dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),physical=physical,
                    scenario=scene,scenario_uid=sid,content_hash=content,
                    rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),family=alias,
                    metadata=dict(split=split,source_group=alias,draw_seed=seed,outcome_blind=True)))
            for i,s in enumerate(chosen):
                s['study_split']=split;s['study_index']=i
                states.append(s)
    p=dict(schema='controller_q_diversity_ff_v1',created_before_new_labels=True,
        cache_inventory_sha256=sha(HERE/'cache_inventory.json'),states=states,controllers=profiles,
        environments=previous['environments'],eta=previous['eta'],eta_source_sha256=sha(MAIN/'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/protocol.json'),
        controller_parameter_names=['speed','goal_braking_distance','passing_turn','radial_repulsion','velocity_damping','field_relaxation'],
        controller_sampling='Outcome-blind Latin hypercube, seed2026100911; same controllers in both scenes',
        controller_roles=roles,physical_representation='existing physical-only masked entity adapter',
        physical_representation_sha256=sha(MAIN/'diagnostics/physical_state_critic_formal_v1/physical_entities.py'),
        context='Seven static policy-function queries per physical state, per-agent goal-relative responses, four fixed independent probe seeds; no simulator steps or eta/outcome inputs',
        bootstrap_unit='physical state; controller-generalization uncertainty separately reported across held controllers',
        split_policy='Physical families disjoint across TRAIN/VAL/FINAL; controller IDs4/5 only source-validation;6/7 label-held-out',
        train_trials=4,validation_trials=8,final_trials=16,final_discovery_indices=list(range(8)),final_evaluation_indices=list(range(8,16)),
        runtime_sha256=sha(HERE/'runtime.py'),FF_operator_sha256=sha(FIELD/'poc/field_controller.py'),
        primary='Q estimation and matched-controller effect prediction on held physical families and held controller parameters',
        scope='Initial FF two-scene controller-learning experiment; final confirmation only after source-only fitting. No claim of four-scene LOSO.',
        models=['eta_only','state_only','context_only','state_context'],seeds=[17,23,41],
        checkpoint_rule='Source validation probability MSE plus 0.5 matched-controller contrast MSE; NLL and eta ranking tracked independently',
        resource_cpu_cap=12,deadline_local='2026-10-09T09:00:00+08:00',
        stop_rule='Preserve all results and stop new execution at deadline; incomplete panels never called confirmation',
        pilot_rule='First two TRAIN states, analytic controllers0/1, first4fixedeta, seeds0/1; pilot labels remainTRAIN. Pilot only measures cost/validity before committing full allocation.')
    assert len({s['state_uid'] for s in states})==len(states)
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXPERIMENT,'Physical-state + controller Q learning, analytic FF diversity',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=12))))
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments'][s['scenario']]['config']),canonical(dict(source_alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXPERIMENT))
        for eta in p['eta']:
            eid,v,hx=eta_identity(eta);db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        for scene,controls in profiles.items():
            for name,ctl in controls.items():
                v=ctl['payload'];prior=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
                assert prior is None or prior[0]==canonical(v),name
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    print(dict(design_frozen=True,physical_states=len(states),controllers_per_scene=9),flush=True)


def case(p,s,name,j,seed):
    eta=[0.,0.,0.] if j==-1 else p['eta'][j];ctl=p['controllers'][s['scenario']][name]['controller_uid']
    sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']))
    key=dict(state=s['state_uid'],eta=eta_identity(eta)[0],controller=ctl,seed=sk)
    return dict(state_uid=key['state'],eta_uid=key['eta'],controller_uid=ctl,seed_key=sk,rollout_uid=uid('roll',key),
        scene=s['scenario'],controller_name=name,split=s['study_split'],state_index=s['study_index'],eta_index=j,eta=eta,seed=seed)


def preflight(stage,scene=None):
    p=protocol();rows=[]
    assert sha(HERE/'runtime.py')==p['runtime_sha256']
    for s in p['states']:
        if scene and s['scenario']!=scene:continue
        if stage=='pilot':
            if s['study_split']!='train' or s['study_index']>=2:continue
            controls=['analytic0','analytic1'];etas=range(4);seeds=range(2)
        elif stage=='source':
            if s['study_split']=='final':continue
            controls=[f'analytic{i}' for i in range(4 if s['study_split']=='train' else 6)]+['flow']
            etas=range(16);seeds=range(4 if s['study_split']=='train' else 8)
        elif stage=='source_budget':
            budget=read(HERE/'execution_budget.json')
            if s['study_split']=='final':continue
            controls=[f'analytic{i}' for i in range(4 if s['study_split']=='train' else 6)]+['flow']
            etas=range(16);seeds=range(budget['train_seeds'] if s['study_split']=='train' else budget['validation_seeds'])
        elif stage=='confirmation_budget':
            budget=read(HERE/'execution_budget.json')
            assert (HERE/'models_frozen.json').exists()
            rule=read(HERE/'confirmation_allocation.json')
            if s['study_split']!='final' or s['study_index']>=rule['states_per_scene']:continue
            controls=budget['confirmation_controllers'];etas=range(16);seeds=range(16)
        elif stage=='safety_confirmation':
            rule=read(HERE/'confirmation_allocation.json')
            assert (HERE/'models_frozen.json').exists()
            if s['study_split']!='final' or s['study_index']>=rule['states_per_scene']:continue
            controls=['analytic6_safety','analytic7_safety'];etas=[-1];seeds=range(16)
        elif stage=='final':
            assert (HERE/'models_frozen.json').exists()
            if s['study_split']!='final':continue
            controls=['analytic0','analytic1','analytic6','analytic7','flow'];etas=range(16);seeds=range(16)
        else:raise ValueError(stage)
        for c in controls:
            for j in etas:
                for seed in seeds:rows.append(case(p,s,c,j,seed))
    # Interleave scenes so a cost interruption cannot leave one scene omitted.
    rows.sort(key=lambda r:(r['split'],r['state_index'],r['eta_index'],r['controller_name'],r['seed'],r['scene']))
    folder=HERE/'batches'/(stage+('_'+scene if scene else ''))
    counts=Counter(requested=len(rows));missing=[];groups=defaultdict(list)
    with connect(True) as db:
        for r in rows:
            old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if old:
                assert not old['conflict_quarantined']
                assert all(old[k]==r[k] for k in ['state_uid','eta_uid','controller_uid','seed_key'])
                counts['numerical_unknown_not_rerun' if old['numerical_failure'] else 'exact_reuse']+=1
            else:missing.append(r)
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing)
    freeze(folder/'cases.json',rows)
    write(folder/'missing_cases.json',missing)
    freeze(folder/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ss) for (s,e,c),ss in groups.items()]))
    write(folder/'exact_identity_preflight.json',dict(counts,protocol_sha256=sha(HERE/'protocol.json'),cases_sha256=sha(folder/'cases.json')))
    print(str(folder),dict(counts),flush=True)


def worker(batch,index,workers):
    from datetime import datetime
    from .runtime import native_runtime,AnalyticField,FIELD
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=protocol();assert sha(HERE/'runtime.py')==p['runtime_sha256']
    assert sha(FIELD/'poc/field_controller.py')==p['FF_operator_sha256']
    folder=HERE/'batches'/batch;pre=read(folder/'exact_identity_preflight.json')
    assert pre['cases_sha256']==sha(folder/'cases.json')
    standard=read(folder/'cache_preflight.json')['summary']
    assert standard['ambiguous']==0
    # The shared planner calls unresolved numerical outcomes 'missing'. They
    # are attempted observations and must not be silently rerun as new draws.
    assert standard['genuinely_missing']==pre['genuinely_missing']+pre.get('numerical_unknown_not_rerun',0)
    states={s['state_uid']:s for s in p['states']};all_rows=read(folder/'missing_cases.json')
    # Short contiguous shards contain both scenes, avoiding fixed parity shards
    # with long Ring tails and idle slots. Each completed task releases its CPU.
    start=len(all_rows)*index//workers;end=len(all_rows)*(index+1)//workers
    rows=all_rows[start:end]
    bases={};runtimes={};buffer=[];new=0;elapsed=[]
    for r in rows:
        if datetime.now().astimezone()>=datetime.fromisoformat(p['deadline_local']):break
        dest=folder/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():
            # Re-journal atomically completed records after an interrupted shard;
            # the merger verifies exact duplicates rather than losing labels.
            buffer.append(read(dest))
            if len(buffer)>=8:
                append_journal(buffer,EXPERIMENT,f'{batch}_{index}');buffer=[]
            continue
        s=states[r['state_uid']];scene=r['scene'];name=r['controller_name']
        if r['split']=='final':assert (HERE/'models_frozen.json').exists()
        if scene not in bases:bases[scene]=native_runtime(scene)
        profile=p['controllers'][scene][name]
        if (scene,name) not in runtimes:
            analytic=AnalyticField
            audit=HERE/'numpy_backend_audit.json'
            if audit.exists():
                from .numpy_backend import NumpyAnalyticField
                proof=read(audit)
                assert proof['every_action_bitwise_equal'] and proof['code_sha256']==sha(HERE/'numpy_backend.py')
                analytic=NumpyAnalyticField
            runtimes[scene,name]=bases[scene] if name=='flow' else analytic(bases[scene],profile['payload']['controller_parameters'])
        # Native scene factories mutate this process-global setting. Restore it
        # for EVERY rollout after runtime construction, including cached objects.
        import jax
        expected_x64=bool(profile['payload']['jax_enable_x64'])
        jax.config.update('jax_enable_x64',expected_x64)
        assert bool(jax.config.x64_enabled)==expected_x64
        rt=runtimes[scene,name];env=rt.make_env(s);t0=time.monotonic();term='timeout';error=None;steps=0;first=None;calls=0
        for step in range(rt.config.max_steps):
            try:
                key=key_for(s,r['seed'],step)
                if name.endswith('_safety'):
                    _,flow,_=rt.nominal(env,key)
                    safe=rt.safety(env);action,_=safe.project(flow);cert=safe.assert_safe(action);n=safe.calls
                else:action,cert,n=rt.action(env,key,np.asarray(r['eta']), 'field')
                calls+=n;steps+=1
                if first is None:first=np.asarray(action).tolist()
                _,_,done,info=env.step(action);term=info['termination']
                if done:break
            except Exception as ex:
                error=dict(step=step,type=type(ex).__name__,message=str(ex)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','obstacle_collision','outer_collision','agent_collision'])
        record=dict(schema='controller_q_diversity_ff_seed_v1',experiment_uid=EXPERIMENT,protocol_sha256=sha(HERE/'protocol.json'),case=r,
            state_uid=r['state_uid'],eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],seed_key=r['seed_key'],
            success=int(summary['collision_free_success'] and term=='success' and error is None),deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),episode_length=steps,outcome=term,
            elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=error,summary=summary,
            host_backend=type(rt).__name__,host_backend_sha256=sha(HERE/'numpy_backend.py') if type(rt).__name__=='NumpyAnalyticField' else p['runtime_sha256'],
            execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=sha(__file__))
        assert not (record['success'] and (record['collision'] or record['numerical_failure']))
        write(dest,record);buffer.append(record);new+=1;elapsed.append(record['elapsed_seconds'])
        if len(buffer)>=8:
            append_journal(buffer,EXPERIMENT,f'{batch}_{index}');buffer=[]
        if new%8==0:print(json.dumps(dict(batch=batch,index=index,new=new,last_seconds=elapsed[-1])),flush=True)
    if buffer:append_journal(buffer,EXPERIMENT,f'{batch}_{index}')
    write(folder/f'worker{index}.json',dict(cases=len(rows),new=new,completed=all((folder/'raw'/(r['rollout_uid']+'.json')).exists() for r in rows),seconds=sum(elapsed)))


def merge():
    p=protocol();expected=sha(HERE/'protocol.json');stats=Counter()
    states={s['state_uid']:s for s in p['states']}
    with connect() as db:
        for path in sorted((DBROOT/'journals'/EXPERIMENT).glob('*.jsonl')):
            source=uid('source',dict(path=str(path),sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?',(source,)).fetchone():continue
            records=[json.loads(v) for v in path.read_text().splitlines()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                    (source,EXPERIMENT,str(path),sha(path),'.jsonl','SEED_EXACT',len(records)))
                added=duplicates=0
                for line,r in enumerate(records,1):
                    c=r['case'];s=states[c['state_uid']]
                    assert r['protocol_sha256']==expected and r['experiment_uid']==EXPERIMENT
                    assert c==case(p,s,c['controller_name'],c['eta_index'],c['seed'])
                    if not path.name.startswith('pilot_'):
                        assert r['execution_semantics_version']=='scene_precision_guard_v1'
                        assert r['jax_enable_x64']==p['controllers'][c['scene']][c['controller_name']]['payload']['jax_enable_x64']
                    assert not(r['success'] and (r['collision'] or r['numerical_failure']))
                    core=tuple(r[k] for k in FLAGS)
                    old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    if old:
                        assert core==tuple(old[k] for k in FLAGS) and r['episode_length']==old['episode_length']
                        duplicates+=1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*core,r['episode_length'],r['outcome'],EXPERIMENT,str(path),'EXACT_REUSE',uid('record',r)))
                        added+=1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?',(added,duplicates,source))
                stats.update(journals=1,inserted=added,duplicates=duplicates)
        db.commit()
    write(HERE/'merge_audits'/f'{time.time_ns()}.json',dict(stats));print(dict(stats),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['design','preflight','worker','merge']);ap.add_argument('--stage');ap.add_argument('--scene');ap.add_argument('--batch');ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    if a.action=='design':design()
    elif a.action=='preflight':preflight(a.stage,a.scene)
    elif a.action=='worker':worker(a.batch,a.index,a.workers)
    else:merge()
