"""Long-return extension of the cached noise-coupling diagnosis: Four TRAIN only."""
import argparse,copy,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_diversity_v1.runtime import FIELD
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.cache_writer import append_journal
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_four_scene_mechanism_v1'
EXPERIMENT='exp_controller_q_noise_coupling_four_v1'
SCENE='four_way_intersection'
FAMILIES=('isotropic','anisotropic','rotate_positive','rotate_negative')
ANCHOR='controller_q_noise_coupling_four_v1_shared_exogenous_stream'
read,write,freeze,sha=old.read,old.write,old.freeze,old.sha

def prepare():
    assert read(ROOT/'controller_noise_coupling_v1/results.json')['gate_passed']
    parent=read(BASE/'protocol.json')
    states=[s for s in parent['states'] if s['scenario']==SCENE and s['study_split']=='train']
    assert len(states)==32
    profiles={}
    for name in FAMILIES:
        original=parent['controllers'][SCENE][name]
        p=copy.deepcopy(original['payload'])
        assert p['rng']['future_root']==2026100403
        p['rng']=dict(future_root=2026100403,key='existing key_for with a single shared source_alias; fold future_index then physical_step',
            shared_source_alias=ANCHOR,seed_key_includes_namespace=True,
            coupling='same exogenous Gaussian stream across physicalstates; same withinstate C/eta coupling; unchanged per-state noise law')
        profiles[name]=dict(controller_uid=uid('ctl',p),payload=p,old_controller_uid=original['controller_uid'])
    # Audit the full shared controller inventory before planning any rollout.
    inventory=[];relevant=[]
    with connect(True) as db:
        for r in db.execute('SELECT controller_uid,config_json,rng_semantics_version FROM controller_config'):
            cfg=json.loads(r['config_json'])
            inventory.append(dict(controller_uid=r['controller_uid'],config_sha256=uid('cfg',cfg),
                base_controller=cfg.get('base_controller'),rng=cfg.get('rng')))
            if cfg.get('base_controller')=='endpoint_matched_response_family_v1':
                z=db.execute('SELECT count(*) n,count(distinct state_uid) s FROM rollout WHERE controller_uid=?',(r['controller_uid'],)).fetchone()
                relevant.append(dict(controller_uid=r['controller_uid'],rng=cfg.get('rng'),outcomes=z['n'],physicalstates=z['s']))
    p=copy.deepcopy(parent)
    p.update(schema='four_train_long_Q_noise_coupling_v1',states=states,controllers={SCENE:profiles},
        shared_source_alias=ANCHOR,train_seeds=[0,1,2,3],validation_seeds=[],eta_indices=[0,5,10,15])
    p['cause_test']=dict(
        history='Stage30 local learning causal contrast passed DB/Four/Ring state_eta on independent noise, all6panels. Triple Cinteraction not generally confirmed. This does NOTyet show long-Q repair. FullDB inventory audited here; existing20endpointprogramprofiles alluse state-alias-dependentstreams, while common static actions have NOterminal outcomes.',
        scope='Only32original naturalFour TRAINstates,4known uncoupled goal-referenceprograms,K4,Q4sharedstreams=2048terminal outcomes. Allphysicalstates/eta/Flow/safety/horizon/evaluatorunchanged;zero newphysicalstates. No VAL/TEST rollout. OriginalTRAINQ2/VALQ4 reusedforindependent-noise diagnosis.',
        scene_choice='Predeclared mechanism localization:Four has independentlyconfirmedlong-Q state/C/eta variation forknownsharedgoal programs, andlocal coupling learninggatepassed. Toyalreadylocal-positive;DB/Ringgoal-Q lackpower/arecensored. Not an all-scene confirmation or selected hard-state subset;all32naturalTRAINstatesretained.',
        intervention='Only exogenous-noise COUPLING acrossphysicalstates; per-stateGaussian driver law unchanged. Useexisting key_for, replacing only its source_alias argument by frozen sharedanchor; actualenvstateunchanged. New RNGsemantic fingerprints prevent mixing old labelidentities.',
        gate_before_training='First estimate long-Q state_eta/triple interaction labelvariance withall6two-keysubsets. ContinuepairedQfitsonlyif >=50%state_eta errorvariance reduction AND nonzero true state_eta signal supported by independent halves; otherwise prune as sufficientlong-Q repair. Unknown outcomes retained, neverrerun orimpute success.',
        prospective_training='Ifeligible:common2-key versusperstate2-keysubset,all6panelsx3initialseeds;identicalaggregatelabels/inputs/optimization/sourceVAL. Fairη-only corresponding TRAINlabels. Exclude fromBOTHtrainarms only cellswithanynumericalunknownamong4keys; reportcoverage;>5%cellcensoring blockscleancontrast ratherthan claimingrepair. No source-cellselection bysuccess.',
        selection='OriginalSOURCEVAL selection/ranking-first rule,neverNLLalone; freeze exacttrainerbeforefit. OriginalphysicalstateswithindependentoldnoiseareDEVELOPMENT,notconfirmation.',
        no_auto_expansion='No automatic other-scene rollouts/LOSO/new512data. Firstread-onlyQvariance gate,thenonlyscientificallyjustifiedfit. Finalconfirmationrequiresuntouchedphysicalstatesandnoise.',
        cache_first=True,expected_new_rollouts=2048,
        literature=read(ROOT/'controller_local_to_return_probe_v1/vector_control/noise_coupling_protocol.json')['literature'])
    hashes={str(FIELD/k):sha(FIELD/k) for k in ('poc/field_controller.py','source/shared_control/hard_projection.py',
        'source/shared_control/basis_families.py','validation_stage2/factorial_controller.py','validation_v2/adapters.py')}
    for f in (ROOT/'controller_response_families_v1/runtime.py',old.HERE/'runtime.py',old.HERE/'numpy_backend.py',
              ROOT/'orthoflow3_tt_ff_matched_learnability_v1/execution.py'):
        hashes[str(f)]=sha(f)
    for name,z in profiles.items():
        q=z['payload']
        assert q['execution_callable_sha256']==sha(ROOT/'controller_response_families_v1/runtime.py')
        assert q['numpy_backend_sha256']==sha(old.HERE/'numpy_backend.py')
        assert q['orthoflow3_sha256']==sha(FIELD/'source/shared_control/basis_families.py')
        for f,h in q['action_code_hashes'].items():assert sha(FIELD/f)==h
    p['source_code_hashes']=hashes;p['worker_code_sha256']=sha(__file__)
    freeze(HERE/'protocol.json',p)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT,'Four TRAIN long-Q noise coupling diagnostic',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cpu_cap=24,pure_Q=True,new_rollouts_max=2048))))
        for name,z in profiles.items():
            v=z['payload']
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (z['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],
                 uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),
                 v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    cases=[]
    for s in states:
        for name,z in profiles.items():
            for j in p['eta_indices']:
                for k in range(4):
                    sk=canonical(dict(future_index=k,future_root=2026100403,shared_source_alias=ANCHOR))
                    key=dict(state=s['state_uid'],eta=eta_identity(p['eta'][j])[0],controller=z['controller_uid'],seed=sk)
                    cases.append(dict(state_uid=key['state'],eta_uid=key['eta'],controller_uid=key['controller'],seed_key=sk,rollout_uid=uid('roll',key),
                        scene=SCENE,controller_name=name,split='train',state_index=s['study_index'],eta_index=j,eta=p['eta'][j],seed=k))
    assert len(cases)==2048
    counts=Counter();missing=[];groups=defaultdict(list)
    with connect(True) as db:
        for r in cases:
            z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if z is None:missing.append(r)
            else:
                assert not z['conflict_quarantined']
                assert all(z[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                counts['cached_unknown' if z['numerical_failure'] else 'exact_reuse']+=1
            groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    counts['genuinely_missing']=len(missing)
    freeze(HERE/'cases.json',cases);freeze(HERE/'missing_cases.json',missing)
    freeze(HERE/'exact_identity_preflight.json',dict(counts,cases_sha256=sha(HERE/'cases.json')))
    freeze(HERE/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]))
    freeze(HERE/'cache_history_audit.json',dict(inventory=inventory,relevant_profiles=relevant,profiles_scanned=len(inventory),
        compatible_terminal_reuse=counts.get('exact_reuse',0),missing=len(missing),physical_states_reused=32,new_physical_states=0,
        reason='No common-exogenous-stream terminal labels for theseexactprogram/state/eta identities. ReusealloldTRAIN/VALlabels for independentnoise evaluation;oldstaticactions are not terminal Q labels. Encoding changesdo notinvalidateoldlabels.'))
    print(dict(counts,profiles_scanned=len(inventory),new_physicalstates=0))

def worker(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json')
    assert sha(__file__)==p['worker_code_sha256']
    assert all(sha(f)==h for f,h in p['source_code_hashes'].items())
    pre=read(HERE/'cache_preflight.json')['summary'];exact=read(HERE/'exact_identity_preflight.json')
    assert pre['ambiguous']==0 and pre['genuinely_missing']==exact['genuinely_missing']+exact.get('cached_unknown',0)
    cases=read(HERE/'missing_cases.json')[index::workers];states={s['state_uid']:s for s in p['states']}
    jax.config.update('jax_enable_x64',False)
    base=native_runtime(SCENE);runtimes={c:EndpointMatchedField(base,c) for c in FAMILIES};buffer=[];start=time.monotonic()
    for r in cases:
        dest=HERE/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():buffer.append(read(dest));continue
        env=runtimes[r['controller_name']].make_env(states[r['state_uid']]);rt=runtimes[r['controller_name']]
        term='timeout';err=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for t in range(rt.config.max_steps):
            try:
                key=key_for(dict(uid=p['shared_source_alias']),r['seed'],t)
                action,_,n=rt.action(env,key,np.asarray(r['eta']),'field');calls+=n;steps+=1
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
            execution_semantics_version='scene_precision_guard_v1_common_exogenous_coupling',jax_enable_x64=bool(jax.config.x64_enabled),
            execution_code_sha256=sha(__file__),host_backend_sha256=sha(old.HERE/'numpy_backend.py'))
        assert not (result['success'] and (result['collision'] or result['numerical_failure']))
        write(dest,result);buffer.append(result)
        if len(buffer)>=4:append_journal(buffer,EXPERIMENT,f'four_{index}');buffer=[]
    if buffer:append_journal(buffer,EXPERIMENT,f'four_{index}')
    write(HERE/f'worker{index}.json',dict(cases=len(cases),seconds=time.monotonic()-start,completed=True))

def merge():
    from .merge import run
    run()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','merge']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=48)
    a=p.parse_args();worker(a.index,a.workers) if a.stage=='worker' else globals()[a.stage]()

