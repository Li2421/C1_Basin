from core import *
from registration import *
from shared_rollout_db.src.planner import preflight
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','B-RHR']
def create(phase):
    assert read(HERE/'ROUND28_REPLAY_GATE.json')['completed']==16 and read(HERE/'ROUND28_REPLAY_GATE.json')['passed']
    if phase!='teacher':assert (HERE/'TRAINING_FROZEN.receipt.json').exists()
    if phase=='main':assert (HERE/'CHOICES_FROZEN.receipt.json').exists()
    source=read(D/'decentralized_pipeline_v1/main_protocol.json');ts=read(HERE/'TEACHERS.json');names=read(HERE/'DISTILLATION_REPORT.json')['valid_names'] if phase=='pilot' else source['controller_order'] if phase=='teacher' else read(HERE/'CAPABILITY_GATE.json')['eligible'];states=read(HERE/('PILOT_STATES.json' if phase=='pilot' else 'EVAL_STATES.json'));p=dict(experiment_uid='exp_teacher_student_macflow_v1b_'+phase,phase=phase,base_profiles={},controller_order=names,states=states,controllers={},eta=source['eta'],runner_sha256=sha(HERE/'episode_extended.py'),pipeline_sha256=sha(HERE/'pipeline.py'));rows=[];registrations=[]
    for n in names:
        idx=source['controller_order'].index(n);v=dict(source['base_profiles'][n]['payload'])
        ck=Path(ts[idx]['checkpoint']) if phase=='teacher' else HERE/'students'/n/'best.pkl'
        if phase!='teacher':assert sha(ck)==read(ck.parent/'COMPLETE.json')['checkpoint_sha256']
        v.update(flow_checkpoint_path=str(ck),flow_checkpoint_sha256=sha(ck));p['base_profiles'][n]=dict(payload=v)
        arms=['safety baseline','S500'] if phase=='pilot' else ['nominal only','safety baseline'] if phase=='teacher' else ARMS
        for arm in arms:
            a=13 if arm in ['F13','S300','S500'] else 10 if arm=='F10' else None;eta_a=p['eta'][a] if a is not None else [0.,0.,0.];eta_b=p['eta'][10] if arm.startswith('S') else eta_a;switch=int(arm[1:]) if arm.startswith('S') else 700;mode='nominal' if arm=='nominal only' else 'rhr_local' if arm=='B-RHR' else 'local'
            payload=dict(v,controller_family='frozen_joint_teacher' if phase=='teacher' else 'decentralized_onestep_student',teacher_checkpoint_sha256=ts[idx]['checkpoint_sha256'],student_index=idx,execution_semantics='round30_native_nominal_only_v1' if mode=='nominal' else 'round30_Dlocal_TT_v1',conditioning='private_student_own_goal_own_noise' if phase!='teacher' else 'original_teacher_joint_information',arm=arm,eta_schedule=dict(a=eta_a,b=eta_b,switch_step=switch),sense_radius=1.5,safety_enabled=mode!='nominal',nominal_source_code_sha256=sha(HERE/'student.py') if phase!='teacher' else v.get('controller_source_sha256',sha(FIELD/'validation_v2/adapters.py')),additional_arms_code_sha256=sha(HERE/'actions.py'),pipeline_sha256=sha(HERE/'pipeline.py'),runner_sha256=sha(HERE/'episode_extended.py'),rng=dict(root=2026100403,key='fold_in(fold_in(fold_in(PRNGKey(root), SHA256(state.uid)[:8]),seed),step)',private_robot_noise='fold_in(key,i)' if phase!='teacher' else 'original joint noise',seed_key='future_index; root/controller and state alias locked by profile/provenance'),jax_enable_x64=False)
            ctl=uid('ctl',payload);name=n+'__'+arm;p['controllers'][name]=dict(controller_uid=ctl,payload=payload)
            registrations.append(('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=sha(ck),orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(pipeline=p['pipeline_sha256'],mode=mode,radius=1.5)),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=payload['conditioning'],rng_semantics_version=canonical(payload['rng']),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')))
            eid,values,hexes=eta_identity(eta_a);registrations.append(('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hexes)))
            for s in states:
                for seed in ([0] if phase=='pilot' else [0,1]):
                    sk=canonical(dict(future_index=seed));rows.append(dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk)),state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,controller_name=name,base_controller_name=n,pipeline=mode,arm=arm,state_index=s['study_index'],seed=seed,eta_a=eta_a,eta_b=eta_b,switch_step=switch,student_index=idx,teacher=phase=='teacher'))
    write(HERE/f'{phase}_protocol.json',p);write(HERE/f'{phase}_CASES.json',rows);register(phase+'_controllers',registrations)
    manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0;write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pf)
    with connect(True) as db:
        missing=[];reuse=[];unknown=[]
        for c in rows:
            found=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if found:
                assert not found['conflict_quarantined'];(unknown if found['numerical_failure'] else reuse).append(c)
            else:missing.append(c)
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_REUSE_CASES.json',dict(exact=reuse,preserved_unknown=unknown));seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_protocol.json',HERE/f'{phase}_MISSING_CASES.json'],before_submission=True,planned=len(rows),missing=len(missing),reuse=len(reuse),preserved_unknown=len(unknown));print(phase,len(rows),'missing',len(missing),flush=True)
if __name__=='__main__':create(sys.argv[1])
