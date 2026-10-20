from core import *
from registration import *
from shared_rollout_db.src.planner import preflight
from audit_utils import source_inventory
BASE=D/'decentralized_pipeline_v1';C=D/'teacher_student_macflow_v1c'
def frozen():
    for p,h in read(HERE/'PREREGISTRATION.receipt.json')['files'].items():assert sha(p)==h,p

def profile(kind,i,arm,pipeline='local',rule=None):
    base=read(BASE/'main_protocol.json');ts=read(HERE/'TEACHERS.json');name=ts[i]['name'] if kind=='student' else kind;v=dict(base['base_profiles'][ts[i if kind=='student' else 1]['name']]['payload']);eta=read(HERE/'K16_AUDIT.json')['eta'];j=10 if arm in ['F10','K10'] else 13 if arm in ['F13','K13','S300','S500'] else int(arm[1:]) if arm.startswith('K') else None;ea=eta[j] if j is not None else [0.,0.,0.];eb=eta[10] if arm in ['S300','S500'] else ea;switch=int(arm[1:]) if arm in ['S300','S500'] else 700;canon='F10' if arm=='K10' else 'F13' if arm=='K13' else arm
    if kind=='student':
        ck=read(HERE/'STUDENTS.json')[str(i)];assert sha(ck['checkpoint'])==ck['checkpoint_sha256'];v.update(flow_checkpoint_path=ck['checkpoint'],flow_checkpoint_sha256=ck['checkpoint_sha256'])
        if canon in ['safety baseline','F10','F13','S300','S500'] and rule is None:
            ref=next(c for c in read(C/'main_CASES.json') if c['base_controller_name']==name and c['arm']==canon);return name,v,ref['controller_uid'],ref['eta_uid'],ea,eb,switch,[]
    else:v.update(flow_checkpoint_path=str(HERE/'CONTROL_DEFINITIONS.json'),flow_checkpoint_sha256=sha(HERE/'CONTROL_DEFINITIONS.json'))
    payload=dict(v,family='round30d_'+kind,index=i if kind=='student' else None,nominal_source_sha256=sha(HERE/('student.py' if kind=='student' else 'controls.py')),pipeline=pipeline,eta_schedule=dict(a=ea,b=eb,switch_step=switch),rule=rule,nominal_kind=kind,controls_definition_sha256=sha(HERE/'CONTROL_DEFINITIONS.json') if kind!='student' else None,fair_rhr_sha256=sha(HERE/'fair_rhr.py') if rule else None,pipeline_sha256=sha(HERE/'pipeline.py'),episode_sha256=sha(HERE/'episode_frozen.py'),rng=dict(root=2026100403,key='fold_in(fold_in(fold_in(PRNGKey(root),sha256(state.uid)[:8]),seed),step)',private_noise='fold_in(key,i)' if kind=='student' else 'none'),jax_enable_x64=False)
    ctl=uid('ctl',payload);eid,values,hexes=eta_identity(ea);rows=[('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v['flow_checkpoint_sha256'],orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(source=sha(HERE/'pipeline.py'),mode=pipeline,radius=1.5 if pipeline=='local' else 'central')),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version='own_obs_goal_noise' if kind=='student' else 'own_goal_position',rng_semantics_version=canonical(payload['rng']),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')),('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hexes))]
    return name,v,ctl,eid,ea,eb,switch,rows

def create(phase):
    frozen();assert read(HERE/'COMPONENT_TESTS.json')['status']=='PASS';base=read(BASE/'main_protocol.json');states=read(HERE/('TUNING_STATES.json' if phase=='tune' else 'EVAL_STATES.json'));plans=[];grid=read(HERE/'RHR_GRID.json')
    if phase=='A':plans=[(kind,None,arm,pipe,None) for kind in ['Z','G'] for pipe in ['local','central'] for arm in ['safety baseline','F10','F13','S300','S500']]
    elif phase=='tune':plans=[(kind,i,f'RHR_grid{k:02}', 'local',rule) for kind,i in [('student',i) for i in range(14)]+[('G',None)] for k,rule in enumerate(grid)]
    elif phase=='fair':
        assert (HERE/'RHR_FROZEN.receipt.json').exists();sel=read(HERE/'RHR_SELECTED.json');plans=[(kind,i,'RHR-fair','local',sel['students' if kind=='student' else 'G']['rule']) for kind,i in [('student',i) for i in range(14)]+[('G',None)]]
    elif phase=='C':plans=[(kind,i,arm,'local',None) for kind,i in [('student',i) for i in range(14)]+[('Z',None)] for arm in [f'K{j}' for j in range(16)]+['S300','S500','safety baseline']]
    else:raise ValueError(phase)
    p=dict(experiment_uid='exp_teacher_student_macflow_v1d_'+phase,phase=phase,states=states,controller_order=[],base_profiles={},controllers={},eta=base['eta']);registrations=[exp_row(phase)];cases=[]
    for kind,i,arm,pipe,rule in plans:
        name,v,ctl,eid,ea,eb,switch,rows=profile(kind,i,arm,pipe,rule);registrations.extend(rows);p['base_profiles'][name]=dict(payload=v)
        if name not in p['controller_order']:p['controller_order'].append(name)
        p['controllers'][name+'__'+pipe+'__'+arm]=dict(controller_uid=ctl)
        for s in states:
            for seed in ([0] if phase=='tune' else [0,1]):
                sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk));cases.append(dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,controller_name=name+'__'+pipe+'__'+arm,base_controller_name=name,pipeline='central' if rule else pipe,display_pipeline='D-local-RHR' if rule else pipe,arm=arm,state_index=s['study_index'],seed=seed,eta_a=ea,eta_b=eb,switch_step=switch,student_index=i,kind=kind,rhr_rule=rule))
    write(HERE/f'{phase}_protocol.json',p);write(HERE/f'{phase}_CASES.json',cases);register(phase,registrations);unique={c['rollout_uid']:c for c in cases};manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in unique.values()]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0;write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pf);missing=[];reuse=[];unknown=[]
    with connect(True) as db:
        for c in unique.values():
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if r:
                assert not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE';(unknown if r['numerical_failure'] else reuse).append(c)
            else:missing.append(c)
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_REUSE_CASES.json',dict(exact=reuse,preserved_unknown=unknown));seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_MISSING_CASES.json',HERE/f'{phase}_protocol.json',HERE/f'{phase}_CASES.json'],before_submission=True,unique_cases=len(unique),missing=len(missing),reuse=len(reuse),unknown=len(unknown));source_inventory('imports_design_'+phase+'.json');print(phase,'missing',len(missing),'reuse',len(reuse),'unknown',len(unknown),flush=True)
if __name__=='__main__':
    try:create(sys.argv[1])
    except (IdentityConflict,FileNotFoundError,ImportError) as e:
        write(HERE/(sys.argv[1]+'_GAP.json'),dict(utc=utc(),status='GAP',reason=str(e),exception=type(e).__name__));sys.exit(86)
