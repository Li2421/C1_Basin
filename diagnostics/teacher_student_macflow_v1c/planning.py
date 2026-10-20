from core import *
from registration import *
from shared_rollout_db.src.planner import preflight
from audit_utils import source_inventory
BASE=D/'decentralized_pipeline_v1';A=D/'teacher_student_macflow_v1';B=D/'teacher_student_macflow_v1b'
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','B-RHR']
from functools import lru_cache
@lru_cache(None)
def original():
    old={}
    for origin,indices in [(A,[8,10,12,13]),(B,[1,7])]:
        names={read(HERE/'TEACHERS.json')[i]['name'] for i in indices}
        for c in read(origin/'main_CASES.json'):
            if c['base_controller_name'] in names:old[(c['base_controller_name'],c['state_index'],c['seed'],c['arm'])]=dict(c,reused_from=str(origin))
    return old

def physical_signature(p):
    if not p:return None
    try:
        vals=[]
        for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
            v=np.asarray(next(p[k] for k in keys if k in p),np.float64).copy()
            if v.shape!=(4,2):return None
            v[v==0]=0;vals.append(v.tobytes().hex())
        return uid('physical',vals)
    except (StopIteration,TypeError,ValueError):return None

def initial():
    assert (HERE/'REGATE_FROZEN.receipt.json').exists()
    from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
    import jax
    jax.config.update('jax_enable_x64',False)
    with connect(True) as db:existing=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
    ids={r['state_uid'] for r in existing};content={r['content_hash'] for r in existing};signatures={physical_signature(json.loads(r['physical_state_json'])) for r in existing if r['physical_state_json']};unresolved=None in signatures;signatures.discard(None)
    for s in read(HERE/'PILOT_STATES.json')+read(HERE/'EVAL_STATES.json'):signatures.add(physical_signature(s['physical']))
    selected=[];rejected=[];native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    for seed in range(2026101503000,2026101504000):
        s=normalized_state(generate('round30c_tune',1,seed,native)[0]);sig=physical_signature(s['physical'])
        if s['state_uid'] in ids or s['content_hash'] in content or sig in signatures:rejected.append(seed);continue
        s['study_index']=len(selected);selected.append(s);signatures.add(sig)
        if len(selected)==16:break
    assert len(selected)==16;write(HERE/'TUNING_STATES.json',selected);write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),DB_rows=len(existing),physical_signatures=len(signatures),unresolved_DB_physical_present=unresolved,chosen_seeds=[s['metadata']['draw_seed'] for s in selected],rejected=rejected,comparisons=['DB uid/content/physical','round28 eval','round30 pilot'],outcomes_read=False))
    register('initial',[exp_row(p) for p in ['pilot','probe','tune','main']]+[state_row(s) for s in selected]);source_inventory('imports_design.json')

def controller(i,arm,speed=None,trigger=None):
    ts=read(HERE/'TEACHERS.json');n=ts[i]['name'];base=read(BASE/'main_protocol.json');v=dict(base['base_profiles'][n]['payload']);ck=read(HERE/'STUDENTS.json')[str(i)];assert sha(ck['checkpoint'])==ck['checkpoint_sha256'];v.update(flow_checkpoint_path=ck['checkpoint'],flow_checkpoint_sha256=ck['checkpoint_sha256']);a=13 if arm in ['F13','S300','S500'] else 10 if arm=='F10' else None;eta_a=base['eta'][a] if a is not None else [0.,0.,0.];eta_b=base['eta'][10] if arm in ['S300','S500'] else eta_a;switch=int(arm[1:]) if arm in ['S300','S500'] else 700;mode='nominal' if arm=='nominal only' else 'rhr_local' if speed is not None or arm=='B-RHR' else 'local'
    # Exact R28 parameters inherit existing6 controller identity, regardless of the new state cohort.
    if i in [1,7,8,10,12,13] and (arm in ARMS and speed is None or speed==.15 and trigger==40):
        ref=original()[(n,0,0,'B-RHR' if speed is not None else arm)];return v,ref['controller_uid'],ref['eta_uid'],eta_a,eta_b,switch,mode,[]
    canonical_arm=('B-RHR' if speed==.15 and trigger==40 else 'B-RHR-tuned') if speed is not None else arm
    payload=dict(v,controller_family='decentralized_onestep_student',student_index=i,teacher_checkpoint_sha256=ts[i]['checkpoint_sha256'],conditioning='own_local_observation_own_goal_own_fold_in_noise',arm=canonical_arm,eta_schedule=dict(a=eta_a,b=eta_b,switch_step=switch),rhr_params=dict(speed=speed if speed is not None else .15,trigger_steps=trigger if trigger is not None else 40,slow_threshold=.01,clockwise=True) if mode=='rhr_local' else None,pipeline_sha256=sha(HERE/'pipeline.py'),runner_sha256=sha(HERE/'episode_extended.py'),student_source_sha256=sha(HERE/'student.py'),original_actions_sha256=sha(HERE/'actions.py'),tuned_code_sha256=sha(HERE/'tuned.py') if speed is not None and (speed,trigger)!=(.15,40) else None,rng=dict(root=2026100403,key='fold_in(fold_in(fold_in(PRNGKey(root),sha256(state.uid)[:8]),seed),step)',private_noise='fold_in(key,i)'),jax_enable_x64=False)
    # Canonical original-RHR identity must also match the (.15,40) tuning point.
    if mode=='rhr_local' and canonical_arm=='B-RHR':payload['rhr_params']=dict(speed=.15,trigger_steps=40,slow_threshold=.01,clockwise=True)
    ctl=uid('ctl',payload);eid,values,hexes=eta_identity(eta_a);rows=[('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=ck['checkpoint_sha256'],orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(pipeline=sha(HERE/'pipeline.py'),mode=mode,radius=1.5)),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=payload['conditioning'],rng_semantics_version=canonical(payload['rng']),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')),('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hexes))];return v,ctl,eid,eta_a,eta_b,switch,mode,rows

def create(phase):
    assert (HERE/'REGATE_FROZEN.receipt.json').exists();ts=read(HERE/'TEACHERS.json');reg=read(HERE/'REGATE.json');valid=reg['valid_names'];fresh=reg['new_valid'];eligible=read(HERE/'CAPABILITY_GATE.json')['eligible'] if phase=='main' else fresh;names=fresh if phase=='pilot' else valid;states=read(HERE/('PILOT_STATES.json' if phase=='pilot' else 'TUNING_STATES.json' if phase=='tune' else 'EVAL_STATES.json'));base=read(BASE/'main_protocol.json');p=dict(experiment_uid='exp_teacher_student_macflow_v1c_'+phase,phase=phase,states=states,controller_order=names,base_profiles={},controllers={},eta=base['eta']);registrations=[];cases=[];old=original()
    if phase=='main':assert (HERE/'CHOICES_FROZEN.receipt.json').exists() and (HERE/'RHR_FROZEN.receipt.json').exists()
    for n in names:
        i=next(j for j,r in enumerate(ts) if r['name']==n)
        if phase=='pilot':arms=[('safety baseline',None,None),('S500',None,None)]
        elif phase=='tune':arms=[(f'RHR_v{speed:.2f}_t{trigger}',speed,trigger) for speed in [.10,.15,.25,.35] for trigger in [20,40,80]]
        else:
            chosen=read(HERE/'RHR_SELECTED.json');arms=[(a,None,None) for a in ARMS] if n in eligible or i in [1,7,8,10,12,13] else [];arms.append(('B-RHR-tuned',chosen['speed'],chosen['trigger_steps']))
        for arm,speed,trigger in arms:
            v,ctl,eid,ea,eb,switch,mode,rows=controller(i,arm,speed,trigger);registrations.extend(rows);p['base_profiles'][n]=dict(payload=v);p['controllers'][n+'__'+arm]=dict(controller_uid=ctl)
            for s in states:
                for seed in ([0] if phase in ['pilot','tune'] else [0,1]):
                    sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk));c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,controller_name=n+'__'+arm,base_controller_name=n,pipeline=mode,arm=arm,state_index=s['study_index'],seed=seed,eta_a=ea,eta_b=eb,switch_step=switch,student_index=i,teacher=False,rhr_speed=speed,rhr_trigger=trigger)
                    if phase=='main' and i in [1,7,8,10,12,13] and arm in ARMS:
                        ref=old[(n,s['study_index'],seed,arm)];assert c['rollout_uid']==ref['rollout_uid'];c['reused_from']=ref['reused_from']
                    cases.append(c)
    write(HERE/f'{phase}_protocol.json',p);write(HERE/f'{phase}_CASES.json',cases);register(phase+'_controllers',registrations);unique={c['rollout_uid']:c for c in cases};manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in unique.values()]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0;write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pf);missing=[];reuse=[];unknown=[]
    with connect(True) as db:
        for c in unique.values():
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if row:
                assert not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE';(unknown if row['numerical_failure'] else reuse).append(c)
            else:missing.append(c)
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_REUSE_CASES.json',dict(exact=reuse,preserved_unknown=unknown));seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_MISSING_CASES.json',HERE/f'{phase}_protocol.json',HERE/f'{phase}_CASES.json'],before_submission=True,unique_cases=len(unique),cases_with_aliases=len(cases),missing=len(missing),reuse=len(reuse),unknown=len(unknown));print(phase,'unique',len(unique),'missing',len(missing),flush=True)
if __name__=='__main__':initial() if sys.argv[1]=='initial' else create(sys.argv[1])
