from core import *
import copy
from collections import defaultdict,Counter
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight

def physical_signature(v):
    if not isinstance(v,dict):return None
    values=[]
    for names in (('positions','initial_positions'),('velocities','initial_velocities'),('goals',)):
        value=next((v[k] for k in names if k in v),None)
        if value is None:return None
        a=np.asarray(value,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():return None
        a[a==0]=0;values.append((a.shape,a.tobytes().hex()))
    return hashlib.sha256(canonical(dict(values=values,timestep=float(v.get('timestep',v.get('step',0))))).encode()).hexdigest()

def create():
    prereg();old=read(P1/'P1_EXECUTION_PROTOCOL.json');r9=read(R9/'protocol.json');assert len(CONTROLLERS)==14 and set(DEV).issubset(CONTROLLERS)
    assert all(old['controllers']['ring_exchange'][n]==r9['base_profiles'][n] for n in DEV)
    sys.path.insert(0,str(FIELD/'source'))
    from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
    native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    states=generate('round10_universal_fresh',64,2026190000,native);freeze(HERE/'PROSPECTIVE_FRESH_STATES.json',states)
    sigs={physical_signature(s['physical']):s['state_uid'] for s in states};assert None not in sigs and len(sigs)==64
    ids={s['state_uid'] for s in states};content={s['content_hash'] for s in states};hits=[];recoverable=0;unresolved=Counter()
    with connect(True) as db:
        historical=db.execute('SELECT state_uid,content_hash,physical_state_json,provenance_json,identity_quality FROM state').fetchall()
        for row in historical:
            if row['state_uid'] in ids or row['content_hash'] in content:hits.append(dict(kind='UID/content',historical_state_uid=row['state_uid']))
            v=read_json(row['physical_state_json']);signature=physical_signature(v)
            if signature is None:unresolved[row['identity_quality']]+=1
            else:
                recoverable+=1
                if signature in sigs:hits.append(dict(kind='physical',historical_state_uid=row['state_uid'],fresh_state_uid=sigs[signature]))
    freeze(HERE/'STATE_DEDUP_AUDIT_V2.json',dict(utc=utc(),generator_sha256=sha(P1/'state_generator.py'),source_environment_sha256=sha(FIELD/'source/ring_exchange/environment.py'),draw_seed_start=2026190000,draw_seed_end=2026190063,fresh_count=64,DB_rows_scanned=len(historical),UID_content_scanned_all=True,recoverable_physical_rows=recoverable,unresolved_physical_by_quality=dict(unresolved),collisions=hits,passed=not bool(hits),scope='All DB identities/content; all recoverable coordinate triples normalized float64/zeros/timestep. Legacy identities without physical coordinates remain explicitly unresolved; no absolute coordinate-nonoverlap claim for these rows.',before_endpoint_opening=True))
    assert not hits,'Duplicate state: stop main; no adaptive redraw'
    profiles=copy.deepcopy(old['controllers']['ring_exchange']);gate_rows=[];main_rows=[]
    def schedule(base,a,b,t,validation=False):
        name=f'{base}__eta{a}to{b}__t{t}';v=copy.deepcopy(old['controllers']['ring_exchange'][base]['payload']);v.update(parent_controller_uid=old['controllers']['ring_exchange'][base]['controller_uid'],eta_schedule=dict(eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_hex=[float_hex(x) for x in old['eta'][a]],eta_b_hex=[float_hex(x) for x in old['eta'][b]],switch_step=t,action_index_rule='a for index<t; b for index>=t; preserve native termination'),schedule_runner_sha256=sha(R9/'run.py'),schedule_core_sha256=sha(R9/'core.py'),execution_semantics='scene_precision_guard_eta_schedule_v1')
        if validation:name+='__round10_io_gate';v['round10_io_validation_sha256']=sha(HERE/'run.py')
        profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=base)
        if base in DEV and not validation:assert profiles[name]==r9['controllers']['ring_exchange'][name]
        return name
    def case(s,base,name,a,b,t,seed,arm,phase):
        sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];ctl=profiles[name]['controller_uid'];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk))
        return dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene='ring_exchange',controller_name=name,base_controller_name=base,split=phase,state_index=s['study_index'],eta_index=a,eta=old['eta'][a],eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_index=a,eta_b_index=b,switch_step=t,seed=seed,arm=arm)
    for base in CONTROLLERS:
        for arm in ARMS:
            if arm=='Freg' and base not in FREG:continue
            if arm.startswith('S'):a,b,t=13,10,int(arm[1:]);name=schedule(base,a,b,t)
            else:a=b=FREG[base] if arm=='Freg' else int(arm[1:]);t=700;name=base
            for s in states:
                for seed in range(4):main_rows.append(case(s,base,name,a,b,t,seed,arm,'prospective_main'))
    gate_states={s['state_uid']:s for s in r9['states']}
    for oldcase in read(R9/'gate_CASES.json'):
        s=gate_states[oldcase['state_uid']];base=oldcase['base_controller_name'];a=oldcase['eta_a_index'];b=oldcase['eta_b_index'];t=oldcase['switch_step'];name=schedule(base,a,b,t,True);c=case(s,base,name,a,b,t,oldcase['seed'],'consistency_gate','historical_gate');c.update(expected_round9_rollout_uid=oldcase['rollout_uid'],expected_eta_index=oldcase['expected_eta_index']);gate_rows.append(c)
    unique_gate_states={c['state_uid'] for c in gate_rows};all_states=states+[s for s in r9['states'] if s['state_uid'] in unique_gate_states]
    assert len(main_rows)==18688 and len(gate_rows)==32 and len({c['rollout_uid'] for c in main_rows+gate_rows})==18720
    protocol=dict(schema='universal_eta_schedule_round10_v1',experiment_uid=EXP,preregistration_sha256=sha(HERE/'PREREGISTRATION_V2.md'),states=all_states,fresh_states=states,controller_order=CONTROLLERS,schedule_dev=DEV,held_out=[n for n in CONTROLLERS if n not in DEV],base_profiles=old['controllers']['ring_exchange'],controllers={'ring_exchange':profiles},eta=old['eta'],environments=old['environments'],flow_adapter_sha256=r9['flow_adapter_sha256'],runner_sha256=sha(R9/'run.py'),core_sha256=sha(HERE/'core.py'),io_binding_sha256=sha(HERE/'run.py'),original_round9_core_sha256=sha(R9/'core.py'),state_generator_sha256=sha(P1/'state_generator.py'),horizon=700,dt=.05,placement='FF',jax_enable_x64=False)
    freeze(HERE/'protocol.json',protocol);freeze(HERE/'main_CASES.json',main_rows);freeze(HERE/'gate_CASES.json',gate_rows)
    # The frozen round9 runner requires a descriptive-stage receipt for non-main execution.
    (HERE/'STAGE0_SUMMARY.md').write_text('Round10 historical inventory only; no fresh outcomes opened. See PREREGISTRATION, STATE_DEDUP_AUDIT and HISTORICAL_INVENTORY. Gate only uses historical round9 states.\n')
    freeze(HERE/'HISTORICAL_INVENTORY.json',dict(utc=utc(),files={str(x):sha(x) for x in [R9/'core.py',R9/'run.py',R9/'protocol.json',R9/'CONSISTENCY_GATE.json',R9/'results.json',R4/'protocol.json',P1/'P1_EXECUTION_PROTOCOL.json',P1/'P1_CHECKPOINT_COMPATIBILITY.json',P1/'state_generator.py']},reuse='Exact14 fixed profiles and12 existing dev schedule IDs; old opened outcomes are background/gate, not prospective endpoints',new_evidence_gap='64 new states x14controllers xfixed/schedules xQ4 and full video traces',schedule_identity_same_format=True))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Prospective universal Ring eta schedule',str(HERE),sha(HERE/'protocol.json'),sha(R9/'run.py'),canonical(dict(main_requests=18688,gate32=True,prospective_registration=read(HERE/'PREREGISTRATION_V2.receipt.json')))))
        assert db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()[0]==sha(HERE/'protocol.json')
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(protocol['environments']['ring_exchange']['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'));db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXP))
        for ctl in profiles.values():
            v=ctl['payload'];oldrow=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone();assert oldrow is None or oldrow[0]==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    for phase,rows in [('gate',gate_rows),('main',main_rows)]:
        groups=defaultdict(list)
        for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
        freeze(HERE/f'{phase}_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]));inventory(phase)

def read_json(v):
    try:return json.loads(v) if v else None
    except (TypeError,ValueError):return None

def inventory(phase):
    prereg();result=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_PREFLIGHT.json',result);assert result['summary']['ambiguous']==0
    rows=read(HERE/f'{phase}_CASES.json');missing=[];cached=0
    with connect(True) as db:
        for c in rows:
            r=db.execute('SELECT conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is None or not r[0]
            if (HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists() or r is not None:cached+=1
            else:missing.append(c)
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_CACHE_INVENTORY.json',dict(utc=utc(),total=len(rows),missing=len(missing),cached=cached,unknown_never_rerun=True,fresh_labels_unopened=True));print(phase,len(rows),len(missing),'missing',cached,'cached')
if __name__=='__main__':create() if sys.argv[1]=='create' else inventory(sys.argv[1])
