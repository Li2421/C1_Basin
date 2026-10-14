from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight
ARMS=['F10','F13','F2','S200','S300','S400','S500','S600']

def inventory(phase):
    prereg();labels_allowed();result=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_PREFLIGHT.json',result);assert result['summary']['ambiguous']==0;rows=read(HERE/f'{phase}_CASES.json');missing=[];reused=[]
    with connect(True) as db:
        for c in rows:
            r=db.execute('SELECT rollout_uid,original_source_file,conflict_quarantined,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is None or not r['conflict_quarantined']
            if (HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists():reused.append(dict(rollout_uid=c['rollout_uid'],kind='local_raw'))
            elif r is not None:reused.append(dict(r))
            else:missing.append(c)
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_CACHE_INVENTORY.json',dict(total=len(rows),genuinely_missing=len(missing),reused=reused,unknown_never_rerun=True,preflight_sha256=sha(HERE/f'{phase}_CACHE_PREFLIGHT.json')));print(phase,len(rows),len(missing))

def create():
    prereg();labels_allowed();assert read(HERE/'STATE_DEDUP_AUDIT.json')['passed'];old=read(R10/'protocol.json');base=read(HERE/'FINAL_BASE_PROFILES.json');profiles=copy.deepcopy(old['base_profiles']);profiles.update(copy.deepcopy(base['profiles']));base_profiles=copy.deepcopy(profiles);states=read(HERE/'FRESH_STATES.json');main=[];gates=[]
    def schedule(n,a,b,t):
        name=f'{n}__eta{a}to{b}__t{t}';v=copy.deepcopy(profiles[n]['payload']);v.update(parent_controller_uid=profiles[n]['controller_uid'],eta_schedule=dict(eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_hex=[float_hex(x) for x in old['eta'][a]],eta_b_hex=[float_hex(x) for x in old['eta'][b]],switch_step=t,action_index_rule='a for index<t; b for index>=t; preserve native termination'),schedule_runner_sha256=sha(HERE/'run.py'),schedule_core_sha256=sha(HERE/'core.py'),execution_semantics='scene_precision_guard_eta_schedule_v1');profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=n);return name
    def case(s,n,name,a,b,t,seed,arm):
        sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];ctl=profiles[name]['controller_uid'];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk))
        return dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene='ring_exchange',controller_name=name,base_controller_name=n,split='prospective_round15_population',state_index=s['study_index'],eta_index=a,eta=old['eta'][a],eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_index=a,eta_b_index=b,switch_step=t,seed=seed,arm=arm)
    for n in base['eligible_controller_order']:
        for arm in ARMS:
            if arm.startswith('S'):a,b,t=13,10,int(arm[1:]);name=schedule(n,a,b,t)
            else:a=b=int(arm[1:]);t=700;name=n
            for s in states:
                for seed in range(4):main.append(case(s,n,name,a,b,t,seed,arm))
    gate_states={s['state_uid']:s for s in old['states']}
    for c in read(R10/'gate_CASES.json'):
        name=c['controller_name']+'__round15_gate';v=copy.deepcopy(old['controllers']['ring_exchange'][c['controller_name']]['payload']);v['round15_io_validation_sha256']=sha(HERE/'run.py');profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=c['base_controller_name']);new=copy.deepcopy(c);new.update(controller_name=name,controller_uid=profiles[name]['controller_uid'],expected_round10_rollout_uid=c['rollout_uid']);new['rollout_uid']=uid('roll',dict(state=new['state_uid'],eta=new['eta_uid'],controller=new['controller_uid'],seed=new['seed_key']));gates.append(new)
    allstates=states+[gate_states[i] for i in dict.fromkeys(c['state_uid'] for c in gates)];assert len(main)==len(base['eligible_controller_order'])*1024 and len(gates)==32
    p=dict(schema='round15_controller_population_v1',experiment_uid=EXP,states=allstates,fresh_states=states,controller_order=base['eligible_controller_order'],full_controller_order=base['eligible_controller_order'],best_controller_order=base['recipe_order'],arms=ARMS,base_profiles=base_profiles,controllers={'ring_exchange':profiles},eta=old['eta'],environments=old['environments'],runner_sha256=sha(HERE/'run.py'),core_sha256=sha(HERE/'core.py'),io_binding_sha256=sha(HERE/'run.py'),original_round10_core_sha256=sha(R10/'core.py'),flow_adapter_sha256=old['flow_adapter_sha256'],horizon=700,dt=.05,jax_enable_x64=False,placement='FF',r5_freeze_sha256=sha(HERE/'R5_FROZEN.receipt.json'),r6_freeze_sha256=sha(HERE/'R6_FROZEN.receipt.json'),analysis_freeze_sha256=sha(HERE/'ANALYSIS_FROZEN.receipt.json'),compatibility_audit_sha256=sha(HERE/'COMPATIBILITY_AUDIT.json'))
    freeze(HERE/'protocol.json',p);freeze(HERE/'main_CASES.json',main);freeze(HERE/'gate_CASES.json',gates);(HERE/'STAGE0_SUMMARY.md').write_text('Round12 sources and final checkpoint/probe/selector freezes complete before historical best32 replay gate. No final outcome produced yet.\n')
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round15 Ring controller population',str(HERE/'main'),sha(HERE/'protocol.json'),sha(HERE/'run.py'),canonical(dict(requests=len(main),gate32=True,prereg_sha256=sha(HERE/'PREREGISTRATION.md'),analysis_frozen=True))))
        assert db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()[0]==sha(HERE/'protocol.json')
        for s in states:
            row=db.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone();assert row is None or row[0]==canonical(s['physical']);db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'));db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXP))
        for ctl in profiles.values():
            v=ctl['payload'];row=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone();assert row is None or row[0]==canonical(v);db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    for phase,rows in [('gate',gates),('main',main)]:
        groups=defaultdict(list)
        for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
        freeze(HERE/f'{phase}_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]));inventory(phase)
if __name__=='__main__':create() if sys.argv[1]=='create' else inventory(sys.argv[1])
