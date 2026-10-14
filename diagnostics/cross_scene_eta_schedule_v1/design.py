from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight

def create():
    prereg();gate();assert read(HERE/'STATE_DEDUP_AUDIT.json')['passed'];old=read(E8/'protocol.json');states=read(HERE/'FRESH_STATES.json');profiles={sc:{n:copy.deepcopy(old['controllers'][sc][n]) for n in NAMES} for sc in SCENES};rows=[]
    for sc in SCENES:
        H=old['environments'][sc]['config']['config']['max_steps'];times=[round(3*H/7),round(5*H/7)]
        for n in NAMES:
            for arm in ARMS:
                if arm.startswith('F'):a=b=int(arm[1:]);t=H;name=n
                else:
                    a=int(arm[1:3]);b=10;t=times[int(arm[-1])-1];name=n+'__'+arm;v=copy.deepcopy(profiles[sc][n]['payload']);v.update(parent_controller_uid=profiles[sc][n]['controller_uid'],eta_schedule=dict(eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_hex=[float_hex(x) for x in old['eta'][a]],eta_b_hex=[float_hex(x) for x in old['eta'][b]],switch_step=t,action_index_rule='a for index<t; b for index>=t'),schedule_wrapper_sha256=sha(HERE/'run.py'),execution_semantics='E8_scene_precision_FF_absolute_eta_switch_v1');profiles[sc][name]=dict(controller_uid=uid('ctl',v),payload=v)
                for s in [q for q in states if q['scenario']==sc]:
                    for seed in range(4):
                        sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];ctl=profiles[sc][name]['controller_uid'];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk));rows.append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene=sc,controller_name=name,base_controller_name=n,split=s['study_split'],state_index=s['study_index'],eta_index=a,eta=old['eta'][a],eta_a=old['eta'][a],eta_b=old['eta'][b],switch_step=t,seed=seed,arm=arm))
    assert len(rows)==13440
    p=dict(schema='round13_cross_scene_eta_schedule_v1',experiment_uid=EXP,states=states,controllers=profiles,base_profiles={sc:{n:old['controllers'][sc][n] for n in NAMES} for sc in SCENES},environments=old['environments'],eta=old['eta'],source_hashes=read(HERE/'HISTORICAL_ASSET_INDEX.json')['assets'],run_sha256=sha(HERE/'run.py'),core_sha256=sha(HERE/'core.py'),scenes=SCENES,controller_order=NAMES,arms=ARMS,prereg_sha256=sha(HERE/'PREREGISTRATION.md'),gate_sha256=sha(HERE/'CONSISTENCY_GATE.json'),dedup_sha256=sha(HERE/'STATE_DEDUP_AUDIT.json'))
    freeze(HERE/'protocol.json',p);freeze(HERE/'main_CASES.json',rows)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round13 four-scene eta schedules',str(HERE/'main'),sha(HERE/'protocol.json'),sha(HERE/'run.py'),canonical(dict(requests=13440,prospective_new_states=True))))
        for s in states:
            prior=db.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone();assert prior is None or prior[0]==canonical(s['physical']);db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments'][s['scenario']]['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'));db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXP))
        for sc in SCENES:
            for ctl in profiles[sc].values():
                v=ctl['payload'];prior=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone();assert prior is None or prior[0]==canonical(v);db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    groups=defaultdict(list)
    for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
    freeze(HERE/'main_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]));pre=preflight(HERE/'main_PLANNED_ROLLOUTS.json');write(HERE/'main_CACHE_PREFLIGHT.json',pre);assert pre['summary']['ambiguous']==0;missing=[];reuse=[]
    with connect(True) as db:
        for c in rows:
            d=db.execute('SELECT original_source_file,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert d is None or not d['conflict_quarantined']
            if (HERE/'main/raw'/f"{c['rollout_uid']}.json").exists():reuse.append(c)
            elif d is not None:reuse.append(dict(cached=c,source=dict(d)))
            else:missing.append(c)
    write(HERE/'main_MISSING_CASES.json',missing);write(HERE/'main_CACHE_INVENTORY.json',dict(planned=13440,genuinely_missing=len(missing),reused=reuse,unknown_never_rerun=True));assert not reuse,'Audit compatible raw import before any submission';print('Main13440 preflight PASS')
if __name__=='__main__':create()
