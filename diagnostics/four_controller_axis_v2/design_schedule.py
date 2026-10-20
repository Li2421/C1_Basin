"""Conditional 5-schedule design under the preregistered headroom gate."""
from __future__ import annotations
import copy,datetime,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight
EXP='exp_round19b_four_controller_axis_schedule_v1'
ARMS=[('S15to10_471',15,10,471),('S15to10_786',15,10,786),
      ('S10to15_471',10,15,471),('S10to15_786',10,15,786),('S13to10_786',13,10,786)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    fixed=read(HERE/'FIXED_RESULTS.json');assert fixed['headroom_gate_passed'] is True
    assert fixed['r5_freeze_sha256']==sha(HERE/'R5_FREEZE.receipt.json')
    base=read(HERE/'FIXED_PROTOCOL.json');states=base['states'];names=base['controller_order'];profiles={};cases=[]
    for name in names:
        parent=base['profiles'][name]
        for arm,a,b,t in ARMS:
            key=name+'__'+arm;v=copy.deepcopy(parent['payload'])
            v.update(parent_controller_uid=parent['controller_uid'],
                eta_schedule=dict(eta_a=base['eta'][a],eta_b=base['eta'][b],eta_a_hex=[float_hex(x) for x in base['eta'][a]],
                    eta_b_hex=[float_hex(x) for x in base['eta'][b]],switch_step=t,
                    action_index_rule='a for index<t; b for index>=t; preserve native termination'),
                schedule_runner_sha256=sha(HERE/'run_schedule.py'),
                execution_semantics='round19b_four_checkpoint_native_eta_schedule_v1')
            cid=uid('ctl',v);profiles[key]=dict(controller_uid=cid,payload=v,parent_name=name,arm=arm)
            for s in states:
                for seed in (0,1):
                    sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']))
                    e=eta_identity(base['eta'][a])[0]
                    rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=cid,seed=sk))
                    cases.append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=cid,seed_key=sk,rollout_uid=rid,
                        scene='four_way_intersection',controller_name=key,base_controller_name=name,
                        state_index=s['study_index'],eta_index=a,eta=base['eta'][a],eta_a=base['eta'][a],eta_b=base['eta'][b],
                        switch_step=t,seed=seed,arm=arm))
    assert len(cases)==25*32*5*2 and len({c['rollout_uid'] for c in cases})==len(cases)
    p=dict(schema='round19b_four_schedule_v1',experiment_uid=EXP,states=states,profiles=profiles,
           controller_order=names,arms=[x[0] for x in ARMS],eta=base['eta'],environments=base['environments'],
           fixed_protocol_sha256=sha(HERE/'FIXED_PROTOCOL.json'),fixed_results_sha256=sha(HERE/'FIXED_RESULTS.json'),
           r5_freeze_sha256=sha(HERE/'R5_FREEZE.receipt.json'),runner_sha256=sha(HERE/'run_schedule.py'),
           adapter_sha256=sha(HERE/'adapter.py'),horizon=1100,jax_enable_x64=False)
    write(HERE/'SCHEDULE_PROTOCOL.json',p);write(HERE/'SCHEDULE_CASES.json',cases)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'Round19b Four schedules',str(HERE/'stage3'),sha(HERE/'SCHEDULE_PROTOCOL.json'),sha(HERE/'run_schedule.py'),
             canonical(dict(requests=len(cases),headroom_gate=sha(HERE/'FIXED_RESULTS.json')))))
        for x in profiles.values():
            v=x['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (x['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],
                 uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),
                 v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
            stored=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(x['controller_uid'],)).fetchone()
            assert stored[0]==canonical(v)
    groups=defaultdict(list)
    for x in cases:groups[x['state_uid'],x['eta_uid'],x['controller_uid']].append(x['seed_key'])
    write(HERE/'SCHEDULE_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ks) for (s,e,c),ks in groups.items()]))
    result=preflight(HERE/'SCHEDULE_PLANNED_ROLLOUTS.json');assert result['summary']['ambiguous']==0
    write(HERE/'SCHEDULE_CACHE_PREFLIGHT.json',result)
    missing=[];existing=[]
    with connect(True) as db:
        for x in cases:
            row=db.execute('SELECT numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(x['rollout_uid'],)).fetchone()
            if row:
                assert not row['conflict_quarantined'];existing.append(dict(uid=x['rollout_uid'],unknown=bool(row['numerical_failure'])))
            else:missing.append(x)
    write(HERE/'SCHEDULE_MISSING_CASES.json',missing)
    write(HERE/'SCHEDULE_CACHE_INVENTORY.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        planned=len(cases),existing=len(existing),genuinely_missing=len(missing),unknown_existing=sum(x['unknown'] for x in existing),
        official_preflight_summary=result['summary'],no_unknown_rerun=True))
    print(json.dumps(dict(stage='schedule_design',planned=len(cases),missing=len(missing))),flush=True)
if __name__=='__main__':main()
