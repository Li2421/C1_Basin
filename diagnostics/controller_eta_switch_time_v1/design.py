from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight

def main():
    prereg();old=read(R4/'protocol.json');assert (HERE/'STAGE0_SUMMARY.md').exists();profiles={};cases=[];gatecases=[]
    for base in CONTROLLERS:
        for a,b in [(10,13),(13,10)]:
            for t in [0]+TIMES+[700]:
                name=f'{base}__eta{a}to{b}__t{t}';v=copy.deepcopy(old['controllers']['ring_exchange'][base]['payload']);v.update(parent_controller_uid=old['controllers']['ring_exchange'][base]['controller_uid'],eta_schedule=dict(eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_hex=[float_hex(x) for x in old['eta'][a]],eta_b_hex=[float_hex(x) for x in old['eta'][b]],switch_step=t,action_index_rule='a for index<t; b for index>=t; preserve native termination'),schedule_runner_sha256=sha(HERE/'run.py'),schedule_core_sha256=sha(HERE/'core.py'),execution_semantics='scene_precision_guard_eta_schedule_v1')
                profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=base)
                for s in old['states'][128:]:
                    for seed in range(4):
                        sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];c=profiles[name]['controller_uid'];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=c,seed=sk));r=dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=c,seed_key=sk,rollout_uid=rid,scene='ring_exchange',controller_name=name,base_controller_name=base,split='eta_switch',state_index=s['study_index'],eta_index=a,eta=old['eta'][a],eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_index=a,eta_b_index=b,switch_step=t,seed=seed)
                        if t in TIMES:cases.append(r)
                        elif s['study_index'] in (0,1) and seed in (0,1):
                            # 8 cases/controller: both pure eta, each state0/1 x seed0/1;
                            # distribute t0/t700 across branches without duplicate pure requests.
                            valid=(a==13 and t==0 and seed==0) or (a==10 and t==700 and seed==1) or (a==13 and t==700 and seed==0) or (a==10 and t==0 and seed==1)
                            if valid:
                                r['split']='consistency_gate';r['expected_eta_index']=b if t==0 else a;gatecases.append(r)
    assert len(cases)==10240 and len(gatecases)==32 and len({c['rollout_uid'] for c in cases+gatecases})==10272
    p=dict(schema='controller_eta_switch_time_protocol_v1',experiment_uid=EXP,preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),states=old['states'][128:],base_profiles={n:old['controllers']['ring_exchange'][n] for n in CONTROLLERS},controllers={'ring_exchange':profiles},eta=old['eta'],environments=old['environments'],flow_adapter_sha256=old['flow_adapter_sha256'],runner_sha256=sha(HERE/'run.py'),core_sha256=sha(HERE/'core.py'),horizon=700,dt=.05,placement='FF',jax_enable_x64=False)
    freeze(HERE/'protocol.json',p);freeze(HERE/'main_CASES.json',cases);freeze(HERE/'gate_CASES.json',gatecases)
    with connect(True) as db:
        existing=db.execute("SELECT controller_uid,config_json FROM controller_config WHERE config_json LIKE '%eta_schedule%' OR config_json LIKE '%switch_step%'").fetchall();write(HERE/'HISTORICAL_SCHEDULE_AUDIT.json',dict(utc=utc(),existing_profiles=[dict(x) for x in existing],new_controller_profiles=len(profiles),decision='Reuse only exact frozen schedule/code identities; fixed eta outcomes reused as endpoints'))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Ring eta switch-time causal mechanism',str(HERE),sha(HERE/'protocol.json'),sha(HERE/'run.py'),canonical(dict(cap=10272,main_cap=10240,consistency_gate=32))))
        assert db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()[0]==sha(HERE/'protocol.json')
        for ctl in profiles.values():
            v=ctl['payload'];cid=ctl['controller_uid'];oldrow=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(cid,)).fetchone();assert oldrow is None or oldrow[0]==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(cid,v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    for phase,rows in [('gate',gatecases),('main',cases)]:
        groups=defaultdict(list)
        for c in rows:groups[(c['state_uid'],c['eta_uid'],c['controller_uid'])].append(c['seed_key'])
        freeze(HERE/f'{phase}_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ss) for (s,e,c),ss in groups.items()]))
        inventory(phase)

def inventory(phase):
    prereg();result=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_PREFLIGHT.json',result);assert result['summary']['ambiguous']==0;rows=read(HERE/f'{phase}_CASES.json');missing=[];counts=dict(raw_cached=0,DB_cached_valid=0,DB_cached_unknown=0,missing=0)
    with connect(True) as db:
        for c in rows:
            out=HERE/phase/'raw'/f"{c['rollout_uid']}.json";r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is None or not r['conflict_quarantined']
            if out.exists():counts['raw_cached']+=1
            elif r is not None:
                counts['DB_cached_unknown' if r['numerical_failure'] else 'DB_cached_valid']+=1
            else:missing.append(c);counts['missing']+=1
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_CACHE_INVENTORY.json',dict(utc=utc(),**counts,preflight_summary=result['summary'],unknown_never_rerun=True));print(phase,counts)
if __name__=='__main__':main() if sys.argv[1]=='create' else inventory(sys.argv[1])
