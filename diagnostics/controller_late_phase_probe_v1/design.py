from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight

def inventory(phase):
    labels_allowed();res=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_PREFLIGHT.json',res);assert res['summary']['ambiguous']==0
    rows=read(HERE/f'{phase}_CASES.json');missing=[];reused=[]
    with connect(True) as db:
        for c in rows:
            r=db.execute('SELECT rollout_uid,original_source_file,conflict_quarantined,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is None or not r['conflict_quarantined']
            if (HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists():reused.append(dict(rollout_uid=c['rollout_uid'],kind='local_existing_raw'))
            elif r is not None:reused.append(dict(r))
            else:missing.append(c)
    write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_CACHE_INVENTORY.json',dict(total=len(rows),genuinely_missing=len(missing),reused=reused,unknown_never_rerun=True,cache_preflight_sha256=sha(HERE/f'{phase}_CACHE_PREFLIGHT.json')))
    print(phase,len(rows),len(missing))

def create():
    prereg();labels_allowed();assert (HERE/'SELECTOR_FROZEN.receipt.json').exists();old=read(R5/'stage_2/protocol.json');r10=read(R10/'protocol.json');profiles=copy.deepcopy(old['controllers']['ring_exchange']);states=[s for s in old['states'] if s['study_split']=='blend_test'];assert len(states)==32;main=[];gates=[]
    for name in r10['controller_order']:assert profiles[name]==r10['base_profiles'][name]
    def schedule(base,a,b,t):
        name=f'{base}__eta{a}to{b}__t{t}';v=copy.deepcopy(profiles[base]['payload']);v.update(parent_controller_uid=profiles[base]['controller_uid'],eta_schedule=dict(eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_hex=[float_hex(x) for x in old['eta'][a]],eta_b_hex=[float_hex(x) for x in old['eta'][b]],switch_step=t,action_index_rule='a for index<t; b for index>=t; preserve native termination'),schedule_runner_sha256=sha(R9/'run.py'),schedule_core_sha256=sha(R9/'core.py'),execution_semantics='scene_precision_guard_eta_schedule_v1')
        profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=base);return name
    def case(s,base,name,a,b,t,seed,arm,si):
        sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(old['eta'][a])[0];ctl=profiles[name]['controller_uid'];rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk))
        return dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene='ring_exchange',controller_name=name,base_controller_name=base,split='blend_test',state_index=si,eta_index=a,eta=old['eta'][a],eta_a=old['eta'][a],eta_b=old['eta'][b],eta_a_index=a,eta_b_index=b,switch_step=t,seed=seed,arm=arm)
    for base in old['controller_order']:
        for t in [300,500]:
            name=schedule(base,13,10,t)
            for si,s in enumerate(states):
                for seed in range(4):main.append(case(s,base,name,13,10,t,seed,f'S{t}',si))
    gatestates={s['state_uid']:s for s in r10['states']}
    for c in read(R10/'gate_CASES.json'):
        name=c['controller_name']+'__round11_gate';v=copy.deepcopy(r10['controllers']['ring_exchange'][c['controller_name']]['payload']);v['round11_io_validation_sha256']=sha(HERE/'run.py');profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=c['base_controller_name']);n=copy.deepcopy(c);n.update(controller_name=name,controller_uid=profiles[name]['controller_uid'],expected_round10_rollout_uid=c['rollout_uid']);n['rollout_uid']=uid('roll',dict(state=n['state_uid'],eta=n['eta_uid'],controller=n['controller_uid'],seed=n['seed_key']));gates.append(n)
    allstates=states+[gatestates[s] for s in dict.fromkeys(c['state_uid'] for c in gates)];assert len(main)==3072 and len(gates)==32
    protocol=dict(schema='round11_blend_schedule_v1',experiment_uid=EXP,states=allstates,test_states=states,controller_order=old['controller_order'],base_profiles=old['controllers']['ring_exchange'],controllers={'ring_exchange':profiles},eta=old['eta'],environments=old['environments'],runner_sha256=sha(R9/'run.py'),core_sha256=sha(HERE/'core.py'),io_binding_sha256=sha(HERE/'run.py'),flow_adapter_sha256=r10['flow_adapter_sha256'],blend_runtime_sha256=sha(R5/'blend_runtime.py'),original_round10_core_sha256=sha(R10/'core.py'),selector_receipt_sha256=sha(HERE/'SELECTOR_FROZEN.receipt.json'),horizon=700,dt=.05,jax_enable_x64=False)
    freeze(HERE/'protocol.json',protocol);freeze(HERE/'main_CASES.json',main);freeze(HERE/'gate_CASES.json',gates);(HERE/'STAGE0_SUMMARY.md').write_text('Gate replays historical32 cases only after R6 and all blend selectors freeze.\n')
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round11 late-phase blend schedule endpoint',str(HERE/'main'),sha(HERE/'protocol.json'),sha(R9/'run.py'),canonical(dict(main_requests=3072,gate32=True,selector_frozen=True))))
        assert db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()[0]==sha(HERE/'protocol.json')
        for s in allstates:
            row=db.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone();assert row and row[0]==canonical(s['physical'])
        for ctl in profiles.values():
            v=ctl['payload'];row=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone();assert row is None or row[0]==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    for phase,rows in [('gate',gates),('main',main)]:
        groups=defaultdict(list)
        for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
        freeze(HERE/f'{phase}_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]));inventory(phase)
if __name__=='__main__':create() if sys.argv[1]=='create' else inventory(sys.argv[1])
