"""Exact new execution identities; official preflight precedes all outcomes."""
from common import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from diagnostics.controller_q_diversity_v1.experiment import case

def create():
    receipt(HERE/'PREREGISTRATION_STAGE_1.md')
    audit=read(HERE/'ENDPOINT_EQUIVALENCE_AUDIT.json');assert audit['status']=='PASS'
    assert audit['runtime_sha256']==sha(HERE/'blend_runtime.py')
    prev=read(PREVIOUS/'protocol.json');extra=read(PREVIOUS/'stage_b/protocol.json')
    profiles=copy.deepcopy(prev['controllers']['ring_exchange']);profiles.update(copy.deepcopy(extra['controllers']['ring_exchange']))
    names=[]
    for ai,A in enumerate(['base_u_macflow','base_u_v10_local_macflow']):
        for bi,B in enumerate(['base_u_plus_w_v4_wide_macflow','base_u_v9_local_macflow']):
            for lam in (.25,.5,.75):
                name=f'blend_pair{2*ai+bi}_lambda{int(lam*100):02d}'
                value=copy.deepcopy(profiles[A]['payload'])
                # All dependency identities are captured. No semantic alias to anchor.
                value.update(base_controller='world_flow_convex_blend_v1',blend=dict(A=A,B=B,lambda_value=lam,**{'lambda':lam},pair=2*ai+bi,
                    A_uid=profiles[A]['controller_uid'],B_uid=profiles[B]['controller_uid'],
                    A_payload=profiles[A]['payload'],B_payload=profiles[B]['payload']),
                    nominal_rule='unprojected common-world Euler; convex blend of anchor-chart terminal clip and float32 speed bound; float64 final bound',
                    initial_rule='same xi normal(key,1x8); convex combination of action denormalization in world chart',
                    numerical_boundary_rule='exact legacy anchor branch at lambda0/1; interior world Euler',
                    execution_callable=str(HERE/'blend_runtime.py'),execution_callable_sha256=sha(HERE/'blend_runtime.py'),controller_source_sha256=sha(HERE/'blend_runtime.py'),
                    conditioning='free true-t0 Ring FF; two exact Flow programs in common world field; unchanged plant/projection/evaluator')
                value.pop('legacy_controller_uid_verified_by_original_function',None)
                profiles[name]=dict(controller_uid=uid('ctl',value),payload=value);names.append(name)
    p=dict(schema='controller_flow_blend_stage1_v1',experiment_uid='exp_controller_flow_blend_v1_stage_1',
        controller_order=names,anchor_order=list(prev['controller_order'])+list(extra['controller_order']),controllers={'ring_exchange':profiles},
        states=prev['states'][:24],eta=prev['eta'],eta_indices=[2,4,10,13],seeds=[0,1,2,3],environments=prev['environments'],
        jax_enable_x64=False,horizon=700,dt=.05,placement='FF',
        preregistration_sha256=sha(HERE/'PREREGISTRATION_STAGE_1.md'),runtime_sha256=sha(HERE/'blend_runtime.py'),
        flow_adapter_sha256=sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py'))
    freeze(HERE/'stage_1/protocol.json',p)
    rows=[case(p,s,n,j,k) for n in names for s in p['states'] for j in p['eta_indices'] for k in p['seeds']]
    assert len(rows)==4608
    # Group by controller (avoids repeated JIT loads); each timing/request shard is balanced across all 12.
    rows.sort(key=lambda r:(r['controller_name'],r['state_index'],r['eta_index'],r['seed']))
    freeze(HERE/'stage_1/CASES.json',rows)
    groups=defaultdict(list)
    for r in rows:groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    freeze(HERE/'stage_1/PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=v) for (s,e,c),v in groups.items()]))
    # Register identities and experiment before official preflight; merge also verifies it.
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (p['experiment_uid'],'Round5 Flow blending regime pilot',str(HERE/'stage_1'),sha(HERE/'stage_1/protocol.json'),sha(__file__),canonical(dict(new_cap=5000))))
        for name in names:
            ctl=profiles[name];v=ctl['payload'];previous=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
            assert previous is None or previous[0]==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    print(json.dumps(dict(stage=1,controllers=12,requests=len(rows),experiment_registered=True)))

def inventory():
    folder=HERE/'stage_1';p=read(folder/'protocol.json');pre=read(folder/'CACHE_PREFLIGHT.json');cases=read(folder/'CASES.json')
    missing=[];unknown=[];cached=[]
    with connect(True) as db:
        for r in cases:
            x=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if x is None:missing.append(r)
            else:
                assert not x['conflict_quarantined']
                assert all(x[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                (unknown if x['numerical_failure'] else cached).append(r['rollout_uid'])
    assert pre['summary']['ambiguous']==0 and pre['summary']['genuinely_missing']==len(missing)+len(unknown)
    assert len(missing)<=5000
    freeze(folder/'MISSING_CASES.json',missing)
    freeze(folder/'CACHE_INVENTORY.json',dict(total=len(cases),missing=len(missing),cached_valid=len(cached),cached_unknown=len(unknown),quarantined=0,new_cap=5000,preflight_sha256=sha(folder/'CACHE_PREFLIGHT.json')))
    print(json.dumps(read(folder/'CACHE_INVENTORY.json')))

if __name__=='__main__':globals()[sys.argv[1]]()
