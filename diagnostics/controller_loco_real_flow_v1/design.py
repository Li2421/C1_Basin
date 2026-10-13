"""Frozen, outcome-blind LOCO design and exact-cache inventory."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from diagnostics.controller_q_diversity_v1.experiment import case
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,eta_identity,uid

NAMES=['base_u_plus_w_v4_wide_macflow','base_u_macflow','base_u_plus_w_v3_macflow',
       'base_u_plus_w_v4_train_targeted_macflow','base_u_v10_local_macflow',
       'base_u_plus_w_v4_macflow','base_u_v9_local_macflow']
EXP='exp_controller_loco_real_flow_v1_stage_a'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(name,obj):
    p=HERE/name
    if p.exists():assert read(p)==obj,('immutable',str(p))
    else:p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def create():
    prereg=HERE/'PREREGISTRATION_STAGE_A.md';receipt=read(HERE/'PREREGISTRATION_STAGE_A.receipt.json')
    assert receipt['sha256']==sha(prereg)
    old=read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json')
    native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    train=generate('loco_train',128,2026140000,native)
    test=generate('loco_test',64,2026150000,native)
    states=train+test
    # UID/content collision check against the *entire* shared database, plus named protocols.
    with connect(True) as db:
        hits=[]
        for s in states:
            row=db.execute('SELECT state_uid FROM state WHERE state_uid=? OR content_hash=? LIMIT 1',(s['state_uid'],s['content_hash'])).fetchone()
            if row:hits.append((s['state_uid'],row['state_uid']))
        old_count=db.execute('SELECT count(*) n FROM state').fetchone()['n']
    named=[native,read(ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/protocol.json'),old,
           read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P2PRIME_PROTOCOL.json')]
    prior={s['state_uid'] for p in named for s in p['states']}
    assert not hits and not prior.intersection(s['state_uid'] for s in states)
    assert len({s['state_uid'] for s in states})==192
    freeze('STATE_DISJOINT_RECEIPT.json',dict(db_existing_states=old_count,db_collisions=hits,
          named_protocols=['native512','development256','P1_24','P2prime352'],named_uid_collisions=0,
          train=128,test=64,train_seed_start=2026140000,test_seed_start=2026150000))
    profiles={name:old['controllers']['ring_exchange'][name] for name in NAMES}
    assert all(profiles[n]['payload']['jax_enable_x64'] is False for n in NAMES)
    protocol=dict(schema='controller_loco_real_flow_stage_a_v1',experiment_uid=EXP,
        preregistration_sha256=sha(prereg),states=states,controllers={'ring_exchange':profiles},
        controller_order=NAMES,eta=old['eta'],eta_indices=list(range(16)),seeds=list(range(8)),
        environments=old['environments'],horizon=700,dt=.05,placement='FF',jax_enable_x64=False,
        prior_P1_protocol_sha256=sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json'),
        source_validation_protocol='diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json',
        flow_adapter_sha256=sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py'))
    freeze('protocol.json',protocol)
    # Complementary pairs guarantee exactly 64 appearances per η over 128 states.
    rng=np.random.default_rng(2026101204);masks={}
    for block in range(64):
        p=rng.permutation(16);masks[train[2*block]['state_uid']]=sorted(map(int,p[:8]));masks[train[2*block+1]['state_uid']]=sorted(map(int,p[8:]))
    assert [sum(j in m for m in masks.values()) for j in range(16)]==[64]*16
    freeze('TRAIN_ETA_MASK.json',masks)
    cases=[];groups=defaultdict(list)
    for name in NAMES:
        for s in states:
            for j in masks[s['state_uid']] if s['study_split']=='loco_train' else range(16):
                for seed in range(4 if s['study_split']=='loco_train' else 8):
                    r=case(protocol,s,name,j,seed);cases.append(r)
                    groups[(r['state_uid'],r['eta_uid'],r['controller_uid'])].append(r['seed_key'])
    assert len(cases)==86016
    cases.sort(key=lambda r:(r['controller_name'],r['split'],r['state_index'],r['eta_index'],r['seed']))
    freeze('CASES.json',cases)
    freeze('PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=v) for (s,e,c),v in groups.items()]))
    print(json.dumps(dict(states=len(states),controllers=len(NAMES),cases=len(cases),protocol_sha256=sha(HERE/'protocol.json'))))

def inventory():
    p=read(HERE/'protocol.json');cases=read(HERE/'CASES.json')
    prior=read(HERE/'CACHE_PREFLIGHT.json')['summary']
    with connect(True) as db:
        missing=[];cached=[];unknown=[];quarantine=[]
        for r in cases:
            z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if z is None:missing.append(r)
            elif z['conflict_quarantined']:quarantine.append(r['rollout_uid'])
            elif z['numerical_failure']:unknown.append(r['rollout_uid'])
            else:cached.append(r['rollout_uid'])
    assert not quarantine and prior['ambiguous']==0
    assert prior['genuinely_missing']==len(missing)+len(unknown)
    assert len(missing)<=90000
    freeze('MISSING_CASES.json',missing)
    freeze('CACHE_INVENTORY.json',dict(total=len(cases),missing=len(missing),cached_valid=len(cached),cached_unknown=len(unknown),quarantine=0,
        preflight_summary=prior,mean_seconds_prior=6.5,predicted_cpu_hours=len(missing)*6.5/3600,
        predicted_wall_hours_22=len(missing)*6.5/(22*3600),new_cap=90000))
    print(json.dumps(read(HERE/'CACHE_INVENTORY.json')))

def register():
    p=read(HERE/'protocol.json');states=p['states'];profiles=p['controllers']['ring_exchange']
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXP,'Real Flow LOCO Stage A',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cap=90000,cpu_workers=22))))
        row=db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()
        assert row['protocol_hash']==sha(HERE/'protocol.json')
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                       (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXP))
        for eta in p['eta']:
            eid,v,hx=eta_identity(eta)
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        for ctl in profiles.values():
            v=ctl['payload'];old=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
            assert old is None or old['config_json']==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    print('registered',EXP)

if __name__=='__main__':
    if sys.argv[1]=='create':create()
    elif sys.argv[1]=='inventory':inventory()
    elif sys.argv[1]=='register':register()
    else:raise ValueError(sys.argv)
