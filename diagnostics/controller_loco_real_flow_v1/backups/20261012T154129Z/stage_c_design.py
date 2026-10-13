"""Conditional independent confirmation under the 10,000-rollout cap."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent
HERE=BASE/'stage_c'
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from diagnostics.controller_q_diversity_v1.experiment import case
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

EXP='exp_controller_loco_real_flow_v1_stage_c'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==x,('immutable',p)
    else:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def create():
    a=read(BASE/'protocol.json');gate=read(BASE/'GATE_RESULTS.json');result=read(BASE/'stage_b/results.json')
    necessary=gate['necessary_folds'];passed=result['primary']['passing_arms']
    assert len(necessary)>=2 and passed,'Stage C gate not satisfied'
    native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    states=generate('loco_confirmation',64,2026160000,native)
    with connect(True) as db:
        collisions=[s['state_uid'] for s in states if db.execute('SELECT 1 FROM state WHERE state_uid=? OR content_hash=? LIMIT 1',(s['state_uid'],s['content_hash'])).fetchone()]
    assert not collisions and len({s['state_uid'] for s in states})==64
    alloc={int(f):64//len(necessary)+(i<64%len(necessary)) for i,f in enumerate(necessary)}
    assert sum(alloc.values())==64
    p=dict(schema='controller_loco_real_flow_stage_c_v1',experiment_uid=EXP,states=states,
           controllers=a['controllers'],controller_order=a['controller_order'],held_folds=necessary,
           passing_arms=passed,allocation=alloc,eta=a['eta'],eta_indices=list(range(16)),seeds=list(range(8)),
           environments=a['environments'],horizon=700,dt=.05,placement='FF',jax_enable_x64=False,
           preregistration_sha256=sha(BASE/'PREREGISTRATION_STAGE_C_AFTER_B.md'),
           stage_a_gate_sha256=sha(BASE/'GATE_RESULTS.json'),stage_b_results_sha256=sha(BASE/'stage_b/results.json'),
           flow_adapter_sha256=sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py'))
    freeze(HERE/'protocol.json',p)
    freeze(HERE/'STATE_DISJOINT_RECEIPT.json',dict(new_states=64,db_collisions=collisions,seed_start=2026160000))
    cases=[];groups=defaultdict(list)
    for fold in necessary:
        name=a['controller_order'][fold]
        for s in states[:alloc[fold]]:
            for j in range(16):
                for k in range(8):
                    r=case(p,s,name,j,k);cases.append(r)
                    groups[(r['state_uid'],r['eta_uid'],r['controller_uid'])].append(r['seed_key'])
    assert len(cases)==8192
    cases.sort(key=lambda r:(r['controller_name'],r['state_index'],r['eta_index'],r['seed']))
    freeze(HERE/'CASES.json',cases)
    freeze(HERE/'PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=v) for (s,e,c),v in groups.items()]))
    print(json.dumps(dict(allocation=alloc,cases=len(cases))))

def inventory():
    pre=read(HERE/'CACHE_PREFLIGHT.json')['summary'];cases=read(HERE/'CASES.json')
    with connect(True) as db:
        missing=[];cached=[];unknown=[];quarantine=[]
        for r in cases:
            z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if z is None:missing.append(r)
            elif z['conflict_quarantined']:quarantine.append(r['rollout_uid'])
            elif z['numerical_failure']:unknown.append(r['rollout_uid'])
            else:cached.append(r['rollout_uid'])
    assert not quarantine and pre['ambiguous']==0 and pre['genuinely_missing']==len(missing)+len(unknown)
    assert len(missing)<=10000
    freeze(HERE/'MISSING_CASES.json',missing)
    freeze(HERE/'CACHE_INVENTORY.json',dict(total=len(cases),missing=len(missing),cached_valid=len(cached),
        cached_unknown=len(unknown),quarantine=0,new_cap=10000,preflight_summary=pre))
    print(json.dumps(read(HERE/'CACHE_INVENTORY.json')))

def register():
    p=read(HERE/'protocol.json')
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXP,'Real Flow LOCO Stage C independent confirmation',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cap=10000,cpu_workers=19))))
        row=db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()
        assert row['protocol_hash']==sha(HERE/'protocol.json')
        for s in p['states']:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                       (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXP))
    print('registered',EXP)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='create':create()
    elif cmd=='inventory':inventory()
    elif cmd=='register':register()
    else:raise ValueError(cmd)
