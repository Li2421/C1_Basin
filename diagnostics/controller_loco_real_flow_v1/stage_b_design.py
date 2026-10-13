"""Conditional seven-additional-source design; never called unless Stage A fails."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent
HERE=BASE/'stage_b'
from diagnostics.controller_q_diversity_v1.experiment import case
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

EXP='exp_controller_loco_real_flow_v1_stage_b'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==x,('immutable',p)
    else:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def create():
    gate=read(BASE/'GATE_RESULTS.json');results=read(BASE/'results.json')
    assert len(gate['necessary_folds'])>=2 and not results['primary']['passing_arms'], 'Stage B gate not satisfied'
    a=read(BASE/'protocol.json');p1=read(ROOT/'diagnostics/controller_q_transfer_round3_v1/P1_EXECUTION_PROTOCOL.json')
    names=[x for x in p1['controller_order'] if x not in a['controller_order']]
    assert len(names)==7
    profiles={n:p1['controllers']['ring_exchange'][n] for n in names}
    p=dict(schema='controller_loco_real_flow_stage_b_v1',experiment_uid=EXP,states=a['states'][:128],
           controllers={'ring_exchange':profiles},controller_order=names,eta=a['eta'],eta_indices=list(range(16)),seeds=list(range(4)),
           environments=a['environments'],horizon=700,dt=.05,placement='FF',jax_enable_x64=False,
           preregistration_sha256=sha(BASE/'PREREGISTRATION_CONDITIONAL_BRANCHES.md'),
           stage_a_gate_sha256=sha(BASE/'GATE_RESULTS.json'),stage_a_results_sha256=sha(BASE/'results.json'),
           flow_adapter_sha256=sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py'))
    freeze(HERE/'protocol.json',p)
    masks=read(BASE/'TRAIN_ETA_MASK.json');cases=[];groups=defaultdict(list)
    for name in names:
        for s in p['states']:
            for j in masks[s['state_uid']]:
                for k in range(4):
                    r=case(p,s,name,j,k);cases.append(r)
                    groups[(r['state_uid'],r['eta_uid'],r['controller_uid'])].append(r['seed_key'])
    assert len(cases)==28672
    cases.sort(key=lambda r:(r['controller_name'],r['state_index'],r['eta_index'],r['seed']))
    freeze(HERE/'CASES.json',cases)
    freeze(HERE/'PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=v) for (s,e,c),v in groups.items()]))
    print(json.dumps(dict(names=names,cases=len(cases))))

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
    assert len(missing)<=30000
    freeze(HERE/'MISSING_CASES.json',missing)
    freeze(HERE/'CACHE_INVENTORY.json',dict(total=len(cases),missing=len(missing),cached_valid=len(cached),
        cached_unknown=len(unknown),quarantine=0,new_cap=30000,preflight_summary=pre))
    print(json.dumps(read(HERE/'CACHE_INVENTORY.json')))

def register():
    p=read(HERE/'protocol.json')
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXP,'Real Flow LOCO Stage B additional source',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(cap=30000,cpu_workers=19))))
        row=db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()
        assert row['protocol_hash']==sha(HERE/'protocol.json')
    print('registered',EXP)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='create':create()
    elif cmd=='inventory':inventory()
    elif cmd=='register':register()
    else:raise ValueError(cmd)
