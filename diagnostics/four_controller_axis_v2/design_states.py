"""Freeze 16 probe and 32 evaluation Four states after hard gates."""
from __future__ import annotations
import datetime,hashlib,json,sqlite3,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import canonical,uid
from four_way_intersection.environment import Config
from four_way_intersection.scenario import sample_initial_state,build_layout
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sig(p):
    if not isinstance(p,dict):return None
    a=[]
    for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
        v=next((p[k] for k in keys if k in p),None)
        if v is None:return None
        x=np.asarray(v,np.float64).copy();x[x==0]=0;a.append((x.shape,x.tobytes().hex()))
    return hashlib.sha256(canonical(dict(arr=a,timestep=float(p.get('timestep',p.get('step',0))))).encode()).hexdigest()
def main():
    for gate in ('G0','G1A','G2'):
        assert read(HERE/f'{gate}_REPORT.json')['status']=='PASS'
    assert read(HERE/'PREREGISTRATION.receipt.json')['preregistration_sha256']==sha(HERE/'PREREGISTRATION.md')
    old=read(ROOT/'diagnostics/controller_ff_factorial_repair_v1/protocol.json')
    info=old['environments']['four_way_intersection'];config=Config(**info['config']['config']);sid=info['scenario_uid']
    goals=build_layout(config).goals.tolist();states=[]
    for split,count,base_seed in (('probe',16,2026101402),('census',32,2026102403)):
        for i in range(count):
            pos,vel,meta=sample_initial_state(config,'development',i,base_seed=base_seed)
            physical=dict(positions=pos.tolist(),velocities=vel.tolist(),goals=goals,timestep=0)
            alias=f'round19b_four_{split}_{i:03d}'
            content=uid('content',dict(physical=physical,reset='empty monitor true-t0'))
            states.append(dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),content_hash=content,
                               scenario='four_way_intersection',scenario_uid=sid,physical=physical,
                               family='round19b_four_'+split,study_split=split,study_index=i,metadata=meta,
                               rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16)))
    signatures={sig(x['physical']) for x in states}
    assert len(signatures)==48
    conn=sqlite3.connect(f'file:{ROOT/"shared_rollout_db/rollout.sqlite"}?mode=ro',uri=True)
    rows=conn.execute('SELECT state_uid,content_hash,physical_state_json,identity_quality FROM state').fetchall()
    collisions=[];recoverable=0;unresolved={}
    uids={x['state_uid'] for x in states};contents={x['content_hash'] for x in states}
    for state_uid,content_hash,physical_json,quality in rows:
        try:p=json.loads(physical_json) if physical_json else None
        except ValueError:p=None
        h=sig(p)
        if h is None:unresolved[quality]=unresolved.get(quality,0)+1
        else:recoverable+=1
        if h in signatures or state_uid in uids or content_hash in contents:collisions.append(state_uid)
    assert not collisions,collisions
    write(HERE/'FRESH_STATES.json',states)
    audit=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               draw_seed_probe=2026101402,draw_seed_census=2026102403,
               fresh_probe=16,fresh_census=32,db_rows=len(rows),recoverable_db_physical_states=recoverable,
               unresolved_db_rows=unresolved,collisions=collisions,
               new_labels_created=0,g0_sha256=sha(HERE/'G0_REPORT.json'),g1a_sha256=sha(HERE/'G1A_REPORT.json'),g2_sha256=sha(HERE/'G2_REPORT.json'))
    write(HERE/'STATE_DEDUP_AUDIT.json',audit)
    write(HERE/'STATE_FREEZE.receipt.json',dict(utc=audit['utc'],fresh_states_sha256=sha(HERE/'FRESH_STATES.json'),audit_sha256=sha(HERE/'STATE_DEDUP_AUDIT.json')))
    print(json.dumps(dict(status='PASS',fresh_probe=16,fresh_census=32,collisions=0)),flush=True)
if __name__=='__main__':main()
