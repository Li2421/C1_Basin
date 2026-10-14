from core import *
from collections import Counter
from shared_rollout_db.src.rollout_db import connect,canonical

def signature(v):
    if not isinstance(v,dict):return None
    values=[]
    for fields in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
        value=next((v[k] for k in fields if k in v),None)
        if value is None:return None
        a=np.asarray(value,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():return None
        a=a.copy();a[a==0]=0;values.append((a.shape,a.tobytes().hex()))
    return hashlib.sha256(canonical(dict(values=values,timestep=float(v.get('timestep',v.get('step',0))))).encode()).hexdigest()

def main():
    prereg()
    from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
    native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json');states=generate('round12_final_confirmation_fresh',64,2026210000,native);freeze(HERE/'FRESH_STATES.json',states);sig={signature(s['physical']):s['state_uid'] for s in states};assert None not in sig and len(sig)==64;ids={s['state_uid'] for s in states};content={s['content_hash'] for s in states};collisions=[];unresolved=Counter();recoverable=0
    with connect(True) as db:old=db.execute('SELECT state_uid,content_hash,physical_state_json,identity_quality FROM state').fetchall()
    for r in old:
        if r['state_uid'] in ids or r['content_hash'] in content:collisions.append(dict(source='DB UID/content',state_uid=r['state_uid']))
        try:v=json.loads(r['physical_state_json']) if r['physical_state_json'] else None
        except ValueError:v=None
        h=signature(v)
        if h is None:unresolved[r['identity_quality']]+=1
        else:
            recoverable+=1
            if h in sig:collisions.append(dict(source='DB physical',state_uid=r['state_uid'],fresh_state_uid=sig[h]))
    checked=[];physical_count=0
    for path in ROOT.glob('diagnostics/**/protocol.json'):
        if path.is_relative_to(HERE):continue
        q=read(path)
        if not isinstance(q,dict):continue
        rows=q.get('states',[]);rows=list(rows.values()) if isinstance(rows,dict) else rows;pool=q.get('pool',[]);pool=list(pool.values()) if isinstance(pool,dict) else pool
        if not isinstance(rows,list):rows=[]
        if not isinstance(pool,list):pool=[]
        for s in rows+[v.get('state',v) for v in pool if isinstance(v,dict)]:
            if not isinstance(s,dict):continue
            h=signature(s.get('physical'))
            if h is None:continue
            physical_count+=1
            if h in sig:collisions.append(dict(source=str(path),fresh_state_uid=sig[h],historical_state_uid=s.get('state_uid',s.get('uid'))))
        checked.append(dict(path=str(path),sha256=sha(path)))
    freeze(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=not bool(collisions),fresh=64,seed_start=2026210000,seed_end=2026210063,all_DB_identity_rows_scanned=len(old),recoverable_physical_DB_rows=recoverable,unresolved_physical_DB_rows=unresolved,historical_protocols=checked,historical_protocol_physical_rows=physical_count,collisions=collisions,success_labels_read_or_generated=False,generator_sha256=sha(P1/'state_generator.py')));assert not collisions,'Overlap: STOP main, no adaptive redraw'
    print(json.dumps(dict(fresh=64,DB_rows=len(old),recoverable=recoverable,collisions=len(collisions))))
if __name__=='__main__':main()
