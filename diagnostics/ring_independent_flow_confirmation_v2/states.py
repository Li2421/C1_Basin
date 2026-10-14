from core import *
from shared_rollout_db.src.rollout_db import uid,canonical,connect
from collections import Counter
from ring_exchange.environment import LocalFrameConfig,sample_instance
prereg();old=read(R10/'protocol.json');sid=old['environments']['ring_exchange']['scenario_uid'];cfg=LocalFrameConfig(**old['environments']['ring_exchange']['config']['config']);states=[]
for i in range(64):
 seed=2026142400+i;ins=sample_instance(seed,'development',cfg);physical=dict(positions=ins.positions.tolist(),velocities=ins.velocities.tolist(),goals=ins.goals.tolist(),timestep=0);alias=f'round14a_ring_independent_{i}';content=uid('content',dict(physical=physical,reset='empty monitor true-t0'));states.append(dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),content_hash=content,scenario='ring_exchange',scenario_uid=sid,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),physical=physical,family='round14a_ring_independent',study_split='prospective_round14a',study_index=i,metadata=dict(draw_seed=seed,outcome_blind=True)))
def sig(p):
 if not isinstance(p,dict):return None
 v=[]
 for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
  a=next((p[k] for k in keys if k in p),None)
  if a is None:return None
  a=np.asarray(a,np.float64).copy();a[a==0]=0;v.append((a.shape,a.tobytes().hex()))
 return hashlib.sha256(canonical(dict(arr=v,timestep=float(p.get('timestep',p.get('step',0))))).encode()).hexdigest()
sigs={sig(s['physical']) for s in states};assert len(sigs)==64;collisions=[];unresolved=Counter();recoverable=0
with connect(True) as db:
 historical=db.execute('SELECT state_uid,content_hash,physical_state_json,identity_quality FROM state').fetchall()
 for r in historical:
  try:p=json.loads(r['physical_state_json']) if r['physical_state_json'] else None
  except ValueError:p=None
  h=sig(p)
  if h is None:unresolved[r['identity_quality']]+=1
  else:recoverable+=1
  if h in sigs or r['state_uid'] in {s['state_uid'] for s in states} or r['content_hash'] in {s['content_hash'] for s in states}:collisions.append(r['state_uid'])
checked=[]
for path in ROOT.glob('diagnostics/**/protocol.json'):
 if path.is_relative_to(HERE):continue
 q=read(path)
 if not isinstance(q,dict):continue
 rows=q.get('states',[]);rows=list(rows.values()) if isinstance(rows,dict) else rows
 if not isinstance(rows,list):continue
 for s in rows:
  if isinstance(s,dict) and sig(s.get('physical')) in sigs:collisions.append(str(path))
 checked.append(dict(path=str(path),sha256=sha(path)))
assert not collisions;freeze(HERE/'FRESH_STATES.json',states);freeze(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=True,fresh=64,all_DB_rows=len(historical),recoverable=recoverable,unresolved=dict(unresolved),collisions=collisions,historical_protocols=checked,no_target_labels=True))
print('Fresh64 dedup PASS')
