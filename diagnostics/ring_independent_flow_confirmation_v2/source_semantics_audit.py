from core import *
prereg();old=read(R10/'protocol.json');checks=[]
for name,ctl in old['base_profiles'].items():
 v=ctl['payload']
 for f,h in v.get('action_code_hashes',{}).items():
  path=Path(f) if Path(f).is_absolute() else FIELD/f;assert sha(path)==h,(name,path);checks.append(dict(controller=name,path=str(path),sha256=h))
write(HERE/'SOURCE_SEMANTICS_AUDIT.json',dict(utc=utc(),passed=True,checks=checks,no_target_labels_read=True))
# Supplement old pool states to the immutable original dedup audit.
from shared_rollout_db.src.rollout_db import canonical
states=read(HERE/'FRESH_STATES.json');sig=lambda p:canonical(p) if isinstance(p,dict) else None;target={sig(s['physical']) for s in states};checked=0;collisions=[]
for path in ROOT.glob('diagnostics/**/protocol.json'):
 if path.is_relative_to(HERE):continue
 q=read(path)
 if not isinstance(q,dict):continue
 pool=q.get('pool',[]);pool=list(pool.values()) if isinstance(pool,dict) else pool
 if not isinstance(pool,list):continue
 for row in pool:
  if not isinstance(row,dict):continue
  s=row.get('state',row);p=s.get('physical') if isinstance(s,dict) else None
  if p is None:continue
  checked+=1
  if sig(p) in target:collisions.append(dict(path=str(path),state=s.get('uid',s.get('state_uid'))))
assert not collisions;write(HERE/'HISTORICAL_POOL_DEDUP_SUPPLEMENT.json',dict(utc=utc(),passed=True,pool_physical_rows_checked=checked,collisions=collisions,no_redrawing_or_target_labels=True))
print('Historical action-code hashes and pool dedup PASS')
