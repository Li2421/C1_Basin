from common import *
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
import copy

def normalized_state(s,validation=None):
 old=copy.deepcopy(s);physical={k:old['physical'][k] for k in ['positions','velocities','goals']}
 old['original_state_uid']=s['state_uid'];old['original_physical']=s['physical'];old['transport_physical']=physical
 # Future stream identity is unchanged: key_for uses original uid string, not DB UID.
 identity=dict(physical=physical,timestep=0,future_root=2026100403,key_token=int(hashlib.sha256(s['uid'].encode()).hexdigest()[:8],16))
 if validation:identity['validation']=validation
 old['content_hash']=uid('content',identity);old['state_uid']=uid('state',dict(scenario=s['scenario_uid'],content=old['content_hash']));return old

def register(stage,p,rows,profiles):
 prereg();inserted=[];tables=['experiment','state','controller_config','eta','scenario'];checks={}
 with connect() as db,transaction(db):
  before={t:[dict(r) for r in db.execute('SELECT * FROM '+t+' ORDER BY 1')] for t in tables};keys={t:db.execute('PRAGMA table_info('+t+')').fetchone()['name'] for t in tables}
  def ins(t,v):
   key=keys[t];old=db.execute('SELECT * FROM '+t+' WHERE '+key+'=?',(v[key],)).fetchone()
   if old:
    assert all(old[k]==x for k,x in v.items()),('UID_CONTENT_CONFLICT',t,v[key]);return
   db.execute('INSERT OR IGNORE INTO '+t+'('+','.join(v)+') VALUES('+','.join('?' for _ in v)+')',list(v.values()));inserted.append({'table':t,'row':dict(db.execute('SELECT * FROM '+t+' WHERE '+key+'=?',(v[key],)).fetchone())})
  ins('experiment',dict(experiment_uid=EXP,name='Round37 geometry addendum',path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(HERE/'registry.py'),metadata_json=canonical({'round':37})))
  for v in p['scenarios'].values():ins('scenario',v)
  for s in p['states']:
   ins('state',dict(state_uid=s['state_uid'],scenario_uid=s['scenario_uid'],source_group=s.get('family',s['uid']),content_hash=s['content_hash'],physical_state_json=canonical(s['transport_physical']),goals_geometry_json=canonical(s['evaluation_config']),provenance_json=canonical(dict(original_state_uid=s['original_state_uid'],original_physical=s['original_physical'],original_uid=s['uid'],rng_namespace=s['rng_namespace'],future_root=2026100403)),identity_quality='CONTENT_EXACT'))
  for c in profiles.values():
   v=c['payload'];cid=c['controller_uid'];old=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(cid,)).fetchone()
   if old:assert old[0]==canonical(v),('controller conflict',cid);continue
   ins('controller_config',dict(controller_uid=cid,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=v['conditioning'],rng_semantics_version=canonical(v['rng']),config_json=canonical(v),compatibility_quality='EXACT_PROFILE'))
  for e in [[0.,0.,0.]]+[p['eta'][i] for i in [10,13,15]]:
   eid,values,hx=eta_identity(e);old=db.execute('SELECT canonical_hex FROM eta WHERE eta_uid=?',(eid,)).fetchone()
   if old:assert old[0]==hx;continue
   ins('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))
  for t,records in before.items():
   h1=hashlib.sha256();h2=hashlib.sha256()
   for r in records:h1.update(canonical(r).encode());h2.update(canonical(dict(db.execute('SELECT * FROM '+t+' WHERE '+keys[t]+'=?',(r[keys[t]],)).fetchone())).encode())
   assert h1.digest()==h2.digest(),t;checks[t]=dict(rows=len(records),before=h1.hexdigest(),after=h2.hexdigest())
 write(HERE/f'{stage}_INSERTED_ROWS.json',inserted);seal(f'{stage}_REGISTRATION.receipt.json',[HERE/f'{stage}_INSERTED_ROWS.json'],preexisting_row_hashes=checks)
 write(HERE/f'{stage}_protocol.json',p);write(HERE/f'{stage}_CASES.json',rows);manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=preflight(manifest);write(HERE/f'{stage}_CACHE_PREFLIGHT.json',pf);assert pf['summary']['ambiguous']==0
 with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
 write(HERE/f'{stage}_MISSING_CASES.json',missing);seal(f'{stage}_PREFLIGHT.receipt.json',[manifest,HERE/f'{stage}_CACHE_PREFLIGHT.json',HERE/f'{stage}_protocol.json',HERE/f'{stage}_MISSING_CASES.json'],before_submission=True,planned=len(rows),missing=len(missing));print(stage,len(rows),len(missing),flush=True)
