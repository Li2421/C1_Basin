from common import *
import copy
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from ring_exchange.environment import RingExchangeEnv,LocalFrameConfig
from shift_generator import sample_instance
Q=read(D/'controller_universal_eta_schedule_v1/protocol.json')
def sig(v):
 try:
  arrays=[]
  for names in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
   a=np.asarray(next(v[k] for k in names if k in v),np.float64).copy();assert a.ndim==2 and a.shape[1]==2;a[a==0]=0;arrays.append([list(a.shape),a.tobytes().hex()])
  return uid('physical',{'arrays':arrays,'timestep':float(v.get('timestep',v.get('step',0)))})
 except (TypeError,KeyError,StopIteration,AssertionError):return None

def register(p,rows,phase):
 inserted=[];tables=['experiment','state','controller_config','eta','scenario'];before={};keys={}
 with connect() as db,transaction(db):
  for t in tables:
   rr=[dict(x) for x in db.execute('SELECT * FROM '+t+' ORDER BY 1')];before[t]=rr;keys[t]=list(rr[0])[0] if rr else db.execute('PRAGMA table_info('+t+')').fetchone()['name']
  def ins(table,values):
   key=keys[table];old=db.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(values[key],)).fetchone()
   if old:
    assert all(old[k]==v for k,v in values.items()),('UID_CONTENT_CONFLICT',table,values[key]);return
   db.execute('INSERT OR IGNORE INTO '+table+'('+','.join(values)+') VALUES('+','.join('?' for _ in values)+')',list(values.values()))
   inserted.append({'table':table,'row':dict(db.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(values[key],)).fetchone())})
  ins('experiment',dict(experiment_uid=EXP,name='Round32 State axis',path=str(HERE),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(HERE/'design.py'),metadata_json=canonical({'round':32})))
  for s in p['states']:
   if phase=='main' and s.get('cohort')=='gate':continue
   physical={k:v for k,v in s['physical'].items() if k!='timestep'}
   ins('state',dict(state_uid=s['state_uid'],scenario_uid=s['scenario_uid'],source_group=s['family'],content_hash=s['content_hash'],physical_state_json=canonical(physical),goals_geometry_json=canonical(Q['environments']['ring_exchange']['config']),provenance_json=canonical(dict(metadata=s['metadata'],rng_namespace=s['rng_namespace'],future_root=2026100403,original_physical=s['physical'])),identity_quality='CONTENT_EXACT'))
  for ctl in p['registered_controllers'].values():
   v=ctl['payload'];cid=ctl['controller_uid'];existing=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(cid,)).fetchone()
   if existing:
    assert existing['config_json']==canonical(v),('Controller conflict',cid);continue
   ins('controller_config',dict(controller_uid=cid,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v['flow_checkpoint_sha256'],orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version=v['conditioning'],rng_semantics_version=canonical(v['rng']),config_json=canonical(v),compatibility_quality='EXACT_PROFILE'))
  for e in [[0.,0.,0.],p['eta'][10],p['eta'][13]]:
   eid,values,hx=eta_identity(e)
   old=db.execute('SELECT * FROM eta WHERE eta_uid=?',(eid,)).fetchone()
   if old:assert old['canonical_hex']==hx;continue
   ins('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))
  checks={}
  for t,rr in before.items():
   key=keys[t];h=hashlib.sha256();h2=hashlib.sha256()
   for row in rr:
    h.update(canonical(row).encode());current=dict(db.execute('SELECT * FROM '+t+' WHERE '+key+'=?',(row[key],)).fetchone());h2.update(canonical(current).encode())
   assert h.digest()==h2.digest(),t;checks[t]={'preexisting_rows':len(rr),'before_sha256':h.hexdigest(),'after_sha256':h2.hexdigest()}
 write(HERE/f'{phase}_INSERTED_ROWS.json',inserted);seal(f'{phase}_REGISTRATION.receipt.json',[HERE/f'{phase}_INSERTED_ROWS.json'],preexisting_content_checks=checks)
 write(HERE/f'{phase}_CASES.json',rows)
 manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,{'requests':[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]});pf=preflight(manifest);write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pf);assert pf['summary']['ambiguous']==0
 with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',(c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'])).fetchone() is None]
 write(HERE/f'{phase}_MISSING_CASES.json',missing);seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_MISSING_CASES.json'],before_submission=True,missing=len(missing));print(phase,len(rows),len(missing),'missing',flush=True)

def case(s,name,arm,seed,ctl,eta,t=700):
 sk=canonical({'future_index':seed});eid=eta_identity(eta)[0];cid=ctl['controller_uid']
 return dict(state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),base_controller_name=name,arm=arm,seed=seed,eta_a=eta,eta_b=Q['eta'][10] if arm.startswith('S') else eta,switch_step=t,state_index=s['study_index'],cohort=s['cohort'])

def setup():
 prereg();native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');cfg=LocalFrameConfig(**native['environment']['config']['config'])
 with connect(True) as db:history=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
 ids={x['state_uid'] for x in history};contents={x['content_hash'] for x in history};sigs=set();unresolved=0
 for row in history:
  signature=sig(json.loads(row['physical_state_json']) if row['physical_state_json'] else None)
  if signature is None:unresolved+=1
  else:sigs.add(signature)
 states=generate('round32_axis1_id',64,202610320000,native)
 for s in states:
  assert s['state_uid'] not in ids and s['content_hash'] not in contents and sig(s['physical']) not in sigs,'1a duplicate';sigs.add(sig(s['physical']));s['cohort']='1a';RingExchangeEnv(cfg).reset(**{k:np.asarray(v) for k,v in s['physical'].items() if k!='timestep'})
 shifted=[];reject=[]
 for seed in range(202610321000,202610322000):
  inst=sample_instance(seed,'round32_shift',cfg)
  try:RingExchangeEnv(cfg).reset(instance=inst)
  except ValueError as exc:reject.append({'seed':seed,'reason':str(exc)});continue
  physical=dict(positions=inst.positions.tolist(),velocities=inst.velocities.tolist(),goals=inst.goals.tolist(),timestep=0);content=uid('content',dict(physical=physical,reset='empty monitor true-t0'));sid=uid('state',dict(scenario=states[0]['scenario_uid'],content=content));signature=sig(physical)
  if sid in ids or content in contents or signature in sigs:reject.append({'seed':seed,'reason':'duplicate'});continue
  alias=f'round32_axis1_shift_{seed}';shifted.append(dict(uid=alias,state_uid=sid,scenario='ring_exchange',scenario_uid=states[0]['scenario_uid'],content_hash=content,physical=physical,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),family=alias,metadata=dict(draw_seed=seed,split='round32_shift',SeedSequence_tag=32015491),study_index=len(shifted),cohort='1b'));sigs.add(signature)
  if len(shifted)==32:break
 assert len(shifted)==32;states+=shifted
 write(HERE/'EVAL_STATES.json',states);write(HERE/'STATE_DEDUP_AUDIT.json',dict(passed=True,DB_rows=len(history),unrecoverable_physical=unresolved,rejections=reject,scope='All DB identity/content and recoverable physical triples; legacy unknown coordinates remain a gap'))
 write(HERE/'HISTORY_AUDIT.json',dict(sources=['R10 protocol+actual main_CASES+full traces','R16 frozen fits/transform','R7 frozen FEATURES R5','R26 aligned T2 predictions and analysis','R2 REPORT'],reuse='Existing best checkpoints, R5, frozen A3/A0/G_E; historical endpoints DEVELOPMENT; no training or redundant probes',new_gap='96 previously unseen states require new labels',known_issue='Do not use obsolete R10 FRESH_STATES; R26 aligned T2 corrected source',DB_rows=len(history)))
 # Exact16 historical R10 F10 cases, predetermined controllers/states/seeds.
 names=['base_u_macflow','base_u_v10_local_macflow','base_u_v9_local_macflow','base_u_plus_w_v4_wide_macflow'];oldrows=read(D/'controller_universal_eta_schedule_v1/main_CASES.json');first={s['state_uid'] for s in Q['fresh_states'][:2]};refs=[c for c in oldrows if c['base_controller_name'] in names and c['state_uid'] in first and c['seed'] in [0,1] and c['arm']=='F10'];assert len(refs)==16
 gstates=[];mapping={}
 for s0 in Q['fresh_states'][:2]:
  s=copy.deepcopy(s0);s['state_uid']=uid('state',dict(round32_gate=s0['state_uid']));s['content_hash']=uid('content',dict(round32_gate=s0['content_hash']));s['family']='round32_reproduction_'+s0['family'];s['cohort']='gate';gstates.append(s);mapping[s0['state_uid']]=s
 p=dict(states=states+gstates,base_profiles=Q['base_profiles'],registered_controllers={n:Q['base_profiles'][n] for n in names},eta=Q['eta'],controller_order=Q['controller_order']);rows=[]
 for c in refs:
  s=mapping[c['state_uid']];row=case(s,c['base_controller_name'],'F10',c['seed'],Q['base_profiles'][c['base_controller_name']],Q['eta'][10]);row['reference_rollout_uid']=c['rollout_uid'];rows.append(row)
 write(HERE/'protocol.json',p);register(dict(p,states=gstates),rows,'gate')

def main_design():
 prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed'];p=read(HERE/'protocol.json');rows=[]
 for n in p['controller_order']:
  for arm in ['nominal_only','safety_baseline','F10','F13','S300','S500']:
   eta=[0.,0.,0.] if arm in ['nominal_only','safety_baseline'] else p['eta'][13 if arm.startswith('S') or arm=='F13' else 10];ctl=copy.deepcopy(Q['base_profiles'][n])
   if arm.startswith('S'):
    key=f'{n}__eta13to10__t{int(arm[1:])}';ctl=copy.deepcopy(Q['controllers']['ring_exchange'][key])
   if arm=='nominal_only':
    v=ctl['payload'];v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round32_nominal_only_v1',nominal_source_sha256=sha(HERE/'arms.py'));ctl['controller_uid']=uid('ctl',v)
   p['registered_controllers'][n+'__'+arm]=ctl
   for s in p['states']:
    if s['cohort']=='gate':continue
    for seed in [0,1]:rows.append(case(s,n,arm,seed,ctl,eta,int(arm[1:]) if arm.startswith('S') else 700))
 assert len(rows)==16128;write(HERE/'protocol.json',p);register(p,rows,'main')
if __name__=='__main__':globals()[sys.argv[1]]()
