from design import *
from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv,sample_instance
OLD=D/'axis4a_geometry_horizon_v1'
def signature(v):
 try:
  parts=[]
  for k in ['positions','velocities','goals']:
   a=np.asarray(v[k],np.float64).copy();a[a==0]=0;parts.append([a.shape,a.tobytes().hex()])
  return uid('physical',parts)
 except (TypeError,KeyError,ValueError):return None

def gate():
 p=protocol();rows=[];ss=read(HERE/'EVAL_STATES_ORIGINAL.json');refs=read(HERE/'SOURCE_CORRECTIONS.json');look={(q['case']['base_controller_name'],q['case']['state_uid'],q['case']['arm'],q['case']['seed']):q for q in refs};names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']
 spec=[(n,0,'F10',0) for n in names]+[(n,1,'F13',1) for n in ['base_u_macflow','base_u_v10_local_macflow']]
 for n,i,a,seed in spec:
  s=state(p,ss[i],700,.9,'round37_R10_gate');c=case(p,s,n,700,.9,a,seed,'round37_R10_gate');q=look[n,ss[i]['state_uid'],a,seed];c.update(reference_path=q['raw_path'],reference_sha256=q['raw_sha256'],reference_group='R10');rows.append(c)
 pp=read(OLD/'pilot_protocol.json');oldstates={s['state_uid']:s for s in pp['states']};cc=read(OLD/'pilot_CASES.json');radii=[.8,.85,.95,1.];spec=[(n,radii[i%4],0,'F10') for i,n in enumerate(names)]+[('base_u_macflow',.85,1,'S500'),('base_u_v10_local_macflow',1.,1,'S500')]
 for n,r,i,a in spec:
  candidates=[c for c in cc if c['base_controller_name']==n and c['r']==r and c['arm']==a];candidates.sort(key=lambda c:oldstates[c['state_uid']]['draw_seed']);ref=candidates[i];s=state(p,oldstates[ref['state_uid']],700,r,'round37_R35_geometry_gate');c=case(p,s,n,700,r,a,0,'round37_R35_geometry_gate');f=OLD/'pilot/raw'/f"{ref['rollout_uid']}.json";c.update(reference_path=str(f),reference_sha256=sha(f),reference_group='R35_geometry');rows.append(c)
 register('gate',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})

def A1():
 p=protocol();rows=[];states=[];seen=set();unrecoverable=0;oldcount=0
 with rollout_db.connect(True) as db:
  for q in db.execute('SELECT state_uid,physical_state_json FROM state'):
   oldcount+=1
   try:sig=signature(json.loads(q['physical_state_json']))
   except (TypeError,ValueError):sig=None
   if sig:seen.add(sig)
   else:unrecoverable+=1
 for s in read(OLD/'GEOMETRY_PILOT_STATES.json'):seen.add(signature(s['physical']))
 attempts=[];candidate=202610354000
 while len(states)<16:
  seed=candidate;candidate+=1;cc=LocalFrameConfig(**config(700,.9));ins=sample_instance(seed,'development',cc);env=RingExchangeEnv(cc);env.reset(ins.positions,velocities=ins.velocities,goals=ins.goals);physical=clean(dict(positions=ins.positions,velocities=ins.velocities,goals=ins.goals));sig=signature(physical);dup=sig in seen;attempts.append(dict(seed=seed,duplicate=dup,physical_hash=sig))
  if dup:continue
  seen.add(sig);alias=f'round35_geometry_pilot_r0.9_{seed}';s=dict(uid=alias,state_uid=uid('state',alias),scenario_uid='new',physical=physical,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),study_index=len(states),family=alias);s=state(p,s,700,.9);s.update(draw_seed=seed,split='pilot',radius=.9);states.append(s)
  for n in read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']:
   for a in ['F10','S500']:
    c=case(p,s,n,700,.9,a,0);c['state_index']=s['study_index'];rows.append(c)
 write(HERE/'A1_STATES.json',states);write(HERE/'A1_STATE_DEDUP.json',dict(utc=utc(),passed=True,old_DB_states=oldcount,old_unrecoverable=unrecoverable,round35_pilots=64,attempts=attempts,accepted=16))
 register('A1',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})

def A2():
 p=protocol();rows=[];checks=[];ss=sorted(read(HERE/'EVAL_STATES_ORIGINAL.json'),key=lambda s:s['state_uid'])[:16]
 for r in [.8,.85,.95,1.]:
  for i,original in enumerate(ss):
   check=dict(radius=r,state_uid=original['state_uid'],state_index=i)
   try:
    env=RingExchangeEnv(LocalFrameConfig(**config(700,r)));z=original['physical'];env.reset(np.asarray(z['positions']),velocities=np.asarray(z['velocities']),goals=np.asarray(z['goals']));check['valid']=True
   except Exception as ex:check.update(valid=False,error=repr(ex));checks.append(check);continue
   checks.append(check);s=state(p,original,700,r);s['study_index']=i
   for n in read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']:
    for a in ['F10','S500']:
     c=case(p,s,n,700,r,a,0);c['state_index']=i;rows.append(c)
 write(HERE/'A2_RESET_VALIDITY.json',checks);write(HERE/'A2_BASELINE_REFERENCES.json',[q for q in read(HERE/'SOURCE_CORRECTIONS.json') if q['case']['state_uid'] in {s['state_uid'] for s in ss} and q['case']['arm'] in ['F10','S500'] and q['case']['seed']==0]);register('A2',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
def GA3():
 eligibility=read(HERE/'CONTROL_RESULTS.json');assert (HERE/'GEOMETRY_ELIGIBILITY_FROZEN.receipt.json').exists();radii=eligibility['GA3']['eligible_radii'];assert eligibility['GA1']['status']=='GENERATOR_INVALID' or not radii
 if not radii:write(HERE/'GA3_NO_REQUESTS.json',dict(utc=utc(),reason='Prespecified conditional trigger absent'));return
 p=protocol();rows=[];checks=[]
 for r in radii:
  for i,original in enumerate(read(HERE/'EVAL_STATES_ORIGINAL.json')):
   try:
    z=original['physical'];env=RingExchangeEnv(LocalFrameConfig(**config(700,r)));env.reset(np.asarray(z['positions']),velocities=np.asarray(z['velocities']),goals=np.asarray(z['goals']));checks.append(dict(r=r,index=i,valid=True))
   except Exception as ex:checks.append(dict(r=r,index=i,valid=False,error=repr(ex)));continue
   s=state(p,original,700,r);s['study_index']=i
   for n in read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']:
    for a in ['nominal_only','safety_baseline','F10','F13','S500']:
     for seed in [0,1]:
      c=case(p,s,n,700,r,a,seed);c['state_index']=i;rows.append(c)
 write(HERE/'GA3_RESET_VALIDITY.json',checks);register('GA3',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
if __name__=='__main__':globals()[sys.argv[1]]()
