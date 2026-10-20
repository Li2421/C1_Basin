from factor_design import *

def signature(v):
 try:
  data=[]
  for k in ['positions','velocities','goals']:
   a=np.asarray(v[k],np.float64).copy();a[a==0]=0;data.append([a.shape,a.tobytes().hex()])
  return uid('physical',data)
 except (TypeError,KeyError,ValueError):return None

def pilot():
 assert (HERE/'HORIZON_NUMERICAL_COMPLETE.receipt.json').exists()
 from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv,sample_instance
 cfg=read(HERE/'DESIGN_CONFIG.json');p=protocol();rows=[];generated=[]
 with rollout_db.connect(True) as db:old=[dict(r) for r in db.execute('SELECT state_uid,physical_state_json FROM state')]
 signatures=set();unrecoverable=0
 for q in old:
  try:sig=signature(json.loads(q['physical_state_json']))
  except (TypeError,ValueError):sig=None
  if sig:signatures.add(sig)
  else:unrecoverable+=1
 collisions=[]
 for radius,offset in zip(cfg['radii'],cfg['pilot_radius_offsets']):
  for i in range(16):
   seed=cfg['pilot_seed_start']+offset+i;cc=LocalFrameConfig(**config(700,radius));ins=sample_instance(seed,'development',cc);env=RingExchangeEnv(cc);env.reset(ins.positions,velocities=ins.velocities,goals=ins.goals);physical=clean(dict(positions=ins.positions,velocities=ins.velocities,goals=ins.goals));sig=signature(physical)
   if sig in signatures:collisions.append(dict(radius=radius,index=i,seed=seed))
   signatures.add(sig);alias=f'round35_geometry_pilot_r{radius}_{seed}';s=dict(uid=alias,state_uid=uid('state',alias),scenario_uid='new',physical=physical,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),study_index=i,family=alias);s=state(p,s,700,radius);s.update(draw_seed=seed,split='pilot',radius=radius);generated.append(s)
   for name in read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']:
    for arm in ['F10','S500']:rows.append(case(p,s,name,700,radius,arm,0))
 write(HERE/'GEOMETRY_PILOT_STATE_DEDUP.json',dict(utc=utc(),passed=not collisions,old_db_states=len(old),unrecoverable=unrecoverable,new_states=64,collisions=collisions));assert not collisions,'Do not redraw'
 write(HERE/'GEOMETRY_PILOT_STATES.json',generated);register('pilot',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
def capability():
 names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];cases=read(HERE/'pilot_CASES.json');out={}
 for r in read(HERE/'DESIGN_CONFIG.json')['radii']:
  bounds=[];details={}
  for name in names:
   rates=[]
   for arm in ['F10','S500']:
    rr=[read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json") for c in cases if c['r']==r and c['base_controller_name']==name and c['arm']==arm];assert len(rr)==16;rates.append([sum(q['success'] for q in rr)/16,sum(q['success']+q['numerical_failure'] for q in rr)/16])
   pair=np.max(rates,axis=0);bounds.append(pair);details[name]=dict(arms=rates,max_bounds=pair)
  med=np.median(bounds,axis=0);out[str(r)]=dict(status='PASS' if med[0]>=.2 else 'FAIL_CONTROLLER_CAPABILITY' if med[1]<.2 else 'INCONCLUSIVE',median_max_success_bounds=med,controllers=details)
 write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),radii=out,threshold=.2));seal('CAPABILITY_GATE_FROZEN.receipt.json',[HERE/'CAPABILITY_GATE.json'],before_geometry_main=True)
def evaluation():
 radii=[float(r) for r,v in read(HERE/'CAPABILITY_GATE.json')['radii'].items() if v['status']=='PASS'];write(HERE/'GEOMETRY_ELIGIBLE_RADII.json',radii)
 if radii:actors('geometry',[(700,r) for r in radii],lambda H:['nominal_only','safety_baseline','F10','F13','S500'])
if __name__=='__main__':globals()[sys.argv[1]]()
