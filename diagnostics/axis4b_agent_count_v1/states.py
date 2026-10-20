from design import *
from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.scenario import Config
from dataclasses import replace

def signature(v):
 try:
  data=[]
  for k in ['positions','velocities','goals']:
   a=np.asarray(v[k],np.float64).copy();assert a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all();a[a==0]=0;data.append([a.shape,a.tobytes().hex()])
  return uid('physical',data)
 except (TypeError,KeyError,ValueError,AssertionError):return None

def main():
 prereg();cfg=read(HERE/'DESIGN_CONFIG.json');assets=read(HERE/'ASSET_AUDIT.json')['assets'];out={};p=protocol();generated=[];errors=[];nested={}
 for n in cfg['N']:
  out[str(n)]={}
  for split,num in [('pilot',16),('eval',32)]:
   ss=[]
   for i in range(num):
    seed=cfg[split+'_seed_start']+i;actual=copy.deepcopy(assets[str(n)]['config']);actual.update(seed=seed,split='dev')
    try:
     env=BottleneckEnv(Config(**actual));physical=clean(dict(positions=env.positions,velocities=env.velocities,goals=env.goals));s=state(p,n,physical,f'round36_{split}_N{n}_{i}_{seed}',actual,None,study_index=i,study_split=split,draw_seed=seed,geometry_validation=env.instance.manifest()['validation']);ss.append(s);generated.append(s);key=(split,i)
     if key in nested:
      previous=nested[key];assert np.array_equal(env.positions[:len(previous[0])],previous[0]) and np.array_equal(env.goals[:len(previous[1])],previous[1]),'Nested task violation'
     nested[key]=(env.positions.copy(),env.goals.copy())
    except ValueError as ex:errors.append(dict(N=n,split=split,index=i,reason=str(ex),not_deadlock=True,no_redraw=True))
   out[str(n)][split]=ss
 with rollout_db.connect(True) as db:old=[dict(r) for r in db.execute('SELECT state_uid,physical_state_json FROM state')]
 signatures=set();unrecoverable=0
 for q in old:
  try:sig=signature(json.loads(q['physical_state_json']))
  except (TypeError,ValueError):sig=None
  if sig:signatures.add(sig)
  else:unrecoverable+=1
 collisions=[]
 for s in generated:
  sig=signature(s['physical']);assert sig
  if sig in signatures:collisions.append(s['uid'])
  signatures.add(sig)
 write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=not collisions,new_states=len(generated),old_db_states=len(old),unrecoverable_old_physical=unrecoverable,collisions=collisions,geometry_errors=errors,nested_tasks_verified=True,fixed_geometry=True));assert not collisions,'No adaptive redraw';write(HERE/'STATES.json',out);print('States generated and audited',len(generated),flush=True)
if __name__=='__main__':main()
