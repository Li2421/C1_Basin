#!/usr/bin/env python3
"""Frozen, decision-targeted acquisition; no execution and no network fitting."""
from audit import *
from families import contains
from scipy.stats import qmc

def main():
 inv=read(HERE/'exact_q64_inventory.csv');have={(r['state_id'],r['eta_key']) for r in inv}
 models=json.load(open(HERE/'family_models_after_round1_complete.json'))
 frozen=[r for r in models if r['model']['family']=='semialgebraic' and not r['state_id'].startswith('DB')]
 probes=[];sampling=[]
 for r in frozen:
  sid=r['state_id'];m=r['model'];seed=int(hashlib.sha256((sid+'retained-validation-v1').encode()).hexdigest()[:8],16)
  z=qmc.Sobol(3,scramble=True,seed=seed).random_base2(16)*np.array([5/3,1,1])+np.array([-7/6,-.5,-.5])
  z=z[contains(m,z,True)];z=np.array([p for p in z if (sid,key(AFF+SCALE*p)) not in have]).reshape(-1,3)
  sampling.append(dict(state_id=sid,retained_pool_count=len(z),status='EMPTY_OR_TOO_SMALL_RETAINED_SET' if len(z)<6 else 'FROZEN_FOR_FRESH_VALIDATION'))
  if len(z)<6:continue
  chosen=[int(np.argmin(np.linalg.norm(z-z.mean(axis=0),axis=1)))];distance=np.linalg.norm(z-z[chosen[0]],axis=1)
  while len(chosen)<6:
   selected=int(np.argmax(distance));chosen.append(selected);distance=np.minimum(distance,np.linalg.norm(z-z[selected],axis=1))
  for n,i in enumerate(chosen):probes.append(dict(state_id=sid,eta=(AFF+SCALE*z[i]).tolist(),phase='semialgebraic_retained_validation_r1',probe_id=f'{sid}_retained{n}',model_sha256=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()))
 dump('semialgebraic/frozen_validation_r1_parameters.json',frozen);write('semialgebraic/frozen_validation_r1_sampling.csv',sampling)
 plan('retained_validation_r1',probes)
 # Cross-scenario mode transfer plus long conditional rays, not a Sobol sweep.
 rows=[];paths=[];panel=read(HERE/'common_eta_panel.csv')
 states=json.load(open(HERE/'double_bottleneck_state_panel.json'))
 for s in states:
  sid=s['state_id'];rr=[r for r in inv if r['state_id']==sid];pos=[r for r in rr if r['B63']=='True'];neg=[r for r in rr if r['B63']=='False']
  assert pos and neg
  P=np.array([(eta(r)-AFF)/SCALE for r in pos]);N=np.array([(eta(r)-AFF)/SCALE for r in neg]);clear=cdist(P,N).min(axis=1)
  ci=max(range(len(pos)),key=lambda i:(clear[i],-i));c=P[ci]
  for p in panel:rows.append(dict(state_id=sid,eta=eta(p).tolist(),phase='db_cross_scenario_common_core_r2',probe_id=f'{sid}_mode{p["mode_id"]}',kind='cross_scenario_mode'))
  for axis in range(3):
   direction=np.eye(3)[axis];a=HS[:,:3]@direction;b=-(HS[:,:3]@c+HS[:,3]);lo=max(b[a< -1e-12]/a[a< -1e-12]);hi=min(b[a>1e-12]/a[a>1e-12]);path=f'{sid}_long_axis{axis}'
   for t in (.9*lo,.5*lo,0.,.5*hi,.9*hi):
    v=eta(pos[ci]) if t==0 else AFF+SCALE*(c+t*direction)
    rows.append(dict(state_id=sid,eta=v.tolist(),phase='db_conditional_extent_r2',probe_id=f'{path}_{t.hex()}',kind='conditional_extent',path_id=path,coordinate=float(t)))
   paths.append(dict(state_id=sid,path_id=path,kind='conditional',axis=axis))
 name='db_geometry2';unique={}
 for r in rows:
  assert inside((np.array(r['eta'])-AFF)/SCALE)[0]
  k=(r['state_id'],key(r['eta']));r['eta_key']=k[1];r['cached_q64']=k in have
  if k not in have:unique.setdefault(k,r)
 write(f'targeted_probe_rounds/{name}/manifest.csv',[dict(r,eta=json.dumps(r['eta'])) for r in rows],['state_id','eta','eta_key','phase','probe_id','kind','path_id','coordinate','cached_q64'])
 write(f'targeted_probe_rounds/{name}/paths.csv',paths)
 tasks=[dict(r,future_index=fi) for r in unique.values() for fi in range(64)]
 out=HERE/'plans'/name;out.mkdir(parents=True,exist_ok=True)
 for sh in range(6):
  with open(out/f'shard{sh}.jsonl','w') as f:
   for i,t in enumerate(tasks):
    if i%6==sh:f.write(json.dumps(t)+'\n')
 dump(f'targeted_probe_rounds/{name}/cost_estimate.json',dict(new_continuations=len(tasks),new_exact_q64=len(unique),shards=6,estimated_wall_seconds=len(tasks)*1.9/6+90,threefold_stop_diagnosis=True,
  decision='Shared actual eta modes across scenarios; conditional interval extent beyond prior .10 neighborhood. Four unchanged source states; no outcome-based source replacement.'))
 print(json.dumps(dict(fresh_retained_eta=len(probes),empty_retained_states=[s['state_id'] for s in sampling if s['retained_pool_count']<6],db_new_eta=len(unique))))
if __name__=='__main__':main()
