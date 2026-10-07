#!/usr/bin/env python3
"""Six fresh geometric coverage points per frozen nonempty retained instance."""
from audit import *
from families import contains
from scipy.stats import qmc
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--name',required=True);ap.add_argument('--namespace',required=True);ap.add_argument('--scenario',choices=['toy','db'],required=True);a=ap.parse_args()
 assert not (HERE/f'targeted_probe_rounds/{a.name}/manifest.csv').exists(),'Do not overwrite frozen validation'
 gates=json.load(open(HERE/f'family_gates_{a.tag}.json'));assert len(gates)==1 and gates[0]['cached_screen_pass']
 inv=read(HERE/'exact_q64_inventory.csv');have={(r['state_id'],r['eta_key']) for r in inv};frozen=[r for r in json.load(open(HERE/f'family_models_{a.tag}.json')) if r['state_id'].startswith('DB')==(a.scenario=='db')]
 rows=[];sampling=[]
 for r in frozen:
  sid=r['state_id'];m=r['model'];seed=int(hashlib.sha256((sid+a.namespace).encode()).hexdigest()[:8],16)
  Z=qmc.Sobol(3,scramble=True,seed=seed).random_base2(16)*np.array([5/3,1,1])+np.array([-7/6,-.5,-.5]);Z=Z[contains(m,Z,True)]
  Z=np.array([z for z in Z if (sid,key(AFF+SCALE*z)) not in have]).reshape(-1,3)
  sampling.append(dict(state_id=sid,pool_points=len(Z),empty_or_insufficient=len(Z)<6,namespace=a.namespace))
  if len(Z)<6:continue
  chosen=[int(np.argmin(np.linalg.norm(Z-Z.mean(axis=0),axis=1)))];d=np.linalg.norm(Z-Z[chosen[0]],axis=1)
  while len(chosen)<6:
   j=int(np.argmax(d));chosen.append(j);d=np.minimum(d,np.linalg.norm(Z-Z[j],axis=1))
  for n,j in enumerate(chosen):rows.append(dict(state_id=sid,eta=(AFF+SCALE*Z[j]).tolist(),phase=a.name+'_retained_validation',probe_id=f'{sid}_{a.name}_{n}',model_sha256=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()))
 dump(f'targeted_probe_rounds/{a.name}/frozen_parameters.json',frozen);write(f'targeted_probe_rounds/{a.name}/sampling.csv',sampling)
 if a.scenario=='toy':plan(a.name,rows)
 else:
  out=HERE/'plans'/a.name;out.mkdir(parents=True,exist_ok=True);tasks=[dict(r,eta_key=key(r['eta']),future_index=f) for r in rows for f in range(64)]
  for sh in range(6):
   with open(out/f'shard{sh}.jsonl','w') as f:
    for n,t in enumerate(tasks):
     if n%6==sh:f.write(json.dumps(t)+'\n')
  write(f'targeted_probe_rounds/{a.name}/manifest.csv',[dict(r,eta=json.dumps(r['eta']),eta_key=key(r['eta'])) for r in rows]);dump(f'targeted_probe_rounds/{a.name}/cost_estimate.json',dict(new_exact_q64=len(rows),new_continuations=len(tasks),shards=6,estimated_wall_seconds=90+len(tasks)*1.9/6))
 print(json.dumps(dict(states=len(frozen),new_eta=len(rows),insufficient_pool=sum(r['empty_or_insufficient'] for r in sampling))))
if __name__=='__main__':main()
