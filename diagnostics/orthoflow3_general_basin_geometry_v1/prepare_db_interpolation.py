#!/usr/bin/env python3
from audit import *
def main():
 inv=read(HERE/'exact_q64_inventory.csv');name='db_independent_interpolation_r3';rows=[];pairs=[];have={(r['state_id'],r['eta_key']) for r in inv}
 for sid in sorted({r['state_id'] for r in inv if r['state_id'].startswith('DB')}):
  pp=[r for r in inv if r['state_id']==sid and r['B63']=='True'];P=np.array([eta(r) for r in pp]);dist=cdist(P/SCALE,P/SCALE)
  candidates=sorted([(dist[i,j],i,j) for i in range(len(P)) for j in range(i)],reverse=True);used=set();selected=[]
  for d,i,j in candidates:
   if i in used or j in used:continue
   values=[(1-alpha)*P[i]+alpha*P[j] for alpha in (.25,.5,.75)]
   if any((sid,key(v)) in have for v in values):continue
   selected.append((i,j,values));used.update([i,j])
   if len(selected)==2:break
  assert len(selected)==2
  for pi,(i,j,values) in enumerate(selected):
   pairs.append(dict(state_id=sid,pair_id=pi,endpoint_a=pp[i]['eta_key'],endpoint_b=pp[j]['eta_key'],distance=float(dist[i,j])))
   for k,v in enumerate(values):rows.append(dict(state_id=sid,eta=v.tolist(),eta_key=key(v),phase=name,kind='heldout_success_pair_interpolation',probe_id=f'{sid}_pair{pi}_{k}',geometry_role_frozen='heldout',pair_id=pi,alpha=(k+1)/4))
 assert len(rows)==24
 out=HERE/'plans'/name;out.mkdir(parents=True,exist_ok=True);tasks=[dict(r,future_index=f) for r in rows for f in range(64)]
 for sh in range(6):
  with open(out/f'shard{sh}.jsonl','w') as f:
   for i,r in enumerate(tasks):
    if i%6==sh:f.write(json.dumps(r)+'\n')
 write(f'targeted_probe_rounds/{name}/manifest.csv',[dict(r,eta=json.dumps(r['eta'])) for r in rows]);write(f'targeted_probe_rounds/{name}/pairs.csv',pairs)
 dump(f'targeted_probe_rounds/{name}/cost_estimate.json',dict(new_exact_q64=24,new_continuations=1536,shards=6,estimated_wall_seconds=580,
  decision='Test cross-scenario filled/interrupted long success connectors on all4 unchanged states; independently heldout positives improve geometric validation, not fitting.',
  role_policy='All new points reserved heldout before outcomes, no hash-based selection; existing roles unchanged. Negative evidence always hard.'))
 print('Frozen24 DB independent interpolation queries; no outcomes inspected.')
if __name__=='__main__':main()
