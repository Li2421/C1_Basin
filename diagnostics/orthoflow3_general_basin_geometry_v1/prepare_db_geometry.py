#!/usr/bin/env python3
from audit import *
def main():
 states=json.load(open(HERE/'double_bottleneck_state_panel.json'));cost=json.load(open(HERE/'double_bottleneck_cost_estimate.json'));centers=np.array(cost['candidate_centers'])
 seen={};durations=[]
 for p in (HERE/'db_raw').glob('*.jsonl'):
  for line in open(p):
   r=json.loads(line);assert r['scientific_outcome_valid'];seen[(r['state_id'],key(r['eta']),r['future_index'])]=r;durations.append(r['wall_seconds'])
 assert len(seen)>=8,'Wait for exact preflight completion'
 values={key([0,0,0]):([0,0,0],'zero')}
 for ci,c in enumerate(centers):
  values[key(c)]=(c.tolist(),'center')
  for axis in range(3):
   for mag in (-.10,-.025,.025,.10):
    v=c+SCALE*np.eye(3)[axis]*mag
    if inside((v-AFF)/SCALE)[0]:values.setdefault(key(v),(v.tolist(),f'center{ci}_axis{axis}_{mag}'))
 for i in range(len(centers)):
  for j in range(i):
   v=(centers[i]+centers[j])/2;values.setdefault(key(v),(v.tolist(),'center_connector'))
 rows=[dict(state_id=s['state_id'],eta=v,eta_key=k,kind=kind,phase='db_geometry1',probe_id=f"{s['state_id']}_{i}") for s in states for i,(k,(v,kind)) in enumerate(values.items())]
 tasks=[dict(r,future_index=f) for r in rows for f in range(64) if (r['state_id'],r['eta_key'],f) not in seen]
 shards=6;p=HERE/'plans/db_geometry1';p.mkdir(parents=True,exist_ok=True)
 for sh in range(shards):
  with open(p/f'shard{sh}.jsonl','w') as f:
   for i,r in enumerate(tasks):
    if i%shards==sh:f.write(json.dumps(r)+'\n')
 write('targeted_probe_rounds/db_geometry1/manifest.csv',[dict(r,eta=json.dumps(r['eta'])) for r in rows])
 estimate={'new_continuations':len(tasks),'exact_eta_per_state':len(values),'state_count':4,'shards':6,'mean_preflight_rollout_seconds':float(np.mean(durations)),
  'estimated_wall_seconds':float(len(tasks)*np.mean(durations)/6+90),'threefold_stop_diagnosis':True,'selection':'historical3 frequent16seed centers, fixed axis offsets and connectors; no new Sobol sweep'}
 dump('targeted_probe_rounds/db_geometry1/cost_estimate.json',estimate);print(estimate)
if __name__=='__main__':main()
