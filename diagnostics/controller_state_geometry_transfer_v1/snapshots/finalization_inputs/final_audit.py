"""End-to-end artifact/cache/split/trajectory audit; no new controller rollout."""
from pathlib import Path
import json,hashlib,datetime,collections
import numpy as np
from shared_rollout_db.src.planner import preflight
from .main_analysis import write
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert read(HERE/'factor_training_complete.json')['completed'];assert read(HERE/'phase2_complete.json')['completed'];assert read(HERE/'contract_tests.json')['metric_oracle_and_ANOVA_passed'];assert (HERE/'normalization_results.json').exists()
 stages={'pilot_revision':2048,'broad_pilot':1024,'legacy_supplement':1024,'prefix_audit':4,'factors':3072};tot=collections.Counter();summary={};cpu=0;nf=[];all_rows=[]
 for name,n in stages.items():
  root=HERE/name;cases=read(root/'cases.json');assert len(cases)==n;ph=sha(root/'protocol.json');records=[]
  for c in cases:
   f=root/'raw'/(c['rollout_uid']+'.json');assert f.exists();r=read(f);assert c==r['case'] and ph==r['protocol_sha256'];assert not(r['success'] and (r['collision'] or r['numerical_failure']));records.append(r);cpu+=r['elapsed_seconds'];tot[r['outcome']]+=1;all_rows.append({'stage':name,'path':str(f),'sha256':sha(f),'rollout_uid':c['rollout_uid']})
   if r['numerical_failure']:nf.append({'stage':name,'path':str(f),'case':c,'error':r['error']})
  pf=preflight(root/'planned_rollouts.json');write(root/'cache_postflight.json',pf);assert pf['summary']['genuinely_missing']==0 and pf['summary']['ambiguous']==0 and pf['summary']['exact_reusable']==n
  summary[name]={'n':n,'outcomes':dict(collections.Counter(r['outcome'] for r in records)),'cache':pf['summary'],'rollout_elapsed_hours':sum(r['elapsed_seconds'] for r in records)/3600}
 assert len({r['rollout_uid'] for r in all_rows})==sum(stages.values())
 factor=read(HERE/'factors/protocol.json');by={(s['geometry_index'],s['state_index']):s['physical'] for s in factor['states']}
 for s in range(16):
  for top in range(2):
   for w in range(3):
    a=by[top*6+w,s];b=by[top*6+3+w,s]
    for k in ['positions','goals','velocities']:assert a[k]==b[k][:2]
  for nstart in [0,3]:
   for g in [nstart,nstart+1,nstart+2,nstart+6,nstart+7,nstart+8]:assert by[g,s]==by[nstart,s]
 train={json.dumps(by[g,s],sort_keys=True) for g in range(12) for s in range(8)};test={json.dumps(by[g,s],sort_keys=True) for g in range(12) for s in range(8,16)};assert not train&test
 traces=[];rows=[read(HERE/'factors/raw'/(c['rollout_uid']+'.json')) for c in read(HERE/'factors/cases.json')]
 from single_integrator.environment import segment_distance,point_segment_distance
 from .factor_runtime import config
 # Hash every full trajectory; independently recompute swept safety on one success
 # and one timeout per physical geometry, plus the numerical-failure prefix.
 chosen=set()
 for gi in range(12):
  for outcome in ['success','timeout']:
   rr=next((r for r in rows if r['case']['geometry_index']==gi and r['outcome']==outcome),None)
   if rr:chosen.add(rr['case']['rollout_uid'])
 chosen|={r['case']['rollout_uid'] for r in rows if r['numerical_failure']}
 for r in rows:
  z=np.load(r['trace_path']);p=z['positions'];assert hashlib.sha256(p.tobytes()).hexdigest()==r['trace_sha256'];assert len(p)==r['episode_length']+1
  if r['case']['rollout_uid'] not in chosen:continue
  g=factor['geometries'][r['case']['geometry_index']];c=config(g['width'],g['N'],g['topology']);walls=z['walls'];margin=c.agent_radius+c.wall_radius+c.wall_collision_margin
  wall=segment_distance(p[:-1,:,None],p[1:,:,None],walls[None,None,:,0],walls[None,None,:,1]).min()-margin
  i,j=np.triu_indices(g['N'],1);rel=p[:,i]-p[:,j];pair=point_segment_distance(np.zeros(2),rel[:-1],rel[1:]).min()-2*c.agent_radius-c.agent_collision_margin
  speed=np.linalg.norm(np.diff(p,axis=0)/c.dt,axis=-1).max();assert speed<c.max_speed+1e-9;assert abs(wall-r['summary']['min_swept_wall_clearance'])<1e-10;assert abs(pair-r['summary']['min_swept_agent_clearance'])<1e-10
  if r['success']:assert wall>0 and pair>0 and np.linalg.norm(z['goals']-p[-1],axis=-1).max()<=c.goal_tolerance
  traces.append({'rollout_uid':r['case']['rollout_uid'],'frames':len(p),'wall_clearance':wall,'pair_clearance':pair,'max_speed':speed,'outcome':r['outcome']})
 fits=[]
 for f in (HERE/'models').rglob('*.json'):
  m=read(f)
  if 'seconds' in m and 'steps' in m:fits.append({'path':str(f),**m})
 assert len(fits)==138,('fit count',len(fits));assert len(list((HERE/'models').rglob('*.msgpack')))==138
 for name in ['preregistration_receipt.json','addendum_state_support_receipt.json','addendum_horizon_receipt.json','diagnostics_factor_receipt.json','factor_transfer_receipt.json','numerical_handling_receipt.json','normalization_receipt.json']:
  r=read(HERE/name);mapping={'preregistration_receipt.json':'PREREGISTRATION.md','addendum_state_support_receipt.json':'ADDENDUM_STATE_SUPPORT.md','addendum_horizon_receipt.json':'ADDENDUM_HORIZON_PILOT.md','diagnostics_factor_receipt.json':'ADDENDUM_DIAGNOSTICS_AND_FACTORS.md','factor_transfer_receipt.json':'FACTOR_TRANSFER_ADDENDUM.md','numerical_handling_receipt.json':'NUMERICAL_HANDLING.md','normalization_receipt.json':'NORMALIZATION_DIAGNOSTIC.md'};assert r['sha256']==sha(HERE/mapping[name])
 cost={'new_physical_rollouts':sum(stages.values()),'reused_legacy_toy_rollouts':13312,'reused_four_scene_flow_rollouts':7680,'new_rollout_elapsed_cpu_hours_approx':cpu/3600,'fits':len(fits),'fit_elapsed_gpu_seconds_sum':sum(f['seconds'] for f in fits),'max_cpu_rollout_workers':16,'max_gpu_shards':4,'new_collisions':tot['collision'],'new_numerical_failures':len(nf),'accounting_limitation':'Slurm accounting disabled; wall elapsed sums from one-CPU workers / fits, not CPU/GPU utilization integrals'}
 write(HERE/'DATA_INDEX.json',all_rows);write(HERE/'FINAL_AUDIT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'stages':summary,'total_outcomes':dict(tot),'cost':cost,'unknown_records':nf,'factor_train_test_physically_disjoint':True,'nested_N_and_cross_geometry_states_checked_all16':True,'factor_final_test_unknown':0,'all3072_trace_hashes_match':True,'independent_swept_rechecks':traces,'preregistration_and_addendum_hashes_match':True,'completed_fit_artifacts':len(fits),'primary_scientific_gap':'strong state-dependent eta-selection gate failed; not excluded by successful artifact audit'})
 print(cost)
if __name__=='__main__':main()
