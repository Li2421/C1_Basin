"""Prespecified secondary exports and checks from the same saved trajectories."""
from core import *
from stats import *
import csv,collections
precision()
def main():
 manifest=read(HERE/'MANIFEST.json');fs={}
 for p in (HERE/'features').glob('shard*.jsonl'):
  for l in p.open():
   f=json.loads(l);fs[f['rollout_uid']]=f
 by={(r['group'],r['controller_index'],r['state_index'],r['seed'],r['arm']):r for r in manifest};arrivals={};prefix_errors=[];prefix_checks=0;exclusive={};episodes=[];timing={};teacher_diffs=[]
 for group in FULL:
  rs=[r for r in manifest if r['group']==group];C=1 if group in ['expert','Z'] else 14;Q=1 if group=='expert' else 2
  def cube(a,k):
   x=np.full((C,32,Q),np.nan)
   for r in rs:
    if r['arm']!=a:continue
    v=k(r,fs[r['rollout_uid']]);x[r['controller_index'],r['state_index'],r['seed']]=v if v is not None else np.nan
   return x
  arrivals[group]={}
  for arm in (['expert'] if group=='expert' else ARMS):
   selected=[r for r in rs if r['arm']==arm and r['success'] and not r['unknown']]
   values=[fs[r['rollout_uid']]['total_time'] for r in selected]
   if not any(r['arm']==arm for r in rs):continue
   arrivals[group][arm]=dict(n_success=len(selected),mean_seconds=mean(cube(arm,lambda r,f:f['total_time'] if r['success'] and not r['unknown'] else None)),quantiles_seconds=np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else None,first_robot_goal_entry_seconds=[[None if a is None else a*.05 for a in fs[r['rollout_uid']]['first_goal_entry_step']] for r in selected])
  if any(r['arm']=='safety baseline' for r in rs):
   timing[group]={k:mean(cube('safety baseline',lambda r,f,k=k:f[k] if r['timeout'] and not r['unknown'] else None)) for k in ['stall_first_step','stall_max_seconds','moving_no_progress_max_seconds']}
  if group in ['TD','V0','V1','V3']:
   exclusive[group]=collections.Counter()
   for r in rs:
    if r['arm']!='nominal only' or not r['collision'] or r['unknown']:continue
    types=fs[r['rollout_uid']]['collision_geometry']['types'];label='unverified' if not types else 'multi' if len(types)>1 else types[0];exclusive[group][label]+=1
  if group in ['TD','V0','V1','V3','Z']:
   for r in rs:
    if r['arm']!='F13':continue
    pair=by.get((group,r['controller_index'],r['state_index'],r['seed'],'S500'))
    if not pair or r['unknown'] or pair['unknown']:continue
    a=fs[r['rollout_uid']];b=fs[pair['rollout_uid']];prefix_checks+=1
    if a['distance_curve'][:51]!=b['distance_curve'][:51] or a['mode_curve'][:51]!=b['mode_curve'][:51]:prefix_errors.append(dict(F13=r['rollout_uid'],S500=pair['rollout_uid']))
  # Per-controller paired student-minus-teacher rates, state-cluster CI.
  if group in ['V0','V1','V3']:
   for i in range(14):
    a=np.zeros((1,32,2));b=np.zeros_like(a);au=np.zeros_like(a);bu=np.zeros_like(a)
    for s in range(32):
     for q in range(2):
      u=by[group,i,s,q,'nominal only'];v=by['TD',i,s,q,'nominal only'];a[0,s,q]=u['success']*(1-u['unknown']);au[0,s,q]=u['unknown'];b[0,s,q]=v['success']*(1-v['unknown']);bu[0,s,q]=v['unknown']
    teacher_diffs.append(dict(group=group,teacher_index=i,teacher=read(HERE/'TEACHER_NAMES.json')[i],paired_difference=paired_success(a,au,b,bu)))
  for r in rs:
   f=fs[r['rollout_uid']];c=f['collision_geometry'];label='unknown' if r['unknown'] else 'success' if r['success'] else 'timeout' if r['timeout'] else 'collision' if r['collision'] else r['outcome'];types=c['types'] if c else [];exclusive_type='unverified' if r['collision'] and not types else 'multi' if len(types)>1 else types[0] if types else ''
   episodes.append(dict(group=FULL[group],group_code=group,teacher_name=r['teacher_name'],controller_index=r['controller_index'],state=r['state_index'],seed=r['seed'],arm=ARM_FULL.get(r['arm'],r['arm']),arm_code=r['arm'],outcome=label,success=r['success'],collision=r['collision'],timeout=r['timeout'],unknown=r['unknown'],collision_type=exclusive_type,mode=['stationary','CW-consistent','CCW-consistent','mixed'][f['mode']],mode100=f['early']['100']['mode'],mode200=f['early']['200']['mode'],mode300=f['early']['300']['mode'],stable_step=f['stable_step'],four_same=f['four_same'],last_progress_seconds=f['last_progress_seconds'],final_unfinished=f['final_unfinished'],opposite_facing_pairs=f['geometric_opposite_pairs'],stall_runs=f['stall_runs'],stall_first_step=f['stall_first_step'],stall_robot_seconds=f['stall_robot_seconds'],moving_no_progress_robot_seconds=f['moving_no_progress_robot_seconds'],trace=r['trace_path'],source_round=pathlib.Path(r['raw_path']).relative_to(D).parts[0],evidence_level='DEVELOPMENT'))
 with (HERE/'EPISODE_METRICS.csv').open('w') as out:
  w=csv.DictWriter(out,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)
 # Same-collision-step location checks require only stored position arrays.
 locations=read(HERE/'analysis3_safety_stalls.json')['location_pairs'];same_step=[]
 for pair in locations:
  raw=read(pair['safety_raw']);f=fs[raw['case']['rollout_uid']];n=by[pair['group'],pair['index'],pair['state'],pair['seed'],'nominal only'];nf=fs[n['rollout_uid']];cg=nf['collision_geometry'];ids=[int(i) for i in cg['robot_positions'] if f['unfinished_mask'][int(i)]]
  with np.load(raw['trace_path']) as z:p=z['positions'][min(pair['collision_step'],len(z['positions'])-1)]
  dd=[float(np.linalg.norm(p[i]-np.asarray(cg['robot_positions'][str(i)]))) for i in ids];same_step.append(dict(group=pair['group'],controller=pair['index'],state=pair['state'],seed=pair['seed'],at_original_collision_step_median_distance=float(np.median(dd)),terminal_median_distance=pair['median_robot_distance']))
 duplicate_checks=[]
 for r in manifest:
  if r['group']!='Z' or r['seed']!=0:continue
  b=by['Z',0,r['state_index'],1,r['arm']]
  with np.load(r['trace_path']) as x,np.load(b['trace_path']) as y:equal=all(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes() for k in ['positions','actions'])
  assert equal;duplicate_checks.append(dict(state=r['state_index'],arm=r['arm'],identical=True))
 assert not prefix_errors,prefix_errors
 write(HERE/'SECONDARY_ANALYSES.json',dict(utc=utc(),arrivals=arrivals,stall_onset=timing,exclusive_collision_types=exclusive,student_minus_teacher_by_controller=teacher_diffs,same_collision_step_locations=same_step,prefix_consistency=dict(pairs_checked=prefix_checks,all_progress_and_mode_samples_equal_through500=True,errors=prefix_errors),zero_deterministic_seed_checks=dict(pairs_checked=len(duplicate_checks),all_positions_and_actions_bytes_equal=True),notes='All are preregistered descriptive secondary metrics; no new trajectory or policy query',precision=precision()));print('secondary exports complete',len(episodes),len(same_step),prefix_checks,flush=True)
if __name__=='__main__':main()
