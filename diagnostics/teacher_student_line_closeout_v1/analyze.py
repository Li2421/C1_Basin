"""Seven prespecified DEVELOPMENT analyses from already-opened labels and traces."""
from core import *
from stats import *
import csv,collections
from scipy.stats import spearmanr
precision()
def main():
 prereg();manifest=read(HERE/'MANIFEST.json');features={};metric_errors=[]
 for p in sorted((HERE/'features').glob('shard*_COMPLETE.json')):
  r=read(p);assert sha(p.with_name(p.name.replace('_COMPLETE.json','.jsonl')))==r['feature_sha256'];metric_errors+=r['errors']
 for p in sorted((HERE/'features').glob('shard*.jsonl')):
  for line in p.open():
   f=json.loads(line);assert f['rollout_uid'] not in features;features[f['rollout_uid']]=f
 groups=list(FULL);counts={g:1 if g in ['Z','expert'] else 14 for g in groups};by={};roles=collections.defaultdict(list)
 for r in manifest:
  key=(r['group'],r['controller_index'],r['state_index'],r['seed'],r['arm']);by[key]=r;roles[r['group']].append(r)
 def cube(g,a,fn,default=np.nan):
  x=np.full((counts[g],32,1 if g=='expert' else 2),default,float)
  for r in roles[g]:
   if r['arm']!=a:continue
   f=features.get(r['rollout_uid']);value=fn(r,f)
   if value is not None:x[r['controller_index'],r['state_index'],r['seed']]=value
  return x
 def lab(g,a,k):return cube(g,a,lambda r,f:r[k],default=0)
 def met(g,a,k):return cube(g,a,lambda r,f:f.get(k) if f else None)
 def mode(g,a,t=None):return cube(g,a,lambda r,f:(f['early'][str(t)]['mode'] if t else f['mode']) if f else None)
 def known(g,a):return cube(g,a,lambda r,f:int(not r['unknown']),default=0).astype(bool)
 def exists(g,a):return any(r['arm']==a for r in roles[g])
 def positive(g,a):return lab(g,a,'success')*(1-lab(g,a,'unknown'))
 def emit(name,obj):write(HERE/(name+'.json'),obj)
 # Complete arm-level outcome rates for publication and reuse.
 paper={};outcomes={}
 for g in groups:
  paper[g]={};outcomes[g]={}
  for a in (['expert'] if g=='expert' else ARMS):
   if not exists(g,a):paper[g][a]=dict(status='MISSING',reason='No compatible historical arm');continue
   s=positive(g,a);u=lab(g,a,'unknown');co=lab(g,a,'collision');to=lab(g,a,'timeout');n=sum(r['arm']==a for r in roles[g]);rounds=sorted({pathlib.Path(r['raw_path']).relative_to(D).parts[0] for r in roles[g] if r['arm']==a});paper[g][a]=dict(success=outcome_rate(s,u),collision=outcome_rate(co*(1-u),u),timeout=outcome_rate(to*(1-u),u),n=n,independent_states=32,controllers=counts[g],source_rounds=rounds,evidence_level='DEVELOPMENT');outcomes[g][a]=dict(success=paper[g][a]['success']['lower']['estimate'],collision=paper[g][a]['collision']['lower']['estimate'],timeout=paper[g][a]['timeout']['lower']['estimate'],unknown=int(u.sum()))
 # Analysis 1: teacher-by-teacher ceiling association, plus stratified sensitivity.
 per_teacher=[];corr={};teacher=positive('TD','nominal only');teacher_safety=positive('TD','safety baseline')
 for i,n in enumerate(read(HERE/'TEACHER_NAMES.json')):
  row=dict(index=i,teacher=n,frame='local' if i in [8,10,12,13] else 'global',teacher_nominal=float(teacher[i].mean()),teacher_safety=float(teacher_safety[i].mean()))
  row.update({v:float(positive(v,'nominal only')[i].mean()) for v in ['V0','V1','V3']});per_teacher.append(row)
 def correlations(x,y):
  if len(x)<3 or np.std(x)==0 or np.std(y)==0:return dict(Pearson=None,Spearman=None,reason='zero variance or fewer than3 controllers')
  return dict(Pearson=float(np.corrcoef(x,y)[0,1]),Spearman=float(spearmanr(x,y).statistic))
 for v in ['V0','V1','V3']:
  x=positive(v,'nominal only');paired=paired_success(x,lab(v,'nominal only','unknown'),teacher,lab('TD','nominal only','unknown'));p=correlations(teacher.mean((1,2)),x.mean((1,2)));tb=draws(teacher,True);xb=draws(x,True);tbc=tb-tb.mean(1,keepdims=True);xbc=xb-xb.mean(1,keepdims=True);div=np.sqrt((tbc*tbc).sum(1)*(xbc*xbc).sum(1));cb=np.divide((tbc*xbc).sum(1),div,out=np.full(len(div),np.nan),where=div>0);p['Pearson_hierarchical_bootstrap']=quantiles(cb);p['student_minus_teacher']=paired;p['by_frame']={frame:correlations(teacher[ids].mean((1,2)),x[ids].mean((1,2))) for frame,ids in [('global',[i for i in range(14) if i not in [8,10,12,13]]),('local',[8,10,12,13])]};corr[v]=p
 a1=dict(per_teacher=per_teacher,comparisons=corr,teacher_nominal=paper['TD']['nominal only'],teacher_safety=paper['TD']['safety baseline'],caveat='A poor teacher and associated student outcomes do not prove a mathematical student performance upper bound.');emit('analysis1_teacher_ceiling',a1)
 # Analysis 2: nominal failures, mode labels, and observed head-on collisions.
 nominal={}
 for g in ['TD','V0','V1','V3']:
  a='nominal only';s=positive(g,a);u=lab(g,a,'unknown');k=known(g,a);m=mode(g,a);valid=np.isfinite(m)&k;success=k&(s==1)&valid;fail=k&(s==0)&valid;agent=cube(g,a,lambda r,f:int('agent' in f['collision_geometry']['types']) if f and r['collision'] else 0,0);head=cube(g,a,lambda r,f:int(f['collision_geometry']['head_on_episode']) if f and r['collision'] else 0,0);opphead=cube(g,a,lambda r,f:int(f['collision_geometry']['opposite_head_on_episode']) if f and r['collision'] else 0,0);pairnum=cube(g,a,lambda r,f:sum(e.get('head_on',False) for e in f['collision_geometry']['events'] if e['type']=='agent') if f and r['collision'] else 0,0);pairden=cube(g,a,lambda r,f:sum(e['type']=='agent' for e in f['collision_geometry']['events']) if f and r['collision'] else 0,0)
  types={}
  for label in ['agent','center','outer','multi','unverified']:
   def fn(r,f,label=label):
    if not r['collision'] or not f:return 0
    t=f['collision_geometry']['types'];return int(len(t)>1 if label=='multi' else not t if label=='unverified' else label in t)
   types[label]=mean(cube(g,a,fn,0))
  nominal[g]=dict(outcomes=paper[g][a],mode_counts={str(i):int(((m==i)&valid).sum()) for i in range(4)},mixed_failure=ratio(((m==3)&fail).astype(float),fail.astype(float),u),mixed_success=ratio(((m==3)&success).astype(float),success.astype(float),u),mixed_overall=ratio(((m==3)&valid).astype(float),valid.astype(float),u),four_robots_same=mean(met(g,a,'four_same'),k),head_on_among_agent_collision=ratio(head,agent),opposite_head_on_among_agent_collision=ratio(opphead,agent),head_on_pair_events=ratio(pairnum,pairden),collision_types=types)
 expert_mode=[features[r['rollout_uid']]['mode'] for r in roles['expert']];a2=dict(groups=nominal,expert=dict(n=32,consistent=sum(m in [1,2] for m in expert_mode),four_robots_same=sum(features[r['rollout_uid']]['four_same'] for r in roles['expert'])),mode_legend={'0':'静止','1':'顺时针一致','2':'逆时针一致','3':'不一致'});emit('analysis2_nominal_failure',a2)
 # Analysis3: timeout phenomenology and collision-location pairing.
 safety={};location_pairs=[];timeout_rows=[]
 for g in ['TD','V0','V1','V3']:
  a='safety baseline';mask=(lab(g,a,'timeout')==1)&known(g,a);safety[g]={key:mean(met(g,a,key),mask) for key in ['last_progress_seconds','literal_last_progress_step','final_unfinished','geometric_opposite_pairs','moving_opposite_pairs','stall_robot_seconds','stall_runs','stall_max_seconds','moving_no_progress_robot_seconds','moving_no_progress_team_seconds']};safety[g]['timeout_episodes']=int(mask.sum());safety[g]['has_geometric_opposite_pair']=mean((met(g,a,'geometric_opposite_pairs')>0).astype(float),mask);safety[g]['has_low_speed_stall']=mean((met(g,a,'stall_runs')>0).astype(float),mask)
  near=cube(g,a,lambda r,f:None);distance=near.copy();comparable=np.zeros_like(near);eligible_pairs=np.zeros_like(near)
  for r in roles[g]:
   if r['arm']!=a or r['unknown'] or not r['timeout']:continue
   f=features.get(r['rollout_uid']);key=(g,r['controller_index'],r['state_index'],r['seed'],'nominal only');n=by.get(key);nf=features.get(n['rollout_uid']) if n else None;ix=(r['controller_index'],r['state_index'],r['seed']);timeout_rows.append(dict(group=g,index=r['controller_index'],state=r['state_index'],seed=r['seed'],**{k:f[k] for k in ['last_progress_seconds','final_unfinished','geometric_opposite_pairs','stall_robot_seconds','moving_no_progress_robot_seconds','final_radii','final_positions']}))
   if not n or not n['collision'] or n['unknown'] or not nf:continue
   eligible_pairs[ix]=1;cg=nf['collision_geometry'];rp=cg['robot_positions'];ids=[i for i in range(4) if str(i) in rp and f['unfinished_mask'][i]]
   if not ids:continue
   dd=np.array([np.linalg.norm(np.asarray(f['last40_mean_positions'][i])-np.asarray(rp[str(i)])) for i in ids]);value=float(np.median(dd));distance[ix]=value;near[ix]=value<=.5;comparable[ix]=1;location_pairs.append(dict(group=g,index=r['controller_index'],state=r['state_index'],seed=r['seed'],median_robot_distance=value,near_025=bool(value<=.25),near_050=bool(value<=.5),near_100=bool(value<=1.),collision_step=cg['step'],collision_location=cg['location'],stall_robot_positions=[f['last40_mean_positions'][i] for i in ids],per_robot_distances=dd.tolist(),nominal_raw=n['raw_path'],safety_raw=r['raw_path']))
  safety[g]['collision_to_timeout']=int(eligible_pairs.sum());safety[g]['location_comparable']=int(comparable.sum());safety[g]['near_original_collision_050']=mean(near);safety[g]['distance_to_original_collision']=mean(distance);safety[g]['distance_quantiles']=np.quantile(distance[np.isfinite(distance)],[0,.25,.5,.75,1]).tolist() if np.isfinite(distance).any() else None;safety[g]['near_sensitivity']={str(t):mean(np.where(np.isfinite(distance),(distance<=t).astype(float),np.nan)) for t in [.25,.5,1.]}
 emit('analysis3_safety_stalls',dict(groups=safety,location_pairs=location_pairs,timeout_rows=timeout_rows))
 # Analysis4: paired arm transitions, early progress and switching.
 correction={};progress_data={}
 for g in ['TD','V0','V1','V3','Z']:
  baseline='safety baseline';correction[g]={};progress_data[g]={}
  for a in ['safety baseline','F13','S500']:
   if not exists(g,a):continue
   k=known(g,a);m=mode(g,a);progress_data[g][a]={}
   for t in [100,200,300]:
    em=mode(g,a,t);prog=cube(g,a,lambda r,f:f['early'][str(t)]['progress'] if f else None);progress_data[g][a][str(t)]=dict(consistent=mean(np.isin(em,[1,2]).astype(float),k),mixed=mean((em==3).astype(float),k),four_same=mean(cube(g,a,lambda r,f:f['early'][str(t)]['four_same'] if f else None),k),progress=mean(prog,k))
   progress_data[g][a]['stable_step']=mean(met(g,a,'stable_step'),k);progress_data[g][a]['consistent_final']=mean(np.isin(m,[1,2]).astype(float),k)
   if a==baseline:continue
   b=mode(g,baseline);valid=k&known(g,baseline)&np.isfinite(m)&np.isfinite(b);eligible=valid&np.isin(b,[0,3]);correction[g][a]=dict(mixed_or_stationary_to_consistent=ratio((eligible&np.isin(m,[1,2])).astype(float),eligible.astype(float)),mixed_to_consistent=ratio((valid&(b==3)&np.isin(m,[1,2])).astype(float),(valid&(b==3)).astype(float)),stationary_to_consistent=ratio((valid&(b==0)&np.isin(m,[1,2])).astype(float),(valid&(b==0)).astype(float)),consistent_rate_difference=mean(np.isin(m,[1,2]).astype(float)-np.isin(b,[1,2]).astype(float),valid),success_gain=paired_success(positive(g,a),lab(g,a,'unknown'),positive(g,baseline),lab(g,baseline,'unknown')),rescue_of_baseline_timeout=ratio(((lab(g,baseline,'timeout')==1)&(positive(g,a)==1)&valid).astype(float),((lab(g,baseline,'timeout')==1)&valid).astype(float)))
 switch={}
 for g in ['TD','V0','V1','V3','Z']:
  pre=met(g,'S500','pre_switch_progress_rate');post=met(g,'S500','post_switch_progress_rate');apre=met(g,'S500','absorbed_pre_switch_rate');apost=met(g,'S500','absorbed_post_switch_rate');fpre=met(g,'F13','pre_switch_progress_rate');fpost=met(g,'F13','post_switch_progress_rate');mask=known(g,'S500')
  switch[g]=dict(observed_600_pre=mean(pre,mask),observed_600_post=mean(post,mask),observed_600_post_minus_pre=difference(post,pre,mask),absorbed_all_post_minus_pre=difference(apost,apre,mask),versus_fixed_F13_difference_in_differences=mean((post-pre)-(fpost-fpre),mask&known(g,'F13')),caveat='conditioned on trajectories observed through600; descriptive survivor selection, not causal switch effect')
 emit('analysis4_correction',dict(paired_transitions=correction,early_progress=progress_data,switch=switch))
 # Analysis5: zero-controller failure and its learned-controller comparators.
 zero={}
 for a in ['safety baseline','F10','F13','S300','S500']:
  k=known('Z',a);m=mode('Z',a);fail=k&(positive('Z',a)==0);zero[a]=dict(outcomes=paper['Z'][a],consistent_failures=mean(np.isin(m,[1,2]).astype(float),fail),mixed_failures=mean((m==3).astype(float),fail),stationary_failures=mean((m==0).astype(float),fail),four_same_failures=mean(met('Z',a,'four_same'),fail),moving_no_progress_failures=mean(met('Z',a,'moving_no_progress_robot_seconds'),fail),stalls_failures=mean(met('Z',a,'stall_robot_seconds'),fail),final_unfinished_failures=mean(met('Z',a,'final_unfinished'),fail))
 emit('analysis5_zero_controller',dict(zero=zero,comparators={v:dict(success=paper[v]['S500'],early=progress_data[v]['S500']) for v in ['TD','V0','V1','V3']},warning='Existing G+CBF+always-on clockwise0.10m/s achieved32/32 in round30d; this zero-controller comparison cannot establish superiority to all hand-designed controllers.'))
 # Analysis6: shared-noise mode vs collisions vs motion without progress, paired by teacher.
 paired_modes={}
 for v,b in [('V1','V0'),('V3','V1')]:
  a='nominal only';valid=known(v,a)&known(b,a);mv=mode(v,a);mb=mode(b,a);newtime=(lab(v,a,'timeout')==1)&valid;both_timeout=newtime&(lab(b,a,'timeout')==1)
  paired_modes[v+'_minus_'+b]=dict(success=paired_success(positive(v,a),lab(v,a,'unknown'),positive(b,a),lab(b,a,'unknown')),mixed_rate_difference=mean((mv==3).astype(float)-(mb==3).astype(float),valid),collision_rate_difference=mean(lab(v,a,'collision')-lab(b,a,'collision'),valid),timeout_rate_difference=mean(lab(v,a,'timeout')-lab(b,a,'timeout'),valid),moving_no_progress_robot_seconds_difference=difference(met(v,a,'moving_no_progress_robot_seconds'),met(b,a,'moving_no_progress_robot_seconds'),valid),moving_no_progress_difference_both_timeout=difference(met(v,a,'moving_no_progress_robot_seconds'),met(b,a,'moving_no_progress_robot_seconds'),both_timeout),early_mixed_differences={str(t):mean((mode(v,a,t)==3).astype(float)-(mode(b,a,t)==3).astype(float),valid) for t in [100,200,300]},mode_transition_counts=[[int(((mb==i)&(mv==j)&valid).sum()) for j in range(4)] for i in range(4)])
 emit('analysis6_relabeling_shared_noise',dict(paired=paired_modes,nominal=nominal))
 # Analysis7: frozen selections and controller-level post-hoc oracle.
 frozen=read(HERE/'FROZEN_CRITIC_REUSE.json')
 for g in groups:
  if g=='expert':continue
  if g in frozen:
   s=np.stack([positive(g,frozen[g][str(i)])[i] for i in range(counts[g])]);u=np.stack([lab(g,frozen[g][str(i)],'unknown')[i] for i in range(counts[g])]);co=np.stack([lab(g,frozen[g][str(i)],'collision')[i] for i in range(counts[g])]);to=np.stack([lab(g,frozen[g][str(i)],'timeout')[i] for i in range(counts[g])]);paper[g]['critic']=dict(success=outcome_rate(s,u),collision=outcome_rate(co*(1-u),u),timeout=outcome_rate(to*(1-u),u),n=int(s.size),controllers=counts[g],independent_states=32,source_rounds=sorted({pathlib.Path(r['raw_path']).relative_to(D).parts[0] for r in roles[g]}),evidence_level='DEVELOPMENT',choices=frozen[g])
  else:paper[g]['critic']=dict(status='MISSING',reason='No compatible frozen A3 selection for this controller and pipeline')
  menus=['F10','F13','S300','S500']
  if all(exists(g,a) for a in menus):
   ss=np.stack([positive(g,a) for a in menus],-1);uu=np.stack([lab(g,a,'unknown') for a in menus],-1);rate=ss.mean((1,2));up=(ss+uu).mean((1,2));ix=rate.argmax(-1);chosen=np.stack([ss[i,...,ix[i]] for i in range(counts[g])]);co=np.stack([lab(g,menus[ix[i]],'collision')[i] for i in range(counts[g])]);to=np.stack([lab(g,menus[ix[i]],'timeout')[i] for i in range(counts[g])]);bl=np.stack([draws(ss[...,j],True) for j in range(4)],-1).max(-1).mean(1);bu=np.stack([draws(ss[...,j]+uu[...,j],True) for j in range(4)],-1).max(-1).mean(1);paper[g]['oracle']=dict(success=dict(lower=dict(estimate=float(rate.max(-1).mean()),**quantiles(bl)),upper=dict(estimate=float(up.max(-1).mean()),**quantiles(bu)),CI_unknown_envelope=[quantiles(bl)['CI95'][0],quantiles(bu)['CI95'][1]],unknown=int(uu.sum())),collision=outcome_rate(co,np.stack([uu[i,...,ix[i]] for i in range(counts[g])])),timeout=outcome_rate(to,np.stack([uu[i,...,ix[i]] for i in range(counts[g])])),n=int(chosen.size),controllers=counts[g],independent_states=32,source_rounds=sorted({pathlib.Path(r['raw_path']).relative_to(D).parts[0] for r in roles[g]}),evidence_level='DEVELOPMENT',selected={str(i):menus[ix[i]] for i in range(counts[g])},oracle_unit='per-controller best aggregate arm; arm reselected within bootstrap')
  else:paper[g]['oracle']=dict(status='MISSING',reason='Incomplete four-arm correction menu')
 emit('analysis7_paper_cells',paper)
 # Curves for plotting, all known terminal traces absorbed only for display.
 curve={}
 for g in groups:
  curve[g]={}
  for a in (['expert'] if g=='expert' else ARMS):
   rs=[r for r in roles[g] if r['arm']==a and not r['unknown'] and r['rollout_uid'] in features]
   if not rs:continue
   fs=[features[r['rollout_uid']] for r in rs];dd=np.asarray([f['distance_curve'] for f in fs]);modes=np.asarray([f['mode_curve'] for f in fs]);arrival=np.asarray([f['T'] if r['success'] else np.inf for r,f in zip(rs,fs)]);times=np.arange(0,701,10);curve[g][a]=dict(steps=times,mean_remaining_distance=dd.mean(0),mean_progress=(dd[:,0,None]-dd).mean(0),consistent_fraction=np.isin(modes,[1,2]).mean(0),mixed_fraction=(modes==3).mean(0),success_cumulative_fraction=(arrival[:,None]<=times).mean(0),n_known=len(rs),unknown=sum(r['unknown'] for r in roles[g] if r['arm']==a),end_absorption_is_display_only=True)
 write(HERE/'PLOT_DATA.json',curve);write(HERE/'METRIC_ERROR_AUDIT.json',dict(errors=metric_errors,expected_unique=len({r['rollout_uid'] for r in manifest}),computed_unique=len(features),kinematic_max_abs_error=max(f['kinematic_max_abs_error'] for f in features.values()),reconstructed_velocity_traces=sum(f['velocities_reconstructed'] for f in features.values())))
 write(HERE/'results.json',dict(utc=utc(),status='ANALYZED',evidence_level='DEVELOPMENT',analyses={'teacher_ceiling':a1,'nominal_failure':a2,'safety_stalls':'analysis3_safety_stalls.json','correction':'analysis4_correction.json','zero_controller':'analysis5_zero_controller.json','shared_noise':'analysis6_relabeling_shared_noise.json','paper_cells':'analysis7_paper_cells.json'},metric_errors=metric_errors,precision=precision(),new_rollouts=0,training_steps=0,DB_opened=False,bootstrap=dict(seed=2026101707,replicates=5000,clusters='controller then state; paired seeds, arms, teacher variants')));seal('ANALYSIS_COMPLETE.receipt.json',[HERE/'results.json']+[HERE/(n+'.json') for n in ['analysis1_teacher_ceiling','analysis2_nominal_failure','analysis3_safety_stalls','analysis4_correction','analysis5_zero_controller','analysis6_relabeling_shared_noise','analysis7_paper_cells']],new_rollouts=0,training_steps=0,DB_writes=0);print('seven analyses complete',utc(),flush=True)
if __name__=='__main__':main()
