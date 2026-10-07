#!/usr/bin/env python3
"""Freeze, run, and aggregate the mandatory fresh-WIDE t=0 extrapolation."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,platform,sys,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay');HERE=ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1';DIRECT=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
for p in (SYSROOT,ROOT):sys.path.insert(0,str(p))
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);N=300;IC_SEED=2026092805;FLOW_SEED=2026092806
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def chash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write_csv(p,rr,fields=None):
 if fields is None:fields=list(rr[0])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
class G(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)
class GLowJ(nn.Module):
 @nn.compact
 def __call__(self,x):
  low=jnp.array([0.,-.5,0.]);high=jnp.array([1.25,.5,.75]);x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return low+nn.sigmoid(nn.Dense(3)(x))*(high-low)
def prepare():
 # Enforce TEST freeze before cohort construction.
 if not (HERE/'intermediate_test_summary.json').exists():raise RuntimeError('TEST not frozen')
 source=ROOT/'diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py';reference=ROOT/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json'
 spec=importlib.util.spec_from_file_location('fresh_source_cm',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 fresh=mod.generate(IC_SEED,N);prior,identities,_,prior_ids,assets=mod.collect_prior();dist=np.linalg.norm(fresh[:,None].astype(float)-prior[None].astype(float),axis=(2,3));ids={f'orthoflow3_center_margin_fresh_v1_{i:04d}' for i in range(N)};seeds={v for _,v,_ in identities};coll={'exact_initial':int(np.sum(dist==0)),'seed':sorted(seeds&{IC_SEED,FLOW_SEED}),'source_id':sorted(ids&prior_ids)}
 if coll['exact_initial'] or coll['seed'] or coll['source_id']:raise RuntimeError(('fresh overlap',coll))
 ref=json.load(open(reference));selected={a:json.load(open(HERE/f'g_{a}/selected_checkpoint.json')) for a in ('center','margin')};low=json.load(open(DIRECT/'selected_checkpoint.json'));utc=datetime.now(timezone.utc).isoformat()
 manifest={'schema':'orthoflow3_center_vs_margin_fresh_wide_v1','frozen_before_rollout':True,'frozen_utc':utc,'episode_count':N,'generator':{'implementation':'authoritative WIDE uniform generator; no rejection','ic_seed':IC_SEED,'source_script':str(source),'source_sha256':sha(source)},'flow_randomness':{'root_seed':FLOW_SEED,'matched_across_controllers':True},'controllers':['safety','g_lowj','g_center','g_margin'],'environment':ref['environment'],'cbf':ref['cbf'],'outcome_protocol':ref['outcome_protocol'],'checkpoint_sha256':{'g_lowj':low['checkpoint_sha256'],'g_center':selected['center']['checkpoint_sha256'],'g_margin':selected['margin']['checkpoint_sha256']},'overlap_audit':{'status':'PASS','collisions':coll,'minimum_l2_distance_to_prior':float(dist.min()),'prior_initial_records_checked':len(prior),'assets_checked':len(set(assets))},'episodes':[{'episode_index':i,'rollout_id':i,'source_id':f'orthoflow3_center_margin_fresh_v1_{i:04d}','initial_positions':p.tolist()} for i,p in enumerate(fresh)]};manifest['content_sha256']=chash(manifest);dump(HERE/'fresh_wide_manifest.json',manifest);print(json.dumps({'frozen':N,'manifest_sha256':sha(HERE/'fresh_wide_manifest.json'),'overlap':manifest['overlap_audit']},indent=2))
def load_new(arm):
 s=json.load(open(HERE/f'g_{arm}/selected_checkpoint.json'));m=G();t=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));return m,serialization.from_bytes(t,Path(s['checkpoint']).read_bytes()),s
def outcome(env,error):
 if error:return 'other',error['type']
 s=env.summary()
 if s['wall_collision']:return 'collision','wall_collision'
 if s['agent_collision']:return 'collision','agent_collision'
 if s['success']:return 'success','success'
 if s['deadlock']:return 'deadlock','strict_deadlock'
 return 'timeout','timeout'
def run(shard,shards):
 jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name','gpu')
 from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
 from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
 from shared_control.basis_families import get_basis_family
 from single_integrator.cbf import CBFConfig,barrier_constraints
 from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
 from single_integrator.evaluate import load_policy
 manifest=json.load(open(HERE/'fresh_wide_manifest.json'));norm=json.load(open(HERE/'normalization.json'));hm=np.array(norm['h_mean']);hs=np.array(norm['h_std']);oldnorm=json.load(open(DIRECT/'normalization.json'));ohm=np.array(oldnorm['h_mean']);ohs=np.array(oldnorm['h_std']);models={a:load_new(a)[:2] for a in ('center','margin')}
 lsel=json.load(open(DIRECT/'selected_checkpoint.json'));lm=GLowJ();lt=lm.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));lp=serialization.from_bytes(lt,Path(lsel['checkpoint']).read_bytes())
 config=Config(**manifest['environment']);cbf=CBFConfig(**manifest['cbf']);policy,prov=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');sample=jax.jit(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]);basis=get_basis_family('orthoflow3');root=int(manifest['flow_randomness']['root_seed'])
 episodes=[e for i,e in enumerate(manifest['episodes']) if i%shards==shard];out=HERE/'raw/fresh_wide';out.mkdir(parents=True,exist_ok=True);path=out/f'shard{shard}.jsonl';rows=[] if not path.exists() else read_jsonl(path);seen={(x['episode_index'],x['controller']) for x in rows};steps=0;started=time.monotonic()
 for ep in episodes:
  episode_key=jax.random.fold_in(jax.random.PRNGKey(np.uint32(root)),int(ep['rollout_id']))
  for ctl in manifest['controllers']:
   if (ep['episode_index'],ctl) in seen:continue
   env=GiveWayEnv(config);env.reset(np.asarray(ep['initial_positions'],float));eta=None;feature=None;jdef=0.;error=None;retries1=retries2=invalid=0;minres=np.inf;maxspeed=-np.inf
   for step in range(config.max_steps):
    try:
     obs=np.asarray(env.observation(),np.float32);key=jax.random.fold_in(episode_key,step);flow=bounded_nominal(np.asarray(sample(jnp.asarray(obs),key),float),config.max_speed);A,lower,_=barrier_constraints(env.snapshot(),cbf);safe,_,r1,_=project_velocity_with_retry(flow,A,lower,config.max_speed,cbf);retries1+=int(r1)
     if step==0:
      feature,_=StartupAwareFeatureBuilder().build(env,{'u_flow':flow,'u_safe':safe},config,cbf)
      if ctl=='safety':eta=np.zeros(3)
      elif ctl=='g_lowj':eta=np.asarray(lm.apply(lp,jnp.asarray(((feature-ohm)/ohs).astype(np.float32))[None])[0],float)
      else:
       arm=ctl[2:];m,p=models[arm];yn=np.asarray(m.apply(p,jnp.asarray(((feature-hm)/hs).astype(np.float32))[None])[0],float);eta=AFF+SCALE*yn
     correction=np.zeros_like(safe) if ctl=='safety' else basis.compute(env.positions,env.goals,safe,config.max_speed).correction(eta);exe,_,r2,_=project_velocity_with_retry(safe+correction,A,lower,config.max_speed,cbf);retries2+=int(r2);res=float(np.min(A@exe.reshape(4)-lower));spd=float(np.max(np.linalg.norm(exe,axis=-1)-config.max_speed));minres=min(minres,res);maxspeed=max(maxspeed,spd)
     if not np.isfinite(exe).all() or res < -cbf.feasibility_tol or spd > cbf.speed_tol:invalid+=1;raise RuntimeError(('invalid projected action',res,spd))
     jdef+=config.dt*float(np.sum((exe-safe)**2));env.step(exe);steps+=1
     if env.done:break
    except Exception as exc:error={'type':type(exc).__name__,'message':str(exc),'step':step};break
   lab,failure=outcome(env,error);summ=env.summary();row={'episode_index':ep['episode_index'],'rollout_id':ep['rollout_id'],'source_id':ep['source_id'],'controller':ctl,'outcome':lab,'failure_type':failure,'success':lab=='success','deadlock':lab=='deadlock','timeout':lab=='timeout','collision':lab=='collision','wall_collision':bool(summ['wall_collision']),'agent_collision':bool(summ['agent_collision']),'other_failure':lab=='other','episode_steps':int(env.step_count),'J_def':jdef,'eta':eta.tolist(),'eta_norm':float(np.linalg.norm(eta)),'eta_query_count':0 if ctl=='safety' else 1,'feature_sha256':hashlib.sha256(np.asarray(feature,dtype=np.float64).tobytes()).hexdigest(),'h0':np.asarray(feature,dtype=float).tolist() if ctl=='safety' else None,'second_projection':True,'first_projection_retries':retries1,'second_projection_retries':retries2,'minimum_linear_residual':None if not np.isfinite(minres) else minres,'maximum_speed_excess':None if not np.isfinite(maxspeed) else maxspeed,'invalid_actions':invalid,'execution_error':error,'manifest_sha256':sha(HERE/'fresh_wide_manifest.json')};rows.append(row)
   with path.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
  print(json.dumps({'episode':ep['episode_index'],'done_controllers':4}),flush=True)
 runtime={'shard':shard,'shards':shards,'new_controller_rollouts':len([x for x in rows if (x['episode_index'],x['controller']) not in set()]),'records':len(rows),'physical_steps_this_process':steps,'wall_seconds':time.monotonic()-started,'outcomes':dict(Counter(x['outcome'] for x in rows)),'devices':[str(x) for x in jax.devices()],'host':platform.node(),'cpu_request':os.environ.get('SLURM_CPUS_PER_TASK')};dump(out/f'shard{shard}_runtime.json',runtime)
def bootstrap_delta(epmaps,a,b,metric,n=10000):
 ids=sorted(epmaps);rng=np.random.default_rng(2026092807);vals=[]
 for _ in range(n):
  ii=rng.integers(0,len(ids),len(ids));sample=[ids[i] for i in ii];vals.append(np.mean([metric(epmaps[i][a],epmaps[i][b]) for i in sample]))
 point=np.mean([metric(epmaps[i][a],epmaps[i][b]) for i in ids]);return {'point':float(point),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'bootstrap_samples':n}
def finalize():
 rr=[]
 for p in sorted((HERE/'raw/fresh_wide').glob('shard*.jsonl')):rr+=read_jsonl(p)
 if len(rr)!=N*4:raise RuntimeError(('incomplete fresh',len(rr)))
 write_csv(HERE/'fresh_wide_results.csv',[{k:(json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v) for k,v in x.items() if k!='h0'} for x in rr])
 byctl={};
 for ctl in ('safety','g_lowj','g_center','g_margin'):
  z=[x for x in rr if x['controller']==ctl];o=Counter(x['outcome'] for x in z);j=[x['J_def'] for x in z if x['success']];byctl[ctl]={'episodes':len(z),'success':o['success'],'strict_deadlock':o['deadlock'],'timeout':o['timeout'],'collision':o['collision'],'other':o['other'],'episode_length_mean':float(np.mean([x['episode_steps'] for x in z])),'successful_J_def_mean':float(np.mean(j)) if j else None,'successful_J_def_median':float(np.median(j)) if j else None,'invalid_actions':sum(x['invalid_actions'] for x in z)}
 ep=defaultdict(dict)
 for x in rr:ep[int(x['episode_index'])][x['controller']]=x
 pair=[]
 for i,z in sorted(ep.items()):
  s=z['safety']['success'];c=z['g_center']['success'];m=z['g_margin']['success'];l=z['g_lowj']['success'];pair.append({'episode_index':i,'safety_success':s,'lowj_success':l,'center_success':c,'margin_success':m,'center_rescue':not s and c,'margin_rescue':not s and m,'center_break':s and not c,'margin_break':s and not m,'center_break_recovered_by_margin':s and not c and m,'new_margin_break':s and c and not m,'center_rescue_preserved_by_margin':not s and c and m,'new_margin_rescue':not s and not c and m})
 write_csv(HERE/'fresh_wide_pairwise.csv',pair)
 for ctl in ('g_lowj','g_center','g_margin'):
  byctl[ctl]['rescue']=sum(not ep[i]['safety']['success'] and ep[i][ctl]['success'] for i in ep);byctl[ctl]['break']=sum(ep[i]['safety']['success'] and not ep[i][ctl]['success'] for i in ep);byctl[ctl]['net_success_delta']=byctl[ctl]['success']-byctl['safety']['success']
 # Support distances in TRAIN-frozen normalized h coordinates.
 norm=json.load(open(HERE/'normalization.json'));hm=np.array(norm['h_mean']);hs=np.array(norm['h_std']);data=list(csv.DictReader(open(HERE/'verified_ball_dataset.csv')));feat=np.load(HERE/'learning_features.npz')['features'];train=np.array([x['split']=='train' for x in data]);tx=(feat[train]-hm)/hs;trows=[x for x in data if x['split']=='train'];support=[]
 for i,z in sorted(ep.items()):
  h=(np.array(z['safety']['h0'])-hm)/hs;ds=np.linalg.norm(tx-h,axis=1);k=int(np.argmin(ds));support.append({'episode_index':i,'nearest_h_distance':float(ds[k]),'nearest_state_id':trows[k]['state_id'],'nearest_label_kind':trows[k]['label_kind'],'nearest_anchor_family':trows[k]['anchor_family'],'center_success':z['g_center']['success'],'margin_success':z['g_margin']['success'],'safety_success':z['safety']['success']})
 write_csv(HERE/'fresh_h_support_distance.csv',support)
 def dist_stats(ctl,success):
  z=[x['nearest_h_distance'] for x in support if x[f'{ctl}_success']==success];return {'n':len(z),'mean':float(np.mean(z)) if z else None,'median':float(np.median(z)) if z else None}
 cis={'success_rate_margin_minus_center':bootstrap_delta(ep,'g_margin','g_center',lambda a,b:float(a['success'])-float(b['success'])),'break_rate_margin_minus_center':bootstrap_delta(ep,'g_margin','g_center',lambda a,b:float((ep[a['episode_index']]['safety']['success'] and not a['success']))-float((ep[b['episode_index']]['safety']['success'] and not b['success'])))}
 summary={'controllers':byctl,'center_vs_margin':{'center_break_recovered_by_margin':sum(x['center_break_recovered_by_margin'] for x in pair),'new_margin_breaks':sum(x['new_margin_break'] for x in pair),'center_rescue_preserved_by_margin':sum(x['center_rescue_preserved_by_margin'] for x in pair),'new_margin_rescues':sum(x['new_margin_rescue'] for x in pair),'bootstrap_95ci':cis},'support_distance':{'all':{'mean':float(np.mean([x['nearest_h_distance'] for x in support])),'median':float(np.median([x['nearest_h_distance'] for x in support])),'min':float(np.min([x['nearest_h_distance'] for x in support])),'max':float(np.max([x['nearest_h_distance'] for x in support]))},'center_success':dist_stats('center',True),'center_failure':dist_stats('center',False),'margin_success':dist_stats('margin',True),'margin_failure':dist_stats('margin',False),'nearest_kind_counts':dict(Counter(x['nearest_label_kind'] for x in support))},'interpretation':'OOD t0 extrapolation: training contains zero true t0 labels'}
 dump(HERE/'fresh_wide_summary.json',summary);print(json.dumps(summary,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','finalize']);ap.add_argument('--shard',type=int);ap.add_argument('--shards',type=int,default=6);a=ap.parse_args()
 if a.stage=='prepare':prepare()
 elif a.stage=='run':run(a.shard,a.shards)
 else:finalize()
if __name__=='__main__':main()
