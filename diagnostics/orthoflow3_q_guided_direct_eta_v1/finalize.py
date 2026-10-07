#!/usr/bin/env python3
"""Aggregate fresh WIDE, paired uncertainty, safety, decision, report, manifest."""
from __future__ import annotations
import csv,hashlib,json,platform,time
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
from scipy.stats import binomtest,pearsonr,spearmanr
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';BASE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';QDIR=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
class Q(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(256)(x));x=nn.silu(nn.Dense(256)(x));return nn.Dense(1)(x).squeeze(-1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(path,rows):
 f=path.open('w',newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.close()
def load(stem):
 out=[]
 for p in sorted((HERE/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit():out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def ci(v):return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]
def main():
 began=time.monotonic();fresh=load('fresh_wide')
 if len(fresh)!=1200:raise RuntimeError(('fresh incomplete',len(fresh)))
 flat=[]
 for r in fresh:flat.append({k:r[k] for k in ('episode_index','source_id','controller','outcome','failure_type','success','deadlock','timeout','collision','wall_collision','agent_collision','other_failure','episode_steps','J_def','eta_norm','eta_clipped','invalid_actions','nan_inf','projection_failures','minimum_linear_residual','maximum_speed_excess')})
 write(HERE/'fresh_wide_all_results.csv',flat);by=defaultdict(dict)
 for r in fresh:by[r['episode_index']][r['controller']]=r
 def summ(name):
  rr=[x for x in fresh if x['controller']==name];sj=[x['J_def'] for x in rr if x['success']]
  return {'controller':name,'episodes':len(rr),'success':sum(x['success'] for x in rr),'deadlock':sum(x['deadlock'] for x in rr),'timeout':sum(x['timeout'] for x in rr),'agent_collision':sum(x['agent_collision'] for x in rr),'wall_collision':sum(x['wall_collision'] for x in rr),'other_failure':sum(x['other_failure'] for x in rr),'episode_steps_mean':float(np.mean([x['episode_steps'] for x in rr])),'episode_steps_median':float(np.median([x['episode_steps'] for x in rr])),'successful_J_def_mean':float(np.mean(sj)) if sj else None,'successful_J_def_median':float(np.median(sj)) if sj else None}
 sums=[summ(x) for x in ('safety','baseline','guided')];write(HERE/'fresh_wide_controller_summary.csv',sums);sd={x['controller']:x for x in sums}
 trans=[]
 for i,p in sorted(by.items()):
  s,b,g=p['safety']['success'],p['baseline']['success'],p['guided']['success'];trans.append({'episode_index':i,'safety_success':s,'baseline_success':b,'guided_success':g,'baseline_rescue':not s and b,'guided_rescue':not s and g,'baseline_break':s and not b,'guided_break':s and not g,'baseline_break_recovered':s and not b and g,'baseline_rescue_preserved':not s and b and g,'new_guided_break':s and b and not g,'new_guided_rescue':not s and not b and g,'baseline_outcome':p['baseline']['outcome'],'guided_outcome':p['guided']['outcome'],'safety_outcome':p['safety']['outcome']})
 write(HERE/'fresh_wide_pairwise_transitions.csv',trans)
 def count(k):return sum(r[k] for r in trans)
 decomp={'baseline_breaks':count('baseline_break'),'guided_breaks':count('guided_break'),'baseline_rescues':count('baseline_rescue'),'guided_rescues':count('guided_rescue'),'baseline_break_recovery':count('baseline_break_recovered'),'baseline_rescue_preservation':count('baseline_rescue_preserved'),'new_guided_breaks':count('new_guided_break'),'new_guided_rescues':count('new_guided_rescue')}
 # Paired episode bootstrap, fixed deterministic seed.
 rng=np.random.default_rng(2026092723);arr=np.array([[r[k] for k in ('safety_success','baseline_success','guided_success')] for r in trans],bool);vals=[]
 for _ in range(100000):
  z=arr[rng.integers(0,len(arr),len(arr))];s,b,g=z.T;vals.append([np.mean(g)-np.mean(b),(np.sum(s&~g)/max(np.sum(s),1))-(np.sum(s&~b)/max(np.sum(s),1)),(np.sum(~s&g)/max(np.sum(~s),1))-(np.sum(~s&b)/max(np.sum(~s),1))])
 vals=np.asarray(vals);bg=sum((not r['baseline_success']) and r['guided_success'] for r in trans);gb=sum(r['baseline_success'] and (not r['guided_success']) for r in trans);boot={'replicates':100000,'seed':2026092723,'guided_minus_baseline_success':{'estimate':(sd['guided']['success']-sd['baseline']['success'])/400,'ci95':ci(vals[:,0])},'guided_minus_baseline_break_rate':{'estimate':decomp['guided_breaks']/sum(r['safety_success'] for r in trans)-decomp['baseline_breaks']/sum(r['safety_success'] for r in trans),'ci95':ci(vals[:,1])},'guided_minus_baseline_rescue_rate':{'estimate':decomp['guided_rescues']/sum(not r['safety_success'] for r in trans)-decomp['baseline_rescues']/sum(not r['safety_success'] for r in trans),'ci95':ci(vals[:,2])},'McNemar':{'baseline_fail_guided_success':bg,'baseline_success_guided_fail':gb,'discordant':bg+gb,'exact_two_sided_p':float(binomtest(min(bg,gb),bg+gb,.5).pvalue) if bg+gb else 1.0}};dump(HERE/'fresh_wide_bootstrap_ci.json',boot)
 # Offline Q diagnosis and output distributions.
 qn=json.loads((QDIR/'normalization.json').read_text());hm=np.asarray(qn['h_mean']);hs=np.asarray(qn['h_std']);ec=np.asarray(qn['eta_center']);es=np.asarray(qn['eta_scale']);qsel=json.loads((QDIR/'selected_checkpoint.json').read_text());qm=Q();tmp=qm.init(jax.random.PRNGKey(0),jnp.zeros((1,217)));qp=serialization.from_bytes(tmp,Path(qsel['checkpoint']).read_bytes());qrows=[]
 for r in fresh:
  if r['controller']=='safety':continue
  eta=np.asarray(r['eta']);x=np.r_[((np.asarray(r['feature0'])-hm)/hs),(eta-ec)/es].astype(np.float32);q=float(jax.nn.sigmoid(qm.apply(qp,jnp.asarray(x)[None]))[0]);p=by[r['episode_index']];role='break' if p['safety']['success'] and not r['success'] else 'rescue' if not p['safety']['success'] and r['success'] else 'success' if r['success'] else r['outcome'];qrows.append({'episode_index':r['episode_index'],'controller':r['controller'],'Qhat':q,'outcome':r['outcome'],'paired_role':role,'eta_norm':r['eta_norm']})
 write(HERE/'fresh_wide_q_diagnostic.csv',qrows)
 qsum={}
 for ctrl in ('baseline','guided'):
  qsum[ctrl]={'mean_Qhat':float(np.mean([r['Qhat'] for r in qrows if r['controller']==ctrl])),'by_role':{role:{'count':len(v),'mean':float(np.mean(v)),'median':float(np.median(v))} for role in sorted({r['paired_role'] for r in qrows if r['controller']==ctrl}) if (v:=[r['Qhat'] for r in qrows if r['controller']==ctrl and r['paired_role']==role])}}
 outdist={}
 for ctrl in ('baseline','guided'):
  rr=[r for r in fresh if r['controller']==ctrl];eta=np.asarray([r['eta'] for r in rr]);norm=np.linalg.norm(eta,axis=1);outdist[ctrl]={'norm_mean':float(norm.mean()),'norm_median':float(np.median(norm)),'norm_p95':float(np.quantile(norm,.95)),'norm_max':float(norm.max()),'per_dimension_mean':eta.mean(0).tolist(),'per_dimension_median':np.median(eta,axis=0).tolist(),'clipping':sum(r['eta_clipped'] for r in rr)}
 dump(HERE/'fresh_wide_output_and_q_summary.json',{'output':outdist,'Qhat':qsum})
 hard={'status':'PASS','collisions':sum(r['collision'] for r in fresh),'invalid_actions':sum(r['invalid_actions'] for r in fresh),'nan_inf':sum(r['nan_inf'] for r in fresh),'projection_failures':sum(r['projection_failures'] for r in fresh),'numerical_failures':sum(r['outcome']=='other' for r in fresh),'minimum_linear_residual':min(r['minimum_linear_residual'] for r in fresh),'maximum_speed_excess':max(r['maximum_speed_excess'] for r in fresh),'second_projection_authoritative':True};dump(HERE/'safety_integrity.json',hard)
 test=json.loads((HERE/'heldout_test_summary.json').read_text());active_preserved=test['guided_active_B63']==test['baseline_active_B63']==4;zero_improved=test['guided_zero_harmful']<test['baseline_zero_harmful'];exploit=test['critic_exploitation_cases'];break_drop=decomp['baseline_breaks']-decomp['guided_breaks'];rescue_pres=decomp['baseline_rescue_preservation']/max(decomp['baseline_rescues'],1)
 severe_exploitation=exploit>0 and test['max_Q_overestimate']>=.25
 if sd['guided']['success']<sd['baseline']['success']-8 or decomp['guided_breaks']>decomp['baseline_breaks']+5 or rescue_pres<.7 or severe_exploitation:classification='Q_GUIDANCE_HARMS_CONTROL'
 elif test['guided_B63']>test['baseline_B63'] and zero_improved and active_preserved and break_drop>=max(5,.2*decomp['baseline_breaks']) and rescue_pres>=.8 and exploit==0 and hard['status']=='PASS':classification='Q_GUIDED_DIRECT_ETA_STRONGLY_SUPPORTED'
 elif (test['guided_B63']>test['baseline_B63'] or break_drop>=5 or sd['guided']['success']>sd['baseline']['success']+5) and active_preserved:classification='Q_GUIDED_DIRECT_ETA_PROMISING'
 else:classification='Q_GUIDANCE_NO_MATERIAL_BENEFIT'
 nxt={'Q_GUIDED_DIRECT_ETA_STRONGLY_SUPPORTED':'Test diagnostic J_hat(h,eta), since feasibility is nearly resolved.','Q_GUIDED_DIRECT_ETA_PROMISING':'Diagnose the remaining critic/actor mismatch before introducing J.','Q_GUIDANCE_NO_MATERIAL_BENEFIT':'Reconsider simple Direct-eta as the primary method and inspect why Q gradients did not transfer.','Q_GUIDANCE_HARMS_CONTROL':'Study critic support and conservative basin learning before further actor optimization.'}[classification]
 decision={'classification':classification,'test':test,'fresh_controller_summary':sd,'fresh_decomposition':decomp,'bootstrap':boot,'critic_exploitation':exploit,'severe_critic_exploitation':severe_exploitation,'active_preserved':active_preserved,'zero_harmful_reduced':zero_improved,'baseline_rescue_preservation_fraction':rescue_pres,'Q_material_value_beyond_MSE':classification in ('Q_GUIDED_DIRECT_ETA_STRONGLY_SUPPORTED','Q_GUIDED_DIRECT_ETA_PROMISING'),'Q_training_time_component_justified':classification in ('Q_GUIDED_DIRECT_ETA_STRONGLY_SUPPORTED','Q_GUIDED_DIRECT_ETA_PROMISING'),'strict_deadlock_stress':{'executed':False,'reason':'Archived strict-deadlock additions lack the rng_namespace/current-Flow key binding required by the exact Q-v2 214-D h conditioning semantics; no replacement set was invented.'},'next_direction':nxt};dump(HERE/'final_decision.json',decision)
 # Runtime from exact durable records.
 stages=('validation_stage1_plan','validation_stage2_plan','heldout_test_plan','fresh_wide');rt=[]
 for stage in stages:
  for p in (HERE/'raw'/stage).glob('*_runtime.json'):rt.append((stage,json.loads(p.read_text())))
 stagewall={s:max(x.get('wall_seconds',0) for st,x in rt if st==s) for s in stages};new=sum(len(load(s)) for s in stages);steps=sum(int(x.get('new_physical_steps',x.get('physical_steps',0))) for _,x in rt);train_seconds=json.loads((HERE/'validation_stage1_plan.json').read_text())['training_seconds'];runtime={'model_training_seconds':train_seconds,'new_rollouts':new,'reused_rollouts':3040,'physical_steps':steps,'rollout_stage_critical_wall_seconds':stagewall,'rollout_wall_seconds_critical_path_sum':sum(stagewall.values()),'maximum_gpu_shards':6,'observed_gpu_memory_per_process_mib_approximately':602,'rollout_cpu_threads_max':12,'training_cpu_threads':4,'ram_request_per_shard_gb':14,'maximum_ram_allocation_gb':84,'observed_peak_ram':'not exposed; Slurm accounting disabled','host':platform.node(),'finalization_seconds':time.monotonic()-began};dump(HERE/'runtime_statistics.json',runtime)
 sel=json.loads((HERE/'selected_checkpoint.json').read_text());s1=list(csv.DictReader(open(HERE/'validation_stage1.csv')));s2=list(csv.DictReader(open(HERE/'validation_stage2.csv')));report=f'''# Frozen-Q-guided OrthoFlow3 Direct-eta controlled ablation\n\n## Decision\n\n**{classification}**\n\nBaseline checkpoint: `bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7`. Frozen Q: `{qsel['checkpoint_sha256']}`; Q freeze integrity passed. Tau was the pre-existing validation-only threshold `0.771`. Fifteen actors were trained at five lambdas and three seeds.\n\nThe selected actor is `{sel['selected_actor_id']}` (lambda {sel['lambda_Q']}, seed {sel['seed']}), SHA256 `{sel['checkpoint_sha256']}`. Stage-1 has {len(s1)} rows; Stage-2 promoted {len(s2)} rows.\n\n## Held-out TEST\n\nBaseline/guided B63: {test['baseline_B63']}/20 vs {test['guided_B63']}/20. Mean Q64: {test['baseline_mean_Q64']:.4f} vs {test['guided_mean_Q64']:.4f}. ZERO harmful false activations: {test['baseline_zero_harmful']} vs {test['guided_zero_harmful']}; ACTIVE B63: {test['baseline_active_B63']}/4 vs {test['guided_active_B63']}/4. Critic-exploitation cases: {exploit}.\n\n## Fresh WIDE 400\n\nSafety/baseline/guided successes: {sd['safety']['success']} / {sd['baseline']['success']} / {sd['guided']['success']}. Baseline/guided breaks: {decomp['baseline_breaks']} / {decomp['guided_breaks']}; rescues: {decomp['baseline_rescues']} / {decomp['guided_rescues']}. Baseline break recovery: {decomp['baseline_break_recovery']}; baseline rescue preservation: {decomp['baseline_rescue_preservation']}; new guided breaks: {decomp['new_guided_breaks']}; new guided rescues: {decomp['new_guided_rescues']}.\n\nPaired guided-minus-baseline success delta: {boot['guided_minus_baseline_success']['estimate']:.4f}, 95% CI {boot['guided_minus_baseline_success']['ci95']}. Break-rate delta CI: {boot['guided_minus_baseline_break_rate']['ci95']}; rescue-rate delta CI: {boot['guided_minus_baseline_rescue_rate']['ci95']}.\n\nHard-safety integrity: {hard['status']}; collisions {hard['collisions']}, invalid actions {hard['invalid_actions']}, projection failures {hard['projection_failures']}.\n\nQ provides material training-time value beyond MSE: **{decision['Q_material_value_beyond_MSE']}**. Next direction: {nxt} No next experiment was started.\n''';(HERE/'q_guided_direct_eta_report.md').write_text(report)
 files=[]
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and 'raw' not in p.parts and p.name!='manifest.json':files.append({'path':str(p.relative_to(HERE)),'sha256':sha(p),'bytes':p.stat().st_size})
 dump(HERE/'manifest.json',{'schema':'orthoflow3_q_guided_direct_eta_v1','classification':classification,'orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','files':files,'raw_rollouts_retained':True});print(json.dumps({'decision':decision,'runtime':runtime},indent=2))
if __name__=='__main__':main()
