#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
from scipy.stats import beta,binomtest

ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent
OLD=D/'orthoflow3_shared_eta_codebook_v1';POINT=D/'orthoflow3_true_t0_point_learning_v1'

def read(p):return list(csv.DictReader(open(p)))
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sh(s):return hashlib.sha256(s.encode()).hexdigest()

class MLP(nn.Module):
 M:int
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(64)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(self.M)(x)

def prepare():
 frozen=json.load(open(OLD/'state_split.json'));used_ids={q['state_id'] for q in frozen['states']};used_groups={q['source_group'] for q in frozen['states']}
 eligible=json.load(open(POINT/'eligible_state_manifest.json'))['selected_states'];cand=[dict(q) for q in eligible if q['state_id'] not in used_ids and q['source_group'] not in used_groups]
 cand.sort(key=lambda q:(sh(q['source_group']),q['state_id']));seen=set();unique=[]
 for q in cand:
  if q['source_group'] not in seen:seen.add(q['source_group']);unique.append(q)
 states=unique[:64]
 if not states:raise RuntimeError('no fresh source-isolated states')
 allf=np.load(POINT/'conditioning_features.npz')['features'];F=np.stack([allf[int(q['feature_index'])] for q in states])
 for i,q in enumerate(states):q['inventory_feature_index']=q['feature_index'];q['dataset_index']=i;q['selection_rank']=i;q['selection_hash']=sh(q['source_group'])
 np.savez_compressed(H/'fresh_state_features.npz',features=F,state_ids=np.array([q['state_id'] for q in states]))
 cb=read(OLD/'codebook_eta.csv');E=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in cb]);M=len(E)
 norm=json.load(open(OLD/'normalization.json'));X=((F-np.array(norm['h_mean']))/np.array(norm['h_std'])).astype(np.float32)
 sel=json.load(open(OLD/'selected_model.json'));ck=Path(sel['checkpoint']);model=MLP(M);template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));params=serialization.from_bytes(template,ck.read_bytes());logits=np.asarray(model.apply(params,jnp.asarray(X)))
 cal=json.load(open(OLD/'calibration.json'));T=float(cal['temperature']) if cal['used'] else 1.;P=1/(1+np.exp(-np.clip(logits/T,-30,30)));tau=float(json.load(open(OLD/'selected_threshold.json'))['tau']);fixed=int(json.load(open(OLD/'selector_summary.json'))['train_prior_common_mode'])
 modes=[];tasks=[]
 for q,p in zip(states,P):
  m=int(np.argmax(p));conf=float(p[m]);ab=conf<tau;eta=np.zeros(3) if ab else E[m]
  modes.append(dict(state_id=q['state_id'],selected_mode=-1 if ab else m,raw_argmax_mode=m,confidence=conf,abstained=ab,fixed_mode=fixed,switched_from_fixed=(ab or m!=fixed),selector_eta1=eta[0],selector_eta2=eta[1],selector_eta3=eta[2]))
  for ctl,e in [('safety',np.zeros(3)),('fixed',E[fixed]),('selector',eta)]:
   for fi in range(64):tasks.append(dict(state_id=q['state_id'],controller=ctl,eta=np.asarray(e).tolist(),future_index=fi,phase='fresh_primary'))
 write('mode_selection.csv',modes)
 manifest={'task':'ORTHOFLOW3_FRESH_SELECTOR_VS_FIXED_MODE_V1','status':'FROZEN_BEFORE_OUTCOMES','requested_states':64,'selected_states':len(states),'available_fresh_source_groups':len(unique),'selection':'SHA256(source_group), state_id; outcome blind','excluded_prior_state_count':len(used_ids),'excluded_prior_source_groups':len(used_groups),'source_leakage':False,'states':states,'frozen':{'codebook_sha256':sha_file(OLD/'codebook_eta.csv'),'checkpoint':str(ck),'checkpoint_sha256':sha_file(ck),'temperature':T,'threshold':tau,'fixed_mode':fixed,'fixed_eta':E[fixed].tolist(),'codebook_size':M}}
 dump('fresh_state_manifest.json',manifest)
 groups=defaultdict(list)
 for t in tasks:groups[t['state_id']].append(t)
 load=[0]*6;owner={}
 for sid in sorted(groups,key=lambda s:sh(s)):
  j=int(np.argmin(load));owner[sid]=j;load[j]+=len(groups[sid])
 p=H/'plans';p.mkdir(exist_ok=True)
 for j in range(6):
  with (p/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[t['state_id']]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 dump('working_state.json',{'status':'PRIMARY_MANIFEST_FROZEN','completed':['frozen_inputs','fresh_cohort','selector_predictions','primary_plans'],'next_action':'run primary matched Q64 controllers','counts':{'states':len(states),'nominal_controller_continuations':len(tasks),'shards':load}})
 write('experiment_ledger.csv',[{'stage':'prepare','status':'complete','states':len(states),'continuations':0,'decision':'fresh manifest frozen before outcomes'}])
 print(json.dumps({'states':len(states),'available':len(unique),'tasks':len(tasks),'shards':load,'switches':sum(r['switched_from_fixed'] for r in modes)},indent=2))

def metric(v):
 out=Counter(x['outcome'] for x in v);k=sum(bool(x['success']) for x in v);js=[float(x['J_def']) for x in v if x['success']]
 return dict(successes=k,trials=len(v),Q64=k/len(v),B63=k>=63,deadlock=out['safe_deadlock'],timeout=out['timeout'],collision=out['collision'],J_def=float(np.mean(js)) if js else '',episode_length=float(np.mean([x['continuation_steps'] for x in v])))

def aggregate():
 rr=[]
 for p in sorted((H/'raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 N=json.load(open(H/'fresh_state_manifest.json'))['selected_states'];expect=N*3*64
 if len(rr)!=expect:raise RuntimeError(('primary incomplete',len(rr),expect))
 by=defaultdict(list)
 for r in rr:by[(r['state_id'],r['controller'])].append(r)
 modes={r['state_id']:r for r in read(H/'mode_selection.csv')};rows=[];pairs=[]
 for sid in sorted(modes):
  mm={ctl:metric(by[(sid,ctl)]) for ctl in ('safety','fixed','selector')};base=dict(state_id=sid,selected_mode=modes[sid]['selected_mode'],confidence=modes[sid]['confidence'],abstained=modes[sid]['abstained'],switched_from_fixed=modes[sid]['switched_from_fixed'])
  for ctl in ('safety','fixed','selector'):
   for k,v in mm[ctl].items():base[f'{ctl}_{k}']=v
  rows.append(base)
  a={(int(x['future_index'])):bool(x['success']) for x in by[(sid,'selector')]};f={(int(x['future_index'])):bool(x['success']) for x in by[(sid,'fixed')]};s={(int(x['future_index'])):bool(x['success']) for x in by[(sid,'safety')]}
  pairs.append(dict(state_id=sid,switched_from_fixed=modes[sid]['switched_from_fixed'],selector_rescue_fixed=sum((not f[i]) and a[i] for i in range(64)),selector_break_fixed=sum(f[i] and (not a[i]) for i in range(64)),selector_net_fixed=sum(int(a[i])-int(f[i]) for i in range(64)),selector_rescue_safety=sum((not s[i]) and a[i] for i in range(64)),selector_break_safety=sum(s[i] and (not a[i]) for i in range(64)),selector_net_safety=sum(int(a[i])-int(s[i]) for i in range(64))))
 write('per_state_results.csv',rows);write('selector_vs_fixed.csv',pairs)
 fail=[r['state_id'] for r in rows if str(r['selector_B63']).lower()!='true']
 dump('primary_summary.json',{'states':N,'selector_nonB63_states':fail,'collisions':sum(int(r[f'{c}_collision']) for r in rows for c in ('safety','fixed','selector'))})
 w=json.load(open(H/'working_state.json'));w.update(status='PRIMARY_COMPLETE',completed=w['completed']+['primary_q64','primary_aggregation'],next_action='conditional oracle audit for selector non-B63 states');dump('working_state.json',w)
 print(json.dumps(json.load(open(H/'primary_summary.json')),indent=2))

def oracle_prepare():
 fail=json.load(open(H/'primary_summary.json'))['selector_nonB63_states'];cb=read(OLD/'codebook_eta.csv');tasks=[]
 for sid in fail:
  for r in cb:
   m=int(r['mode_id']);e=[float(r[f'eta{i}']) for i in (1,2,3)]
   for fi in range(64):tasks.append(dict(state_id=sid,controller=f'mode_{m:02d}',mode_id=m,eta=e,future_index=fi,phase='selector_failure_oracle'))
 groups=defaultdict(list)
 for t in tasks:groups[(t['state_id'],t['controller'])].append(t)
 load=[0]*6;owner={}
 for g in sorted(groups,key=lambda x:sh('|'.join(x))):j=int(np.argmin(load));owner[g]=j;load[j]+=len(groups[g])
 p=H/'oracle_plans';p.mkdir(exist_ok=True)
 for j in range(6):
  with (p/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[(t['state_id'],t['controller'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 dump('selector_failure_oracle_manifest.json',{'frozen_after_primary':True,'selector_nonB63_states':fail,'modes':len(cb),'nominal_continuations':len(tasks),'shards':load})
 if not fail:write('selector_failure_oracle_audit.csv',[],['state_id','selected_mode','selected_Q64','best_mode','best_Q64','B63_modes','diagnosis'])
 print(json.dumps({'failure_states':len(fail),'tasks':len(tasks),'shards':load}))

def oracle_aggregate():
 fail=json.load(open(H/'primary_summary.json'))['selector_nonB63_states']
 if not fail:return
 rr=[]
 for p in sorted((H/'oracle_raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 expect=len(fail)*12*64
 if len(rr)!=expect:raise RuntimeError(('oracle incomplete',len(rr),expect))
 by=defaultdict(list)
 for r in rr:by[(r['state_id'],int(r['mode_id']))].append(r)
 ms={r['state_id']:r for r in read(H/'mode_selection.csv')};ps={r['state_id']:r for r in read(H/'per_state_results.csv')};out=[]
 for sid in fail:
  vals=[(m,metric(by[(sid,m)])) for m in range(12)];b=[m for m,q in vals if q['B63']];best=max(vals,key=lambda z:(z[1]['Q64'],-z[0]));out.append(dict(state_id=sid,selected_mode=ms[sid]['selected_mode'],selected_Q64=ps[sid]['selector_Q64'],best_mode=best[0],best_Q64=best[1]['Q64'],B63_modes=';'.join(map(str,b)),diagnosis='SELECTOR_PREDICTION_FAILURE' if b else 'CODEBOOK_COVERAGE_FAILURE'))
 write('selector_failure_oracle_audit.csv',out)

def summarize(rows,ctl):
 return dict(controller=ctl,B63_states=sum(str(r[f'{ctl}_B63']).lower()=='true' for r in rows),states=len(rows),mean_Q64=float(np.mean([float(r[f'{ctl}_Q64']) for r in rows])),total_success=sum(int(r[f'{ctl}_successes']) for r in rows),trials=sum(int(r[f'{ctl}_trials']) for r in rows),deadlock=sum(int(r[f'{ctl}_deadlock']) for r in rows),timeout=sum(int(r[f'{ctl}_timeout']) for r in rows),collision=sum(int(r[f'{ctl}_collision']) for r in rows),J_def=float(np.average([float(r[f'{ctl}_J_def']) for r in rows if r[f'{ctl}_J_def']!=''],weights=[int(r[f'{ctl}_successes']) for r in rows if r[f'{ctl}_J_def']!=''])),episode_length=float(np.mean([float(r[f'{ctl}_episode_length']) for r in rows])))

def finalize():
 rows=read(H/'per_state_results.csv');pairs=read(H/'selector_vs_fixed.csv');N=len(rows);summ=[summarize(rows,c) for c in ('safety','fixed','selector')];write('controller_summary.csv',summ)
 d=np.array([float(r['selector_Q64'])-float(r['fixed_Q64']) for r in rows]);db=np.array([(str(r['selector_B63']).lower()=='true')-(str(r['fixed_B63']).lower()=='true') for r in rows],float);rng=np.random.default_rng(20260929);idx=rng.integers(0,N,size=(200000,N));boot=d[idx].mean(1);bootb=db[idx].mean(1);R=sum(int(r['selector_rescue_fixed']) for r in pairs);B=sum(int(r['selector_break_fixed']) for r in pairs);disc=R+B
 exact={'discordant_pairs':disc,'selector_wins':R,'fixed_wins':B,'two_sided_p':float(binomtest(R,disc,.5).pvalue) if disc else 1.,'selector_win_fraction_ci95':[float(beta.ppf(.025,R,disc-R+1)) if R else 0.,float(beta.ppf(.975,R+1,disc-R)) if B else 1.]}
 stats={'paired_state_bootstrap_replicates':200000,'seed':20260929,'selector_minus_fixed_mean_Q64':float(d.mean()),'mean_Q64_ci95':np.quantile(boot,[.025,.975]).tolist(),'selector_minus_fixed_B63_rate':float(db.mean()),'B63_rate_ci95':np.quantile(bootb,[.025,.975]).tolist(),'continuation_paired_exact':exact};dump('paired_statistics.json',stats)
 mode=read(H/'mode_selection.csv');counts=Counter(int(r['selected_mode']) for r in mode);sw=[r for r in rows if str(r['switched_from_fixed']).lower()=='true'];swids={r['state_id'] for r in sw};sp=[r for r in pairs if r['state_id'] in swids];fixed=next(x for x in summ if x['controller']=='fixed');selector=next(x for x in summ if x['controller']=='selector');audit=read(H/'selector_failure_oracle_audit.csv') if (H/'selector_failure_oracle_audit.csv').exists() else []
 pred_fail=sum(r.get('diagnosis')=='SELECTOR_PREDICTION_FAILURE' for r in audit);cov_fail=sum(r.get('diagnosis')=='CODEBOOK_COVERAGE_FAILURE' for r in audit)
 if audit and cov_fail>=math.ceil(len(audit)/2):classification='CODEBOOK_COVERAGE_LIMITED'
 elif selector['B63_states']<fixed['B63_states'] or selector['mean_Q64']<fixed['mean_Q64'] or B>R:classification='SELECTOR_GENERALIZATION_WEAK'
 elif fixed['B63_states']/N>=.90 and (selector['B63_states']-fixed['B63_states']<=2 or stats['selector_minus_fixed_mean_Q64']<.02):classification='FIXED_MODE_NEARLY_SUFFICIENT'
 elif stats['mean_Q64_ci95'][0]>0 and selector['B63_states']>fixed['B63_states'] and sum(int(r['selector_rescue_fixed']) for r in sp)>sum(int(r['selector_break_fixed']) for r in sp):classification='SELECTOR_ADVANTAGE_REPLICATED'
 else:classification='UNDERRESOLVED'
 switch_summary={'selection_counts':{str(k):v for k,v in sorted(counts.items())},'fixed_mode':int(json.load(open(H/'fresh_state_manifest.json'))['frozen']['fixed_mode']),'selected_fixed_fraction':counts[int(json.load(open(H/'fresh_state_manifest.json'))['frozen']['fixed_mode'])]/N,'switched_states':len(sw),'switched_selector_B63':sum(str(r['selector_B63']).lower()=='true' for r in sw),'switched_selector_nonB63':sum(str(r['selector_B63']).lower()!='true' for r in sw),'switched_fixed_nonB63_states':sum(str(r['fixed_B63']).lower()!='true' for r in sw),'switched_fixed_nonB63_rescued_to_B63':sum(str(r['fixed_B63']).lower()!='true' and str(r['selector_B63']).lower()=='true' for r in sw),'switched_states_with_more_successes':sum(int(r['selector_successes'])>int(r['fixed_successes']) for r in sw),'switched_states_with_fewer_successes':sum(int(r['selector_successes'])<int(r['fixed_successes']) for r in sw),'switched_rescue_continuations':sum(int(r['selector_rescue_fixed']) for r in sp),'switched_break_continuations':sum(int(r['selector_break_fixed']) for r in sp)}
 safety_pair={'rescue':sum(int(r['selector_rescue_safety']) for r in pairs),'break':sum(int(r['selector_break_safety']) for r in pairs),'net':sum(int(r['selector_net_safety']) for r in pairs)}
 dump('final_decision.json',{'classification':classification,'states':N,'controller_summary':summ,'paired_statistics':stats,'selector_vs_safety':safety_pair,'mode_switching':switch_summary,'selector_failure_diagnosis':{'prediction_failure_states':pred_fail,'codebook_coverage_failure_states':cov_fail},'state_conditioned_selection_necessary':'YES' if classification=='SELECTOR_ADVANTAGE_REPLICATED' else ('NO_OR_MARGINAL' if classification=='FIXED_MODE_NEARLY_SUFFICIENT' else 'NOT_ESTABLISHED'),'next_step':'Do not retrain automatically; retain the frozen selector and test it on a genuinely different compatible scenario/state distribution before any local-margin extension.'})
 lines=['# Fresh selector versus fixed common mode','',f'Classification: **{classification}**','',f'Fresh source-isolated states: {N}. Frozen codebook/model/temperature/threshold/fixed eta were unchanged.','', '|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|J_def|Episode length|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 lines += [f"|{x['controller']}|{x['B63_states']}/{N}|{x['mean_Q64']:.4f}|{x['total_success']}/{x['trials']}|{x['deadlock']}|{x['timeout']}|{x['collision']}|{x['J_def']:.4f}|{x['episode_length']:.1f}|" for x in summ]
 lines += ['',f"Selector vs fixed: rescue {R}, break {B}, net {R-B} matched continuations. Mean-Q64 gain {stats['selector_minus_fixed_mean_Q64']:.4f}, paired state-bootstrap 95% CI [{stats['mean_Q64_ci95'][0]:.4f}, {stats['mean_Q64_ci95'][1]:.4f}].",f"Selector vs Safety: rescue {safety_pair['rescue']}, break {safety_pair['break']}, net {safety_pair['net']}.",'',f"Mode selections: {dict(sorted(counts.items()))}. Selector chose fixed mode on {counts[switch_summary['fixed_mode']]}/{N} states and switched on {len(sw)}.",f"Among switched states, selector was B63 on {switch_summary['switched_selector_B63']}/{len(sw) if sw else 0}. All {switch_summary['switched_fixed_nonB63_states']} fixed-mode non-B63 exception states were rescued to B63; switching produced {switch_summary['switched_rescue_continuations']} rescue and {switch_summary['switched_break_continuations']} break continuations.",'',f"Conditional audit: {pred_fail} selector-prediction failures; {cov_fail} codebook-coverage failures.",'',f"State-conditioned mode selection necessary: **{json.load(open(H/'final_decision.json'))['state_conditioned_selection_necessary']}**."]
 (H/'final_report.md').write_text('\n'.join(lines)+'\n')
 runtime=[]
 for p in list((H/'raw').glob('*runtime.json'))+list((H/'oracle_raw').glob('*runtime.json')):runtime.append(json.load(open(p)))
 dump('runtime_statistics.json',{'nominal_primary_continuations':N*3*64,'new_continuations_recorded':sum(int(x['new_continuations']) for x in runtime),'physical_steps':sum(int(x['physical_steps']) for x in runtime),'worker_wall_seconds':sum(float(x['wall_seconds']) for x in runtime),'max_gpu_shards':6})
 w=json.load(open(H/'working_state.json'));w.update(status='COMPLETE',completed=w['completed']+['conditional_oracle','paired_statistics','final_report'],next_action='none');dump('working_state.json',w)
 arts={str(p.relative_to(H)):sha_file(p) for p in H.rglob('*') if p.is_file() and not any(z in str(p) for z in ('/raw/','/runs/','/logs/','__pycache__','/plans/','/oracle_plans/'))};dump('manifest.json',{'task':'ORTHOFLOW3_FRESH_SELECTOR_VS_FIXED_MODE_V1','status':'COMPLETE','artifacts':arts})
 print(json.dumps(json.load(open(H/'final_decision.json')),indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','aggregate','oracle_prepare','oracle_aggregate','finalize']);a=ap.parse_args();{'prepare':prepare,'aggregate':aggregate,'oracle_prepare':oracle_prepare,'oracle_aggregate':oracle_aggregate,'finalize':finalize}[a.stage]()
if __name__=='__main__':main()
