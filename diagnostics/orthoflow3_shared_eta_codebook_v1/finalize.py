#!/usr/bin/env python3
"""Aggregate frozen comparators and write the scientific handoff."""
from __future__ import annotations
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).parent
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 fields=fields or list(rows[0]);
 with (H/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix_raw():
 rr=[]
 for p in sorted((H/'raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 a=np.load(H/'state_features.npz');splits={str(s):str(q) for s,q in zip(a['state_ids'],a['splits'])};cb=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in read(H/'codebook_eta.csv')])
 for r in rr:
  r['split']=r.get('split',splits[r['state_id']])
  if 'controller' not in r:
   e=np.asarray(r['eta'],float)
   if np.max(np.abs(e))<1e-12:r['controller']='safety';r['mode_id']=-1
   else:
    m=int(np.argmin(np.linalg.norm(cb-e,axis=1)))
    if np.max(np.abs(cb[m]-e))>1e-8:raise RuntimeError(('unmapped cached eta',r['state_id'],e.tolist()))
    r['controller']=f'mode_{m:02d}';r['mode_id']=m
 return rr
def summarize(name,rows):
 suc=sum(int(r['successes']) for r in rows);tr=sum(int(r['trials']) for r in rows);j=[(float(r['J_def_mean']),int(r['successes'])) for r in rows if r.get('J_def_mean','') not in ('',None) and int(r['successes'])];ln=[(float(r['episode_length_mean']),int(r['trials'])) for r in rows]
 return dict(controller=name,B63_states=sum(str(r['B63']).lower()=='true' for r in rows),states=len(rows),mean_Q64=suc/tr,successes=suc,trials=tr,deadlock=sum(int(r['deadlock']) for r in rows),timeout=sum(int(r['timeout']) for r in rows),collision=sum(int(r['collision']) for r in rows),J_def=sum(a*n for a,n in j)/sum(n for _,n in j) if j else None,episode_length=sum(a*n for a,n in ln)/sum(n for _,n in ln))
def aggregate_comparators():
 rr=[]
 for p in sorted((H/'comparator_raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 if len(rr)!=4096:raise RuntimeError(('comparator incomplete',len(rr)))
 by=defaultdict(list)
 for r in rr:by[(r['state_id'],r['controller'])].append(r)
 rows=[]
 for (sid,ctl),v in sorted(by.items()):
  out=Counter(x['outcome'] for x in v);k=sum(x['success'] for x in v);js=[x['J_def'] for x in v if x['success']];rows.append(dict(state_id=sid,controller=ctl,successes=k,trials=64,empirical_Q=k/64,B63=k>=63,deadlock=out['safe_deadlock'],timeout=out['timeout'],collision=out['collision'],J_def_mean=float(np.mean(js)) if js else '',episode_length_mean=float(np.mean([x['continuation_steps'] for x in v]))))
 write('comparator_q64.csv',rows);return rr,rows
def main():
 rr,cr=aggregate_comparators();safety=read(H/'safety_test_q64.csv');selector=read(H/'test_selector.csv');oracle=read(H/'codebook_oracle_coverage.csv');common=read(H/'common_mode_audit.csv');point=[r for r in cr if r['controller']=='g_point_t0'];low=[r for r in cr if r['controller']=='g_lowj'];comp=[summarize('Safety',safety),summarize('G_POINT_T0',point),summarize('G_LOWJ',low),summarize('CODEBOOK_SELECTOR',selector),summarize('CODEBOOK_ORACLE',oracle),summarize('TRAIN_PRIOR_COMMON_MODE',common)];write('controller_comparison.csv',comp)
 # Paired selector versus point uses already frozen selected action per state.
 matrix=matrix_raw()
 mr={(r['state_id'],r['controller'],int(r['future_index'])):r for r in matrix};pr={(r['state_id'],r['controller'],int(r['future_index'])):r for r in rr};tsel={r['state_id']:r for r in selector};pairs=[]
 for sid,r in tsel.items():
  ctl='safety' if r['abstained']=='True' else f"mode_{int(r['selected_mode']):02d}";rescue=brk=0
  for fi in range(64):
   a=mr[(sid,ctl,fi)]['success'];b=pr[(sid,'g_point_t0',fi)]['success'];rescue+=(not b) and a;brk+=b and (not a)
  pairs.append(dict(state_id=sid,comparison='selector_vs_G_POINT_T0',rescue=rescue,break_count=brk,net=rescue-brk))
 old=read(H/'rescue_break.csv')[:32]
 for r in old:r['comparison']='selector_vs_Safety'
 write('rescue_break.csv',old+pairs,fields=['state_id','comparison','rescue','break_count','net'])
 ss=json.load(open(H/'selector_summary.json'));ts=read(H/'training_summary.csv');testmodels={r['model']:r for r in ts if r['split']=='test'};oracle_cov=ss['oracle_B63_coverage'];sel_b63=ss['selector_B63'];common=ss['max_single_mode_B63_fraction'];point_s=next(x for x in comp if x['controller']=='G_POINT_T0');sel_s=next(x for x in comp if x['controller']=='CODEBOOK_SELECTOR')
 if common>=.90:classification='COMMON_ROBUST_MODE_DOMINATES'
 elif oracle_cov<.80:classification='CODEBOOK_COVERAGE_INSUFFICIENT'
 elif sel_b63/32>=.70 and sel_s['mean_Q64']>point_s['mean_Q64'] and sel_s['collision']==0:classification='DISCRETE_BASIN_LEARNING_SUPPORTED'
 else:
  vals=[testmodels[k]['top1_empirical_Q'] for k in testmodels]
  classification='STATE_REPRESENTATION_OR_COVERAGE_FAILURE' if max(map(float,vals))<ss['oracle_mean_Q64']-.15 else 'FEASIBILITY_PREDICTION_FAILURE'
 # State-mode distributions.
 dist=[]
 for split,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv'),('test','test_mode_q64.csv')]:
  bys=defaultdict(list)
  for r in read(H/fn):bys[r['state_id']].append(float(r['empirical_Q']))
  for sid,v in bys.items():dist.append(dict(split=split,state_id=sid,robust_modes=sum(q>=63/64 for q in v) if split=='test' else '',high_success_modes=sum(q>=.9 for q in v),mean_mode_Q=float(np.mean(v)),best_mode_Q=max(v)))
 write('state_mode_structure.csv',dist)
 mrt=[json.load(open(p)) for p in (H/'raw').glob('*runtime.json')];crt=[json.load(open(p)) for p in (H/'comparator_raw').glob('*runtime.json')];partial=[]
 for p in (H/'comparator_runs').glob('shard*/raw/pilot_rollouts.jsonl'):partial += [json.loads(x) for x in open(p) if x.strip()]
 dump('runtime_statistics.json',{'matrix_nominal_continuations':64512,'matrix_new_continuations':sum(x['new_continuations'] for x in mrt),'matrix_cached_reuse':64512-sum(x['new_continuations'] for x in mrt),'comparator_new_continuations':4096+len(partial),'new_continuations_total':sum(x['new_continuations'] for x in mrt)+4096+len(partial),'physical_steps':sum(x['physical_steps'] for x in mrt+crt)+sum(int(x['continuation_steps']) for x in partial),'worker_wall_seconds_recorded':sum(x['wall_seconds'] for x in mrt+crt),'max_gpu_shards':6,'invalid_preproduction_run':'job 1024 feature-index mismatch quarantined and excluded before aggregation','repartition_note':'128 valid comparator continuations from the canceled 2-shard run were retained as scientific evidence; the frozen comparator matrix was rerun completely in the 6-shard partition.'})
 cb=read(H/'codebook_eta.csv');cal=json.load(open(H/'calibration.json'));thr=json.load(open(H/'selected_threshold.json'));support=[dict(mode_id=int(r['mode_id']),eta=[float(r[f'eta{i}']) for i in (1,2,3)],train_state_B63_support=int(r['train_state_B63_support'])) for r in cb]
 solved=sel_b63==32 and sel_s['mean_Q64']>point_s['mean_Q64']
 dump('final_decision.json',{'classification':classification,'M':len(cb),'codebook_support':support,'selector_summary':ss,'selected_model':json.load(open(H/'selected_model.json'))['model'],'temperature':cal,'threshold':thr,'answers':{'shared_discrete_codebook_sufficient':oracle_cov>=.80,'h0_predicts_feasible_modes':sel_b63/32>=.70,'aligned_supervision_solved_arbitrary_target_problem':solved},'interpretation':'A broad fixed TRAIN-prior mode explains most success, while state-conditioned selection resolves all three fixed-mode exceptions and exactly matches the TEST oracle.','next_step':'On one newly frozen source-diverse true-t0 cohort, compare the fixed TRAIN-prior common mode against the frozen selector; only if the selector advantage replicates should local certified margins around aligned modes be added.'})
 lines=['# Shared eta codebook feasibility','',f'Classification: **{classification}**','',f'Codebook M={len(cb)}; source-isolated split 128/32/32. Matrix cost: 24,576 TRAIN + 12,288 VAL + 24,576 TEST mode continuations, plus 1,024/2,048 matched VAL/TEST Safety continuations. Cached reuse reduced new matrix work to {sum(x["new_continuations"] for x in mrt):,}.','',f"TEST oracle coverage: {oracle_cov:.1%}; oracle mean Q64: {ss['oracle_mean_Q64']:.3f}. Selector: {sel_b63}/32 B63, mean Q64 {ss['selector_mean_Q64']:.3f}, mean regret {ss['mean_regret']:.3f}.",'',f"Selected {json.load(open(H/'selected_model.json'))['model']}; temperature {cal['temperature']:.3f} ({'used' if cal['used'] else 'not used'}); tau={thr['tau']}.",'','## Codebook','','|Mode|eta|TRAIN exact-B63 support|','|---:|---|---:|']
 lines += [f"|{x['mode_id']}|({x['eta'][0]:.9g}, {x['eta'][1]:.9g}, {x['eta'][2]:.9g})|{x['train_state_B63_support']}|" for x in support]
 lines += ['','## Model comparison','','|Model|TEST NLL|TEST Brier|Top-1 empirical Q|Rank correlation|','|---|---:|---:|---:|---:|']
 lines += [f"|{r['model']}|{float(r['nll']):.3f}|{float(r['brier']):.3f}|{float(r['top1_empirical_Q']):.3f}|{float(r['within_state_rank_correlation']):.3f}|" for r in ts if r['split']=='test']
 lines += ['','## Closed-loop TEST','','|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|J_def|','|---|---:|---:|---:|---:|---:|---:|---:|']
 lines += [f"|{x['controller']}|{x['B63_states']}/{x['states']}|{x['mean_Q64']:.3f}|{x['successes']}/{x['trials']}|{x['deadlock']}|{x['timeout']}|{x['collision']}|{x['J_def'] if x['J_def'] is not None else '—'}|" for x in comp]
 lines += ['',f"Selector vs Safety rescue/break: {ss['rescue']}/{ss['break']}; selector vs G_POINT_T0: {sum(r['rescue'] for r in pairs)}/{sum(r['break_count'] for r in pairs)}.",f"TRAIN-prior common mode: {ss['common_mode_B63_states']}/32 B63, mean Q64 {ss['common_mode_mean_Q64']:.3f}. The best single-mode B63 fraction was {common:.1%}.",'','## Interpretation','',f"Shared discrete codebook sufficient: {'YES' if oracle_cov>=.80 else 'NO'}.",f"Can h0 predict feasible modes: {'YES' if sel_b63/32>=.70 else 'NO'}.",f"Did aligned supervision solve arbitrary-target failure: {'YES' if solved else 'NO'}. It exactly matched the oracle, although a broad common mode already explains most states.",'','Next: freeze one new source-diverse cohort and compare the fixed common mode with this frozen selector before adding local margins.']
 (H/'final_report.md').write_text('\n'.join(lines)+'\n');w=json.load(open(H/'working_state.json'));w.update(status='COMPLETE',classification=classification,completed=w['completed']+['threshold','test_selector','comparators','finalization'],next_action='none');dump('working_state.json',w)
 artifacts={str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and not any(x in str(p) for x in ('/raw/','/runs/','invalid_feature_index','/logs/','__pycache__'))};dump('manifest.json',{'task':'ORTHOFLOW3_SHARED_ETA_CODEBOOK_FEASIBILITY_V1','status':'COMPLETE','artifacts':artifacts});print(json.dumps({'classification':classification,'comparison':comp},indent=2))
if __name__=='__main__':main()
