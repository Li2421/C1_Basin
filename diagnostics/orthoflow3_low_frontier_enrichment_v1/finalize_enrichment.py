#!/usr/bin/env python3
"""Analyze only confirmed B63 candidates after the frozen promotion plan."""
import csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT=Path('/home/zhihan/research/Basin_C1'); OUT=ROOT/'diagnostics/orthoflow3_low_frontier_enrichment_v1'; OLD=ROOT/'diagnostics/orthoflow3_analytic_mindef_stability_v1'; WIDTH=np.array([.75,1.,.75]); EPS=1e-12
def write(n,rows):
 k=sorted({x for r in rows for x in r}) if rows else ['status'];
 with (OUT/n).open('w',newline='') as f:w=csv.DictWriter(f,k);w.writeheader();w.writerows(rows)
def corr(x,y,f):
 if len(x)<2 or np.std(x)==0 or np.std(y)==0:return float('nan')
 return float(f(x,y).statistic)
def raw_records():
 # Existing audit has an exact cache union for historical B63 evidence.
 result={}
 for base in (ROOT/'diagnostics/orthoflow3_representation_migration_v1',ROOT/'diagnostics/orthoflow3_local_basin_continuity_v1',ROOT/'diagnostics/orthoflow3_bilateral_canonical_audit_v1',OLD,OUT):
  if not (base/'raw').exists():continue
  for p in (base/'raw').rglob('*.jsonl'):
   for line in p.read_text().splitlines():
    r=json.loads(line); k=(r['state_id'],tuple(round(float(x),15) for x in r['eta']),int(r['seed']),int(r['rng_namespace']))
    if k in result and (result[k]['success'],result[k]['outcome'])!=(r['success'],r['outcome']):raise RuntimeError(('cache disagreement',k))
    result.setdefault(k,r)
 return result
def proxy_from_rows(rr,eta):
 eta=np.array(eta); g=[];ex=[]
 for r in rr:
  fs=r.get('first_step')
  if fs:
   gv=np.asarray(fs['g_raw']);uv=np.asarray(fs['u_exec']);sv=np.asarray(fs['u_safe']);g.append(float(gv@gv));ex.append(float((uv-sv)@(uv-sv)))
 return {'R_eta_raw':float(eta@eta),'R_eta_norm':float((eta/WIDTH)@(eta/WIDTH)),'R_gram':float(np.mean(g)),'R_exec1':float(np.mean(ex))}
def main():
 states=json.loads((OUT/'active_state_manifest.json').read_text())['states']; sm={s['state_id']:s for s in states}; records=raw_records(); by=defaultdict(list)
 for (sid,eta,seed,ns),r in records.items():
  if sid in sm:by[(sid,eta)].append(r)
 candidates=[]
 for (sid,eta),rr in by.items():
  seeds={int(x['seed']) for x in rr};succ=[x for x in rr if x['success']]
  if len(seeds)>=64 and len(succ)>=63:
   row={'state_id':sid,'pair_rank':sm[sid]['pair_rank'],'side':sm[sid]['side'],'eta':json.dumps(list(eta)),'n_seeds':len(seeds),'successes':len(succ),'true_J':float(np.mean([x['J_def'] for x in succ])),'new_enrichment_candidate':any(x.get('stage') in ('screen','promote') for x in rr)}
   row.update(proxy_from_rows(rr,eta)); candidates.append(row)
 candidates.sort(key=lambda x:(x['state_id'],x['eta']))
 write('enriched_candidate_table.csv',candidates)
 bystate=defaultdict(list)
 for r in candidates:bystate[r['state_id']].append(r)
 # Promotion B63 table includes the screened/promoted status of each new coordinate.
 promotion=[]
 plan=json.loads((OUT/'promotion_plan.json').read_text())
 for a in plan['arms']:
  eta=tuple(round(float(x),15) for x in a['eta']); rr=by[(a['state_id'],eta)];succ=sum(x['success'] for x in rr); promotion.append({'state_id':a['state_id'],'sobol_index':a['sobol_index'],'eta':json.dumps(list(eta)),'seeds_total':len({x['seed'] for x in rr}),'successes_total':succ,'B63':len({x['seed'] for x in rr})>=64 and succ>=63,'role':a['role']})
 write('promoted_b63_results.csv',promotion)
 proxies=['R_eta_raw','R_eta_norm','R_gram','R_exec1']; rank=[]; regrets=[]; inv=[]
 selected={}
 for sid,cs in bystate.items():
  cs.sort(key=lambda x:x['eta']); y=np.array([float(x['true_J']) for x in cs]);best=float(y.min());selected[sid]={}
  for p in proxies:
   x=np.array([float(c[p]) for c in cs]);rho=corr(x,y,spearmanr);tau=corr(x,y,kendalltau);ix=int(x.argmin()); selected[sid][p]=cs[ix]
   rank.append({'state_id':sid,'n_candidates':len(cs),'proxy':p,'spearman':rho,'kendall':tau})
   regrets.append({'state_id':sid,'n_candidates':len(cs),'proxy':p,'best_true_J':best,'selected_true_J':float(y[ix]),'absolute_regret':float(y[ix]-best),'regret_ratio':float(y[ix]/max(best,EPS)),'within_5pct':bool(y[ix]<=1.05*best+EPS),'within_10pct':bool(y[ix]<=1.10*best+EPS),'within_20pct':bool(y[ix]<=1.20*best+EPS)})
   for c in cs:
    # New low-proxy candidates whose J rank is worse than their proxy rank.
    pr=1+sum(float(q[p])<float(c[p]) for q in cs);jr=1+sum(float(q['true_J'])<float(c['true_J']) for q in cs)
    if c['new_enrichment_candidate'] and jr>pr:
     inv.append({'state_id':sid,'proxy':p,'eta':c['eta'],'proxy_rank':pr,'true_J_rank':jr,'true_J':c['true_J'],'best_true_J':best,'J_ratio':float(c['true_J'])/max(best,EPS),'severity_rank_gap':jr-pr})
 write('enriched_rank_correlations.csv',rank);write('enriched_selection_regret.csv',regrets);write('low_proxy_inversions.csv',inv)
 # Direct previous-to-enriched comparison, separate source rows to avoid claims from a pooled average.
 oldrank=list(csv.DictReader(open(OLD/'within_state_rank_correlations.csv'))); oldreg=list(csv.DictReader(open(OLD/'proxy_selection_regret.csv'))); compare=[]
 for p in proxies:
  o=[r for r in oldrank if r['proxy']==p and r['stratum']=='ACTIVE_REQUIRED'];n=[r for r in rank if r['proxy']==p]
  orr=[r for r in oldreg if r['proxy']==p and r['stratum']=='ACTIVE_REQUIRED'];nrr=[r for r in regrets if r['proxy']==p]
  compare.append({'proxy':p,'previous_n_states':len(o),'enriched_n_states':len(n),'previous_mean_spearman':float(np.nanmean([float(x['spearman']) for x in o])),'enriched_mean_spearman':float(np.nanmean([x['spearman'] for x in n])),'previous_mean_kendall':float(np.nanmean([float(x['kendall']) for x in o])),'enriched_mean_kendall':float(np.nanmean([x['kendall'] for x in n])),'previous_median_J_ratio':float(np.median([float(x['regret_ratio']) for x in orr])),'enriched_median_J_ratio':float(np.median([x['regret_ratio'] for x in nrr])),'previous_within_10pct':float(np.mean([x['within_10pct']=='True' for x in orr])),'enriched_within_10pct':float(np.mean([x['within_10pct'] for x in nrr]))})
 write('previous_vs_enriched.csv',compare)
 # Descriptive trajectory explanation for at most five largest new inversion cases.
 expl=[]
 for z in sorted(inv,key=lambda r:r['J_ratio'],reverse=True)[:5]:
  eta=tuple(round(float(x),15) for x in json.loads(z['eta']));rr=by[(z['state_id'],eta)];best=min(bystate[z['state_id']],key=lambda x:x['true_J']); br=by[(z['state_id'],tuple(round(float(x),15) for x in json.loads(best['eta'])))];
  def desc(x):
   return {'mean_steps':float(np.mean([r['continuation_steps'] for r in x])),'mean_J':float(np.mean([r['J_def'] for r in x if r['success']])), 'mean_J_per_step':float(np.mean([r['J_def']/max(r['continuation_steps'],1) for r in x if r['success']])), 'mean_second_projection_retries':float(np.mean([r.get('second_projection_retries',0) for r in x]))}
  expl.append({**z,'proxy_candidate':json.dumps(desc(rr)),'true_J_best_candidate':best['eta'],'true_J_best':json.dumps(desc(br))})
 write('trajectory_explanations.csv',expl)
 counts={sid:len(cs) for sid,cs in bystate.items()}; resolved3=sum(x>=3 for x in counts.values());resolved4=sum(x>=4 for x in counts.values())
 # Failure confirmation is intentionally conservative: require resolved candidates and
 # non-positive mean rank evidence OR median regret above 10% for every proxy.
 proxy_summary={}
 for p in proxies:
  rs=[r for r in rank if r['proxy']==p and r['n_candidates']>=3];gs=[r for r in regrets if r['proxy']==p and r['n_candidates']>=3]
  proxy_summary[p]={'n':len(rs),'spearman_median':float(np.nanmedian([x['spearman'] for x in rs])),'spearman_mean':float(np.nanmean([x['spearman'] for x in rs])),'kendall_median':float(np.nanmedian([x['kendall'] for x in rs])),'kendall_mean':float(np.nanmean([x['kendall'] for x in rs])),'J_ratio_median':float(np.median([x['regret_ratio'] for x in gs])),'J_ratio_mean':float(np.mean([x['regret_ratio'] for x in gs])),'J_ratio_p90':float(np.quantile([x['regret_ratio'] for x in gs],.9)),'within_5pct':float(np.mean([x['within_5pct'] for x in gs])),'within_10pct':float(np.mean([x['within_10pct'] for x in gs])),'within_20pct':float(np.mean([x['within_20pct'] for x in gs]))}
 if resolved3<8:classification='UNDERRESOLVED'
 elif all(v['spearman_mean']>0 and v['J_ratio_median']<=1.10 for v in proxy_summary.values()):classification='ANALYTIC_MINDEF_RESCUED'
 elif any(v['spearman_mean']>0 for v in proxy_summary.values()):classification='MIXED_ANALYTIC_MINDEF_EVIDENCE'
 else:classification='ANALYTIC_MINDEF_FAILURE_CONFIRMED'
 # Some early screen jobs were safely cancelled after their atomic records were
 # written, so their process-runtime sidecars do not exist. Count immutable raw
 # records rather than under-reporting completed continuation work.
 newrows=[]
 for stage in ('screen','promote'):
  for p in (OUT/'raw'/stage).glob('*.jsonl'):
   newrows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 runs=[json.load(open(x)) for x in (OUT/'raw').rglob('*_runtime.json')]
 runtime={'new_continuations':len(newrows),'physical_steps':sum(int(x['continuation_steps']) for x in newrows),'wall_seconds_max_completed_shard':max([x['wall_seconds'] for x in runs],default=0),'wall_time_note':'Initial screening jobs were cancelled only after atomic raw records were persisted and resumed without duplicate tuples; their sidecars were not written. Scheduler accounting is recorded separately.','gpu_shards':2,'cpu_threads':2,'memory_per_shard_gib_limit':12}
 (OUT/'runtime_statistics.json').write_text(json.dumps(runtime,indent=2)+'\n')
 decision={'classification':classification,'states':len(states),'new_B63_counts':{sid:sum(x['new_enrichment_candidate'] for x in cs) for sid,cs in bystate.items()},'final_B63_counts':counts,'states_ge3':resolved3,'states_ge4':resolved4,'proxy_summary':proxy_summary,'low_proxy_high_true_J_inversions':len(inv),'learned_J_scientifically_justified_to_test':classification=='ANALYTIC_MINDEF_FAILURE_CONFIRMED','next_experiment':'If failure is confirmed, test (do not deploy) a trajectory-level J_hat(h,eta) on frozen candidate evidence; otherwise retain model-free analytical selection investigation.'}
 (OUT/'audit_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 report=['# OrthoFlow3 low-frontier enrichment audit','',f"**{classification}**. Exactly 10 frozen ACTIVE-required states were reused. {resolved3}/10 have >=3 and {resolved4}/10 have >=4 confirmed B63 candidates.",'','| Proxy | mean Spearman | mean Kendall | median J ratio | ≤5% | ≤10% | ≤20% |','|---|---:|---:|---:|---:|---:|---:|']
 for p,v in proxy_summary.items():report.append(f"| {p} | {v['spearman_mean']:.3f} | {v['kendall_mean']:.3f} | {v['J_ratio_median']:.3f} | {v['within_5pct']:.0%} | {v['within_10pct']:.0%} | {v['within_20pct']:.0%} |")
 report += ['','Promotions were selected before B63 and without true J: lowest raw eta norm and lowest start-Gram energy among distinct 8/8 candidates. The full-horizon J comparison uses only B63-feasible candidates and authoritative mean successful `J_def`.',f"\nNew branch work: {runtime['new_continuations']} continuations, {runtime['physical_steps']} physical steps; maximum completed-shard wall {runtime['wall_seconds_max_completed_shard']:.1f}s. Low-proxy/high-true-J inversions: {len(inv)}."]
 (OUT/'low_frontier_enrichment_report.md').write_text('\n'.join(report)+'\n')
 files=[]
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.name!='manifest.json':files.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
 (OUT/'manifest.json').write_text(json.dumps({'classification':classification,'files':files},indent=2)+'\n')
 print(json.dumps({'classification':classification,'counts':counts,'summary':proxy_summary,'runtime':runtime},indent=2))
if __name__=='__main__':main()
