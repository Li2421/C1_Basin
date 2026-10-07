#!/usr/bin/env python3
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,kendalltau
ROOT=Path('/home/zhihan/research/Basin_C1');OUT=ROOT/'diagnostics/orthoflow3_mindef_independent_replication_v1';M=ROOT/'diagnostics/orthoflow3_representation_migration_v1';LOW=ROOT/'diagnostics/orthoflow3_low_frontier_enrichment_v1';WIDTH=np.array([.75,1,.75]);EPS=1e-12
def write(n,rs):
 k=sorted({z for r in rs for z in r}) if rs else ['status']
 with open(OUT/n,'w',newline='') as f:w=csv.DictWriter(f,k);w.writeheader();w.writerows(rs)
def co(x,y,f):return float(f(x,y).statistic) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else float('nan')
def main():
 states=json.load(open(OUT/'independent_active_state_manifest.json'))['states'];ids={s['state_id'] for s in states};sm={s['state_id']:s for s in states}
 raw={}
 for base in (M,OUT):
  if not (base/'raw').exists():continue
  for p in (base/'raw').rglob('*.jsonl'):
   for l in p.read_text().splitlines():
    r=json.loads(l);k=(r['state_id'],tuple(round(float(x),15) for x in r['eta']),int(r['seed']),int(r['rng_namespace']))
    if r['state_id'] in ids:raw.setdefault(k,r)
 by=defaultdict(list)
 for (sid,e,seed,ns),r in raw.items():by[(sid,e)].append(r)
 cand=[]
 for (sid,e),rr in by.items():
  succ=[r for r in rr if r['success']];n=len({r['seed'] for r in rr})
  if n>=64 and len(succ)>=63:
   g=[];ex=[]
   for r in rr:
    fs=r.get('first_step');
    if fs:
     a=np.asarray(fs['g_raw']);b=np.asarray(fs['u_exec']);c=np.asarray(fs['u_safe']);g.append(float(a@a));ex.append(float((b-c)@(b-c)))
   eta=np.array(e); cand.append({'state_id':sid,'source_trajectory':sm[sid]['source_trajectory'],'eta':json.dumps(list(e)),'n_seeds':n,'successes':len(succ),'true_J':float(np.mean([r['J_def'] for r in succ])),'R_eta_raw':float(eta@eta),'R_eta_norm':float((eta/WIDTH)@(eta/WIDTH)),'R_gram':float(np.mean(g)),'R_exec1':float(np.mean(ex)),'new_candidate':any(r.get('stage') in ('screen','promotion') for r in rr)})
 write('independent_candidate_table.csv',cand);bs=defaultdict(list)
 for r in cand:bs[r['state_id']].append(r)
 prom=[]
 for a in json.load(open(OUT/'promotion_plan.json'))['arms']:
  e=tuple(round(float(x),15) for x in a['eta']);rr=by[(a['state_id'],e)];prom.append({'state_id':a['state_id'],'eta':json.dumps(list(e)),'sobol_index':a['sobol_index'],'role':a['role'],'seeds_total':len({r['seed'] for r in rr}),'successes_total':sum(r['success'] for r in rr),'B63':len({r['seed'] for r in rr})>=64 and sum(r['success'] for r in rr)>=63})
 write('promoted_b63_results.csv',prom)
 proxies=['R_eta_raw','R_eta_norm','R_gram','R_exec1']; ranks=[];reg=[];inv=[]
 for sid,cs in bs.items():
  cs.sort(key=lambda x:x['eta']);y=np.array([float(x['true_J']) for x in cs]);best=y.min()
  for p in proxies:
   x=np.array([float(r[p]) for r in cs]);ix=int(x.argmin());ranks.append({'state_id':sid,'n_candidates':len(cs),'proxy':p,'spearman':co(x,y,spearmanr),'kendall':co(x,y,kendalltau)})
   reg.append({'state_id':sid,'n_candidates':len(cs),'proxy':p,'best_true_J':float(best),'selected_true_J':float(y[ix]),'J_ratio':float(y[ix]/max(best,EPS)),'within_5pct':bool(y[ix]<=1.05*best+EPS),'within_10pct':bool(y[ix]<=1.10*best+EPS),'within_20pct':bool(y[ix]<=1.2*best+EPS)})
   for c in cs:
    pr=1+sum(float(q[p])<float(c[p]) for q in cs);jr=1+sum(float(q['true_J'])<float(c['true_J']) for q in cs)
    if c['new_candidate'] and jr>pr:inv.append({'state_id':sid,'proxy':p,'eta':c['eta'],'proxy_rank':pr,'true_J_rank':jr,'severity_rank_gap':jr-pr,'true_J':c['true_J'],'best_true_J':float(best),'J_ratio':float(c['true_J'])/max(best,EPS)})
 write('independent_rank_correlations.csv',ranks);write('independent_selection_regret.csv',reg);write('independent_proxy_inversions.csv',inv)
 exp=[]
 for z in sorted(inv,key=lambda x:x['J_ratio'],reverse=True)[:5]:
  e=tuple(round(float(x),15) for x in json.loads(z['eta'])); rr=by[(z['state_id'],e)]; best=min(bs[z['state_id']],key=lambda x:x['true_J']);br=by[(z['state_id'],tuple(round(float(x),15) for x in json.loads(best['eta'])))]
  def d(v):return {'mean_termination_steps':float(np.mean([r['continuation_steps'] for r in v])),'mean_J':float(np.mean([r['J_def'] for r in v if r['success']])), 'mean_J_per_step':float(np.mean([r['J_def']/max(r['continuation_steps'],1) for r in v if r['success']])), 'mean_second_projection_retries':float(np.mean([r.get('second_projection_retries',0) for r in v]))}
  exp.append({**z,'proxy_candidate':json.dumps(d(rr)),'true_J_best_eta':best['eta'],'true_J_best':json.dumps(d(br))})
 write('trajectory_explanations.csv',exp)
 summ={}
 for p in proxies:
  a=[r for r in ranks if r['proxy']==p and r['n_candidates']>=3];b=[r for r in reg if r['proxy']==p and r['n_candidates']>=3];
  summ[p]={'n':len(a),'spearman_mean':float(np.nanmean([r['spearman'] for r in a])),'spearman_median':float(np.nanmedian([r['spearman'] for r in a])),'spearman_p25':float(np.nanquantile([r['spearman'] for r in a],.25)),'spearman_p75':float(np.nanquantile([r['spearman'] for r in a],.75)),'kendall_mean':float(np.nanmean([r['kendall'] for r in a])),'kendall_median':float(np.nanmedian([r['kendall'] for r in a])),'J_ratio_median':float(np.median([r['J_ratio'] for r in b])),'J_ratio_mean':float(np.mean([r['J_ratio'] for r in b])),'J_ratio_p90':float(np.quantile([r['J_ratio'] for r in b],.9)),'J_ratio_max':float(np.max([r['J_ratio'] for r in b])),'within_5pct':float(np.mean([r['within_5pct'] for r in b])),'within_10pct':float(np.mean([r['within_10pct'] for r in b])),'within_20pct':float(np.mean([r['within_20pct'] for r in b]))}
 original=json.load(open(LOW/'audit_decision.json'))['proxy_summary']; oldinv=list(csv.DictReader(open(LOW/'low_proxy_inversions.csv')));comp=[]
 for p in proxies:comp.append({'proxy':p,'original_mean_spearman':original[p]['spearman_mean'],'original_median_spearman':original[p]['spearman_median'],'original_mean_kendall':original[p]['kendall_mean'],'original_median_J_ratio':original[p]['J_ratio_median'],'original_mean_J_ratio':original[p]['J_ratio_mean'],'original_p90_J_ratio':original[p]['J_ratio_p90'],'original_within_5pct':original[p]['within_5pct'],'original_within_10pct':original[p]['within_10pct'],'original_within_20pct':original[p]['within_20pct'],'original_inversions':sum(r['proxy']==p for r in oldinv),'independent_mean_spearman':summ[p]['spearman_mean'],'independent_median_spearman':summ[p]['spearman_median'],'independent_mean_kendall':summ[p]['kendall_mean'],'independent_median_J_ratio':summ[p]['J_ratio_median'],'independent_mean_J_ratio':summ[p]['J_ratio_mean'],'independent_p90_J_ratio':summ[p]['J_ratio_p90'],'independent_within_5pct':summ[p]['within_5pct'],'independent_within_10pct':summ[p]['within_10pct'],'independent_within_20pct':summ[p]['within_20pct'],'independent_inversions':sum(r['proxy']==p for r in inv)})
 write('original_vs_replication.csv',comp)
 count={k:len(v) for k,v in bs.items()};ge3=sum(v>=3 for v in count.values());ge4=sum(v>=4 for v in count.values())
 if ge3<8:cl='INDEPENDENT_REPLICATION_UNDERRESOLVED'
 # A median raw-norm regret of one does not establish reliable selection when
 # one third of independent states exceed 10% regret and the tail is near 2x.
 # The primary question is whether *a simple proxy* reliably identifies min J;
 # it is falsified when raw is only moderate/tail-risky and both control-energy
 # proxies are non-positive with substantial regret.
 elif summ['R_eta_raw']['within_10pct']<.80 and summ['R_eta_raw']['J_ratio_mean']>1.15 and summ['R_gram']['spearman_mean']<0 and summ['R_exec1']['spearman_mean']<=0:cl='ANALYTIC_FAILURE_INDEPENDENTLY_REPLICATED'
 elif all(summ[p]['J_ratio_median']<=1.1 and summ[p]['spearman_mean']>0 for p in proxies):cl='ANALYTIC_PROXY_GENERALIZES_WELL'
 else:cl='REPLICATION_MIXED'
 new=[]
 for stage in ('screen','promotion'):
  for f in (OUT/'raw'/stage).glob('*.jsonl'):new += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
 sidecars=[json.load(open(x)) for x in (OUT/'raw').rglob('*_runtime.json')]
 rt={'new_continuations':len(new),'physical_steps':sum(r['continuation_steps'] for r in new),'wall_seconds_max_shard_per_stage':max([x['wall_seconds'] for x in sidecars],default=0),'wall_seconds_sum_of_stage_maxima':sum(max([x['wall_seconds'] for x in sidecars if x['stage']==stage],default=0) for stage in ('screen','promotion')),'gpu_shards':2,'cpu_threads':2,'gpu_memory_mib_observed_approx':1203,'ram_limit_per_shard_gib':12}
 (OUT/'runtime_statistics.json').write_text(json.dumps(rt,indent=2)+'\n')
 decision={'classification':cl,'target_active':10,'achieved_active':len(states),'B63_counts':count,'states_ge3':ge3,'states_ge4':ge4,'proxy_summary':summ,'inversion_count':len(inv),'affected_states':len({r['state_id'] for r in inv}),'worst_inversion_ratio':max([r['J_ratio'] for r in inv],default=float('nan')),'learned_J_diagnostic_justified':cl=='ANALYTIC_FAILURE_INDEPENDENTLY_REPLICATED','next_step':'If independently replicated, design a frozen diagnostic J_hat(h,eta) prediction experiment; do not deploy or train it here.'}
 (OUT/'replication_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 report=['# Independent OrthoFlow3 analytical-min-def replication','',f"**{cl}**. {len(states)} state-identity-disjoint ACTIVE-required states were resolved; {ge3} have >=3 and {ge4} have >=4 B63 candidates.",'','| Proxy | mean / median Spearman | mean / median Kendall | median J ratio | <=10% |','|---|---:|---:|---:|---:|']
 for p,v in summ.items():report.append(f"| {p} | {v['spearman_mean']:.3f} / {v['spearman_median']:.3f} | {v['kendall_mean']:.3f} / {v['kendall_median']:.3f} | {v['J_ratio_median']:.3f} | {v['within_10pct']:.0%} |")
 report += ['',f"Inversions: {len(inv)} across {len({r['state_id'] for r in inv})} states; worst ratio {max([r['J_ratio'] for r in inv],default=float('nan')):.3f}.",f"New execution: {rt['new_continuations']} continuation rollouts and {rt['physical_steps']} physical steps. No model was trained."]
 (OUT/'independent_replication_report.md').write_text('\n'.join(report)+'\n')
 files=[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'];(OUT/'manifest.json').write_text(json.dumps({'classification':cl,'files':files},indent=2)+'\n')
 print(json.dumps({'classification':cl,'counts':count,'summary':summ,'inversions':len(inv),'runtime':rt},indent=2))
if __name__=='__main__':main()
