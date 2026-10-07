"""Cached model-free proxy, Gram, projection, regret, and stability analysis."""
import csv,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,kendalltau,pearsonr
ROOT=Path('/home/zhihan/research/Basin_C1');H=ROOT/'diagnostics/orthoflow3_analytic_mindef_stability_v1';M=ROOT/'diagnostics/orthoflow3_representation_migration_v1';C=ROOT/'diagnostics/orthoflow3_local_basin_continuity_v1';BIL=ROOT/'diagnostics/orthoflow3_bilateral_canonical_audit_v1';SYS=Path('/home/zhihan/research/02_C1_Toy_GiveWay');sys.path[:0]=[str(ROOT),str(SYS)]
from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from shared_control.basis_families import get_basis_family
from single_integrator.environment import Config
WIDTH=np.array([.75,1.,.75]); EPS=1e-12
def write(name,rows):
 keys=sorted({k for r in rows for k in r}) if rows else ['status']
 with (H/name).open('w',newline='') as f:q=csv.DictWriter(f,keys);q.writeheader();q.writerows(rows)
def allraw():
 out={}
 for base in (M,C,BIL,H):
  for d in (base/'raw').glob('*') if (base/'raw').exists() else []:
   if not d.is_dir():continue
   for f in d.glob('*.jsonl'):
    for line in f.read_text().splitlines():
     r=json.loads(line);k=(r['state_id'],tuple(np.round(r['eta'],15)),int(r['seed']),int(r['rng_namespace']))
     if k in out and (out[k]['success'],out[k]['outcome'])!=(r['success'],r['outcome']):raise RuntimeError(('cache disagreement',k))
     out.setdefault(k,r)
 return out
def corr(x,y,kind):
 if len(x)<2 or np.std(x)==0 or np.std(y)==0:return float('nan')
 return float({'spearman':spearmanr,'kendall':kendalltau,'pearson':pearsonr}[kind](x,y).statistic)
def main():
 state_doc=json.loads((H/'state_manifest.json').read_text());states=state_doc['states']; pairs=state_doc['pairs'];raw=allraw();by=defaultdict(list)
 for (sid,eta,seed,ns),r in raw.items():by[(sid,eta)].append(r)
 robust=defaultdict(list)
 for (sid,eta),rr in by.items():
  seeds={int(r['seed']) for r in rr};success=sum(bool(r['success']) for r in rr)
  if len(seeds)>=64 and success>=63:
   success_rows=[r for r in rr if r['success']]; robust[sid].append({'eta':np.array(eta),'rows':rr,'successes':success,'J':float(np.mean([r['J_def'] for r in success_rows]))})
 config=Config(**json.loads((ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json').read_text())['environment']);basis=get_basis_family('orthoflow3');sm={s['state_id']:s for s in states}
 grams={};gramrows=[];projection=[];proxyrows=[];dimrows=[]
 for s in states:
  sid=s['state_id'];env=restore_full(Path(s['state_file']),config); seed_safe={}
  for cand in robust[sid]:
   for r in cand['rows']:
    if r.get('first_step') and r['first_step'].get('u_safe') is not None:seed_safe.setdefault(int(r['seed']),np.array(r['first_step']['u_safe']).reshape(2,2))
  mats=[]
  for seed,safe in seed_safe.items():
   fields=basis.compute(env.positions,env.goals,safe,config.max_speed);cols=np.stack([np.asarray(v).reshape(-1) for v in fields.values],axis=1);m=cols.T@cols;mats.append(m)
   gramrows.append({'state_id':sid,'side':s['side'],'pair_rank':s['pair_rank'],'seed':seed,'M11':m[0,0],'M22':m[1,1],'M33':m[2,2],'M12':m[0,1],'M13':m[0,2],'M23':m[1,2],'cos12':m[0,1]/math.sqrt(max(m[0,0]*m[1,1],EPS)),'cos13':m[0,2]/math.sqrt(max(m[0,0]*m[2,2],EPS)),'cos23':m[1,2]/math.sqrt(max(m[1,1]*m[2,2],EPS))})
  if not mats:continue
  Mbar=np.mean(mats,axis=0);grams[sid]=Mbar;zero=any(np.linalg.norm(c['eta'])<1e-14 for c in robust[sid]);stratum='ZERO_SUFFICIENT' if zero else 'ACTIVE_REQUIRED'
  for ci,c in enumerate(robust[sid]):
   eta=c['eta'];rawp=float(eta@eta);normp=float((eta/WIDTH)@(eta/WIDTH));gp=[];ep=[];rat=[]
   for r in c['rows']:
    fs=r.get('first_step');
    if not fs:continue
    g=np.array(fs['g_raw']);safe=np.array(fs['u_safe']);exe=np.array(fs['u_exec']);gn=float(g@g);en=float((exe-safe)@(exe-safe));gp.append(gn);ep.append(en);rat.append(math.sqrt(en)/(math.sqrt(gn)+EPS));projection.append({'state_id':sid,'eta':json.dumps(eta.tolist()),'seed':r['seed'],'ratio':rat[-1],'raw_g_norm2':gn,'executed_correction_norm2':en,'violation_gt_1e_7':rat[-1]>1+1e-7})
   gram=float(np.mean(gp));exec1=float(np.mean(ep));proxyrows.append({'state_id':sid,'side':s['side'],'pair_rank':s['pair_rank'],'stratum':stratum,'candidate_index':ci,'eta':json.dumps(eta.tolist()),'successes':c['successes'],'true_J':c['J'],'R_eta_raw':rawp,'R_eta_norm':normp,'R_gram':gram,'R_exec1':exec1})
   diag=[eta[i]**2*Mbar[i,i] for i in range(3)];cross=[2*eta[0]*eta[1]*Mbar[0,1],2*eta[0]*eta[2]*Mbar[0,2],2*eta[1]*eta[2]*Mbar[1,2]];total=sum(diag)+sum(cross)
   dimrows.append({'state_id':sid,'candidate_index':ci,'stratum':stratum,'goal_diag':diag[0],'flow_perp_diag':diag[1],'relative_diag':diag[2],'cross12':cross[0],'cross13':cross[1],'cross23':cross[2],'R_gram_from_M':total,'diag_abs_fraction':sum(abs(x) for x in diag)/(sum(abs(x) for x in diag+cross)+EPS),'cross_abs_fraction':sum(abs(x) for x in cross)/(sum(abs(x) for x in diag+cross)+EPS)})
 write('basis_gram_statistics.csv',gramrows);write('projection_compatibility.csv',projection);write('candidate_proxy_values.csv',proxyrows);write('dimension_contributions.csv',dimrows)
 write('candidate_eta_manifest.csv',[{'state_id':sid,'robust_candidate_count':len(cs),'etas':json.dumps([c['eta'].tolist() for c in cs])} for sid,cs in robust.items() if sid in sm])
 proxies=['R_eta_raw','R_eta_norm','R_gram','R_exec1'];within=[];regret=[];selected={}
 for sid in sm:
  rr=[r for r in proxyrows if r['state_id']==sid]
  if len(rr)<2:continue
  y=[float(r['true_J']) for r in rr];best=min(y);selected[sid]={'true_J':rr[int(np.argmin(y))]}
  for p in proxies:
   x=[float(r[p]) for r in rr];idx=int(np.argmin(x));j=y[idx];selected[sid][p]=rr[idx]
   within.append({'state_id':sid,'stratum':rr[0]['stratum'],'proxy':p,'n':len(rr),'spearman':corr(x,y,'spearman'),'kendall':corr(x,y,'kendall'),'pearson':corr(x,y,'pearson')})
   regret.append({'state_id':sid,'stratum':rr[0]['stratum'],'proxy':p,'best_true_J':best,'selected_true_J':j,'absolute_regret':j-best,'regret_ratio':j/max(best,EPS),'within_5pct':j<=1.05*best+EPS,'within_10pct':j<=1.10*best+EPS,'within_20pct':j<=1.20*best+EPS})
 write('within_state_rank_correlations.csv',within);write('proxy_selection_regret.csv',regret)
 pooled=[]
 for p in proxies:
  for stratum in ('ALL','ACTIVE_REQUIRED'):
   rr=proxyrows if stratum=='ALL' else [r for r in proxyrows if r['stratum']==stratum]
   pooled.append({'scope':stratum,'proxy':p,'n':len(rr),'spearman':corr([float(r[p]) for r in rr],[float(r['true_J']) for r in rr],'spearman'),'kendall':corr([float(r[p]) for r in rr],[float(r['true_J']) for r in rr],'kendall'),'pearson':corr([float(r[p]) for r in rr],[float(r['true_J']) for r in rr],'pearson')})
 write('proxy_vs_true_j.csv',pooled)
 stability=[];switch=[]
 anchor={s['pair_rank']:s for s in states if s['side']=='anchor'};neighbor={s['pair_rank']:s for s in states if s['side']=='neighbor_t+4'}
 for pair in pairs:
  a=anchor[pair['pair_rank']]['state_id'];b=neighbor[pair['pair_rank']]['state_id']
  if a not in selected or b not in selected:continue
  for p in ['true_J']+proxies:
   ea=np.array(json.loads(selected[a][p]['eta']));eb=np.array(json.loads(selected[b][p]['eta']));stability.append({'pair_rank':pair['pair_rank'],'offset_steps':4,'selector':p,'normalized_eta_movement':float(np.linalg.norm((ea-eb)/WIDTH))})
  tj=next(r for r in stability if r['pair_rank']==pair['pair_rank'] and r['selector']=='true_J')['normalized_eta_movement'];switch.append({'pair_rank':pair['pair_rank'],'true_J_eta_movement':tj,'classification':'NO_LARGE_SWITCH' if tj<.25 else 'UNDERRESOLVED','best_gap_anchor':sorted([c['J'] for c in robust[a]])[1]-sorted([c['J'] for c in robust[a]])[0],'best_gap_neighbor':sorted([c['J'] for c in robust[b]])[1]-sorted([c['J'] for c in robust[b]])[0]})
 write('local_selection_stability.csv',stability);write('optimizer_switch_analysis.csv',switch)
 summary={'states_total':len(states),'states_with_ge2_robust':len(selected),'pairs_with_both_resolved':len({r['pair_rank'] for r in stability}),'robust_counts':{sid:len(robust[sid]) for sid in sm},'projection':{'median':float(np.median([r['ratio'] for r in projection])),'p95':float(np.quantile([r['ratio'] for r in projection],.95)),'max':float(np.max([r['ratio'] for r in projection])),'violations':sum(r['violation_gt_1e_7'] for r in projection)},'within':within,'regret':regret,'pooled':pooled}
 (H/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ('states_total','states_with_ge2_robust','pairs_with_both_resolved','projection')},indent=2))
if __name__=='__main__':main()
