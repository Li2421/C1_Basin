"""Validation-only Pareto selection and historical k1 offline evaluation."""
from __future__ import annotations
import csv,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/gphi_dagger_k1_diagnostic_v1'; DATA=ROOT/'diagnostics/gphi_training_dataset_dagger_k1_v1'; BASE=ROOT/'diagnostics/gphi_training_dataset_startup_complete_v1'; CAP=ROOT/'diagnostics/strict_deadlock_success_basin_capacity_v1'
OLD=ROOT/'diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz'; V3=18816; STARTUP=26432; COVERAGE=27136
sys.path.insert(0,str(ROOT)); import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
from diagnostics.gphi_training_startup_complete_v1.pipeline import numpy_predict_checkpoint
from diagnostics.gphi_training_dataset_v1.build_states import restore_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig,barrier_constraints
from single_integrator.environment import Config
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def writejson(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def frontier(rows):
 out=[];bw=float('inf')
 for r in sorted(rows,key=lambda x:(x['startup_validation_state_grouped_mean_l2'],x['warm_validation_state_grouped_mean_l2'],x['epoch'],x['seed'])):
  if r['warm_validation_state_grouped_mean_l2']<bw:out.append(dict(r));bw=r['warm_validation_state_grouped_mean_l2']
 return out
def metrics(pred,target,state_id,mask):return core.subset_metrics(pred[mask],target[mask],state_id[mask])
def cosine(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d>1e-14 else np.nan
def main():
 started=time.monotonic(); rows=[]; runt=[]
 for seed in (17,23,41):
  p=json.load(open(HERE/f'seed{seed}/progress.json'))
  if p['status']!='COMPLETED' or p['epoch']!=1200:raise RuntimeError((seed,p))
  for s in readcsv(HERE/f'seed{seed}/epoch_metrics.csv'):
   rows.append({k:(int(v) if k in ('seed','epoch') else float(v)) for k,v in s.items()})
  runt.append(json.load(open(HERE/f'seed{seed}/runtime.json')))
 f=frontier(rows);mins=min(r['startup_validation_state_grouped_mean_l2'] for r in rows);minw=min(r['warm_validation_state_grouped_mean_l2'] for r in rows)
 for r in f:
  r['startup_relative_regret']=r['startup_validation_state_grouped_mean_l2']/mins-1;r['warm_relative_regret']=r['warm_validation_state_grouped_mean_l2']/minw-1;r['max_relative_regret']=max(r['startup_relative_regret'],r['warm_relative_regret']);r['sum_relative_regret']=r['startup_relative_regret']+r['warm_relative_regret']
 pick=min(f,key=lambda r:(r['max_relative_regret'],r['sum_relative_regret'],r['all_validation_state_grouped_mean_l2'],r['epoch'],r['seed']))
 writecsv(HERE/'checkpoint_pareto.csv',f); src=HERE/f"seed{pick['seed']}/checkpoints/epoch_{pick['epoch']:04d}.npz"; dst=HERE/'best_dagger_k1_checkpoint.npz';shutil.copy2(src,dst); digest=sha(dst)
 selection={'selection_frozen_before_diagnostics':True,'selection_frozen_before_test_or_diagnostic_evaluation':True,'selection_rule':'validation-only startup/warm Pareto minimax regret','selected_seed':pick['seed'],'selected_epoch':pick['epoch'],'selected_validation_metrics':pick,'checkpoint':str(dst),'checkpoint_sha256':digest,'historical_k1_used_for_selection':False,'fresh6_used_for_selection':False,'strict_deadlock_diagnostics_used_for_selection':False,'test_used_for_selection':False,'closed_loop_used_for_selection':False}
 writejson(HERE/'selected_checkpoint.json',selection)
 with np.load(DATA/'samples.npz',allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
 test=np.flatnonzero(a['split']=='test');startup=(test>=V3)&(test<STARTUP);ret=[]
 for model,ck in [('OLD_COVERAGE',OLD),('NEW_K1',dst)]:
  pred=numpy_predict_checkpoint(ck,a['features'][test])
  for cohort,mask in [('ALL',np.ones(len(test),bool)),('STARTUP',startup),('WARM_V3',~startup)]:ret.append({'model':model,'cohort':cohort,**metrics(pred,a['targets'][test],a['state_id'][test],mask)})
 writecsv(HERE/'startup_warm_retention.csv',ret)
 cap=json.load(open(CAP/'strict_deadlock_manifest.json'));cfg=Config(**cap['environment']);cbf=CBFConfig(**cap['cbf']);states=json.load(open(HERE/'learner_k1_state_manifest.json'))['states']; valid=[r for r in states if r['teacher_valid']]
 indices=np.arange(COVERAGE,COVERAGE+len(valid));out=[]
 for model,ck in [('OLD_COVERAGE',OLD),('NEW_K1',dst)]:
  pred=numpy_predict_checkpoint(ck,a['features'][indices]); per=[]
  for j,(ix,r) in enumerate(zip(indices,valid)):
   env=restore_full(HERE/r['state_file'],cfg);A,lo,_=barrier_constraints(env.snapshot(),cbf);safe=a['u_safe'][ix];act,_,_,_=project_velocity_with_retry(safe+pred[j].reshape(2,2),A,lo,cfg.max_speed,cbf);pex=(act-safe).reshape(4);target=a['targets'][ix]
   per.append({'model':model,'case_id':r['case_id'],'source_state_id':r['source_state_id'],'state_id':r['state_id'],'flow_seed':r['flow_seed'],'raw_target_L2':float(np.linalg.norm(pred[j]-target)),'executed_action_L2':float(np.linalg.norm(pex-target)),'cosine':cosine(pex,target),'norm_ratio':float(np.linalg.norm(pex)/np.linalg.norm(target)) if np.linalg.norm(target)>1e-14 else np.nan})
  out.extend(per)
  for case in sorted(set(r['case_id'] for r in per))+['ALL']:
   q=per if case=='ALL' else [r for r in per if r['case_id']==case]
   out.append({'model':model,'case_id':case,'source_state_id':'AGGREGATE','state_id':'AGGREGATE','flow_seed':-1,'raw_target_L2':float(np.mean([r['raw_target_L2'] for r in q])),'executed_action_L2':float(np.mean([r['executed_action_L2'] for r in q])),'cosine':float(np.nanmean([r['cosine'] for r in q])),'norm_ratio':float(np.nanmean([r['norm_ratio'] for r in q]))})
 writecsv(HERE/'historical_k1_error_old_vs_new.csv',out)
 oldt={r['cohort']:r for r in ret if r['model']=='OLD_COVERAGE'};newt={r['cohort']:r for r in ret if r['model']=='NEW_K1'}
 (HERE/'training_report.md').write_text(f"""# Minimal k1 DAgger training\n\nSelected strictly by startup/warm validation Pareto: seed {pick['seed']}, epoch {pick['epoch']}, SHA256 `{digest}`. No strict-deadlock or fresh-6 diagnostic entered selection.\n\n| test cohort | old mean L2 | new mean L2 |\n|---|---:|---:|\n| startup | {float(oldt['STARTUP']['state_grouped_mean_l2']):.6f} | {float(newt['STARTUP']['state_grouped_mean_l2']):.6f} |\n| warm | {float(oldt['WARM_V3']['state_grouped_mean_l2']):.6f} | {float(newt['WARM_V3']['state_grouped_mean_l2']):.6f} |\n""")
 writejson(HERE/'training_runtime.json',{'seed_runs':runt,'parallel_wall_seconds':max(r['wall_seconds'] for r in runt),'sum_seed_seconds':sum(r['wall_seconds'] for r in runt),'selection_seconds':time.monotonic()-started,'gpu_shards':3})
 print(json.dumps({'status':'FROZEN','seed':pick['seed'],'epoch':pick['epoch'],'sha256':digest},indent=2))
if __name__=='__main__':main()
