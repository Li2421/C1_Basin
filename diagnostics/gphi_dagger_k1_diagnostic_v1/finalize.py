"""Finalize offline, dense, and WIDE diagnostics for minimal k1 aggregation."""
from __future__ import annotations
import csv,glob,hashlib,json,math,time,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/gphi_dagger_k1_diagnostic_v1'; OLD_DIR=ROOT/'diagnostics/gphi_retrained_dense_strict_deadlock_v1'; FRESH_REF=ROOT/'diagnostics/gphi_h8_fresh_unseen_generalization_v1'; COVERAGE=ROOT/'diagnostics/gphi_strict_deadlock_coverage_retrain_v1'; CAP=ROOT/'diagnostics/strict_deadlock_success_basin_capacity_v1'
OLD_CK=COVERAGE/'best_strict_deadlock_coverage_checkpoint.npz'; NEW_CK=HERE/'best_dagger_k1_checkpoint.npz'; EXPECTED='83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a'
sys.path.insert(0,str(ROOT));from diagnostics.gphi_training_startup_complete_v1.pipeline import numpy_predict_checkpoint
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
def writecsv(p,r):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def writejson(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def jsonl(p):return[json.loads(x) for x in p.read_text().splitlines() if x]
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d>1e-14 else np.nan
def agg(rows,key):return float(np.nanmean([r[key] for r in rows]))
def main():
 start=time.monotonic()
 if sha(NEW_CK)!=EXPECTED:raise RuntimeError('new checkpoint hash')
 # Fresh-6 offline k1 error (generated only after checkpoint selection).
 source=json.load(open(HERE/'source_manifest.json')); cap=json.load(open(CAP/'strict_deadlock_manifest.json'));cfg=Config(**cap['environment']);cbf=CBFConfig(**cap['cbf'])
 fresh=[]
 for state in source['fresh6_heldout_sources']:
  rows=jsonl(HERE/'raw_fresh6_k1'/f"{state['state_id']}.jsonl")
  if len(rows)!=64:raise RuntimeError((state['state_id'],len(rows)))
  fresh+=rows
 freshout=[]
 for model,ck in [('OLD_COVERAGE',OLD_CK),('NEW_K1',NEW_CK)]:
  feats=[];targets=[]
  for r in fresh:
   with np.load(HERE/r['tuple_file'],allow_pickle=False) as z:feats.append(z['feature']);targets.append(z['target'])
  pred=numpy_predict_checkpoint(ck,np.stack(feats)); per=[]
  for p,t,r in zip(pred,targets,fresh):
   env=restore_full(HERE/r['state_file'],cfg);A,lo,_=barrier_constraints(env.snapshot(),cbf)
   with np.load(HERE/r['tuple_file'],allow_pickle=False) as z:safe=np.asarray(z['u_safe'])
   act,_,_,_=project_velocity_with_retry(safe+p.reshape(2,2),A,lo,cfg.max_speed,cbf);pe=(act-safe).reshape(4)
   per.append({'model':model,'case_id':r['case_id'],'source_state_id':r['source_state_id'],'state_id':r['state_id'],'flow_seed':r['flow_seed'],'raw_target_L2':float(np.linalg.norm(p-t)),'executed_action_L2':float(np.linalg.norm(pe-t)),'cosine':cos(pe,t),'norm_ratio':float(np.linalg.norm(pe)/np.linalg.norm(t))})
  freshout+=per
  for case in sorted(set(r['case_id'] for r in per))+['ALL']:
   q=per if case=='ALL' else [r for r in per if r['case_id']==case]
   freshout.append({'model':model,'case_id':case,'source_state_id':'AGGREGATE','state_id':'AGGREGATE','flow_seed':-1,'raw_target_L2':agg(q,'raw_target_L2'),'executed_action_L2':agg(q,'executed_action_L2'),'cosine':agg(q,'cosine'),'norm_ratio':agg(q,'norm_ratio')})
 writecsv(HERE/'fresh6_k1_error_old_vs_new.csv',freshout)
 # Dense old-vs-new paired state summaries.
 states=json.load(open(OLD_DIR/'source_manifest.json'))['states'];rob=[];exact=[]
 def rawmap(directory,sid):return jsonl(directory/'raw'/f'{sid}.jsonl')
 for s in states:
  old=rawmap(OLD_DIR,s['state_id']);new=rawmap(HERE,s['state_id'])
  for c in ('H1','L8'):
   orr=[r for r in old if r['condition']==c and r['flow_mode']=='robust'];nrr=[r for r in new if r['condition']==c and r['flow_mode']=='robust']
   os=sum(r['outcome']=='success' for r in orr);ns=sum(r['outcome']=='success' for r in nrr)
   rob.append({'case_id':s['case_id'],'benchmark':s['benchmark'],'state_id':s['state_id'],'condition':c,'old_successes':os,'new_successes':ns,'old_B63':os>=63,'new_B63':ns>=63,'delta_successes':ns-os})
   oe=next(r for r in old if r['condition']==c and r['flow_mode']=='exact');ne=next(r for r in new if r['condition']==c and r['flow_mode']=='exact')
   exact.append({'case_id':s['case_id'],'benchmark':s['benchmark'],'state_id':s['state_id'],'condition':c,'old_outcome':oe['outcome'],'new_outcome':ne['outcome'],'old_J_def':oe['J_def'],'new_J_def':ne['J_def']})
 writecsv(HERE/'robust64_old_vs_new.csv',rob);writecsv(HERE/'exact_flow_old_vs_new.csv',exact)
 # Matched teacher-error growth from dense H1 active logs.
 growth=[]
 for model,directory in [('OLD_COVERAGE',OLD_DIR),('NEW_K1',HERE)]:
  for cohort in ('historical','fresh_unseen','combined'):
   selected=[s for s in states if cohort=='combined' or s['benchmark']==cohort]
   arrays=[]
   for s in selected:
    with np.load(directory/'active_logs'/f"{s['state_id']}__H1__robust.npz",allow_pickle=False) as z:
     arrays.append({k:np.asarray(z[k]) for k in ('local_step','executed_l2','cosine','norm_ratio')})
   for k in (0,1,2,4,8,16,32):
    vals={name:np.concatenate([a[name][a['local_step']==k] for a in arrays]) for name in ('executed_l2','cosine','norm_ratio')}
    growth.append({'model':model,'cohort':cohort,'step':k,'count':len(vals['executed_l2']),'executed_L2_mean':float(np.mean(vals['executed_l2'])),'executed_L2_median':float(np.median(vals['executed_l2'])),'cosine_mean':float(np.nanmean(vals['cosine'])),'norm_ratio_mean':float(np.nanmean(vals['norm_ratio']))})
 writecsv(HERE/'teacher_error_growth_old_vs_new.csv',growth)
 # WIDE development regression paired with frozen Safety outcomes.
 safety=[r for r in readcsv(FRESH_REF/'per_episode_results.csv') if r['condition']=='Safety']; sm={int(r['episode_index']):r for r in safety}
 wide=[]
 for p in sorted((HERE/'runs/fresh200/raw/h8').glob('episode_*.json')):wide.append(json.load(open(p)))
 if len(wide)!=200:raise RuntimeError(('wide',len(wide)))
 success=sum(r['outcome']=='success' for r in wide);timeouts=sum(r['outcome']=='timeout' for r in wide);dead=sum(r['outcome']=='deadlock' for r in wide);coll=sum(r['outcome']=='collision' for r in wide)
 rescue=brk=to_rescue=dl_rescue=0
 for r in wide:
  s=sm[int(r['episode_index'])]; ss=s['success']=='True'; ns=r['outcome']=='success';rescue+=int(not ss and ns);brk+=int(ss and not ns);to_rescue+=int(s['outcome']=='timeout' and ns);dl_rescue+=int(s['outcome']=='deadlock' and ns)
 j=np.asarray([r['J_def'] for r in wide],float)
 wide_metrics={'cohort_status':'DEVELOPMENT_REGRESSION_SET','episodes':200,'safety':{'success':135,'timeout':59,'strict_deadlock':6},'new_H8':{'success':success,'Q':success/200,'timeout':timeouts,'strict_deadlock':dead,'collision':coll},'paired':{'rescue':rescue,'break':brk,'net':rescue-brk},'timeout_rescue':{'rescued':to_rescue,'total':59,'rate':to_rescue/59},'strict_deadlock_rescue':{'rescued':dl_rescue,'total':6,'rate':dl_rescue/6},'J_def':{'mean':float(j.mean()),'median':float(np.median(j)),'P95':float(np.quantile(j,.95)),'max':float(j.max())},'references':{'original_c298_H8_success':188,'coverage_340b_H8_success':176}}
 writejson(HERE/'wide_regression_metrics.json',wide_metrics)
 # Safety and interpretation.
 newdense=[r for s in states for r in rawmap(HERE,s['state_id'])]; hard={'status':'PASS','dense_collisions':sum(r['outcome']=='collision' for r in newdense),'wide_agent_collisions':sum(bool(r.get('agent_collision')) for r in wide),'wide_wall_collisions':sum(bool(r.get('wall_collision')) for r in wide),'execution_errors':sum(r['outcome']=='execution_error' for r in newdense),'invalid_actions':sum(int(r.get('invalid_actions',0)) for r in wide),'nan_inf_events':sum(int(r.get('nan_inf_events',0)) for r in wide),'projection_solver_failures':sum(int(r.get('projection_failures',0)) for r in wide)}
 if any(v for k,v in hard.items() if k not in ('status',)):hard['status']='FAIL'
 writejson(HERE/'hard_safety_checks.json',hard)
 hist=readcsv(HERE/'historical_k1_error_old_vs_new.csv');h={r['model']:r for r in hist if r['case_id']=='ALL'};f={r['model']:r for r in freshout if r['case_id']=='ALL'}
 new_h1=sum(r['new_successes'] for r in rob if r['condition']=='H1');new_b63=sum(r['new_B63'] in (True,'True') for r in rob if r['condition']=='H1');new_exact=sum(r['new_outcome']=='success' for r in exact if r['condition']=='H1')
 hist_new=sum(r['new_successes'] for r in rob if r['condition']=='H1' and r['benchmark']=='historical');fresh_new=sum(r['new_successes'] for r in rob if r['condition']=='H1' and r['benchmark']=='fresh_unseen')
 hist_drop=float(h['NEW_K1']['executed_action_L2']) < .2*float(h['OLD_COVERAGE']['executed_action_L2']);fresh_drop=float(f['NEW_K1']['executed_action_L2']) < .5*float(f['OLD_COVERAGE']['executed_action_L2'])
 if hist_drop and not fresh_drop: classification='K1_HISTORICAL_MEMORIZATION_ONLY'
 elif hist_drop and new_h1>841 and success<160:classification='K1_AUGMENTATION_CAUSES_NEGATIVE_TRANSFER'
 elif hist_drop and new_h1>841:classification='K1_COVARIATE_SHIFT_HYPOTHESIS_SUPPORTED'
 else:classification='K1_FIXES_LOCAL_ERROR_NOT_CLOSED_LOOP'
 broader=hist_drop and fresh_drop
 # Runtime summary.
 def loads(pattern):return[json.load(open(p)) for p in glob.glob(str(HERE/pattern))]
 collect=loads('runtime_collect_shard*.json');freshrt=loads('runtime_collect_fresh6_shard*.json');train=json.load(open(HERE/'training_runtime.json'));dense=loads('runtime_shard*.json');widert=loads('runtime_closed_loop_shard*.json')
 from datetime import datetime
 wide_wall=max((datetime.fromisoformat(r['finished_utc'])-datetime.fromisoformat(r['started_utc'])).total_seconds() for r in widert)
 runtime={'gpu_shards_peak':6,'cpu_cores_peak':12,'memory_requested_GB_peak':100,'historical_collection_wall_s':max(r['elapsed_s'] for r in collect),'fresh6_collection_wall_s':max(r['elapsed_s'] for r in freshrt),'training_parallel_wall_s':train['parallel_wall_seconds'],'dense_wall_s':max(r['elapsed_seconds'] for r in dense),'wide_wall_s':wide_wall,'wide_completed_episodes':sum(r['completed_tuples'] for r in widert),'final_analysis_s':time.monotonic()-start}
 runtime['major_compute_stages_wall_s_sum']=runtime['historical_collection_wall_s']+runtime['fresh6_collection_wall_s']+runtime['training_parallel_wall_s']+runtime['dense_wall_s']+runtime['wide_wall_s']
 writejson(HERE/'runtime_statistics.json',runtime)
 oldg={(r['model'],int(r['step'])):r for r in growth if r['cohort']=='combined'}
 report=f"""# Minimal k=1 DAgger diagnostic\n\n## Result\n\n**{classification}**\n\n704 historical learner-visited k=1 states were generated; the frozen source eta remained valid for 704/704. The immutable dataset added 704 states/samples with no weighting or oversampling. Selected checkpoint: seed 23, epoch 992, `{EXPECTED}`.\n\n| metric | old coverage | new k1 |\n|---|---:|---:|\n| historical k1 executed L2 | {float(h['OLD_COVERAGE']['executed_action_L2']):.6f} | {float(h['NEW_K1']['executed_action_L2']):.6f} |\n| fresh-6 k1 executed L2 | {float(f['OLD_COVERAGE']['executed_action_L2']):.6f} | {float(f['NEW_K1']['executed_action_L2']):.6f} |\n| dense strict-deadlock success | 841/1088 | {new_h1}/1088 |\n| B63 states | 0/17 | {new_b63}/17 |\n| exact strict-deadlock | 15/17 | {new_exact}/17 |\n\n## Teacher-error growth\n\n| step | old L2 | new L2 | old cosine | new cosine |\n|---:|---:|---:|---:|---:|\n"""
 for k in (0,1,2,4,8,16,32):
  o=oldg[('OLD_COVERAGE',k)];n=oldg[('NEW_K1',k)];report+=f"| {k} | {float(o['executed_L2_mean']):.6f} | {float(n['executed_L2_mean']):.6f} | {float(o['cosine_mean']):.6f} | {float(n['cosine_mean']):.6f} |\n"
 report+=f"""\n## WIDE development regression\n\nNew H8: {success}/200 (Q={success/200:.3f}); timeout rescue {to_rescue}/59; strict-deadlock rescue {dl_rescue}/6; breaks {brk}; collisions {coll}; mean J_def {j.mean():.6f}. References: original checkpoint 188/200, coverage checkpoint 176/200.\n\nBroader on-policy aggregation justified: **{str(broader).upper()}**, but one-layer k1 aggregation is not sufficient and caused strict-deadlock closed-loop regression. The measured error front did not jump at k2; it grew gradually and was largest at k32 among the registered points. Smallest next diagnostic: without training, substitute the frozen eta teacher only at k32 on the historical dense rollouts and then return immediately to the frozen k1 G_phi. This tests whether the later mismatch is causally important before collecting another training layer; fresh-6 remains untouched.\n"""
 (HERE/'dagger_k1_report.md').write_text(report)
 # Manifest last.
 required=['source_manifest.json','learner_k1_state_manifest.json','teacher_validity_audit.csv','k1_teacher_targets.csv','augmented_dataset_manifest.json','overlap_audit.json','training_report.md','checkpoint_pareto.csv','selected_checkpoint.json','historical_k1_error_old_vs_new.csv','fresh6_k1_error_old_vs_new.csv','robust64_old_vs_new.csv','exact_flow_old_vs_new.csv','teacher_error_growth_old_vs_new.csv','wide_regression_metrics.json','hard_safety_checks.json','runtime_statistics.json','dagger_k1_report.md']
 manifest={'schema':'gphi_dagger_k1_diagnostic_v1','status':'COMPLETE','classification':classification,'checkpoint_sha256':EXPECTED,'broader_dagger_justified':broader,'files':{n:{'sha256':sha(HERE/n),'bytes':(HERE/n).stat().st_size} for n in required}}
 writejson(HERE/'manifest.json',manifest)
 print(json.dumps({'classification':classification,'historical_k1_new':h['NEW_K1']['executed_action_L2'],'fresh_k1_new':f['NEW_K1']['executed_action_L2'],'dense_success':new_h1,'B63':new_b63,'exact':new_exact,'wide_success':success,'broader_dagger':broader},indent=2))
if __name__=='__main__':main()
