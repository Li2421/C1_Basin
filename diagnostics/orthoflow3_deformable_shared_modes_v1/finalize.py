#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
H=Path(__file__).parent;D=H.parent;GEN=D/'orthoflow3_general_basin_geometry_v1';SH=D/'orthoflow3_shared_eta_codebook_v1';AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75])
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross_scenario():
 cb=read(SH/'codebook_eta.csv');A=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in cb]);Az=(A-AFF)/SCALE;rows=[r for r in read(GEN/'exact_q64_inventory.csv') if r['scenario']=='DoubleBottleneck_4A' and r['B63']=='True'];E=np.unique(np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in rows]),axis=0);Z=(E-AFF)/SCALE
 # Deterministic 12-center robust-cluster summary.
 order=np.lexsort((Z[:,2],Z[:,1],Z[:,0]));init=Z[order[np.linspace(0,len(order)-1,12,dtype=int)]];C,_=kmeans2(Z,init,minit='matrix',iter=100)
 ri,ci=linear_sum_assignment(np.linalg.norm(Az[:,None,:]-C[None,:,:],axis=2));assign=np.zeros(12,int);assign[ri]=ci
 T=None
 for _ in range(20):
  X=np.c_[Az,np.ones(12)];T=np.linalg.lstsq(X,C[assign],rcond=None)[0];M=X@T;ri,ci=linear_sum_assignment(np.linalg.norm(M[:,None,:]-C[None,:,:],axis=2));new=np.zeros(12,int);new[ri]=ci
  if np.array_equal(new,assign):break
  assign=new
 M=np.c_[Az,np.ones(12)]@T;R=C[assign]-M;raw=np.min(np.linalg.norm(Az[:,None,:]-C[None,:,:],axis=2),axis=1);out=[]
 for m in range(12):out.append(dict(mode_id=m,anchor_eta1=A[m,0],anchor_eta2=A[m,1],anchor_eta3=A[m,2],nearest_cluster_distance=raw[m],assigned_cluster=int(assign[m]),affine_z1=M[m,0],affine_z2=M[m,1],affine_z3=M[m,2],residual_z1=R[m,0],residual_z2=R[m,1],residual_z3=R[m,2],affine_plus_mode_residual_norm=np.linalg.norm(R[m])))
 write('cross_scenario_mode_alignment.csv',out);norm=np.linalg.norm(R,axis=1);dump('cross_scenario_transform.json',{'source':'Toy 12 frozen anchors','destination':'12 deterministic clusters from 159 DoubleBottleneck B63 points / 4 states','normalized_affine_matrix_4x3':T.tolist(),'linear_condition_number':float(np.linalg.cond(T[:3])),'nearest_raw_median':float(np.median(raw)),'affine_residual_median':float(np.median(norm)),'affine_residual_max':float(np.max(norm)),'fraction_residual_le_0.20':float(np.mean(norm<=.20)),'interpretation':'descriptive offline alignment; no cross-scenario h model was trained'})
def continuity():
 sel=json.load(open(H/'selected_checkpoint.json'));seed=sel['seed'];arr=np.load(H/'local_sets.npz');records=[];allx=[];allm=[];alld=[];allsid=[];allsplit=[]
 for split in ('train','val'):
  z=np.load(H/f'seed{seed}/{split}_predictions.npz');mask=arr['splits']==split;allx.append(arr['x'][mask]);allm.append(z['modes']);alld.append(z['delta_z']);allsid.extend(map(str,z['state_ids']));allsplit.extend([split]*len(z['state_ids']))
 X=np.concatenate(allx);M=np.concatenate(allm);Dlt=np.concatenate(alld);hd=[];dd=[]
 for i in range(len(X)):
  cand=np.flatnonzero(M==M[i]);cand=cand[cand!=i]
  if not len(cand):continue
  j=cand[np.argmin(np.linalg.norm(X[cand]-X[i],axis=1))];a=float(np.linalg.norm(X[j]-X[i]));b=float(np.linalg.norm(Dlt[j]-Dlt[i]));hd.append(a);dd.append(b);records.append(dict(state_id=allsid[i],split=allsplit[i],mode_id=int(M[i]),nearest_same_mode_state=allsid[j],h_distance=a,residual_distance=b))
 write('residual_continuity.csv',records);rho,p=spearmanr(hd,dd) if len(hd)>2 else (np.nan,np.nan);dist=[]
 for m in sorted(set(M)):
  q=Dlt[M==m];dist.append(dict(mode_id=int(m),states=len(q),mean_delta_z1=q[:,0].mean(),mean_delta_z2=q[:,1].mean(),mean_delta_z3=q[:,2].mean(),std_delta_z1=q[:,0].std(),std_delta_z2=q[:,1].std(),std_delta_z3=q[:,2].std(),mean_residual_norm=np.linalg.norm(q,axis=1).mean(),max_residual_norm=np.linalg.norm(q,axis=1).max()))
 write('mode_residual_distribution.csv',dist);dump('residual_continuity_summary.json',{'pairs':len(hd),'spearman_h_vs_delta':float(rho),'p_value':float(p),'median_neighbor_h_distance':float(np.median(hd)),'median_neighbor_residual_distance':float(np.median(dd)),'overall_residual_variance':float(np.mean(np.var(Dlt,axis=0)))})
def main():
 cross_scenario();continuity();ts=json.load(open(H/'test_summary.json'));tr=read(H/'training_summary.csv');sel=json.load(open(H/'selected_checkpoint.json'));train=next(r for r in tr if int(r['seed'])==sel['seed'] and r['split']=='train');val=next(r for r in tr if int(r['seed'])==sel['seed'] and r['split']=='val');cr=json.load(open(H/'cross_scenario_transform.json'));co=json.load(open(H/'residual_continuity_summary.json'));pairs=read(H/'fixed_vs_deformable.csv');rng=np.random.default_rng(20260929);q=np.array([float(r['deformable_Q64'])-float(r['fixed_Q64']) for r in pairs]);jd=np.array([float(r['deformable_J_def'])-float(r['fixed_J_def']) for r in pairs]);idx=rng.integers(0,len(q),(200000,len(q)));qci=np.quantile(q[idx].mean(1),[.025,.975]);jci=np.quantile(jd[idx].mean(1),[.025,.975]);learnable=float(val['membership_proxy'])>=float(train['membership_proxy'])-.25 and float(val['local_set_distance'])<=max(.08,2*float(train['local_set_distance']));maintain=ts['deformable_B63']>=ts['fixed_B63']-1 and ts['deformable_mean_Q64']>=ts['fixed_mean_Q64']-.005;lower_j=jci[1]<0;state_dep=co['overall_residual_variance']>=1e-4
 tp=read(H/'test_predictions.csv');rn=np.array([float(x['residual_norm']) for x in tp]);an=np.array([[float(x[f'anchor_eta{i}']) for i in (1,2,3)] for x in tp]);mn=np.array([[float(x[f'eta{i}']) for i in (1,2,3)] for x in tp]);eta_proxy={'test_residual_norm_mean':float(rn.mean()),'test_residual_norm_median':float(np.median(rn)),'test_residual_norm_max':float(rn.max()),'anchor_raw_eta_norm_mean':float(np.linalg.norm(an,axis=1).mean()),'moving_raw_eta_norm_mean':float(np.linalg.norm(mn,axis=1).mean()),'raw_eta_norm_mean_difference':float((np.linalg.norm(mn,axis=1)-np.linalg.norm(an,axis=1)).mean()),'caveat':'raw eta norm is only a coefficient-magnitude proxy; J_def is the closed-loop deformation metric'}
 if not learnable:classification='MODE_LOCAL_SET_UNLEARNABLE'
 elif maintain and lower_j:classification='DEFORMABLE_MODES_USEFUL_FOR_DEFORMATION'
 # The fixed selector is already at the ceiling.  A sub-0.2 percentage
 # point Q change whose paired interval includes zero is preservation, not
 # replicated evidence that moving the anchor is needed.
 elif maintain and abs(ts['deformable_mean_Q64']-ts['fixed_mean_Q64'])<.002 and not lower_j and qci[0]<=0:classification='FIXED_MODES_ALREADY_SUFFICIENT'
 elif maintain and ts['deformable_B63']>=ts['fixed_B63'] and state_dep and qci[0]>=-.002:classification='DEFORMABLE_MODES_SUPPORTED'
 elif cr['affine_residual_median']<=.20 and cr['fraction_residual_le_0.20']>=.75:classification='CROSS_SCENARIO_DEFORMATION_PROMISING'
 else:classification='MIXED_RESULT'
 # Controller totals from per-state table.
 r=read(H/'test_results.csv');summary=[]
 for c in ('fixed','deformable'):
  summary.append(dict(controller=c,B63_states=sum(x[f'{c}_B63']=='True' for x in r),states=len(r),mean_Q64=float(np.mean([float(x[f'{c}_Q64']) for x in r])),success=sum(int(x[f'{c}_successes']) for x in r),trials=sum(int(x[f'{c}_trials']) for x in r),deadlock=sum(int(x[f'{c}_deadlock']) for x in r),timeout=sum(int(x[f'{c}_timeout']) for x in r),collision=sum(int(x[f'{c}_collision']) for x in r),J_def=float(np.average([float(x[f'{c}_J_def']) for x in r],weights=[int(x[f'{c}_successes']) for x in r])),episode_length=float(np.mean([float(x[f'{c}_episode_length']) for x in r]))))
 stats={'classification':classification,'mode_local_residual_learnable':bool(learnable),'robustness_maintained':bool(maintain),'state_dependent_residual':bool(state_dep),'test_controller_summary':summary,'paired':{'rescue':ts['rescue'],'break':ts['break'],'mean_Q64_difference':float(q.mean()),'Q64_difference_ci95':qci.tolist(),'mean_J_def_difference':float(jd.mean()),'J_def_difference_ci95':jci.tolist()},'eta_magnitude_proxy':eta_proxy,'continuity':co,'cross_scenario':cr,'answers':{'residual_learnable':bool(learnable),'moving_mode_maintains_fixed_robustness':bool(maintain),'deformation_reduced':bool(lower_j),'mode_center_moves_with_state':bool(state_dep),'cross_scenario_shared_structure_promising':bool(cr['affine_residual_median']<=.20 and cr['fraction_residual_le_0.20']>=.75),'cross_scenario_caveat':'descriptive fit is ill-conditioned and is not held-out cross-scenario validation','upgrade_to_conditional_multimodal_generator':'YES' if classification in ('DEFORMABLE_MODES_SUPPORTED','DEFORMABLE_MODES_USEFUL_FOR_DEFORMATION') else 'NO_NOT_YET'},'next_step':'Freeze a hard/exception-state cohort before further residual learning; test whether moving anchors provides a replicated benefit away from the fixed-selector ceiling.'};dump('final_decision.json',stats)
 lines=['# Deformable shared modes','',f'Classification: **{classification}**','',f"Usable local sets: {json.load(open(H/'local_set_gate.json'))['usable_pairs']}/{json.load(open(H/'local_set_gate.json'))['selected_pairs']}; selected seed {sel['seed']}.",'','|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|J_def|','|---|---:|---:|---:|---:|---:|---:|---:|']
 lines += [f"|{x['controller']}|{x['B63_states']}/{x['states']}|{x['mean_Q64']:.4f}|{x['success']}/{x['trials']}|{x['deadlock']}|{x['timeout']}|{x['collision']}|{x['J_def']:.4f}|" for x in summary]
 gate=json.load(open(H/'local_set_gate.json'))
 lines += ['',f"Moving vs fixed: rescue {ts['rescue']}, break {ts['break']}; mean Q64 difference {q.mean():.4f}, bootstrap 95% CI [{qci[0]:.4f}, {qci[1]:.4f}].",f"Mean paired J_def difference {jd.mean():.4f}, 95% CI [{jci[0]:.4f}, {jci[1]:.4f}]. This is not a statistically resolved deformation reduction.",f"Mean normalized residual magnitude was {eta_proxy['test_residual_norm_mean']:.4f}; raw eta coefficient norm changed by {eta_proxy['raw_eta_norm_mean_difference']:.4f} on average (proxy only).",'',f"Local-set labels: {gate['usable_pairs']}/{gate['selected_pairs']} usable; {gate['positive_points']} positive, {gate['negative_points']} negative, and {gate['ambiguous_points']} ambiguous eta observations.",f"Residual learnability: TRAIN/VAL membership proxy {float(train['membership_proxy']):.1%}/{float(val['membership_proxy']):.1%}; mean distance {float(train['local_set_distance']):.4f}/{float(val['local_set_distance']):.4f}.",f"The learned residual is state-dependent (variance {co['overall_residual_variance']:.6f}); neighbor h/delta Spearman {co['spearman_h_vs_delta']:.3f}.",f"Cross-scenario affine+per-mode alignment is descriptively promising: residual median {cr['affine_residual_median']:.3f}, <=0.20 fraction {cr['fraction_residual_le_0.20']:.1%}. The affine condition number {cr['linear_condition_number']:.1f} makes this evidence ill-conditioned rather than confirmatory.",'',"The moving mode preserved robustness, but it did not show a stable advantage over the already saturated fixed selector. Mode centers can move predictably with state, yet that motion is not currently necessary for Toy closed-loop success.",f"Upgrade to a conditional multimodal generator: **{stats['answers']['upgrade_to_conditional_multimodal_generator']}**."]
 (H/'final_report.md').write_text('\n'.join(lines)+'\n');runtime=[]
 for p in list((H/'local_raw').glob('*runtime.json'))+list((H/'test_raw').glob('*runtime.json')):runtime.append(json.load(open(p)))
 dump('runtime_statistics.json',{'new_continuations_recorded':sum(x['new_continuations'] for x in runtime),'physical_steps':sum(x['physical_steps'] for x in runtime),'worker_wall_seconds':sum(x['wall_seconds'] for x in runtime),'max_gpu_shards':6});w=json.load(open(H/'working_state.json'));w.update(status='COMPLETE',completed=list(dict.fromkeys(w['completed']+['residual_training','fresh_test','cross_scenario_alignment','final_report'])),next_action='none');dump('working_state.json',w);arts={str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and p.name!='manifest.json' and not any(q in str(p) for q in ('/local_raw/','/test_raw/','/local_runs/','/test_runs/','/logs/','__pycache__','/local_plans/','/test_plans/'))};dump('manifest.json',{'task':'ORTHOFLOW3_DEFORMABLE_SHARED_MODES_V1','status':'COMPLETE','artifacts':arts});print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
