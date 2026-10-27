"""Frozen-only evaluation on untouched-in-this-run validation labels and paired bootstrap."""
from common import *
from design import *
from pool_features import pool,menu_curve
import numpy as np
METHOD_NAMES={'main':'新critic·近邻输入','no_neighbour':'新critic·无近邻输入','original':'新critic·原网络结构','old_pool':'新critic·补充前训练池','global_fixed':'一个全局η（训练池里最好）','knn1':'抄最近1个训练场景','knn2':'抄最近2个训练场景','knn3':'抄最近3个训练场景','scene_oracle':'每个场景自己的最好η（参照）','copy_oracle':'照抄对的训练场景（覆盖上界，参照）','copy_old_oracle':'补充前训练池的覆盖上界（参照）','old_v2':'学习选η v2（上一轮，参照）'}
for g in 'ABCDE':METHOD_NAMES['add_'+g]='新critic·逐类加入'+g
RUN_METHODS={'main':'full_neighbour','no_neighbour':'full_no_neighbour','original':'full_original','old_pool':'old_neighbour',**{'add_'+g:'add_'+g for g in 'ABCDE'}}

def verify_freeze():
 receipt=read(HERE/'FROZEN.receipt.json')
 for p,h in receipt['files'].items():assert sha(p)==h,p
 return receipt

def score_one(y,u,choice):
 if np.asarray(choice).ndim==0:
  v=y[:,:,int(choice)];up=v+u[:,:,int(choice)]
 else:
  choice=np.asarray(choice,int);v=np.take_along_axis(y,choice[...,None],axis=-1)[...,0];up=v+np.take_along_axis(u,choice[...,None],-1)[...,0]
 return v.mean(0),up.mean(0)

def bootstrap(arr,hierarchical,seed):
 M,S,K=arr.shape;counts=np.rint(arr*3).astype(np.int16);assert np.allclose(arr*3,counts,rtol=0,atol=1e-12);rng=np.random.default_rng(seed);out=np.empty((5000,K));indices=[]
 for b in range(0,5000,100):
  count=min(100,5000-b);scene=rng.integers(M,size=(count,M))if hierarchical else np.broadcast_to(np.arange(M),(count,M));state=rng.integers(S,size=(count,M,S));out[b:b+count]=counts[scene[:,:,None],state].sum((1,2),dtype=np.int64)/(M*S*3);indices.append(dict(scene=scene.tolist(),state=state.tolist()))
 return out,indices

def main():
 frozen=verify_freeze();assert frozen['validation_labels_read']is False
 # First validation label-file read in this run, after every model/prediction/selection was frozen.
 access=dict(utc=utc(),path=str(HET/'LABEL_TABLES.json'),freeze_receipt_sha256=sha(HERE/'FROZEN.receipt.json'),first_label_parse_for_scoring_after_freeze=True)
 write(HERE/'VALIDATION_LABEL_ACCESS.receipt.json',access);tables=read_evidence(HET/'LABEL_TABLES.json');inputs=read(HERE/'VALIDATION_INPUTS.json');evals=inputs['evaluation'];names=list(inputs['scene_inputs']);all_choices=read(HERE/'ALL_CHOICES.json');legacy=read(HERE/'OLD_V2_REFERENCE.json');runs={r['id']:r for r in read(HERE/'RUNS.json')};source_full=read(HERE/'datasets'/f"{runs['full_neighbour']['dataset']}.json")['names'];source_old=read(HERE/'datasets'/f"{runs['old_neighbour']['dataset']}.json")['names']
 methods=list(RUN_METHODS)+['global_fixed','knn1','knn2','knn3','scene_oracle','copy_oracle','copy_old_oracle','old_v2'];arr=np.zeros((8,16,len(methods)));upper=np.zeros_like(arr);full_rates={};per_scene={};per_cell={};copy_details={}
 for ni,n in enumerate(names):
  row=tables[n];state_indices=row['state_indices'];assert all_choices['full_neighbour'][n]['state_indices']==state_indices;positions=[state_indices.index(i)for i in evals[n]];y=np.asarray(row['y'],float);u=np.asarray(row['unknown'],float);assert np.array_equal(y,np.rint(y));ye=y[:,positions];ue=u[:,positions];selected={};full_rates[n]={};per_cell[n]={};copy_details[n]={}
  for m in methods:
   if m in RUN_METHODS:choice=all_choices[RUN_METHODS[m]][n]['selected_index']
   elif m in ['global_fixed','knn1','knn2','knn3']:choice=all_choices['full_neighbour'][n]['baselines']['global'if m=='global_fixed'else m]['index']
   elif m=='old_v2':
    choice=legacy['scenes'][n]['selected_index'];assert legacy['scenes'][n]['state_indices']==evals[n]
   elif m=='scene_oracle':
    counts=ye.sum(0);a=(counts.sum(0)[None]-counts).argmax(-1);choice=np.repeat(a[None],3,axis=0)
   else:
    sources=source_full if m=='copy_oracle'else source_old;candidates=[(s,int(np.nanargmax(menu_curve(s)[0])))for s in sources];q=ye.mean((0,1));which=max(range(len(candidates)),key=lambda i:q[candidates[i][1]]);source,choice=candidates[which];copy_details[n][m]=dict(source=source,candidate=menu()[choice],source_best_candidates=[dict(source=s,candidate=menu()[a])for s,a in candidates])
   v,up=score_one(ye,ue,choice);mi=methods.index(m);arr[ni,:,mi]=v;upper[ni,:,mi]=up
   selected[m]=dict(candidate=menu()[int(choice)]if np.asarray(choice).ndim==0 else None,indices=np.asarray(choice).tolist(),rate=float(v.mean()),unknown_success_upper=float(up.mean()))
   if m=='scene_oracle':
    counts=y.sum(0);a=(counts.sum(0)[None]-counts).argmax(-1);value,_=score_one(y,u,np.repeat(a[None],3,axis=0));full_rates[n][m]=float(value.mean())
   elif m=='old_v2':
    assert abs(v.mean()-legacy['scenes'][n]['rate'])<1e-12;selected[m]['rate']=legacy['scenes'][n]['rate'];full_rates[n][m]=None
   else:full_rates[n][m]=float(score_one(y,u,choice)[0].mean())
  for m,rid in RUN_METHODS.items():
   indices=np.asarray(all_choices[rid][n]['per_cell_indices']);v,up=score_one(ye,ue,indices[:,positions]);per_cell[n][m]=dict(rate=float(v.mean()),unknown_success_upper=float(up.mean()),full_state_rate=float(score_one(y,u,indices)[0].mean()))
  per_scene[n]=dict(state_indices=evals[n],methods=selected)
 states,si=bootstrap(arr,False,2026102671);hier,hi=bootstrap(arr,True,2026102672);np.savez_compressed(HERE/'BOOTSTRAP.npz',states=states,scene_states=hier,methods=np.array(methods),scenes=np.array(names));write(HERE/'BOOTSTRAP_INDICES.json',dict(fixed_states=si,hierarchical=hi))
 comparisons={};point=np.rint(arr*3).sum((0,1),dtype=np.int64)/(8*16*3);global_i=methods.index('global_fixed');knn_i=methods.index('knn2');old_i=methods.index('old_pool')
 for i,m in enumerate(methods):
  r=dict(rate=float(point[i])if m!='old_v2'else legacy['mean_rate'],unknown_success_upper=float(upper[:,:,i].mean()),full_state_mean_rate=None if m=='old_v2'else float(np.mean([full_rates[n][m]for n in names])),comparisons={})
  for label,j in [('global_fixed',global_i),('knn2',knn_i),('old_pool',old_i)]:
   r['comparisons'][label]=dict(difference=float(r['rate']-point[j]),ci_states=np.percentile(states[:,i]-states[:,j],[2.5,97.5]).tolist(),ci_scene_states=np.percentile(hier[:,i]-hier[:,j],[2.5,97.5]).tolist(),per_scene={n:float(arr[k,:,i].mean()-arr[k,:,j].mean())for k,n in enumerate(names)})
  comparisons[m]=r
 main=comparisons['main'];dg=main['comparisons']['global_fixed'];dk=main['comparisons']['knn2'];do=main['comparisons']['old_pool'];worst=min(dg['per_scene'].values());useful=dg['ci_states'][0]>0;not_harm=worst>=-.1;supplement=do['difference']>0 and do['ci_states'][0]>0;near='明显超过'if dk['difference']>0 and dk['ci_states'][0]>0 else'持平或略好'if dk['difference']>=0 else'不如';fixed=useful and not_harm and supplement and near!='不如'
 decisions=dict(critic_fixed_on_this_validation_set=bool(fixed),useful=bool(useful),beats_neighbour=near,supplement_useful=bool(supplement),not_harmful=bool(not_harm),worst_scene_difference=float(worst),validation_scenes_previously_used_in_recipe_development=True,requires_untouched_scene_set_for_final_claim=True,thresholds_use_unrounded_json=True)
 # Learning curves use the same frozen per-subset choices; no fraction is selected as a replacement method.
 curves=[];additions={}
 for run in read(HERE/'RUNS.json'):
  if run['role']!='learning_curve':continue
  values={m:[]for m in ['critic','knn2','global_fixed']}
  for n in names:
   row=tables[n];pos=[row['state_indices'].index(i)for i in evals[n]];y=np.asarray(row['y'])[:,pos];ch=all_choices[run['id']][n]
   for m,a in [('critic',ch['selected_index']),('knn2',ch['baselines']['knn2']['index']),('global_fixed',ch['baselines']['global']['index'])]:values[m].append(float(y[:,:,a].mean()))
  curves.append(dict(id=run['id'],fraction=run['fraction'],repeat=run['repeat'],training_scenes=len(read(HERE/'datasets'/f"{run['dataset']}.json")['names']),rates={m:float(np.mean(v))for m,v in values.items()},per_scene={m:dict(zip(names,v))for m,v in values.items()}))
 for g in 'ABCDE':additions[g]=dict(mean_change=comparisons['add_'+g]['comparisons']['old_pool'],per_scene={n:float(per_scene[n]['methods']['add_'+g]['rate']-per_scene[n]['methods']['old_pool']['rate'])for n in names})
 # Grouped training-pool CV; insufficiently observed selected candidates remain explicit missing entries.
 cv=[]
 for fold in read(HERE/'CV_FOLDS.json')['folds']:
  for n in fold['targets']:
   q,coverage=menu_curve(n);eligible=int(np.isfinite(q).sum())>=40;row=dict(fold=fold['fold'],scene=n,qualifying_candidates=int(np.isfinite(q).sum()),eligible=eligible,methods={})
   if eligible:
    mainc=all_choices[f"cv{fold['fold']}_neighbour"][n];noc=all_choices[f"cv{fold['fold']}_no_neighbour"][n];indices={'main':mainc['selected_index'],'no_neighbour':noc['selected_index'],**{m:mainc['baselines']['global'if m=='global_fixed'else m]['index']for m in ['global_fixed','knn1','knn2','knn3']}}
    for m,a in indices.items():row['methods'][m]=dict(candidate=menu()[a],regret=float(np.nanmax(q)-q[a])if np.isfinite(q[a])else None,selected_candidate_coverage=float(coverage[a]),missing_reason=None if np.isfinite(q[a])else'Selected candidate lacks50% coverage; no rollout added.')
   cv.append(row)
 write(HERE/'STATISTICS.json',dict(utc=utc(),method_names=METHOD_NAMES,method_order=methods,scenes=names,comparisons=comparisons,per_scene=per_scene,per_cell=per_cell,full_state_per_scene=full_rates,coverage_upper_details=copy_details,learning_curves=curves,class_additions=additions,training_pool_cv=cv,decisions=decisions,bootstrap=dict(replicates=5000,state_seed=2026102671,hierarchical_seed=2026102672,paired_indices_all_methods=True),old_v2_reference_policy='Published point estimates transcribed verbatim. New paired contrasts bootstrap the same old frozen49 choices against the existing observed labels; no retraining, inference, reselection or rollout.'))
 write(HERE/'TEST_DECISION.json',dict(utc=utc(),**decisions,main_vs_global=dg,main_vs_knn2=dk,main_vs_old_pool=do));print('DECISIONS',decisions,flush=True)
if __name__=='__main__':main()
