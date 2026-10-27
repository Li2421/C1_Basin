"""All run plans and source-only datasets fixed before fitting or validation scoring."""
from common import *
from design import *
from pool_features import pool,rows,dataset,target_x
import numpy as np

def main():
 previous=read_evidence(HET/'STATISTICS.json');legacy=dict(mean_rate=previous['comparisons']['v2']['rate'],scenes={n:dict(rate=r['rates']['v2'],selected_index=r['methods']['v2']['selected_index'],state_indices=r['state_indices'],candidate_etas=r['candidate_etas'])for n,r in previous['per_scene'].items()},source=str(HET/'STATISTICS.json'),point_estimates_reused_verbatim=True)
 write(HERE/'OLD_V2_REFERENCE.json',legacy)
 allnames=sorted(pool());old=[n for n in allnames if not pool()[n]['new']];validation=read(HERE/'VALIDATION_INPUTS.json')['scene_inputs'];runs=[];tasks=[];definitions={}
 def add(name,kind,names,seeds,role,targets=None,target_type='validation',**extra):
  ident=uid('dataset',sorted(names));definition=definitions.get(ident)
  if definition is None:definition=dataset(ident,names);definitions[ident]=definition
  data={}
  if target_type=='validation':data=validation
  else:
   for n in targets:
    p=pool()[n];d=rows(n);data[n]=dict(h=d['h'].tolist(),c_prime=p['c_prime'],state_indices=list(range(p['state_count'])),controllers=list(CONTROL_IDS))
  xs={};metadata={}
  for n,v in data.items():
   x,shape,near=target_x(definition,v);xs[n]=x;metadata[n]=dict(shape=list(shape),state_indices=v['state_indices'],near=near)
  path=HERE/'target_inputs'/f'{name}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**xs)
  run=dict(id=name,kind=kind,dataset=ident,seeds=list(seeds),role=role,target_type=target_type,targets=list(data),target_inputs_path=str(path),target_inputs_sha256=sha(path),target_metadata=metadata,**extra);runs.append(run)
  for seed in seeds:tasks.append(dict(id=name,kind=kind,dataset=ident,seed=int(seed),target_inputs_path=str(path),target_inputs_sha256=sha(path)))
 add('full_neighbour','neighbour',allnames,range(10),'main');add('full_no_neighbour','no_neighbour',allnames,range(10),'no_neighbour');add('full_original','original',allnames,range(10),'original');add('old_neighbour','neighbour',old,range(10),'old_pool')
 for group in 'ABCDE':add('add_'+group,'neighbour',old+[n for n in allnames if pool()[n]['group']==group],range(5),'class_addition',group=group)
 rng=np.random.default_rng(2026102673);curves=[]
 for fraction in [.25,.5,.75]:
  for repeat in range(5):
   count=max(3,int(round(fraction*len(allnames))));chosen=sorted(rng.choice(allnames,count,replace=False).tolist());name=f'curve_{round(fraction*100):02d}_{repeat}'
   curves.append(dict(id=name,fraction=fraction,repeat=repeat,scenes=chosen));add(name,'neighbour',chosen,range(3),'learning_curve',fraction=fraction,repeat=repeat)
 grouped={}
 for n in allnames:grouped.setdefault('lit_crowded_random_flow'if pool()[n]['source_version']=='lit_crowded_random_flow'else n,[]).append(n)
 group_names=sorted(grouped);rng=np.random.default_rng(2026102674);rng.shuffle(group_names);folds=[]
 for i in range(6):
  gs=group_names[i::6];held=sorted(n for g in gs for n in grouped[g]);sources=[n for n in allnames if n not in held];folds.append(dict(fold=i,groups=gs,targets=held,sources=sources))
  for kind in ['neighbour','no_neighbour']:add(f'cv{i}_{kind}',kind,sources,range(10),'cv',targets=held,target_type='pool',fold=i)
 write(HERE/'RUNS.json',runs);write(HERE/'TRAIN_TASKS.json',tasks);write(HERE/'CV_FOLDS.json',dict(seed=2026102674,shuffled_groups=group_names,folds=folds));write(HERE/'LEARNING_CURVE_SUBSETS.json',dict(seed=2026102673,subsets=curves))
 paths=[HERE/n for n in ['TRAINING_POOL.json','OLD_TRAINING_POOL.json','VALIDATION_INPUTS.json','RUNS.json','TRAIN_TASKS.json','CV_FOLDS.json','LEARNING_CURVE_SUBSETS.json','CONVENTIONS.json','OLD_V2_REFERENCE.json','train_critic.py','critic_models.py','pool_features.py','inference.py','prepare_training.py']]
 paths.extend(HERE/'datasets'/f'{k}.json'for k in definitions);paths.extend(Path(v['path'])for v in definitions.values());paths.extend(Path(r['target_inputs_path'])for r in runs);paths.extend(Path(p['rows_path'])for p in pool().values())
 write(HERE/'TRAINING_PLAN.json',dict(runs=len(runs),fits=len(tasks),datasets=len(definitions),full_variants=40,class_addition_fits=25,curve_fits=45,cv_fits=120,cv_uses_general10_seed_rule=True,maximum_parallel=14,maximum_total_cpus=18,no_generators=True,validation_labels_unopened=True))
 seal('TRAINING_FROZEN.receipt.json',list(dict.fromkeys(paths)),source_only_preprocessing=True,all_run_plans_prespecified=True,validation_labels_read=False)
 print('Training ready:',len(tasks),'fits,',len(definitions),'source-only datasets.',flush=True)
if __name__=='__main__':main()
