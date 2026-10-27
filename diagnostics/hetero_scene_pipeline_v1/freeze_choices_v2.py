"""Freeze only requested methods, with unchanged v2 model selection/inference."""
from common import *
from predict_v2 import load,raw_scores,whole_scene,scene_proposals
from descriptors_v2 import target_guard
from anchored import proposal_priors
from design import CONTROL_IDS
import numpy as np
def select_models():
    selected={}
    for task in read(HERE/'TRAIN_OUTER_TASKS.json'):
        key=(task['fold'],task['mode'])
        if key in selected:continue
        folder=HERE/'folds'/task['fold']/task['mode'];rows=[read(folder/f'seed{seed}'/'FIT.receipt.json')for seed in(17,23,41)]
        for r in rows:assert sha(r['checkpoint'])==r['sha256']and r['source_only_early_stopping']and not r['heldout_labels_used']
        best=min(rows,key=lambda r:(r['validation_loss'],r['seed']));write(folder/'SELECTED.json',dict(selected=best,all_seeds=rows,tie_rule='Minimum source-internal validation loss, then smallest seed.'))
        selected[key]=best
    return selected

def choice(index,menu,keys,C,S,**extra):
    if index is None:return dict(indices=None,applicable=False,**extra)
    a=np.full((C,S),index,int)if np.asarray(index).ndim==0 else np.asarray(index,int)
    assert a.shape==(C,S)
    unique,counts=np.unique(a,return_counts=True)
    return dict(indices=a.tolist(),selected_index=int(a[0,0])if np.all(a==a[0,0])else None,applicable=True,
        selected_candidates=[dict(index=int(i),key=keys[i],eta=menu[i].tolist(),cells=int(count))for i,count in zip(unique,counts)],**extra)

def evaluation_data(data):
 d=copy.deepcopy(data);positions=d['evaluation_state_positions']
 for field in ['h','state_indices']:d[field]=[d[field][i]for i in positions]
 d['evaluation_state_positions']=list(range(len(positions)));return d

def main():
 assert(HERE/'train_outer_COMPLETE.json').exists();assert(HERE/'LAMBDA_SELECTION_SUMMARY.json').exists();select_models()
 inputs=read(HERE/'INPUTS_V2.json');eta=np.asarray(inputs['etas'],float);keys=inputs['eta_keys'];receipts=[]
 for fold in read(HERE/'FOLDS.json'):
  folder=HERE/'folds'/fold['name'];receipt=HERE/'FOLD_FROZEN'/f"{fold['name']}.receipt.json"
  if receipt.exists():
   for p,h in read(receipt)['files'].items():assert sha(p)==h,p
   receipts.append(receipt);continue
  p=read(folder/'PRIORS.json');ls=read(folder/'LAMBDA_SELECTION.json');lam=ls['full']['selected_lambda'];full,fit=load(fold['name'],'full');gparams,gfit=load(fold['name'],'generator');choices={};proposals={}
  for name in fold['targets']:
   data=evaluation_data(inputs['scene_inputs'][name]);C,S=len(data['controllers']),len(data['h']);bounds=fold['ranges'];guard_inputs={name:data};guard=target_guard(guard_inputs,name,bounds)
   prop=scene_proposals(gparams,gfit,fold,data,name);proposals[name]=prop;menu=np.concatenate([eta,np.asarray(prop['etas']).reshape(-1,3)]);candidate_keys=keys+['proposal_'+str(i)for i in range(5)];p0=np.r_[p['p0'],proposal_priors(prop['etas'],eta,p)]
   raw=raw_scores(full,fold,data,menu);best,logits,scene_score=whole_scene(raw,p0,lam,'full',guard);b49,l49,s49=whole_scene(raw[:,:,:49],p['p0'],lam,'full',guard)
   bn,ln,sn=whole_scene(raw[:,:,:49],p['p0'],lam,'full',guard,False,False)
   methods={'v2':choice(best,menu,candidate_keys,C,S,shrinkage=lam,scene_fallback=guard['scene_out_of_range']),'v2_49':choice(b49,menu,candidate_keys,C,S,shrinkage=lam,scene_fallback=guard['scene_out_of_range']),'v2_49_no_fallback':choice(bn,menu,candidate_keys,C,S,shrinkage=lam),'global_fixed':choice(p['best_index'],menu,candidate_keys,C,S),'knn':choice(keys.index(fold['baselines'][name]['knn']),menu,candidate_keys,C,S),'safety':choice(0,menu,candidate_keys,C,S)}
   # Same frozen predictor, prior, lambda and controller inputs; replace only c' and h.
   generic=copy.deepcopy(data);generic['c_prime']=np.mean([inputs['scene_inputs'][n]['c_prime']for n in fold['sources']],axis=0).tolist();generic['h']=np.repeat(np.asarray(fold['normalization']['h_mean'])[None],S,axis=0).tolist()
   generic_guard=target_guard({name:generic},name,bounds);gr=raw_scores(full,fold,generic,eta);bg,lg,sg=whole_scene(gr,p['p0'],lam,'full',generic_guard)
   methods['generic_input']=choice(bg,menu,candidate_keys,C,S,shrinkage=lam,generic_scene_out_of_range=generic_guard['scene_out_of_range'])
   if lam==0 or guard['scene_out_of_range']:assert methods['v2']['selected_index']==p['best_index']
   choices[name]=dict(controllers=data['controllers'],controller_types=data['controller_types'],state_indices=data['state_indices'],candidate_etas=menu.tolist(),candidate_keys=candidate_keys,methods=methods,guard=guard,shrinkage=lam,scene_scores=dict(v2=scene_score.tolist(),v2_49=s49.tolist(),v2_49_no_fallback=sn.tolist(),generic_input=sg.tolist()),generic_input=dict(c_prime=generic['c_prime'],h=generic['h'][0],guard=generic_guard,h_mean_scope='Actual fitting-state mean from source-only normalization; c_prime unweighted source-scene mean.'),prior_probabilities=p0.tolist())
  write(folder/'PROPOSALS.json',proposals);write(folder/'CHOICES.json',choices)
  paths=[folder/n for n in ['definition.json','PRIORS.json','LAMBDA_SELECTION.json','PROPOSALS.json','CHOICES.json','source_labels.npz']]
  paths.extend(HERE/n for n in ['INPUTS_V2.json','GUARD.json','TRAINING_FROZEN.receipt.json','freeze_choices_v2.py','predict_v2.py','anchored.py','descriptors_v2.py','networks.py','METHOD_PARITY_CHECK.json'])
  for mode in ['full','generator']:
   paths.append(folder/mode/'SELECTED.json')
   for seed in [17,23,41]:paths.extend(folder/mode/f'seed{seed}'/n for n in ['FIT.receipt.json','history.json','checkpoint.msgpack'])
  for link in fold['inner_links']['full'].values():
   nf=HERE/'folds'/link['training_node'];paths.extend(nf/n for n in ['definition.json','PRIORS.json','source_labels.npz']);paths.extend(nf/'full'/'seed17'/n for n in ['FIT.receipt.json','history.json','checkpoint.msgpack'])
  seal(str(receipt.relative_to(HERE)),list(dict.fromkeys(paths)),fold=fold['name'],sources=fold['sources'],targets=fold['targets'],source_only_models_priors_lambda_and_normalization=True,all_choices_and_proposals_frozen=True,proposal_rollouts_before_freeze=0)
  receipts.append(receipt);print('FROZEN',fold['name'],flush=True)
 write(HERE/'ALL_FOLDS_FROZEN.receipt.json',dict(utc=utc(),folds=len(receipts),files={str(p):sha(p)for p in receipts},proposal_rollouts_before_all_folds_frozen=0,models_priors_bandwidth_lambdas_guards_proposals_and_all_choices_frozen=True))
if __name__=='__main__':main()
