"""Outcome-free per-seed predictions, then average logits before candidate selection."""
from common import *
from critic_models import model
from flax import serialization
import jax,jax.numpy as jnp,numpy as np

def predict_seed(task,fit):
 folder=HERE/'fits'/task['id']/f"seed{task['seed']}";receipt=folder/'PREDICTIONS.receipt.json'
 if receipt.exists():
  r=read(receipt);assert sha(r['path'])==r['sha256'];return r
 assert sha(task['target_inputs_path'])==task['target_inputs_sha256'];assert sha(fit['checkpoint'])==fit['sha256'];net=model(task['kind']);template=net.init(jax.random.PRNGKey(task['seed']),jnp.zeros((1,27),jnp.float32),training=False);params=serialization.from_bytes(template,Path(fit['checkpoint']).read_bytes());apply=jax.jit(lambda p,x:net.apply(p,x,training=False));out={}
 with np.load(task['target_inputs_path'],allow_pickle=False)as z:
  for name in z.files:
   out[name]=np.asarray(apply(params,jnp.asarray(z[name])),float);assert np.isfinite(out[name]).all(),('NONFINITE_PREDICTION',task,name)
 path=folder/'PREDICTIONS.npz';np.savez_compressed(path,**out);r=dict(utc=utc(),path=str(path),sha256=sha(path),checkpoint_sha256=fit['sha256'],input_sha256=task['target_inputs_sha256'],validation_labels_read=False);write(receipt,r);return r

def aggregate():
 assert(HERE/'TRAIN_COMPLETE.json').exists();paths=[HERE/'TRAINING_FROZEN.receipt.json'];all_choices={};from pool_features import baseline,pool
 for run in read(HERE/'RUNS.json'):
  values={n:[]for n in run['targets']}
  for seed in run['seeds']:
   folder=HERE/'fits'/run['id']/f'seed{seed}';fit=read(folder/'FIT.receipt.json');pred=read(folder/'PREDICTIONS.receipt.json');assert sha(pred['path'])==pred['sha256']
   with np.load(pred['path'],allow_pickle=False)as z:
    for n in values:values[n].append(z[n])
   paths.extend(folder/n for n in ['FIT.receipt.json','PREDICTIONS.receipt.json','PREDICTIONS.npz','checkpoint.msgpack','history.json'])
  predictions={};choices={};definition=read(HERE/'datasets'/f"{run['dataset']}.json")
  for n,v in values.items():
   shape=tuple(run['target_metadata'][n]['shape']);mean=np.mean(v,axis=0).reshape(shape);prob=np.exp(-np.logaddexp(0,-mean));scores=prob.mean((0,1));a=int(scores.argmax());predictions[n+'__mean_logits']=mean;predictions[n+'__probability']=prob
   c=(read(HERE/'VALIDATION_INPUTS.json')['scene_inputs'][n]['c_prime']if run['target_type']=='validation'else pool()[n]['c_prime']);baselines={('global'if k is None else'knn'+str(k)):baseline(definition['names'],np.asarray(c),k)for k in [None,1,2,3]}
   choices[n]=dict(selected_index=a,selected_candidate=actions()[a],per_cell_indices=prob.argmax(-1).tolist(),scores=scores.tolist(),baselines=baselines,state_indices=run['target_metadata'][n]['state_indices'],near=run['target_metadata'][n]['near'])
  folder=HERE/'predictions'/run['id'];folder.mkdir(parents=True,exist_ok=True);p=folder/'ENSEMBLE.npz';np.savez_compressed(p,**predictions);write(folder/'CHOICES.json',choices);paths.extend([p,folder/'CHOICES.json']);all_choices[run['id']]=choices
 write(HERE/'ALL_CHOICES.json',all_choices);paths.extend(HERE/n for n in ['ALL_CHOICES.json','inference.py','critic_models.py','pool_features.py','CONVENTIONS.json','STATISTICS_READY.receipt.json','statistics_pool.py','OLD_V2_REFERENCE.json'])
 seal('FROZEN.receipt.json',list(dict.fromkeys(paths)),all_predictions_and_choices_frozen=True,validation_labels_read=False,no_generator=True,ensemble='Mean10 logits except explicitly3/5 seed diagnostics; sigmoid after averaging.')
 print('All predictions and choices frozen; validation labels may now be read.',flush=True)
if __name__=='__main__':aggregate()
