from common import *
from features import physical12,action7,apply_transform
from train_v2 import Critic
from flax import serialization
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
def main():
 prereg();assert not (HERE/'main_SUBMISSION.json').exists();files=[]
 for sc in SCENES:
  receipt=HERE/f'{sc}_MODEL_FROZEN.receipt.json'
  for path,digest in read(receipt)['files'].items():assert sha(path)==digest,path
  files.append(receipt)
 cap=read(HERE/'CAPABILITY_GATE.json')['eligible'];states=read(HERE/'STATES.json');r5=np.load(HERE/'R5_FEATURES.npz');rkeys=r5['controller_keys'].tolist();source=np.load(HERE/'SOURCE_TRAINING_DATA.npz');net=Critic();choices={};source_mean={};preds={}
 for target in SCENES:
  scene_values=[];coverage={}
  for scene in SCENES:
   if scene==target:continue
   vals=[]
   for arm in MENU:
    mask=(source['scenes']==scene)&(source['arms']==arm);controllers=sorted(set(source['controller_keys'][mask].tolist()));rates=[]
    for ctl in controllers:
     mm=mask&(source['controller_keys']==ctl);rates.append([np.mean(source['y'][mm]/source['q'][mm]),np.mean((source['y'][mm]+source['q'][mm]-source['n'][mm])/source['q'][mm])])
    vals.append(np.mean(rates,axis=0) if rates else [np.nan,np.nan]);coverage[scene+'/'+arm]=len(controllers)
   scene_values.append(vals)
  with np.errstate(invalid='ignore'):mean=np.nanmean(np.asarray(scene_values),axis=0)
  best=int(mean[:,0].argmax());possible=[MENU[i] for i in range(7) if mean[i,1]>=mean[:,0].max()];source_mean[target]=dict(chosen=MENU[best],possible=possible,success_ranges=mean,source_scenes=[s for s in SCENES if s!=target],coverage=coverage,scene_equal=True)
  names=cap[target]
  if not names:choices[target]={};continue
  ss=states[target]['eval'];features=[]
  for name in names:
   key=target+'::'+name;cf=r5['features'][rkeys.index(key)];features.append([[np.concatenate([physical12(target,s),cf,action7(arm)]) for arm in MENU] for s in ss])
  X=np.asarray(features,np.float32);selected=read(HERE/'fits'/target/'SELECTED_STEP.json')['selected_step'];pred=[]
  for seed in [17,23,41]:
   folder=HERE/'fits'/target/'full'/f'seed{seed}';tr={k:v for k,v in np.load(folder/'transform.npz').items()};path=folder/f'step{selected}.msgpack';pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,24),jnp.float32)),path.read_bytes());pred.append(np.asarray(jax.nn.sigmoid(net.apply(pa,jnp.asarray(apply_transform(X.reshape(-1,24),tr))))).reshape(len(names),32,7));files.extend([path,folder/'transform.npz'])
  pred=np.asarray(pred);preds[target]=pred;mean=pred.mean((0,2));choices[target]={n:MENU[int(mean[i].argmax())] for i,n in enumerate(names)}
 np.savez_compressed(HERE/'FROZEN_PREDICTIONS.npz',**preds);write(HERE/'FROZEN_CHOICES.json',dict(utc=utc(),controllers=choices,source_mean=source_mean,eligible=cap,arms=MENU,new_evaluation_labels_read=False));seal('CHOICES_FROZEN.receipt.json',files+[HERE/'FROZEN_PREDICTIONS.npz',HERE/'FROZEN_CHOICES.json',HERE/'freeze_choices.py',HERE/'R5_FEATURES.npz'],before_new_evaluation_labels=True);print('LOSO/source choices frozen',flush=True)
if __name__=='__main__':main()
