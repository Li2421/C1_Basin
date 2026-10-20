from common import *
from train_mixed import MixedNet,build_features
from diagnostics.ring_controller_critic_v1 import fit as old
from flax import serialization
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
prereg();assert (HERE/'R5_FROZEN.receipt.json').exists()
for path,digest in read(HERE/'MIXED_FROZEN.receipt.json')['files'].items():assert sha(path)==digest,path
states=read(HERE/'EVAL_STATES.json');z=np.load(HERE/'R5_FEATURES.npz');names=z['controller_names'].tolist();r5=z['features'];f=D/'ring_controller_critic_v1';tr={k:v for k,v in np.load(f/'fits/full_transform.npz').items()};steps=read(f/'fits/selected_steps.json');idx=[old.ARMS.index(a) for a in MENU];pred={};choices={n:{} for n in names};files=[]
for model in ['A0','A3']:
 values=[]
 for seed in old.SEEDS:
  path=f/'fits/full'/model/f'seed{seed}'/f'step{steps[model]}.msgpack';files.append(path);values.append(old.predict(model,seed,path,tr,old.physical23(states).astype(np.float32),r5))
 values=np.asarray(values)[:,:,:,idx];pred[model]=values;mean=values.mean((0,2))
 for i,name in enumerate(names):choices[name][model+'_full']=MENU[int(mean[i].argmax())];choices[name][model+'_fixed2']=MENU[int(mean[i,:2].argmax())];choices[name][model+'_fixed3']=MENU[int(mean[i,:3].argmax())]
X=build_features(states,r5,1);net=MixedNet();values=[];selected=read(HERE/'MIXED_SELECTED_STEP.json')['selected_step']
for seed in old.SEEDS:
 path=HERE/'mixed/full'/f'seed{seed}'/f'step{selected}.msgpack';files.append(path);pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,32),jnp.float32)),path.read_bytes());values.append(np.asarray(jax.nn.sigmoid(net.apply(pa,jnp.asarray(X.reshape(-1,32))))).reshape(len(names),32,6))
pred['mixed']=np.asarray(values);mean=pred['mixed'].mean((0,2))
for i,name in enumerate(names):choices[name]['mixed_full']=MENU[int(mean[i].argmax())]
ge=read(f/'predictions/G_E_FROZEN_CHOICES.json');x=np.delete(r5,np.s_[105:110],axis=1)[:,tr['r5_keep']];x=np.where(np.isfinite(x),x,tr['r5_mean']);pc=((x-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T;eh=(pc-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean']
for i,name in enumerate(names):choices[name].update(G_E='F10' if eh[i]<.05 else 'S500',S500='S500',F10='F10')
np.savez_compressed(HERE/'FROZEN_PREDICTIONS.npz',**pred,GE_Ehat=eh,controller_names=names,state_uids=[s['state_uid'] for s in states],arms=MENU);write(HERE/'FROZEN_CHOICES.json',dict(utc=utc(),controllers=choices,controller_order=names,arms=MENU,new_evaluation_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled)));seal('CHOICES_FROZEN.receipt.json',[HERE/'FROZEN_PREDICTIONS.npz',HERE/'FROZEN_CHOICES.json',HERE/'freeze_choices.py',HERE/'R5_FEATURES.npz',f/'fits/full_transform.npz']+files,before_new_evaluation_labels=True);print('Frozen selections',len(names))
