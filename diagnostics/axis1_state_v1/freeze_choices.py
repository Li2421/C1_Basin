from common import *
from diagnostics.ring_controller_critic_v1 import fit as old
import jax
jax.config.update('jax_enable_x64',False)
prereg();states=read(HERE/'EVAL_STATES.json');names=read(HERE/'protocol.json')['controller_order'];z=np.load(D/'controller_behavior_repr_v1/FEATURES.npz');nn=z['controller_names'].tolist();r5=z['R5'][[nn.index(n) for n in names]].astype(np.float32)
f=D/'ring_controller_critic_v1';tr={k:v for k,v in np.load(f/'fits/full_transform.npz').items()};st=read(f/'fits/selected_steps.json');ge=read(f/'predictions/G_E_FROZEN_CHOICES.json');menu=['F10','F13','S300','S500'];idx=[old.ARMS.index(a) for a in menu];pred={};choices={};files=[HERE/'PREREGISTRATION.md',HERE/'EVAL_STATES.json',D/'controller_behavior_repr_v1/FEATURES.npz',f/'fits/full_transform.npz',f/'fits/selected_steps.json',f/'predictions/G_E_FROZEN_CHOICES.json',f/'fit.py',D/'controller_q_transfer_round3_v1/state_generator.py',HERE/'freeze_choices.py']
for model in ['A3','A0']:
 values=[]
 for seed in old.SEEDS:
  ck=f/'fits/full'/model/f'seed{seed}'/f'step{st[model]}.msgpack';files.append(ck);values.append(old.predict(model,seed,ck,tr,old.physical23(states).astype(np.float32),r5))
 probs=np.asarray(values).mean(0)[:,:,idx];pred[model]=probs
 for cohort in ['1a','1b']:
  si=[i for i,s in enumerate(states) if s['cohort']==cohort];p=probs[:,si];choices[model+'_'+cohort]=dict(controller=np.argmax(p.mean(1),axis=-1),state=np.argmax(p,axis=-1),state_uids=[states[i]['state_uid'] for i in si])
xx=np.delete(r5,np.s_[105:110],axis=1)[:,tr['r5_keep']];xx=np.where(np.isfinite(xx),xx,tr['r5_mean']);coords=((xx-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T;eh=(coords-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean'];choices['G_E']=np.where(eh<.05,0,3)
np.savez_compressed(HERE/'FROZEN_PREDICTIONS.npz',**pred,controller_names=names,state_uids=[s['state_uid'] for s in states],menu=menu,GE_Ehat=eh)
write(HERE/'FROZEN_CHOICES.json',dict(controller_names=names,menu=menu,choices=choices));files.extend([HERE/'FROZEN_PREDICTIONS.npz',HERE/'FROZEN_CHOICES.json']);seal('CHOICES_FROZEN.receipt.json',files,main_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled));print('Choices frozen before any new main labels')
