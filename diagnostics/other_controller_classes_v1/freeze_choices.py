from core import *
from diagnostics.ring_controller_critic_v1 import fit as old
import jax
jax.config.update('jax_enable_x64',False)
prereg();assert (HERE/'R5_FROZEN.receipt.json').exists();p=read(HERE/'probe_protocol.json');z=np.load(HERE/'R5_FEATURES.npz');names=z['controller_names'].tolist();r5=z['features'];states=read(HERE/'EVAL_STATES.json');physical=old.physical23(states).astype(np.float32);tr={k:v for k,v in np.load(D/'ring_controller_critic_v1/fits/full_transform.npz').items()};steps=read(D/'ring_controller_critic_v1/fits/selected_steps.json');armorder=['F10','F13','F2','S200','S300','S500'];idx=[old.ARMS.index(a) for a in armorder];pred={};choices={n:{} for n in names};files=[]
for model in ['A0','A3']:
 vals=[]
 for seed in old.SEEDS:
  ck=D/'ring_controller_critic_v1/fits/full'/model/f'seed{seed}'/f'step{steps[model]}.msgpack';files.append(ck);vals.append(old.predict(model,seed,ck,tr,physical,r5))
 vals=np.asarray(vals)[:,:,:,idx];pred[model]=vals;mean=vals.mean((0,2))
 for i,n in enumerate(names):choices[n][model+'_full']=armorder[int(mean[i].argmax())];choices[n][model+'_fixed']=armorder[int(mean[i,:3].argmax())]
ge=read(D/'ring_controller_critic_v1/predictions/G_E_FROZEN_CHOICES.json');x=np.delete(r5,np.s_[105:110],axis=1)[:,tr['r5_keep']];x=np.where(np.isfinite(x),x,tr['r5_mean']);pc=((x-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T;eh=(pc-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean']
for i,n in enumerate(names):choices[n].update(G_E='F10' if eh[i]<.05 else 'S500',G_E_Ehat=eh[i],S500='S500',F10='F10')
np.savez_compressed(HERE/'BLIND_PREDICTIONS.npz',**pred,controller_names=names,state_uids=[s['state_uid'] for s in states],arms=armorder)
write(HERE/'FROZEN_CHOICES.json',dict(utc=utc(),controllers=choices,controller_order=names,arm_order=armorder,state_uids=[s['state_uid'] for s in states],evaluation_labels_read=False,GE_training='R15 source-only ridge1 frozen R16',jax_enable_x64=bool(jax.config.x64_enabled)))
seal('CHOICES_FROZEN.receipt.json',[HERE/'BLIND_PREDICTIONS.npz',HERE/'FROZEN_CHOICES.json',HERE/'freeze_choices.py',HERE/'R5_FEATURES.npz',D/'ring_controller_critic_v1/fits/full_transform.npz']+files,evaluation_labels_read=False,all_imports_then_precision_restored=True);print('Crossclass choices frozen',len(names),flush=True)
