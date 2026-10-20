from core import *
from guard import guard
from audit_utils import source_inventory
import critic_fit as old
G=guard()
def main():
 assert (HERE/'R5_FROZEN.receipt.json').exists();z=np.load(HERE/'R5_FEATURES.npz');names=z['controller_names'].tolist();r5=z['features'];physical=old.physical23(read(HERE/'EVAL_STATES.json')).astype(np.float32);tr={k:v for k,v in np.load(D/'ring_controller_critic_v1/fits/full_transform.npz').items()};steps=read(D/'ring_controller_critic_v1/fits/selected_steps.json');arms=['F10','F13','S300','S500'];idx=[old.ARMS.index(a) for a in arms];pred={};choices={n:{} for n in names};files=[]
 for model in ['A0','A3']:
  vals=[]
  for seed in old.SEEDS:
   ck=D/'ring_controller_critic_v1/fits/full'/model/f'seed{seed}'/f'step{steps[model]}.msgpack';files.append(ck);vals.append(old.predict(model,seed,ck,tr,physical,r5))
  vals=np.asarray(vals)[:,:,:,idx];pred[model]=vals;mean=vals.mean((0,2))
  for i,n in enumerate(names):choices[n][model+'_full']=arms[int(mean[i].argmax())]
 ge=read(D/'ring_controller_critic_v1/predictions/G_E_FROZEN_CHOICES.json');x=np.delete(r5,np.s_[105:110],axis=1)[:,tr['r5_keep']];x=np.where(np.isfinite(x),x,tr['r5_mean']);pc=((x-tr['r5_mean'])/tr['r5_scale'])@tr['pca'].T;eh=(pc-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean']
 for i,n in enumerate(names):choices[n].update(G_E='F10' if eh[i]<.05 else 'S500',G_E_Ehat=eh[i],S500='S500')
 np.savez_compressed(HERE/'BLIND_PREDICTIONS.npz',**pred,controller_names=names,arms=arms);write(HERE/'FROZEN_CHOICES.json',dict(utc=utc(),controllers=choices,controller_order=names,arm_order=arms,evaluation_labels_read=False,guard=guard()));seal('CHOICES_FROZEN.receipt.json',[HERE/'BLIND_PREDICTIONS.npz',HERE/'FROZEN_CHOICES.json',HERE/'freeze_choices.py',HERE/'R5_FEATURES.npz',D/'ring_controller_critic_v1/fits/full_transform.npz']+files,before_main_labels=True);source_inventory('imports_choices.json')
if __name__=='__main__':main()
