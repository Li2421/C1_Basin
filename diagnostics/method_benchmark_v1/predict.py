from core import *
from diagnostics.ring_controller_critic_v1 import fit as old
import jax
jax.config.update('jax_enable_x64',False)
def r5_for(tag,names):
 files={'T1':('ring_independent_flow_confirmation_v2/R5_FINAL_FEATURES.npz','features'),'T2':('controller_behavior_repr_v1/FEATURES.npz','R5'),'T3':('controller_new_checkpoint_confirmation_v1/fp32_n8/R5_FINAL_FEATURES.npz','features'),'T4':('ring_critic_new_recipe_confirmation_v1/R5_FEATURES.npz','features'),'T5':('ring_eta2_gap_v1/R5_FEATURES.npz','features')}
 path,key=files[tag];z=np.load(D/path);nn=z['controller_names'].tolist();return z[key][[nn.index(n) for n in names]].astype(np.float32)
def main():
 prereg();assert (HERE/'MODEL_METHOD_FREEZE.receipt.json').exists()
 for path,digest in read(HERE/'MODEL_METHOD_FREEZE.receipt.json')['files'].items():assert sha(path)==digest
 p=read(HERE/'INPUT_IDENTITIES.json');tr={k:v for k,v in np.load(HERE/'M3_TRANSFORM.npz').items()};tr16={k:v for k,v in np.load(D/'ring_controller_critic_v1/fits/full_transform.npz').items()};st16=read(D/'ring_controller_critic_v1/fits/selected_steps.json');step=read(HERE/'M3_SELECTED_STEP.json')['selected_step'];ge=read(D/'ring_controller_critic_v1/predictions/G_E_FROZEN_CHOICES.json');allchoices={}
 for tag in ['T1','T2','T3','T4','T5']:
  names=p['groups'][tag];states=p['states'][tag];physical=old.physical23(states).astype(np.float32);r5=r5_for(tag,names);r3=np.asarray([np.nanmean(np.load(HERE/'r3'/f'{n}.npz')['R3'],axis=0) for n in names],np.float32);padded=np.concatenate((r3[:,:105],np.zeros((len(names),5),np.float32),r3[:,105:]),axis=1);pred={}
  for model in ['M3','M4','A0']:
   values=[]
   for seed in old.SEEDS:
    ck=HERE/'fits/full'/f'seed{seed}'/f'step{step}.msgpack' if model=='M3' else D/'ring_controller_critic_v1/fits/full'/('A3' if model=='M4' else 'A0')/f'seed{seed}'/f'step{st16["A3" if model=="M4" else "A0"]}.msgpack'
    values.append(old.predict('A0' if model=='A0' else 'A3',seed,ck,tr if model=='M3' else tr16,physical,padded if model=='M3' else r5))
   pred[model]=np.asarray(values)
  clean=np.delete(r5,np.s_[105:110],axis=1)[:,tr16['r5_keep']];clean=np.where(np.isfinite(clean),clean,tr16['r5_mean']);coords=((clean-tr16['r5_mean'])/tr16['r5_scale'])@tr16['pca'].T;eh=(coords-np.asarray(ge['xmean']))@np.asarray(ge['coefficient'])+ge['ymean'];ges=['F10' if x<.05 else 'S500' for x in eh]
  np.savez_compressed(HERE/f'{tag}_PREDICTIONS.npz',**pred,controller_names=names,state_uids=[s['state_uid'] for s in states],arms=old.ARMS,GE_Ehat=eh,GE_choices=ges)
  allchoices[tag]=dict(controller_names=names,GE=ges,physical_state_uids=[s['state_uid'] for s in states])
 write(HERE/'BLIND_PREDICTION_INDEX.json',allchoices);seal('PREDICTIONS_FROZEN.receipt.json',[HERE/f'{t}_PREDICTIONS.npz' for t in allchoices]+[HERE/'predict.py',HERE/'BLIND_PREDICTION_INDEX.json'],test_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled));print('Blind predictions frozen',flush=True)
if __name__=='__main__':main()
