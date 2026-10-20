from core import *
import time
from diagnostics.ring_controller_critic_v1 import fit as old
import jax
jax.config.update('jax_enable_x64',False)
def setup():
 prereg();p=read(HERE/'INPUT_IDENTITIES.json');data=old.source_data()
 assert data['names']==p['groups']['SOURCE'];raw=[]
 for n in data['names']:
  f=HERE/'r3'/f'{n}.npz';r=read(f.with_suffix('.json'));assert r['features_sha256']==sha(f) and not r['jax_enable_x64'];raw.append(np.nanmean(np.load(f)['R3'],axis=0))
 raw=np.asarray(raw,np.float32)
 # R16 deletes positions105:110. Insert five inert placeholders so every R3
 # coordinate is retained, keeping its fitting and feature code unchanged.
 data['r5']=np.concatenate((raw[:,:105],np.zeros((len(raw),5),np.float32),raw[:,105:]),axis=1)
 return data
def cv(k,seed):
 data=setup();recipes=sorted(set(data['recipes']));hi=np.flatnonzero(np.isin(data['recipes'],[recipes[k],recipes[k+7]]));si=np.flatnonzero(~np.isin(data['recipes'],[recipes[k],recipes[k+7]]));tr=old.transform_fit(data,si)
 folder=HERE/'fits'/f'fold{k}'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(folder/'transform.npz',**tr)
 old.fit_one('A3',seed,data,si,hi,tr,None,folder)
def full():
 data=setup();hist=[read(HERE/'fits'/f'fold{k}'/f'seed{s}'/'fit.json')['validation_nll'] for k in range(7) for s in old.SEEDS];means=np.mean(hist,axis=0);step=old.STEPS[int(np.argmin(means))];write(HERE/'M3_SELECTED_STEP.json',dict(selected_step=step,candidate_steps=old.STEPS,mean_validation_nll=means,source_only=True));si=np.arange(56);tr=old.transform_fit(data,si);np.savez_compressed(HERE/'M3_TRANSFORM.npz',**tr)
 for seed in old.SEEDS:old.fit_one('A3',seed,data,si,si,tr,None,HERE/'fits/full'/f'seed{seed}',step)
 files=list((HERE/'r3').glob('*'))+list((HERE/'fits').rglob('*.msgpack'))+[HERE/'M3_TRANSFORM.npz',HERE/'M3_SELECTED_STEP.json',HERE/'PREREGISTRATION.md',HERE/'INPUT_IDENTITIES.json',HERE/'train_m3.py',HERE/'static_features.py',HERE/'core.py']
 assert len(list((HERE/'r3').glob('*.npz')))==len(read(HERE/'INPUT_IDENTITIES.json')['profiles'])
 seal('R3_FROZEN.receipt.json',list((HERE/'r3').glob('*')),test_labels_read=False,simulator_steps=0)
 seal('MODEL_METHOD_FREEZE.receipt.json',files,selected_step=step,fit_seeds=old.SEEDS,all_method_hyperparameters_in_preregistration=True,test_labels_read=False,precision_after_all_imports=bool(jax.config.x64_enabled))
 print('M3 AND ALL METHODS FROZEN',step,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='cv':cv(int(sys.argv[2]),int(sys.argv[3]))
 else:full()
