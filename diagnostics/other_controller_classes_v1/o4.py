from core import *
import inspect
from diagnostics.ring_controller_critic_v1 import fit as old
import jax
jax.config.update('jax_enable_x64',False)
source=inspect.getsource(old.fit_one);assert ' / 4)' in source;source=source.replace(' / 4)',' / 2)');namespace=dict(old.__dict__);exec(compile(source,'round27_Q2_fit_one','exec'),namespace);fit_one=namespace['fit_one']
def main():
 prereg();p=read(HERE/'main_protocol.json');names=p['controller_order'];arms=p['arms'];states=p['states'];cases=read(HERE/'main_CASES.json');y=np.zeros((len(names),32,8),np.float32);n=y.copy();seen=set()
 for c in cases:
  r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");i=names.index(c['base_controller_name']);s=c['state_index'];a=old.ARMS.index(c['arm']);key=(i,s,a,c['seed']);assert key not in seen;seen.add(key);y[i,s,a]+=r['success'];n[i,s,a]+=not r['numerical_failure']
 z=np.load(HERE/'R5_FEATURES.npz');nn=z['controller_names'].tolist();r5=z['features'][[nn.index(v) for v in names]];step=read(D/'ring_controller_critic_v1/fits/selected_steps.json')['A3'];choices={};files=[]
 for kind in ['M','P']:
  indices=[i for i,name in enumerate(names) if name.startswith(kind) and name[1:].isdigit()]
  if len(indices)<2:continue
  for hi in indices:
   si=np.array([i for i in indices if i!=hi]);data=dict(names=names,physical=old.physical23(states).astype(np.float32),r5=r5,r6=np.zeros((len(names),7),np.float32),y=y,n=n);tr=old.transform_fit(data,si);folder=HERE/'o4'/names[hi];folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'transform.npz',**tr);pred=[]
   for seed in old.SEEDS:
    dest=folder/f'seed{seed}';ck=dest/f'step{step}.msgpack'
    if not ck.exists():fit_one('A3',seed,data,si,si,tr,None,dest,step)
    pred.append(old.predict('A3',seed,ck,tr,data['physical'],r5[[hi]])[0]);files.append(ck)
   mean=np.mean(pred,axis=(0,1));idx=[old.ARMS.index(a) for a in arms];selected=arms[int(np.argmax(mean[idx]))];choices[names[hi]]=dict(action=selected,source_controllers=[names[i] for i in si],selected_step=step,source_only_normalization=True,held_controller_labels_used_for_fit_or_selection=False);np.savez_compressed(folder/'predictions.npz',probability=np.asarray(pred),arms=old.ARMS)
 write(HERE/'O4_CHOICES.json',dict(utc=utc(),controllers=choices,selected_step_from_R16=step,evidence='DEVELOPMENT_LOCO; no held-controller labels for model fitting or selection',objective='Q2 valid-count BCE divided by2',fit_code_sha256=hashlib.sha256(source.encode()).hexdigest(),checkpoint_files=[str(x) for x in files]));print('O4 complete',len(choices),flush=True)
if __name__=='__main__':main()
