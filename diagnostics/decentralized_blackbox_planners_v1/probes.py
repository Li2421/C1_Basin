from common import *
from audit_utils import source_inventory
G=guard()
def freeze():
 p=read(HERE/'probe_protocol.json');names=p['controller_order'];f=np.full((len(names),16,2,5),np.nan,np.float32);index=[];unknown=[]
 for c in read(HERE/'probe_CASES.json'):
  dest=HERE/'probe/raw'/f"{c['rollout_uid']}.json";r=read(dest);assert not r['terminal_labels_read'];assert sha(r['trace_path'])==r['trace_sha256']
  if r['complete']:
   z=np.load(r['trace_path']);pos=z['positions'];acts=z['actions'];goals=z['goals'];initial=np.linalg.norm(goals-pos[0],axis=-1).sum();final=np.linalg.norm(goals-pos[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(acts,axis=-1);f[names.index(c['base_controller_name']),c['state_index'],[10,13].index(c['eta_index'])]=[(initial-final)/initial,z['projection_active'].mean(),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
  else:unknown.append(r)
  index.append(dict(case=c,raw_sha256=sha(dest),trace_sha256=r['trace_sha256']))
 np.savez_compressed(HERE/'R5_FEATURES.npz',features=f.reshape(len(names),160),controller_names=names);write(HERE/'R5_RAW_INDEX.json',index);write(HERE/'probe_POSTFLIGHT.json',dict(utc=utc(),complete_records=len(index),unknown=unknown,task_labels=0,guard=guard()));seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json'],before_any_evaluation_labels=True);source_inventory('imports_probe_freeze.json')
if __name__=='__main__':freeze()
