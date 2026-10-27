"""New training labels only; validation labels remain unopened."""
from common import *
from design import *
from scenes import features
from descriptors_v2 import descriptor
from stages import read_result
import numpy as np

def main():
 assert(HERE/'ROLLOUTS_COMPLETE.json').exists();lookup={}
 for c in read(HERE/'main_CASES.json'):lookup[(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],c['requested_action'])]=c
 tables={};pool=read(HERE/'OLD_TRAINING_POOL.json')
 for scene in read(HERE/'SCENES.json'):
  name=scene['name'];y=np.zeros((3,8,49));u=np.zeros_like(y);ids=[]
  for ci,ctl in enumerate(CONTROL_IDS):
   for si in range(8):
    for ai,key in enumerate(ETA_KEYS):
     c=lookup[(name,ctl,si,key)];r=read_result('main',c);y[ci,si,ai]=r['success'];u[ci,si,ai]=r['numerical_failure'];ids.append(c['rollout_uid'])
  tables[name]=dict(controllers=list(CONTROL_IDS),state_indices=list(range(8)),eta_keys=ETA_KEYS,y=y.tolist(),unknown=u.tolist(),rollout_uids=ids)
  h=np.asarray([features(dict(scene,states=[s]))[-6:]for s in scene['states']]);state=[];etas=[];ctls=[];succ=[];unknown=[]
  for si in range(8):
   for ci in range(3):
    for ai,a in enumerate(menu()):state.append(si);ctls.append(ci);etas.append(a['eta_a']);succ.append(y[ci,si,ai]);unknown.append(u[ci,si,ai])
  path=HERE/'pool_rows'/f'{name}.npz';np.savez_compressed(path,state=np.array(state),eta=np.array(etas),ctl=np.array(ctls),y=np.array(succ),n=np.ones(len(succ),int),success=np.array(succ),unknown=np.array(unknown),h=h)
  pool.append(dict(name=name,source='New formally registered missing-only rollouts',source_version=name,N=scene['config']['num_agents'],config=scene['config'],states=scene['states'],state_count=8,cells=len(succ),eta_count=49,c_prime=descriptor(scene).tolist(),mean_success=float(np.mean(succ)),new=True,group=scene['group'],rows_path=str(path),rows_sha256=sha(path),specification=scene['specification']))
 pool.sort(key=lambda r:r['name'])
 for row in pool:assert not row['name'].startswith(FORBIDDEN)and not row.get('static_definition','').startswith(FORBIDDEN)
 write(HERE/'LABEL_TABLES.json',tables);write(HERE/'TRAINING_POOL.json',pool)
 # Validation inputs contain physical features and identities only.
 inputs=read_evidence(HET/'INPUTS_V2.json');evaluation=read_evidence(HET/'EVALUATION_STATES.json');write(HERE/'VALIDATION_INPUTS.json',dict(scene_inputs=inputs['scene_inputs'],etas=inputs['etas'],eta_keys=inputs['eta_keys'],evaluation=evaluation['scenes'],labels_read=False))
 print('Training pool assembled:',len(pool),'scenes,',sum(p['cells']for p in pool),'cells; validation labels unopened.',flush=True)
if __name__=='__main__':main()
