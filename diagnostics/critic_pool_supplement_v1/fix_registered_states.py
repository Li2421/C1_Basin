from common import *
from design import OLD
from scenes import features
from descriptors_v2 import descriptor
import numpy as np
pool=read(HERE/'OLD_TRAINING_POOL.json');static={s['name']:s for p in [OLD/'SCENES.json',ROOT/'diagnostics/eta_diverse_scene_test_v1/SCENES.json']for s in read_evidence(p)};snapshot=[];changes={}
with dbm.connect(True)as db:
 for g in pool:
  n=g['config']['num_agents'];ids=g['scenario_uids'];records=db.execute('SELECT state_uid,scenario_uid,physical_state_json FROM state WHERE scenario_uid IN ('+','.join('?'for _ in ids)+')',ids).fetchall();snapshot.extend(dict(r)for r in records);physical={uid('physical',json.loads(r['physical_state_json'])):json.loads(r['physical_state_json'])for r in records};oldstates=g['states'];states=[dict(scene=g['config']['scene'],index=i,N=n,**p,physical_id=pid)for i,(pid,p)in enumerate(sorted(physical.items()))]
  if len(states)==g['state_count']:continue
  assert all(len(s['positions'])==n for s in states);scene=copy.deepcopy(static[g['static_definition']]);scene['name']=g['name'];scene['config']=g['config'];scene['states']=states;scene.setdefault('moving',False);scene['features']=features(scene);by={s['physical_id']:i for i,s in enumerate(states)}
  with np.load(g['rows_path'],allow_pickle=False)as z:data={k:z[k]for k in z.files}
  data['state']=np.array([by[oldstates[int(i)]['physical_id']]for i in data['state']]);data['h']=np.array([features(dict(scene,states=[s]))[-6:]for s in states]);tmp=Path(g['rows_path']).with_suffix('.fix.npz');np.savez_compressed(tmp,**data);tmp.replace(g['rows_path']);changes[g['name']]=dict(before=g['state_count'],after=len(states),observed_cells_unchanged=g['cells']);g.update(states=states,state_count=len(states),c_prime=descriptor(scene).tolist(),rows_sha256=sha(g['rows_path']),all_registered_states_in_description=True)
write(HERE/'REGISTERED_TRAINING_STATES.receipt.json',dict(states=snapshot,all_registered_states=True,labels_not_imputed=True));write(HERE/'OLD_TRAINING_POOL.json',pool);write(HERE/'ALL_REGISTERED_STATES_FIX.receipt.json',dict(utc=utc(),changes=changes,no_model_training_before_fix=True,observed_training_cells_unchanged=True));write(HERE/'OLD_POOL_READY.receipt.json',dict(utc=utc(),sha256=sha(HERE/'OLD_TRAINING_POOL.json'),source_cells=sum(g['cells']for g in pool),all_registered_states_included=True));print('Corrected all-state features:',changes)
