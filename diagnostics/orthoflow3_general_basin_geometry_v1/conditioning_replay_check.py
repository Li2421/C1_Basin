#!/usr/bin/env python3
"""No environment step: compare historical batch32 and current batch1 h0 replay."""
import importlib.util,sys,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from audit import STATES,POINT,D,dump
from run_rollout_shard import LEGACY
spec=importlib.util.spec_from_file_location('replay_legacy',LEGACY);wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
states={s:dict(r,feature_index=r['dataset_index']) for s,r in STATES.items() if s.startswith('T0_WIDE')};features=np.load(POINT/'point_learning_arrays.npz')['features']
out=HERE/'conditioning_check';out.mkdir(exist_ok=True);m,O=wrapper.loadmod(out,states);o=O(states,features,np.zeros(3),np.ones(3))
import jax
import jax.numpy as jnp
records=[]
for sid,s in states.items():
 env=o.restore_full(Path(s['state_file']),o.config);obs=np.asarray(env.observation(),dtype=np.float32);key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['flow_seed']),s['rng_namespace']),0)
 traces=[]
 for batch in (1,32,64):
  obsbatch=np.repeat(obs[None],batch,axis=0);keys=np.repeat(np.asarray(key)[None],batch,axis=0)
  action=np.asarray(o.sample(jnp.asarray(obsbatch),jnp.asarray(keys)))[0];flow=o.bounded(action,o.config.max_speed);A,lower,_=o.barrier(env.snapshot(),o.cbf)
  safe,*_=o.project(flow,A,lower,o.config.max_speed,o.cbf);h,_=o.FeatureBuilder().build(o.FiniteHistoryView(env),{'u_flow':flow,'u_safe':safe},o.config,o.cbf)
  traces.append({'batch':batch,'h_sha256':hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest(),'feature_max_error':float(np.max(abs(h-features[s['dataset_index']]))),
   'action':action.tolist(),'flow':flow.tolist(),'safe':safe.tolist(),'h':h.tolist()})
 records.append({'state_id':sid,'canonical_h':s['feature_sha256'],'checks':traces,'action_difference_32_1':float(np.max(abs(np.array(traces[0]['action'])-traces[1]['action']))),
  'safe_difference_32_1':float(np.max(abs(np.array(traces[0]['safe'])-traces[1]['safe']))),'frozen_protocol_tolerance':1e-10})
dump('conditioning_replay_diagnosis.json',{'new_rollouts':0,'physical_steps':0,'records':records,'all_within_frozen_protocol_tolerance':all(t['feature_max_error']<=1e-10 for r in records for t in r['checks'])})
print(json.dumps({'states':len(records),'max_h_error':max(t['feature_max_error'] for r in records for t in r['checks']),'max_action_diff':max(r['action_difference_32_1'] for r in records),'new_rollouts':0}))
