#!/usr/bin/env python3
"""Frozen joint4A controller, exact current Flow fixed, matched future randomness."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import json, time, sys, argparse, hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import jax
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--shard',type=int,required=True);args=ap.parse_args()
 states={s['state_id']:s for s in json.load(open(HERE/'double_bottleneck_state_panel.json'))};tasks=[json.loads(x) for x in open(HERE/'plans'/args.plan/f'shard{args.shard}.jsonl')]
 for p,h in old.EXPECTED.items():assert old.sha(p)==h,(p,'integrity')
 dataset=old.FlowBC4ADataset(next(iter(states.values()))['dataset'],'all');policy,_=old.load_checkpoint(old.CHECKPOINT,dataset.environment_fingerprint)
 out=HERE/'db_raw';out.mkdir(exist_ok=True);dest=out/f'{args.plan}_{args.shard}.jsonl';known={}
 for p in out.glob('*.jsonl'):
  for line in open(p):
   r=json.loads(line);known[(r['state_id'],tuple(r['eta']),r['future_index'])]=r
 for t in tasks:
  if (t['state_id'],tuple(t['eta']),t['future_index']) in known:continue
  s=states[t['state_id']];episode=dataset.by_family[s['family_id']][0];ns=s['rng_namespace'];future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['future_root']),ns),t['future_index'])
  current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['initial_flow_root']),ns),0)
  class FixedCurrent:
   def __init__(self):self.step=0;self.h=None;self.conditioning=None
   def sample_actions(self,obs,unused):
    stepkey=current if self.step==0 else jax.random.fold_in(future,self.step)
    action=policy.sample_actions(obs,stepkey)
    if self.step==0:
     self.h=hashlib.sha256(np.asarray(obs,dtype=np.float64).tobytes()+np.asarray(action,dtype=np.float64).tobytes()).hexdigest()
     self.conditioning={'state_id':s['state_id'],'observation':np.asarray(obs,dtype=np.float64).tolist(),'current_flow_action':np.asarray(action,dtype=np.float64).tolist(),'h_sha256':self.h,
       'initial_positions':episode.initial_positions.tolist(),'initial_velocities':episode.initial_velocities.tolist(),'episode_path':str(episode.path),'source_group':s['source_group'],'config':dataset.config}
    self.step+=1;return action
  wrapper=FixedCurrent();job={'job_id':t['probe_id'],'stage':t['phase'],'representation':'P1-OrthoFlow3','theta':t['eta'],'seed':s['future_root'],'rollout_id':ns,'episode_id':s['state_id'],'family_id':s['family_id']}
  r=old.run_one(wrapper,dataset,episode,job,3.303687238760696)
  conditioning_path=out/f"conditioning_{s['state_id']}_{args.shard}.json"
  if conditioning_path.exists():assert json.load(open(conditioning_path))['h_sha256']==wrapper.h
  else:conditioning_path.write_text(json.dumps(wrapper.conditioning,indent=2)+'\n')
  r.update(t,h_conditioning_identifier=wrapper.h,source_group=s['source_group'],scenario='DoubleBottleneck_4A',future_root_seed=s['future_root'],current_root_seed=s['initial_flow_root'])
  with open(dest,'a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps({k:r.get(k) for k in ('state_id','future_index','outcome','wall_seconds','scientific_outcome_valid')}),flush=True)
  if not r['scientific_outcome_valid']:raise RuntimeError('Projection execution failure; do not interpret as non-B63')
if __name__=='__main__':main()
