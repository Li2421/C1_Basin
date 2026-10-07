#!/usr/bin/env python3
"""Exact DB Q64 runner for frozen Stage-A held-out eta predictions."""
import os
os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, json, sys, time
from pathlib import Path
import jax, numpy as np
HERE=Path(__file__).parent; ROOT=HERE.parents[1]; DB=HERE.parent/'orthoflow3_db_shared_mode_transfer_v1'; sys.path.insert(0,str(ROOT))
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old

def states(): return {s['state_id']:s for s in json.load(open(DB/'db_state_split.json'))['states']}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,required=True); a=ap.parse_args(); ss=states()
 tasks=[json.loads(x) for x in open(HERE/'plans'/'stage_a_q64'/f'shard{a.shard}.jsonl')]
 for p,h in old.EXPECTED.items(): assert old.sha(p)==h,(p,'integrity')
 paths=sorted({ss[t['state_id']]['dataset'] for t in tasks}); datasets={p:old.FlowBC4ADataset(p,'all') for p in paths}; first=datasets[paths[0]]
 policy,_=old.load_checkpoint(old.CHECKPOINT,first.environment_fingerprint)
 out=HERE/'raw'; out.mkdir(exist_ok=True); dest=out/f'stage_a_q64_{a.shard}.jsonl'; known={}
 for p in out.glob('stage_a_q64_*.jsonl'):
  for line in open(p):
   r=json.loads(line); known[(r['state_id'],r['family'],int(r['mode_id']),int(r['fold']),int(r['future_index']))]=1
 start=time.time(); new=steps=0
 for t in tasks:
  key=(t['state_id'],t['family'],int(t['mode_id']),int(t['fold']),int(t['future_index']))
  if key in known: continue
  s=ss[t['state_id']]; ds=datasets[s['dataset']]; ep=ds.by_family[s['family_id']][0]; ns=s['rng_namespace']
  future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['future_root']),ns),t['future_index']); current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['initial_flow_root']),ns),0)
  class Fixed:
   def __init__(self): self.step=0
   def sample_actions(self,obs,unused):
    k=current if self.step==0 else jax.random.fold_in(future,self.step); self.step+=1; return policy.sample_actions(obs,k)
  job={'job_id':t['probe_id'],'stage':t['phase'],'representation':'P1-OrthoFlow3','theta':t['eta'],'seed':s['future_root'],'rollout_id':ns,'episode_id':s['state_id'],'family_id':s['family_id']}
  r=old.run_one(Fixed(),ds,ep,job,3.303687238760696); r.update(t,source_group=s['source_group'],scenario='DoubleBottleneck_4A')
  with dest.open('a') as f: f.write(json.dumps(r)+'\n')
  known[key]=1; new+=1; steps+=int(r.get('episode_steps',0))
  if new%64==0: print(json.dumps({'shard':a.shard,'new':new,'last_state':t['state_id'],'family':t['family'],'mode':t['mode_id']}),flush=True)
  if not r['scientific_outcome_valid']: raise RuntimeError('invalid projection execution')
 (out/f'stage_a_q64_{a.shard}_runtime.json').write_text(json.dumps({'shard':a.shard,'new_continuations':new,'physical_steps':steps,'wall_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__': main()
