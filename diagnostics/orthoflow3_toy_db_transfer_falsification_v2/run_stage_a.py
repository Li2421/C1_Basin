#!/usr/bin/env python3
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time
from pathlib import Path
import jax
H=Path(__file__).parent;ROOT=H.parents[1];DB=H.parent/'orthoflow3_db_shared_mode_transfer_v1';sys.path.insert(0,str(ROOT))
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old
from shared_rollout_db.src.integration import journal_db
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();tasks=[json.loads(x) for x in open(H/'plans'/'stage_a'/f'shard{a.shard}.jsonl')];states={s['state_id']:s for s in json.load(open(DB/'db_state_split.json'))['states']}
 if not tasks:return
 for p,h in old.EXPECTED.items():assert old.sha(p)==h
 paths=sorted({states[t['state_id']]['dataset'] for t in tasks});ds={p:old.FlowBC4ADataset(p,'all') for p in paths};policy,_=old.load_checkpoint(old.CHECKPOINT,next(iter(ds.values())).environment_fingerprint)
 dest=H/'raw'/f'stage_a_{a.shard}.jsonl';known={}
 if dest.exists():
  for x in open(dest):r=json.loads(x);known[r['probe_id']]=r
 new=[];st=time.time();steps=0
 for t in tasks:
  if t['probe_id'] in known:continue
  s=states[t['state_id']];d=ds[s['dataset']];ep=d.by_family[s['family_id']][0];ns=s['rng_namespace'];future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['future_root']),ns),t['future_index']);current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['initial_flow_root']),ns),0)
  class Fixed:
   def __init__(self):self.step=0
   def sample_actions(self,obs,unused):k=current if self.step==0 else jax.random.fold_in(future,self.step);self.step+=1;return policy.sample_actions(obs,k)
  job={'job_id':t['probe_id'],'stage':t['phase'],'representation':'P1-OrthoFlow3','theta':t['eta'],'seed':s['future_root'],'rollout_id':ns,'episode_id':s['state_id'],'family_id':s['family_id']};r=old.run_one(Fixed(),d,ep,job,3.303687238760696);r.update(t,scenario='DoubleBottleneck_4A',source_group=s['source_group'],future_root_seed=s['future_root'],rng_namespace=ns)
  with dest.open('a') as f:f.write(json.dumps(r)+'\n')
  new.append(r);steps+=int(r.get('episode_steps',0))
  if len(new)%128==0:journal_db(new[-128:],H)
  if not r['scientific_outcome_valid']:raise RuntimeError('invalid')
 rem=len(new)%128
 if rem:journal_db(new[-rem:],H)
 (H/'raw'/f'stage_a_{a.shard}_runtime.json').write_text(json.dumps({'new_continuations':len(new),'physical_steps':steps,'wall_seconds':time.time()-st},indent=2)+'\n')
 print(json.dumps({'shard':a.shard,'new':len(new)}))
if __name__=='__main__':main()
