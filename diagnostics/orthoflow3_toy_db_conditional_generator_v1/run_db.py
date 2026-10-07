#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np

H=Path(__file__).parent;D=H.parent;SRC=D/'orthoflow3_db_shared_mode_transfer_v1';ROOT=H.parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SRC))
import jax
import run_db as src
old=src.old

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
    tasks=[json.loads(x) for x in open(H/'plans'/a.phase/'db'/f'shard{a.shard}.jsonl') if x.strip()];dest=H/'raw'/a.phase;dest.mkdir(parents=True,exist_ok=True)
    if not tasks:
        (dest/f'db_shard{a.shard}.jsonl').write_text('');(dest/f'db_shard{a.shard}_runtime.json').write_text(json.dumps({'tasks':0,'new_continuations':0,'physical_steps':0,'wall_seconds':0})+'\n');return
    states={s['state_id']:s for s in json.load(open(SRC/'db_state_split.json'))['states']};paths=sorted({states[t['state_id']]['dataset'] for t in tasks});datasets={p:old.FlowBC4ADataset(p,'all') for p in paths};first=datasets[paths[0]];policy,_=old.load_checkpoint(old.CHECKPOINT,first.environment_fingerprint)
    for p,h in old.EXPECTED.items():assert old.sha(p)==h,(p,'integrity')
    started=time.time();new=0;steps=0;out=dest/f'db_shard{a.shard}.jsonl'
    with out.open('w') as sink:
      for t in tasks:
        s=states[t['state_id']];dataset=datasets[s['dataset']];episode=dataset.by_family[s['family_id']][0];ns=s['rng_namespace'];future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['future_root']),ns),t['future_index']);current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['initial_flow_root']),ns),0)
        class FixedCurrent:
          def __init__(self):self.step=0;self.h=None
          def sample_actions(self,obs,unused):
            stepkey=current if self.step==0 else jax.random.fold_in(future,self.step);action=policy.sample_actions(obs,stepkey)
            if self.step==0:self.h=hashlib.sha256(np.asarray(obs,dtype=np.float64).tobytes()+np.asarray(action,dtype=np.float64).tobytes()).hexdigest()
            self.step+=1;return action
        wrapper=FixedCurrent();job={'job_id':t['probe_id'],'stage':t['phase'],'representation':'P1-OrthoFlow3','theta':t['eta'],'seed':s['future_root'],'rollout_id':ns,'episode_id':s['state_id'],'family_id':s['family_id']}
        r=src.run_one_with_jdef(wrapper,dataset,episode,job,3.303687238760696);r.update(t,h_conditioning_identifier=wrapper.h,source_group=s['source_group'],scenario='DB',future_root_seed=s['future_root'],current_root_seed=s['initial_flow_root']);sink.write(json.dumps(r,sort_keys=True)+'\n');sink.flush();new+=1;steps+=int(r.get('episode_steps',0))
        if not r['scientific_outcome_valid']:raise RuntimeError('Projection execution failure')
    (dest/f'db_shard{a.shard}_runtime.json').write_text(json.dumps({'phase':a.phase,'scenario':'DB','tasks':len(tasks),'new_continuations':new,'physical_steps':steps,'wall_seconds':time.time()-started},indent=2)+'\n')
    print(json.dumps({'phase':a.phase,'scenario':'DB','shard':a.shard,'tasks':len(tasks),'new':new,'steps':steps}))
if __name__=='__main__':main()
