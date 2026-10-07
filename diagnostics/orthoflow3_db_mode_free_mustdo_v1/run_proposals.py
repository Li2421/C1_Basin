#!/usr/bin/env python3
"""Frozen DB generator proposals Q16; writes append-only journals per batch."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import jax,numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old
from shared_rollout_db.src.cache_writer import append_journal
spec=importlib.util.spec_from_file_location('frozen_db_runner',ROOT/'diagnostics/orthoflow3_db_shared_mode_transfer_v1/run_db.py')
runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
HARD=ROOT/'diagnostics/orthoflow3_db_generator_necessity_v1'
EXPERIMENT='orthoflow3_db_mode_free_mustdo_v1'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);args=ap.parse_args()
    frozen=json.loads((HERE/'db_frozen_proposals.json').read_text())['states']
    states={s['state_id']:s for s in json.loads((HARD/'hard_state_manifest.json').read_text())['states']}
    assert len(frozen)==len(states)==48
    for p,h in old.EXPECTED.items():assert old.sha(p)==h,(p,'integrity')
    datasets={p:old.FlowBC4ADataset(p,'all') for p in sorted({s['dataset'] for s in states.values()})}
    policy,_=old.load_checkpoint(old.CHECKPOINT,next(iter(datasets.values())).environment_fingerprint)
    out=HERE/'raw';out.mkdir(exist_ok=True)
    dest=out/f'proposals_shard{args.shard}.jsonl'
    completed=set()
    for prior in out.glob('proposals_shard*.jsonl'):
        completed.update((r['state_id'],r['sample_index'],r['future_index'])
                         for r in map(json.loads,prior.read_text().splitlines()))
    pending=[];count=0;t0=time.monotonic()
    for i,row in enumerate(frozen):
      s=states[row['state_id']];dataset=datasets[s['dataset']];episode=dataset.by_family[s['family_id']][0]
      reference=json.loads((HARD/'raw'/f"conditioning_{s['state_id']}.json").read_text())
      for j,eta in enumerate(row['eta']):
       if (i*16+j)%args.shards!=args.shard:continue
       for fi in range(16):
        if (s['state_id'],j,fi) in completed:continue
        future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['future_root']),s['rng_namespace']),fi)
        current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['initial_flow_root']),s['rng_namespace']),0)
        class FixedCurrent:
         def __init__(self):self.step=0;self.h=None
         def sample_actions(self,obs,unused):
          stepkey=current if self.step==0 else jax.random.fold_in(future,self.step)
          action=policy.sample_actions(obs,stepkey)
          if self.step==0:
           o=np.asarray(obs,dtype=np.float64);a=np.asarray(action,dtype=np.float64)
           self.h=hashlib.sha256(o.tobytes()+a.tobytes()).hexdigest()
           if self.h!=reference['h_sha256']:raise RuntimeError(f"true-t0 conditioning mismatch {s['state_id']}")
          self.step+=1;return action
        wrapper=FixedCurrent()
        job={'job_id':f"{s['state_id']}_sample{j}_{fi}",'stage':'db_hard_mode_free_proposals_q16',
             'representation':'P1-OrthoFlow3','theta':eta,'seed':s['future_root'],
             'rollout_id':s['rng_namespace'],'episode_id':s['state_id'],'family_id':s['family_id']}
        r=runner.run_one_with_jdef(wrapper,dataset,episode,job,3.303687238760696)
        if not r['scientific_outcome_valid']:raise RuntimeError(f"numerical/projection failure {s['state_id']} sample{j} seed{fi}")
        r.update({'state_id':s['state_id'],'eta':eta,'future_index':fi,'sample_index':j,
                  'h_conditioning_identifier':wrapper.h,'source_group':s['source_group'],
                  'scenario':'DoubleBottleneck_4A','future_root_seed':s['future_root'],
                  'current_root_seed':s['initial_flow_root'],'controller':'generator_proposal'})
        with dest.open('a') as f:f.write(json.dumps(r,sort_keys=True)+'\n')
        pending.append(r);count+=1
        if len(pending)>=32:append_journal(pending,EXPERIMENT,f'proposal_shard{args.shard}');pending.clear()
        if count%128==0:print(json.dumps({'shard':args.shard,'completed':count,'seconds':round(time.monotonic()-t0,1)}),flush=True)
    if pending:append_journal(pending,EXPERIMENT,f'proposal_shard{args.shard}')
    print(json.dumps({'shard':args.shard,'completed':count,'seconds':round(time.monotonic()-t0,1)}),flush=True)
if __name__=='__main__':main()
