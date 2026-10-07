#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time,types
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent
TOY=D/'orthoflow3_shared_eta_codebook_v1';POINT=D/'orthoflow3_true_t0_point_learning_v1';sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.integration import journal_toy

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
 tasks=[json.loads(x) for x in open(H/'plans'/f'shard{a.shard}.jsonl') if x.strip()]
 allst=json.load(open(H/'toy_state_split.json'))['states'];features=np.load(TOY/'state_features.npz')['features'];ids={t['state_id'] for t in tasks}
 states={q['state_id']:dict(q,feature_index=q['dataset_index']) for q in allst if q['state_id'] in ids}
 # Frozen authoritative true-t0 driver; batch enlargement changes throughput only.
 text=(POINT/'run_eval_shard.py').read_text().replace('range(0,len(tasks),64)','range(0,len(tasks),256)').replace('chunk=tasks[begin:begin+64]','chunk=tasks[begin:begin+256]')
 driver=types.ModuleType(f'structured_q_shard{a.shard}');driver.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,driver.__file__,'exec'),driver.__dict__)
 out=H/'runs'/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True);m,O=driver.loadmod(out,states);m.CAP_CONT=1000000;m.CAP_STEPS=500000000
 o=O(states,features,np.zeros(3),np.ones(3));started=time.time();res=o.ensure(tasks,'structured_continuous_q_v1')
 dest=H/'raw';dest.mkdir(exist_ok=True);rows=[]
 for t,r in zip(tasks,res):
  z={k:v for k,v in r.items() if k!='_source'}
  z.update({k:t[k] for k in ('matrix_partition','state_split','probe_id','eta_uid','eta_split','probe_type','target_trials','controller')})
  z.update(scenario='ToyGiveWay',representation='P1-OrthoFlow3',future_root_seed=2026092811)
  rows.append(z)
 with (dest/f'shard{a.shard}.jsonl').open('w') as f:
  for z in rows:f.write(json.dumps(z,sort_keys=True)+'\n')
 journal=journal_toy(rows,H,2026092811)
 runtime={'shard':a.shard,'tasks':len(tasks),'runner_new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-started,'journal':str(journal)}
 (dest/f'shard{a.shard}_runtime.json').write_text(json.dumps(runtime,indent=2)+'\n');print(json.dumps(runtime))
if __name__=='__main__':main()
