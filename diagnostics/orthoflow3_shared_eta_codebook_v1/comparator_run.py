#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1'
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();tasks=[json.loads(x) for x in open(H/f'comparator_plans6/shard{a.shard}.jsonl') if x.strip()];allst=json.load(open(H/'state_split.json'))['states'];ids={t['state_id'] for t in tasks};states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in allst if x['state_id'] in ids};features=np.load(H/'state_features.npz')['features'];spec=importlib.util.spec_from_file_location('point_driver',POINT/'run_eval_shard.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);out=H/f'comparator_runs6/shard{a.shard}';out.mkdir(parents=True,exist_ok=True)
 # Seed every new shard with the 128 valid records completed before the
 # 2->6 shard repartition. The driver's exact-key cache prevents reruns.
 cache=out/'raw/pilot_rollouts.jsonl'
 if not cache.exists():
  cache.parent.mkdir(parents=True,exist_ok=True);seen=set();rows=[]
  for old in sorted((H/'comparator_runs').glob('shard*/raw/pilot_rollouts.jsonl')):
   for line in open(old):
    r=json.loads(line);key=(r['state_id'],r['controller'],int(r['future_index']))
    if key not in seen:seen.add(key);rows.append(r)
  with cache.open('w') as f:
   for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 m,O=mod.loadmod(out,states);m.CAP_CONT=100000;m.CAP_STEPS=100000000;o=O(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,'codebook_comparators');dest=H/'comparator_raw';dest.mkdir(exist_ok=True)
 with open(dest/f'shard{a.shard}.jsonl','w') as f:
  for r in res:f.write(json.dumps({k:v for k,v in r.items() if k!='_source'},sort_keys=True)+'\n')
 dump(dest/f'shard{a.shard}_runtime.json',{'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st});print(json.dumps({'shard':a.shard,'tasks':len(tasks),'new':o.new}))
if __name__=='__main__':main()
