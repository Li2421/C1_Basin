#!/usr/bin/env python3
import importlib.util, argparse, json, time, hashlib, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
ROOT=HERE.parents[1];POINT=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
LEGACY=ROOT/'diagnostics/orthoflow3_analytic_basin_margin_learning_v1/run_rollout_shard.py'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
 spec=importlib.util.spec_from_file_location('frozen_wrapper',LEGACY);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 tasks=[json.loads(x) for x in open(HERE/'plans'/a.plan/f'shard{a.shard}.jsonl')]
 if not tasks:return
 states={s['state_id']:dict(s,feature_index=s['dataset_index']) for s in json.load(open(POINT/'final_source_split.json'))['states']}
 features=np.load(POINT/'point_learning_arrays.npz')['features'];out=HERE/'runs'/a.plan/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True)
 old,O=m.loadmod(out,states);old.CAP_CONT=float('inf');old.CAP_STEPS=float('inf')
 class Oracle(O):
  def _load_prior(self):
   super()._load_prior()
   for line in open(HERE/'partial_continuation_cache.jsonl'):
    r=json.loads(line);self._insert(r,'exact_partial_cache')
 o=Oracle(states,features,np.zeros(3),np.ones(3));start=time.time();result=o.ensure(tasks,a.plan)
 assert all(not r.get('execution_error') for r in result),'Invalid execution; do not label.'
 dest=HERE/'raw'/a.plan;dest.mkdir(parents=True,exist_ok=True)
 with open(dest/f'shard{a.shard}.jsonl','w') as f:
  for row in result:f.write(json.dumps({k:v for k,v in row.items() if k!='_source'})+'\n')
 rt={'plan':a.plan,'shard':a.shard,'tasks':len(tasks),'new_continuations':o.new,'reused_continuations':len(tasks)-o.new,'physical_steps':o.steps,'wall_seconds':time.time()-start}
 (dest/f'shard{a.shard}_runtime.json').write_text(json.dumps(rt,indent=2)+'\n');print(json.dumps(rt))
if __name__=='__main__':main()
