#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,types
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1';FRESH=D/'orthoflow3_fresh_selector_vs_fixed_v1'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['local','test'],required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();plans=H/('local_plans' if a.phase=='local' else 'test_plans');tasks=[json.loads(x) for x in open(plans/f'shard{a.shard}.jsonl') if x.strip()];dest=H/('local_raw' if a.phase=='local' else 'test_raw');dest.mkdir(exist_ok=True)
 if not tasks:(dest/f'shard{a.shard}.jsonl').write_text('');(dest/f'shard{a.shard}_runtime.json').write_text(json.dumps({'tasks':0,'new_continuations':0,'physical_steps':0,'wall_seconds':0})+'\n');return
 if a.phase=='local':allst=json.load(open(D/'orthoflow3_shared_eta_codebook_v1/state_split.json'))['states'];features=np.load(H/'frozen_arrays.npz')['features']
 else:allst=json.load(open(FRESH/'fresh_state_manifest.json'))['states'];features=np.load(FRESH/'fresh_state_features.npz')['features']
 ids={t['state_id'] for t in tasks};states={q['state_id']:dict(q,feature_index=q['dataset_index']) for q in allst if q['state_id'] in ids};text=(POINT/'run_eval_shard.py').read_text().replace("range(0,len(tasks),64)","range(0,len(tasks),256)").replace("chunk=tasks[begin:begin+64]","chunk=tasks[begin:begin+256]");driver=types.ModuleType(f'deform_{a.phase}_driver');driver.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,driver.__file__,'exec'),driver.__dict__);out=H/(f'{a.phase}_runs')/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True);m,O=driver.loadmod(out,states);m.CAP_CONT=1000000;m.CAP_STEPS=500000000;o=O(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,f'deformable_{a.phase}')
 with (dest/f'shard{a.shard}.jsonl').open('w') as f:
  for t,r in zip(tasks,res):
   z={k:v for k,v in r.items() if k!='_source'};z.update({k:t[k] for k in ('controller','mode_id') if k in t});
   if 'candidate_index' in t:z['candidate_index']=t['candidate_index']
   f.write(json.dumps(z,sort_keys=True)+'\n')
 (dest/f'shard{a.shard}_runtime.json').write_text(json.dumps({'phase':a.phase,'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st},indent=2)+'\n');print(json.dumps({'phase':a.phase,'shard':a.shard,'tasks':len(tasks),'new':o.new,'steps':o.steps}))
if __name__=='__main__':main()

