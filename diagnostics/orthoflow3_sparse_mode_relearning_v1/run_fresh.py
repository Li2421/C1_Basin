#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,types
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1';F=D/'orthoflow3_fresh_selector_vs_fixed_v1'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fresh','audit'],required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();plans=H/('fresh_plans' if a.phase=='fresh' else 'audit_plans');tasks=[json.loads(x) for x in open(plans/f'shard{a.shard}.jsonl') if x.strip()];dest=H/('fresh_raw' if a.phase=='fresh' else 'audit_raw');dest.mkdir(exist_ok=True)
 if not tasks:(dest/f'shard{a.shard}.jsonl').write_text('');(dest/f'shard{a.shard}_runtime.json').write_text(json.dumps({'tasks':0,'new_continuations':0,'physical_steps':0,'wall_seconds':0})+'\n');return
 allst=json.load(open(F/'fresh_state_manifest.json'))['states'];features=np.load(F/'fresh_state_features.npz')['features'];ids={t['state_id'] for t in tasks};states={q['state_id']:dict(q,feature_index=q['dataset_index']) for q in allst if q['state_id'] in ids};text=(POINT/'run_eval_shard.py').read_text().replace('range(0,len(tasks),64)','range(0,len(tasks),256)').replace('chunk=tasks[begin:begin+64]','chunk=tasks[begin:begin+256]');driver=types.ModuleType('sparse_fresh_driver');driver.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,driver.__file__,'exec'),driver.__dict__);out=H/(a.phase+'_runs')/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True);m,O=driver.loadmod(out,states);m.CAP_CONT=100000;m.CAP_STEPS=500000000;o=O(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,'sparse_'+a.phase)
 with (dest/f'shard{a.shard}.jsonl').open('w') as f:
  for t,r in zip(tasks,res):z={k:v for k,v in r.items() if k!='_source'};z['mode_id']=t['mode_id'];z['controller']=t['controller'];f.write(json.dumps(z,sort_keys=True)+'\n')
 (dest/f'shard{a.shard}_runtime.json').write_text(json.dumps({'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st},indent=2)+'\n');print(json.dumps({'phase':a.phase,'shard':a.shard,'tasks':len(tasks),'new':o.new,'steps':o.steps}))
if __name__=='__main__':main()
