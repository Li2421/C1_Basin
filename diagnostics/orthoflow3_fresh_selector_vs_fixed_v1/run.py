#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,types
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1'
def key(t):return (t['state_id'],tuple(float(x) for x in t['eta']),int(t['future_index']))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['primary','oracle'],required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();plan='plans' if a.phase=='primary' else 'oracle_plans';tasks=[json.loads(x) for x in open(H/plan/f'shard{a.shard}.jsonl') if x.strip()]
 if not tasks:
  dest=H/('raw' if a.phase=='primary' else 'oracle_raw');dest.mkdir(exist_ok=True);(dest/f'shard{a.shard}.jsonl').write_text('');(dest/f'shard{a.shard}_runtime.json').write_text(json.dumps({'tasks':0,'unique_tasks':0,'new_continuations':0,'physical_steps':0,'wall_seconds':0})+'\n');return
 allst=json.load(open(H/'fresh_state_manifest.json'))['states'];ids={t['state_id'] for t in tasks};states={q['state_id']:dict(q,feature_index=q['dataset_index']) for q in allst if q['state_id'] in ids};features=np.load(H/'fresh_state_features.npz')['features']
 text=(POINT/'run_eval_shard.py').read_text().replace("range(0,len(tasks),64)","range(0,len(tasks),256)").replace("chunk=tasks[begin:begin+64]","chunk=tasks[begin:begin+256]")
 driver=types.ModuleType('fresh_driver_256');driver.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,driver.__file__,'exec'),driver.__dict__)
 out=H/('runs' if a.phase=='primary' else 'oracle_runs')/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True);m,O=driver.loadmod(out,states);m.CAP_CONT=1000000;m.CAP_STEPS=500000000
 # Deduplicate identical eta executions (notably selector=fixed) while still
 # emitting one controller-labelled result per requested comparison.
 uniq={}
 for t in tasks:uniq.setdefault(key(t),t)
 u=list(uniq.values());o=O(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(u,f'fresh_{a.phase}');got={key(t):r for t,r in zip(u,res)};dest=H/('raw' if a.phase=='primary' else 'oracle_raw');dest.mkdir(exist_ok=True)
 with (dest/f'shard{a.shard}.jsonl').open('w') as f:
  for t in tasks:
   r={k:v for k,v in got[key(t)].items() if k!='_source'};r['controller']=t['controller'];r['eta']=t['eta'];
   if 'mode_id' in t:r['mode_id']=t['mode_id']
   f.write(json.dumps(r,sort_keys=True)+'\n')
 (dest/f'shard{a.shard}_runtime.json').write_text(json.dumps({'phase':a.phase,'tasks':len(tasks),'unique_tasks':len(u),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st},indent=2)+'\n');print(json.dumps({'phase':a.phase,'shard':a.shard,'tasks':len(tasks),'unique':len(u),'new':o.new,'steps':o.steps}))
if __name__=='__main__':main()
