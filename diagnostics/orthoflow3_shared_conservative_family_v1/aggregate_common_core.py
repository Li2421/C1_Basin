#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows,fields=None):
 with open(HERE/p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):(HERE/p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(v):return np.asarray(v,dtype='<f8').tobytes().hex()
def main():
 man=read(HERE/'fresh_common_core_manifest.csv');planned={(r['state_id'],r['eta_key']) for r in man};groups=defaultdict(dict);files=sorted((HERE/'raw/common_core').glob('shard*.jsonl'));assert len(files)==2
 for p in files:
  for line in open(p):
   r=json.loads(line);assert not r.get('execution_error');k=(r['state_id'],key(r['eta']));assert k in planned;fi=int(r['future_index']);assert fi not in groups[k] or groups[k][fi]['success']==r['success'];groups[k][fi]=r
 assert set(groups)==planned
 rows=[]
 for k,g in sorted(groups.items()):
  assert set(g)==set(range(64)),(k,len(g));rr=[g[i] for i in range(64)];su=sum(r['success'] for r in rr);m=next(r for r in man if (r['state_id'],r['eta_key'])==k)
  rows.append(dict(state_id=k[0],scenario='ToyGiveWay_2A',mode_id=m['mode_id'],eta_key=k[1],eta1=m['eta1'],eta2=m['eta2'],eta3=m['eta3'],successes=su,Q64=su/64,B63=su>=63,deadlock=sum(r['outcome'] in ('deadlock','safe_deadlock','strict_deadlock') for r in rr),timeout=sum(r['outcome']=='timeout' for r in rr),collision=sum(r['outcome']=='collision' for r in rr),physical_steps=sum(r['continuation_steps'] for r in rr)))
 write('fresh_common_core_q64.csv',rows)
 summary=[]
 for mode in sorted(set(r['mode_id'] for r in rows)):
  q=[r for r in rows if r['mode_id']==mode];toy=sum(r['B63'] for r in q);overall=toy+4
  summary.append(dict(mode_id=mode,Toy_B63=toy,Toy_states=8,DB_B63=4,DB_states=4,overall_B63=overall,overall_states=12,overall_fraction=overall/12,Toy_fraction=toy/8,DB_fraction=1.,shared_core_criterion=overall>=11 and toy>=7))
 write('common_core_summary.csv',summary)
 rt=[json.load(open(p)) for p in sorted((HERE/'raw/common_core').glob('shard*_runtime.json'))]
 dump('common_core_decision.json',dict(any_shared_common_core=any(r['shared_core_criterion'] for r in summary),criterion='>=11/12 overall,>=7/8 Toy,>=4/4 DB exact B63',modes=summary,new_exact_Q64=len(rows),new_continuations=sum(r['new_continuations'] for r in rt),reused_continuations=sum(r['reused_continuations'] for r in rt),physical_steps=sum(r['physical_steps'] for r in rt),worker_wall_seconds=sum(r['wall_seconds'] for r in rt),critical_wall_seconds=max(r['wall_seconds'] for r in rt),max_GPU_shards=2))
 print(json.dumps(dict(modes=summary)))
if __name__=='__main__':main()
