#!/usr/bin/env python3
"""Run only the already-frozen 64-seed targeted probes for one t0 state."""
from __future__ import annotations
import argparse,csv,importlib.util,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_t0_basin_shape_v1';T0=ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1';COMP=ROOT/'diagnostics/orthoflow3_t0_basin_completion_v1';MULTI=ROOT/'diagnostics/orthoflow3_t0_multiball_basin_learning_v1';RUNNER=T0/'run_t0_anchor.py';FUTURE=2026092811;BATCH=64
def write(p,rows,fields=None):
 fields=fields or (list(rows[0]) if rows else ['state_id'])
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def key(e):return np.asarray(e,dtype=np.float64).tobytes().hex()
def tasks(sid,e,n,**kw):return [{'state_id':sid,'eta':np.asarray(e,dtype=np.float64).tolist(),'future_index':i,**kw} for i in range(n)]
def summary(rows):return sum(bool(x['success']) for x in rows),len(rows)

def load_old(out,sid):
 spec=importlib.util.spec_from_file_location('shape_t0_runner',RUNNER);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m=m.load_patched()
 source=(m.OLD/'pilot6.py').read_text();a='for begin in range(0,len(tasks),32):';b='chunk=tasks[begin:begin+32]'
 if a not in source or b not in source:raise RuntimeError('batch patch anchor')
 source=source.replace(a,f'for begin in range(0,len(tasks),{BATCH}):').replace(b,f'chunk=tasks[begin:begin+{BATCH}]')
 ospec=importlib.util.spec_from_loader('shape_old_runtime',loader=None);old=importlib.util.module_from_spec(ospec);old.__file__=str(m.OLD/'pilot6.py');sys.modules[ospec.name]=old;exec(compile(source,old.__file__,'exec'),old.__dict__)
 old.HERE=out;old.FUTURE_ROOT=FUTURE;old.CAP_CONT=5000;old.CAP_STEPS=2500000
 class Oracle(old.Oracle):
  def _load_prior(self_inner):
   st=self_inner.states[sid]
   paths=[T0/'anchor_runs'/sid/'raw/pilot_rollouts.jsonl',COMP/'raw'/sid/'raw/pilot_rollouts.jsonl',MULTI/'stage_a'/sid/'raw/pilot_rollouts.jsonl']
   for p in paths:
    if not p.exists():continue
    for line in p.read_text().splitlines():
     if line.strip():
      r=json.loads(line)
      if r.get('state_id')==sid and r.get('h_conditioning_identifier')==st['h_conditioning_identifier'] and r.get('source_group')==st['source_group']:self_inner._insert(r,'exact_cached')
   if self_inner.record_path.exists():
    for line in self_inner.record_path.read_text().splitlines():
     if line.strip():
      r=json.loads(line);self_inner.rows.append(r);self_inner._insert(r,'pilot')
    self_inner.new=len(self_inner.rows);self_inner.steps=sum(int(r['continuation_steps']) for r in self_inner.rows)
 return old,Oracle

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--stage',choices=['initial','refinement'],required=True);a=ap.parse_args()
 states=json.load(open(HERE/'fixed8_manifest.json'))['states'];st=states[a.index];sid=st['state_id'];out=HERE/'state_runs'/sid;out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
 target_path=HERE/f'{a.stage}_targets_{sid}.json'
 targets=json.load(open(target_path)) if target_path.exists() else []
 old,Oracle=load_old(out,sid);features=np.load(T0/'synthetic_qdir/conditioning_features.npz')['features'];oracle=Oracle({sid:st},features,np.zeros(3),np.ones(3));start=time.time();result=[]
 for t in targets:
  eta=np.array([t['eta1'],t['eta2'],t['eta3']],float);rows=oracle.ensure(tasks(sid,eta,64,probe_id=t['probe_id'],probe_stage=a.stage,category=t.get('category',''),eta_key_float64=t['eta_key_float64']),f'{a.stage}_q64')
  suc,n=summary(rows);outs=Counter(x['outcome'] for x in rows)
  result.append({**t,'successes':suc,'trials':n,'Q64':suc/64,'B63':suc>=63,'deadlock':outs['safe_deadlock'],'timeout':outs['timeout'],'collision':outs['collision'],'numerical':outs['other_numerical'],'new_trials_used':sum(x['_source']=='pilot' for x in rows),'cached_trials_used':sum(x['_source']!='pilot' for x in rows)})
 write(out/f'{a.stage}_results_{sid}.csv',result)
 (out/f'{a.stage}_runtime.json').write_text(json.dumps({'state_id':sid,'stage':a.stage,'targets':len(targets),'new_continuations':oracle.new,'physical_steps':oracle.steps,'wall_seconds':time.time()-start,'outcomes':dict(Counter(r['outcome'] for r in oracle.rows))},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'state_id':sid,'stage':a.stage,'targets':len(targets),'new':oracle.new,'steps':oracle.steps,'wall':time.time()-start},indent=2),flush=True)
if __name__=='__main__':main()
