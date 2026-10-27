"""Read-only replay certification; no database labels or new rollouts are registered."""
from common import *
import numpy as np
from execution import execute
import accel_mode
ARRAYS=('positions','velocities','nominal','safe','actions','projection_rewrite','swept_clearance','clearance_types')
FIELDS=('success','collision','timeout','numerical_failure','outcome','episode_length','constraint_violation_steps','sampled_speed_violation_steps','restoration_steps','minimum_swept_clearance','final_goal_distances')
def check_record(task):
 out=HERE/'certification'/f"{task['rollout_uid']}.json"
 if out.exists():return
 raw=read(task['raw']);c=raw['case'];accel_mode.ENABLED=True;accel_mode.CHECK=True
 before=accel_mode.MATRIX_CHECKS;r,a=execute(c);bad=[]
 with np.load(raw['trace_path'],allow_pickle=False)as old:
  for k in ARRAYS:
   if k not in old:bad.append('missing_saved_array:'+k);continue
   aa=old[k];bb=a[k]
   if aa.shape!=bb.shape or aa.dtype!=bb.dtype or aa.tobytes()!=bb.tobytes():bad.append(k)
 for k in FIELDS:
  if canonical(r[k])!=canonical(raw[k]):bad.append(k)
 write(out,dict(rollout_uid=c['rollout_uid'],raw_path=task['raw'],raw_sha256=sha(task['raw']),trace_path=raw['trace_path'],trace_sha256=sha(raw['trace_path']),stratum=task['stratum'],passed=not bad,mismatches=bad,matrix_checks=accel_mode.MATRIX_CHECKS-before,original_elapsed_seconds=raw['elapsed_seconds'],accelerated_elapsed_seconds=r['elapsed_seconds']))
 if bad:raise RuntimeError(('ACCELERATION_REPLAY_MISMATCH',out,bad))

def prepare():
 from collections import defaultdict,Counter
 source=ROOT/'diagnostics/critic_pool_supplement_v1';tables=read_evidence(source/'LABEL_TABLES.json');scenes={s['name']:s for s in read_evidence(source/'SCENES.json')};groups=defaultdict(list)
 for name,t in tables.items():
  group=scenes[name]['group']
  if group not in ['A','B','C','E']:continue
  y=np.asarray(t['y']);u=np.asarray(t['unknown']);C,S,A=y.shape;ids=np.asarray(t['rollout_uids']).reshape(C,S,A)
  for ci,si,ai in np.ndindex(y.shape):
   result='unknown'if u[ci,si,ai]else'success'if y[ci,si,ai]else'failure';key=(group,t['controllers'][ci],result)
   uid_=str(ids[ci,si,ai]);groups[key].append(dict(rollout_uid=uid_,raw=str(source/'main/raw'/f'{uid_}.json'),stratum=key))
 rng=np.random.default_rng(2026102698)
 for rows in groups.values():rng.shuffle(rows)
 keys=sorted(groups);chosen=[];cursor=Counter()
 # Every available layer first receives30, then balanced round-robin to1000.
 for k in keys:
  chosen.extend(groups[k][:30]);cursor[k]=min(30,len(groups[k]))
 while len(chosen)<1000:
  for k in keys:
   if cursor[k]<len(groups[k]):chosen.append(groups[k][cursor[k]]);cursor[k]+=1
   if len(chosen)>=1000:break
 assert len(chosen)==1000 and all(Path(t['raw']).exists()for t in chosen)
 write(HERE/'certify_TASKS.json',chosen);write(HERE/'CERTIFICATION_PLAN.json',dict(total=1000,strata={str(k):dict(available=len(v),selected=cursor[k])for k,v in groups.items()},seed=2026102698,all_matrices_checked=True,minimum_required_matrix_checks=10000))

def units():
 import jax
 from np_threefry import fold_in
 rng=np.random.default_rng(2026102699);pairs=[]
 for _ in range(3000):
  key=rng.integers(0,2**32,size=2,dtype=np.uint32);d=int(rng.integers(0,2**32,dtype=np.uint64));ref=np.asarray(jax.random.fold_in(key,d));got=fold_in(key,d);assert ref.dtype==got.dtype and ref.tobytes()==got.tobytes()
 # Actual legacy key chain and controller folding on every step of maximum horizon.
 for statehash in ['a1b2c3d4','00000000','ffffffff']:
  key=np.asarray(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(statehash,16)),0))
  for t in range(600):
   ref=jax.random.fold_in(key,t);got=fold_in(key,t);assert np.asarray(ref).tobytes()==got.tobytes()
   for i in range(12):assert np.asarray(jax.random.fold_in(ref,i)).tobytes()==fold_in(got,i).tobytes()
 write(HERE/'RNG_CERTIFICATION.json',dict(passed=True,random_pairs=3000,chains=3,steps_per_chain=600,robots_per_step=12,root_seed=2026102004))

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='units':units()
