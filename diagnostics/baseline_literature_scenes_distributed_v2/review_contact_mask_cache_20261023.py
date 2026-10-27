"""Engineering-only candidate review: exact bit-key memo of the contact gate."""
from experiment import *
from atomic_write import write
import ast,difflib,importlib,time,collections
import numpy as np
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
source=HERE/'correction_revision_20261021.py';target=HERE/'correction_revision_mask_cache_candidate_20261023.py';before=source.read_text()
insert='''from functools import lru_cache
@lru_cache(maxsize=1)
def _contact_from_exact_bits(n,position_bits,goal_bits,tolerance_bits,radius_bits,force_all_open):
    p=np.frombuffer(position_bits,np.float64).reshape(n,2)
    g=np.frombuffer(goal_bits,np.float64).reshape(n,2)
    result=contact_mask(p,g,np.frombuffer(tolerance_bits,np.float64)[0],np.frombuffer(radius_bits,np.float64)[0],force_all_open)
    return result

def _cached_contact_mask(positions,goals,tolerance,radius,force_all_open=False):
    p=np.asarray(positions,np.float64);g=np.asarray(goals,np.float64)
    return _contact_from_exact_bits(len(p),p.tobytes(),g.tobytes(),np.float64(tolerance).tobytes(),np.float64(radius).tobytes(),force_all_open).copy()

'''
a='def distributed_fields(';assert a in before;after=before.replace(a,insert+a)
a='if not contact_mask(positions,goals,tolerance,radius,force_all_open)[i]:';b='if not _cached_contact_mask(positions,goals,tolerance,radius,force_all_open)[i]:';assert a in after;after=after.replace(a,b)
ast.parse(after);target.write_text(after)
(HERE/(target.name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=source.name,tofile=target.name)))
import correction_revision_20261021 as original
candidate=importlib.import_module(target.stem)
from states import LiteratureConfig
fixtures=[];source_rows=[]
# Deterministic hash order, saved evidence only; cover all configured N and scenes.
seen=set()
for scene in NAMES:
 stage='distributed_revised_fixed_'+scene
 for path in sorted((HERE/stage/'raw').glob('*.json')):
  raw=read(path);c=raw['case'];n=c['state']['N'];key=(scene,n)
  if key in seen:continue
  with np.load(raw['trace_path']) as z:
   if not len(z['safe']):continue
   cfg=LiteratureConfig(**c['scenario_payload']['config']);Rc=original.contact_radius(cfg)
   for t in sorted(set([0,len(z['safe'])//2,len(z['safe'])-1])):fixtures.append((z['positions'][t].copy(),z['goals'].copy(),z['safe'][t].copy(),cfg.max_speed,cfg.goal_tolerance,Rc))
  source_rows.append(dict(raw=str(path),sha256=sha(path),trace_sha256=raw['trace_sha256']));seen.add(key)
  if scene!='lit_crowded_random' or len([x for x in seen if x[0]==scene])==7:break
checks=0
for p,g,v,speed,tol,Rc in fixtures:
 for forced in (False,True):
  for i in range(len(p)):
   a=original.distributed_fields(p,g,v,i,speed,tol,Rc,forced);b=candidate.distributed_fields(p,g,v,i,speed,tol,Rc,forced)
   assert a.dtype==b.dtype and a.tobytes()==b.tobytes();checks+=1
# Warm native imports and compare paired order to reduce machine-load bias.
def bench(module):
 start=time.monotonic()
 for _ in range(30):
  for p,g,v,speed,tol,Rc in fixtures:
   for i in range(len(p)):module.distributed_fields(p,g,v,i,speed,tol,Rc)
 return time.monotonic()-start
bench(original);bench(candidate)
timings=[]
for order in [(original,candidate),(candidate,original),(original,candidate)]:
 values={m.__name__:bench(m) for m in order};timings.append(dict(original_seconds=values[original.__name__],candidate_seconds=values[candidate.__name__]))
old=sum(x['original_seconds'] for x in timings);new=sum(x['candidate_seconds'] for x in timings)
result=dict(utc=utc(),sources=source_rows,covered_scene_N=sorted(seen),fixtures=len(fixtures),bitwise_field_checks=checks,paired_timings=timings,helper_speedup=old/new,scope='Distributed field helper only; not a complete episode or pipeline speedup',pure_exact_bit_cache=True,semantics_unchanged_in_checked_fields=True,research_rollouts=0,new_training=0,deployed=False,decision='Candidate review only; do not alter sealed execution imports or identities without separate deployment proof')
write(HERE/'CONTACT_MASK_CACHE_CANDIDATE_REVIEW_20261023.json',result)
seal('CONTACT_MASK_CACHE_CANDIDATE_REVIEW_20261023.receipt.json',[Path(__file__),target,HERE/(target.name+'.diff'),HERE/'CONTACT_MASK_CACHE_CANDIDATE_REVIEW_20261023.json'],engineering_only=True,research_rollouts=0,deployed=False)
print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)
