"""Monitor landed revised fixed evidence; partial pooled counts are not final rates."""
from experiment import *
from atomic_write import write
import collections,csv,time,fcntl

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 with (HERE/'LIVE_REVISED_FIXED_SUCCESS.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(HERE/'LIVE_REVISED_FIXED_SUCCESS_PID.json',dict(utc=utc(),pid=os.getpid()))
  cache={}
  while not (HERE/'V2_COMPLETE.json').exists():
   groups=collections.defaultdict(collections.Counter);plans=collections.Counter();seen=0;unknowns=[]
   for layer in ('distributed','central'):
    for scene in NAMES:
     stage=f'{layer}_revised_fixed_{scene}';manifest=HERE/f'{stage}_CASES.json'
     if not manifest.exists():continue
     for c in read(manifest):
      ctl=c['controller_payload']['nominal'];plans[(scene,layer,ctl['type_code'],ctl['controller'],c['method'])]+=1
     for path in (HERE/stage/'raw').glob('*.json'):
      if path not in cache:
       raw=read(path);c=raw['case'];ctl=c['controller_payload']['nominal']
       cache[path]=((scene,layer,ctl['type_code'],ctl['controller'],c['method']),{k:int(raw[k]) for k in ('success','collision','timeout','numerical_failure')},raw.get('error'))
      key,outcomes,error=cache[path];groups[key].update(outcomes);groups[key]['landed']+=1;seen+=1
      if outcomes['numerical_failure'] and len(unknowns)<20:unknowns.append(dict(raw=str(path),error=error))
   rows=[]
   for key,planned in sorted(plans.items()):
    counts=groups[key];n=counts['landed'];rows.append(dict(scene=key[0],layer=key[1],type=key[2],controller=key[3],method=key[4],planned=planned,landed=n,remaining=planned-n,success=counts['success'],collision=counts['collision'],timeout=counts['timeout'],unknown=counts['numerical_failure'],partial_pooled_success_rate=counts['success']/n if n else None,complete=n==planned))
   write(HERE/'LIVE_REVISED_FIXED_SUCCESS.json',dict(utc=utc(),scope='Landed revised fixed requests only; includes deterministic aliases. Partial pooled rates are selected completed cases, not final type-equal or CI estimates.',landed=seen,unknown=sum(r['unknown'] for r in rows),unknown_examples=unknowns,controllers=rows))
   if rows:
    temp=HERE/'LIVE_REVISED_FIXED_CONTROLLER_SUCCESS.csv.tmp'
    with temp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(HERE/'LIVE_REVISED_FIXED_CONTROLLER_SUCCESS.csv')
   print(utc(),'landed',seen,'unknown',sum(r['unknown'] for r in rows),flush=True)
   if '--once' in sys.argv:return
   time.sleep(60)
if __name__=='__main__':main()
