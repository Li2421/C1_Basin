from experiment import *
from atomic_write import write
from resources_corrections_release_20261023 import snapshot
from result_inventory import find_raw
import collections,time
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
start=time.monotonic();terms=[]
ctls={c['controller']:c for c in read(HERE/'CONTROLLER_MANIFEST.json')}
deterministic={'gradient','prioritized_astar','cbs'}
for layer in ('distributed','central'):
 for scene in NAMES:
  old=f'{layer}_fixed_{scene}';new=f'{layer}_revised_fixed_{scene}'
  cases=read(HERE/f'{old}_CASES.json');grouped=collections.defaultdict(list)
  for c in cases:grouped[(c['controller_payload']['nominal']['controller'],c['method'])].append(c)
  landed={p.stem for p in (HERE/new/'raw').glob('*.json')};rows=[];estimate=0.
  for (ctl,method),values in grouped.items():
   samples=values[::max(1,len(values)//8)][:8];timings=[]
   for c in samples:
    path=find_raw(c['rollout_uid'],old,scene);r=read(path);t=float(r.get('elapsed_seconds',0))
    if t>0 and not r.get('deterministic_source'):timings.append(t)
   if not timings:
    for c in values:
     r=read(find_raw(c['rollout_uid'],old,scene));t=float(r.get('elapsed_seconds',0))
     if t>0 and not r.get('deterministic_source'):timings.append(t);break
   mean=sum(timings)/len(timings) if timings else 0.
   # Completion counting uses the new case identities, where already planned.
   rows.append(dict(controller=ctl,method=method,old_mean_seconds=mean,sample_n=len(timings),logical=len(values)))
  newcases=read(HERE/f'{new}_CASES.json') if (HERE/f'{new}_CASES.json').exists() else None
  remaining=collections.Counter((c['controller_payload']['nominal']['controller'],c['method']) for c in newcases if c['rollout_uid'] not in landed) if newcases else collections.Counter({k:len(v) for k,v in grouped.items()})
  for row in rows:
   count=remaining[(row['controller'],row['method'])]
   # Deterministic seed1 is an exact result alias; count half of paired requests.
   nominal=ctls[row['controller']];typ=nominal['type_code'];isdet=typ in ('gradient','prioritized_astar','cbs') or row['controller'].startswith(('gradient','prioritized_astar','cbs'))
   physical=count/2 if isdet else count
   factor=.8 if row['controller'].startswith('cbs') else 1.
   row.update(remaining_logical=count,physical_equivalent=physical,cache_factor=factor,estimated_cpu_seconds=physical*row['old_mean_seconds']*factor)
   estimate+=row['estimated_cpu_seconds']
  terms.append(dict(scope=new,remaining_logical=sum(remaining.values()),historical_analogue_cpu_hours=estimate/3600,terms=rows))
  print(new,'cpu_hours',round(estimate/3600,2),flush=True)
# Learning train states and revised semantics are different: use old scene training costs, explicitly as analogue.
for scene in NAMES:
 stage='learning_labels_'+scene;byctl=collections.defaultdict(list)
 paths=sorted((HERE/stage/'raw').glob('*.json'))
 for path in paths[::max(1,len(paths)//1200)]:
  r=read(path);c=r['case'];t=float(r.get('elapsed_seconds',0))
  if t>0 and not r.get('deterministic_source'):byctl[c['controller_payload']['nominal']['controller']].append(t)
 stats=[sum(v)/len(v) for v in byctl.values()]
 # 3 fixed schedules exactly reuse the D fixed results only in narrow/door qualified training controllers.
 missing={'lit_narrow_passage':5632,'lit_crowded_random':7168,'lit_circle_five':7168+342,'lit_doorway':5824}[scene]
 mean=sum(stats)/len(stats) if stats else 60.
 terms.append(dict(scope='learning_labels_revised_20261021_'+scene,remaining_new_estimate=missing,historical_analogue_cpu_hours=missing*mean/3600,controller_means={k:sum(v)/len(v) for k,v in byctl.items()},not_representative_warning='Old circle landed labels are selected toward cheap controllers; partial means may underestimate CBS tails' if 'circle' in scene else None))
fixed=sum(t['historical_analogue_cpu_hours'] for t in terms if 'fixed' in t['scope'])
labels=sum(t['historical_analogue_cpu_hours'] for t in terms if 'labels' in t['scope'])
write(HERE/'RUNTIME_FORECAST_REVISION_EVIDENCE_20261023.json',dict(utc=utc(),analysis_seconds=time.monotonic()-start,resources=snapshot(),terms=terms,fixed_cpu_hours=fixed,labels_cpu_hours=labels,fixed_label_wall_analogue_hours_at23=(fixed+labels)/23,warning='Planning analogue only. Revised trajectories, hard controller tails, critic fits, selected evaluation and diagnosis/statistics alter wall time; optional dynamics have no production timings. No new rollout or training.'))
print('SUM',fixed,labels,(fixed+labels)/23,flush=True)
