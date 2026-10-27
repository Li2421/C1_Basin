"""Controller-wise runtime analogue; separate physical executions from logical seed aliases."""
from experiment import *
from atomic_write import write
from resources_corrections_release_20261023 import snapshot
from study_cases import qualified
import collections,time
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
started=time.monotonic();manifest=read(HERE/'CONTROLLER_MANIFEST.json');ctlmap={c['controller']:c for c in manifest};det={'gradient_mpc','prioritized_astar','conflict_based_search','goal_reference'}
types=collections.defaultdict(list)
for c in manifest:types[c['type_code']].append(c)
training=[c for kind in sorted(types) for c in types[kind][:2]];assert len(training)==16
oldforecast=read(HERE/'RUNTIME_FORECAST_REVISION_EVIDENCE_20261023.json');fixedcost={(t['scope'].replace('_revised_','_'),r['controller'],r['method']):r['old_mean_seconds'] for t in oldforecast['terms'] if 'fixed' in t['scope'] for r in t['terms']}
newfixed={};labels={};zeros=collections.Counter();terms=[]
for scene in NAMES:
 for stage,kind in [('distributed_revised_fixed_'+scene,'newfixed'),('learning_labels_'+scene,'oldlabels')]:
  samples=collections.defaultdict(list)
  for path in (HERE/stage/'raw').glob('*.json'):
   raw=read(path);c=raw['case'];ctl=c['controller_payload']['nominal']['controller'];seconds=float(raw.get('elapsed_seconds',0));index=c.get('menu_index')
   if kind=='oldlabels' and (index==0 or c['method']=='safety'):zeros[(scene,ctl)]+=1;continue
   if seconds<=0 or raw.get('deterministic_source'):continue
   factor=.8 if ctlmap[ctl]['type_code']=='conflict_based_search' and not raw.get('pure_geometry_cache_equivalence_receipt_sha256') else 1.
   samples[ctl].append(seconds*factor)
  if kind=='newfixed':newfixed[scene]={k:sum(v)/len(v) for k,v in samples.items()}
  else:labels[scene]={k:dict(n=len(v),mean_seconds=sum(v)/len(v)) for k,v in samples.items()}
# Fixed requests: deterministic seed1 produces no new physical execution.
for layer in ('distributed','central'):
 for scene in NAMES:
  stage=f'{layer}_revised_fixed_{scene}';oldstage=f'{layer}_fixed_{scene}';path=HERE/f'{stage}_CASES.json';cases=read(path if path.exists() else HERE/f'{oldstage}_CASES.json');landed={p.stem for p in (HERE/stage/'raw').glob('*.json')};counts=collections.Counter()
  for c in cases:
   if path.exists() and c['rollout_uid'] in landed:continue
   ctl=c['controller_payload']['nominal']['controller']
   if ctlmap[ctl]['type_code'] in det and c['seed']==1:continue
   counts[(ctl,c['method'])]+=1
  cost=0.
  for (ctl,method),count in counts.items():
   mean=fixedcost[(oldstage,ctl,method)]
   if layer=='distributed' and ctl in newfixed[scene]:mean=newfixed[scene][ctl]
   elif ctlmap[ctl]['type_code']=='conflict_based_search':mean*=.8
   cost+=mean*count
  terms.append(dict(scope=stage,remaining_physical_executions=len([None for key,n in counts.items() for _ in range(n)]),analogue_cpu_hours=cost/3600))
for scene in NAMES:
 cost=0.;detail=[];q=qualified(scene)
 for ctlrow in training:
  ctl=ctlrow['controller'];source=labels[scene].get(ctl)
  if source:mean=source['mean_seconds'];origin='Same controller,old11-action labels; partial circle state sample explicitly uncertain'
  else:
   fallback=[fixedcost[('distributed_fixed_'+scene,ctl,m)] for m in ('fixed_eta10','fixed_eta13','human_progress') if ('distributed_fixed_'+scene,ctl,m) in fixedcost]
   assert fallback,(scene,ctl);mean=sum(fallback)/len(fallback);origin='Same controller,old fixed main states; label states differ'
   if ctlrow['type_code']=='conflict_based_search':mean*=.8
  old_actions=32*11
  fixed_actions=0 if scene in ('lit_narrow_passage','lit_doorway') and ctl in q else 32*3
  fixed_mean=newfixed[scene].get(ctl,mean)
  zero_missing=max(0,32-zeros[(scene,ctl)]) if scene=='lit_circle_five' else 0
  cpu=old_actions*mean+fixed_actions*fixed_mean+zero_missing*mean
  detail.append(dict(controller=ctl,old_action_missing=old_actions,new_fixed_action_missing=fixed_actions,eta0_missing=zero_missing,old_action_analogue_seconds=mean,new_fixed_analogue_seconds=fixed_mean,source=origin,estimated_cpu_hours=cpu/3600));cost+=cpu
 terms.append(dict(scope='learning_labels_revised_20261021_'+scene,analogue_cpu_hours=cost/3600,controllers=detail))
fits={}
for scene in NAMES:
 values=[]
 for path in (HERE/'critic_fits'/scene).glob('*/HISTORY.json'):
  hist=read(path)
  if hist:values.append(hist[-1]['seconds'])
 if values:fits[scene]=dict(completed_source_fits=len(values),sum_fit_seconds=sum(values),mean_fit_seconds=sum(values)/len(values))
fixed=sum(t['analogue_cpu_hours'] for t in terms if 'fixed' in t['scope']);label=sum(t['analogue_cpu_hours'] for t in terms if 'labels' in t['scope'])
out=dict(utc=utc(),analysis_seconds=time.monotonic()-started,resources=snapshot(),terms=terms,total_fixed_analogue_cpu_hours=fixed,total_labels_analogue_cpu_hours=label,compute_only_balanced_hours_at23=(fixed+label)/23,old_gpu_fit_timings=fits,not_a_wall_time_guarantee=True,uncertainties=['Circle training states16..47 differ from fixed0..15; old labels cover only6of16controllers and partialCBS1 states. Each of16now counted explicitly,never average only observed controllers.','Selected evaluation action and exact reuse unknown until model frozen; add evaluation/model/saved-diagnosis/statistics and serial phase tails.','Remaining whole-run optional dynamics have no revised production throughput.','No blanket transfer of distributed gate speedup to centralized trajectories.','Old forecast deterministic CBS seed1 count and partial-circle-controller averaging were overly conservative; superseded by explicit type codes and16controller cost terms.'],new_rollouts=0,new_training=0)
write(HERE/'RUNTIME_FORECAST_CONTROLLERWISE_20261023.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='terms'}),flush=True)
print('Stage cost CPU hours',[(t['scope'],round(t['analogue_cpu_hours'],2)) for t in terms],flush=True)
