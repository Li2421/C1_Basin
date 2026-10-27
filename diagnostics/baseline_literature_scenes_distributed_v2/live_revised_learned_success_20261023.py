"""Completed L3 point estimates on matched independent evaluation states only."""
from experiment import *
from atomic_write import write
from result_inventory import find_raw
from study_cases import qualified
from coordinator_correction_configured_20261023 import stage_done
import collections

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});old=read(HERE/'COMPLETED_SCENE_RESULTS.json')['results'];output=[]
 for scene in NAMES:
  stage='learning_eval_revised_20261021_'+scene
  if not stage_done(stage):continue
  q=qualified(scene);cases=[c for c in read(HERE/(stage+'_CASES.json')) if c['controller_payload']['nominal']['controller'] in q];keys={(c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index'],c['seed']) for c in cases}
  fixed=[c for c in read(HERE/f'distributed_revised_fixed_{scene}_CASES.json') if (c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index'],c['seed']) in keys]
  rows=[];cache={}
  for c in [*cases,*fixed]:
   rid=c['rollout_uid']
   if rid not in cache:
    paths=[HERE/stage/'raw'/(rid+'.json'),HERE/f'distributed_revised_fixed_{scene}'/'raw'/(rid+'.json'),HERE/f'distributed_baseline_{scene}'/'raw'/(rid+'.json')];path=next((x for x in paths if x.exists()),None) or find_raw(rid,scene=scene);cache[rid]=read(path)
   raw=cache[rid];ctl=c['controller_payload']['nominal'];rows.append(dict(method=c['method'],controller=ctl['controller'],type=ctl['type_code'],success=raw['success'],upper=min(1,raw['success']+raw['numerical_failure']),unknown=raw['numerical_failure'],collision=raw['collision']))
  summaries=[]
  for method in ('learned_scene','fixed_eta10','fixed_eta13','human_progress'):
   rr=[r for r in rows if r['method']==method];controllers=collections.defaultdict(list)
   for r in rr:controllers[(r['type'],r['controller'])].append(r)
   types=collections.defaultdict(list)
   for (t,ctl),values in controllers.items():types[t].append((sum(r['success'] for r in values)/len(values),sum(r['upper'] for r in values)/len(values)))
   lower=sum(sum(v[0] for v in vv)/len(vv) for vv in types.values())/len(types);upper=sum(sum(v[1] for v in vv)/len(vv) for vv in types.values())/len(types)
   summaries.append(dict(method=method,success_lower=lower,success_upper=upper,qualified_evaluation_requests=len(rr),unknown=sum(r['unknown'] for r in rr),collision=sum(r['collision'] for r in rr)))
  base=next((r for r in old[scene]['learned_evaluation'].get('statistics',[]) if (r['type'],r['layer'],r['method'])==('type_equal','distributed','safety')),None)
  output.append(dict(scene=scene,evaluation_state_keys=sorted({(c['state']['N'],c['state']['index']) for c in cases}),baseline_success_same_state_scope=None if base is None else base['success_rate'],summaries=summaries))
 result=dict(utc=utc(),scope='Completed revised distributed L3,qualified controllers,type-equal matched independent evaluation states;point estimates and unknown bounds,CI pending native final statistics',scenes=output)
 write(HERE/'LIVE_REVISED_LEARNED_POINT_ESTIMATES_20261023.json',result)
 for row in output:print(row['scene'],'baseline',row['baseline_success_same_state_scope'],[(x['method'],round(x['success_lower'],6),round(x['success_upper'],6),x['unknown']) for x in row['summaries']],flush=True)
if __name__=='__main__':main()
