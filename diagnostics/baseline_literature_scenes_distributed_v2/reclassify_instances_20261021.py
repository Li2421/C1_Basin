"""User-revised instance taxonomy from saved baseline outcomes/diagnoses only."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import qualification
import numpy as np,collections,time

GRID=[(W,tau) for W in (2.5,5.,10.) for tau in (.02,.05,.1)]
LABELS=('安全层导致的全局失败','墙前陷阱实例','controller级失败')
def category(eligible,upper,counts):
 if not eligible:return '未通过能力门'
 if upper>.05:return '非全局失败实例'
 total=sum(counts.values())
 if total and counts['robot_filter_equilibrium_deadlock']>total/2:return LABELS[0]
 if total and counts['wall_filter_equilibrium']>total/2:return LABELS[1]
 return LABELS[2]

def small_diagnosis(data):
 cut=data.find(b',\n  "census":')
 return json.loads(data if cut<0 else data[:cut]+b'\n}')

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});os.nice(10)
 verify_receipt('CORRECTION_REVISION_DEFINITION_20261021.receipt.json');verify_receipt('COMPLETED_DIAGNOSIS_NUMBERS.receipt.json')
 started=time.monotonic();q={scene:{r['controller'] for r in qualification(scene) if r['qualified']} for scene in NAMES}
 groups={};total=0;source_path=HERE/'INSTANCE_CLASSES_REVISED_SOURCES_20261021.jsonl';assert not source_path.exists()
 with source_path.open('w') as sources:
  for layer in ('central','distributed'):
   for scene in NAMES:
    stage=f'{layer}_baseline_{scene}';analysis='analysis_'+stage
    verify_receipt(analysis+'/WORK_QUEUE.receipt.json')
    for entry in read(HERE/analysis/'GROUP_INDEX.json'):
     for row in read(entry['file']):
      path=Path(row['path']);data=path.read_bytes();actual=hashlib.sha256(data).hexdigest();assert actual==row['sha256'],path
      raw=json.loads(data);assert raw['rollout_uid']==row['rollout_uid'];c=raw['case'];ctl=c['controller_payload']['nominal'];name=ctl['controller'];n=c['state']['N'];tol=c['scenario_payload']['config']['goal_tolerance']
      dp=HERE/'DIAGNOSTICS'/f"{raw['rollout_uid']}.json";db=dp.read_bytes();diag=small_diagnosis(db)
      assert diag['trace_sha256']==raw['trace_sha256']
      assert diag['source']['sha256']==actual,('Diagnosis raw provenance mismatch',dp,path)
      key=(scene,layer,name)
      if key not in groups:groups[key]=dict(scene=scene,layer=layer,controller=name,type=ctl['type_code'],qualified=name in q[scene],episodes=0,successes=0,unknowns=0,counts={g:collections.Counter() for g in GRID},at_least_one_4a_prime=False)
      g=groups[key];g['episodes']+=1;g['successes']+=raw['success'];g['unknowns']+=raw['numerical_failure'];g['at_least_one_4a_prime']|='robot_filter_equilibrium_deadlock' in diag['local']
      if not raw['success']:
       if 'final_goal_distances' in raw:distance=np.asarray(raw['final_goal_distances'])
       else:
        with np.load(raw['trace_path']) as z:distance=np.linalg.norm(z['positions'][-1]-np.asarray(c['state']['goals']),axis=-1)
       missing=distance>tol;assert len(missing)==n
       grid={(x['window_seconds'],x['cone_tolerance']):x['local'] for x in diag['grid']}
       if not grid and raw['numerical_failure']:grid={x:['unknown']*n for x in GRID}
       assert set(grid)==set(GRID),(dp,set(grid))
       for point,cats in grid.items():
        assert len(cats)==n
        g['counts'][point].update(cat for cat,failed in zip(cats,missing) if failed)
      sources.write(canonical(dict(rollout_uid=raw['rollout_uid'],stage=stage,raw=str(path),raw_sha256=actual,diagnostic=str(dp),diagnostic_sha256=hashlib.sha256(db).hexdigest(),trace_sha256=raw['trace_sha256']))+'\n');total+=1
     write(HERE/'INSTANCE_CLASSES_REVISED_PROGRESS.json',dict(utc=utc(),logical_baselines_processed=total,stage=stage,new_rollouts=0))
 instances=[]
 for key,g in sorted(groups.items()):
  lo=g['successes']/g['episodes'];hi=(g['successes']+g['unknowns'])/g['episodes'];counts=g.pop('counts');primary=category(g['qualified'],hi,counts[(5.,.05)])
  sensitivity=[dict(window_seconds=W,cone_tolerance=tau,classification=category(g['qualified'],hi,counts[(W,tau)]),unarrived_agent_episode_categories=dict(counts[(W,tau)]),unarrived_agent_episodes=sum(counts[(W,tau)].values())) for W,tau in GRID]
  other=groups[(g['scene'],'distributed' if g['layer']=='central' else 'central',g['controller'])]
  instances.append(dict(**g,success_lower=lo,success_upper=hi,classification=primary,old_threshold_only_controller_failure=bool(g['qualified'] and hi<=.05),remove_in_revised_exclusion=bool(g['qualified'] and primary==LABELS[2]),robot_deadlock_evidence_eligible=bool(g['qualified'] and primary!=LABELS[2]),state_related_failure_definite=bool(g['qualified'] and lo>.05 and hi<.95),other_layer_success_lower=other['successes']/other['episodes'],other_layer_success_upper=(other['successes']+other['unknowns'])/other['episodes'],primary_unarrived_agent_episode_categories=dict(counts[(5.,.05)]),sensitivity=sensitivity,sensitivity_changes=any(x['classification']!=primary for x in sensitivity)))
 discrimination=[]
 for scene in NAMES:
  for layer in ('central','distributed'):
   rr=[r for r in instances if r['scene']==scene and r['layer']==layer and r['qualified']]
   discrimination.append(dict(scene=scene,layer=layer,qualified_instances=len(rr),controller_level_failure_instances=sum(r['classification']==LABELS[2] for r in rr),state_related_failure_instances=sum(r['state_related_failure_definite'] for r in rr),safety_layer_global_failure_instances=sum(r['classification']==LABELS[0] for r in rr),wall_trap_instances=sum(r['classification']==LABELS[1] for r in rr),instances_with_4a_prime=sum(r['at_least_one_4a_prime'] for r in rr),robot_deadlock_evidence_eligible_instances=sum(r['robot_deadlock_evidence_eligible'] and r['at_least_one_4a_prime'] for r in rr)))
 write(HERE/'INSTANCE_CLASSES_REVISED.json',dict(utc=utc(),evidence='Posthoc user-requested taxonomy revision; original sealed evidence preserved',primary=dict(window_seconds=5.,cone_tolerance=.05),definition='Qualified baseline success upper<=.05: majority unarrived robot-episodes4a-prime => safety-layer global failure; majority4a-double-prime => wall trap; otherwise controller failure. Only latter excluded.',instances=instances,discrimination=discrimination,sensitivity_changes=[r for r in instances if r['sensitivity_changes']],logical_baselines=total,new_rollouts=0,new_training=0,seconds=time.monotonic()-started))
 text='# 修订实例归类（原轨迹重算）\n\n原COMPLETED_*与原controller级口径保持不变。这里按用户修正指令，用失败局中未到达机器人·局的互斥分类多数重新判断；只从去除分析中去掉controller级失败，不把安全层导致的全局失败排除出死锁证据。主值W=5s、τ=.05；上界≤5%为候选。旁证列同实例另一安全层成功率。\n\n| 场景 | 层 | 合格数 | controller级失败 | state相关失败 | 安全层全局失败 | 墙前陷阱 | 有4a′且可作证据 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
 for r in discrimination:text+='| '+' | '.join(map(str,(NAMES[r['scene']],r['layer'],r['qualified_instances'],r['controller_level_failure_instances'],r['state_related_failure_instances'],r['safety_layer_global_failure_instances'],r['wall_trap_instances'],r['robot_deadlock_evidence_eligible_instances'])))+' |\n'
 text+='\n## 低成功上界实例及旁证\n\n| 场景 | 层 | controller | 修订类 | 本层成功上下界 | 另一层成功上下界 | 未到达机器人·局分类 |\n|---|---|---|---|---|---|---|\n'
 for r in instances:
  if r['old_threshold_only_controller_failure']:text+='| '+' | '.join(map(str,(NAMES[r['scene']],r['layer'],r['controller'],r['classification'],[r['success_lower'],r['success_upper']],[r['other_layer_success_lower'],r['other_layer_success_upper']],r['primary_unarrived_agent_episode_categories'])))+' |\n'
 text+='\n## 九格敏感性改变的实例\n\n'
 changed=[r for r in instances if r['sensitivity_changes']]
 if not changed:text+='无。\n'
 for r in changed:text+='- '+str((r['scene'],r['layer'],r['controller']))+'：'+str([(x['window_seconds'],x['cone_tolerance'],x['classification']) for x in r['sensitivity']])+'\n'
 text+='\n完整384个场景/层/实例、九格计数与排除标记见JSON。只读取已有raw与DIAGNOSTICS，没有新rollout或训练；timeout本身不作为死锁。\n'
 (HERE/'INSTANCE_CLASSES_REVISED.md').write_text(text)
 seal('INSTANCE_CLASSES_REVISED.receipt.json',[HERE/n for n in ('reclassify_instances_20261021.py','INSTANCE_CLASSES_REVISED.json','INSTANCE_CLASSES_REVISED.md','INSTANCE_CLASSES_REVISED_SOURCES_20261021.jsonl','COMPLETED_DIAGNOSIS_NUMBERS.receipt.json')],new_rollouts=0,new_training=0,primary_W=5.,primary_tau=.05)
 print(json.dumps(discrimination,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
