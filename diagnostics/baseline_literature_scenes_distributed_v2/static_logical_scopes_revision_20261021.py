"""Reaggregate static terminal audit by logical study scope, retaining exact-reuse arms."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from study_cases import qualified
import collections
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
verify_receipt('CORRECTION_REVISION_STATIC_WANDERING_20261021.receipt.json')
audit=read(HERE/'CORRECTION_REVISION_STATIC_WANDERING_20261021.json');hover={r['rollout_uid']:r for r in audit['hovers']};rows=[]
for layer in ('central','distributed'):
 for scene in NAMES:
  q=qualified(scene)
  for kind in ('baseline','fixed'):
   stage=f'{layer}_{kind}_{scene}';cases=read(HERE/f'{stage}_CASES.json');count=collections.defaultdict(collections.Counter)
   for c in cases:
    ctl=c['controller_payload']['nominal']['controller'];key=(c['method'],'qualified' if ctl in q else 'unqualified');v=count[key];v['episodes']+=1
    h=hover.get(c['rollout_uid'])
    if h:
     v['wandering_episodes']+=1;v['unarrived_robots']+=sum(h['unarrived']);v['contact_off_unarrived_robots']+=sum(a and not b for a,b in zip(h['unarrived'],h['contact_open']))
   for (method,qualification),v in count.items():rows.append(dict(stage=stage,scene=scene,layer=layer,method=method,qualification=qualification,**v,wandering_fraction=v['wandering_episodes']/v['episodes'],contact_off_fraction=v['contact_off_unarrived_robots']/v['unarrived_robots'] if v['unarrived_robots'] else None))
write(HERE/'CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021.json',dict(utc=utc(),rows=rows,scope='Exact original baseline/fixed logical case manifests; whitebox and capability background excluded, exact-reuse logical arms retained. All-UID static audit remains unchanged.',new_rollouts=0))
text='# 原主线范围的徘徊统计\n\n原全UID静态审计包含能力门、白盒背景与标签等全部保存数据；本表按原baseline/fixed精确逻辑清单重聚合，避免将白盒背景并入黑盒baseline。两个配对seed各自保留。原静态审计不改。\n\n| 场景 | 层 | 方法 | 合格性 | 局数 | 徘徊比例 | 开关关闭的未到达机器人比例 |\n|---|---|---|---|---:|---:|---:|\n'
for r in rows:
 rate=r['contact_off_fraction'];text+='| '+' | '.join(map(str,(NAMES[r['scene']],r['layer'],r['method'],r['qualification'],r['episodes'],f"{100*r['wandering_fraction']:.4f}%",'N/A' if rate is None else f'{100*rate:.4f}%')))+' |\n'
(HERE/'CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021.md').write_text(text)
seal('CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021.receipt.json',[HERE/n for n in ('static_logical_scopes_revision_20261021.py','CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021.json','CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021.md','CORRECTION_REVISION_STATIC_WANDERING_20261021.receipt.json')],new_rollouts=0)
unit=read(HERE/'CORRECTION_REVISION_UNIT_GATES_20261021.json');assert unit['passed'] and len(unit['replays'])==32 and all(r['passed'] for r in unit['replays'])
seal('CORRECTION_REVISION.receipt.json',[HERE/n for n in ('CORRECTION_REVISION.md','CORRECTION_REVISION_DEFINITION_20261021.receipt.json','CORRECTION_REVISION_UNIT_GATES_20261021.json','CORRECTION_REVISION_REPLAY_PLAN_20261021.json','CORRECTION_REVISION_ALGEBRA_TESTS_20261021.json','test_correction_revision_20261021.py','CORRECTION_REVISION_SOURCE_DIFF_20261021.json','correction_revision_20261021.py','runtime_correction_revised_20261021.py','central_correction_revised_20261021.py','dynamics_correction_revised_20261021.py','CORRECTION_REVISION_STATIC_WANDERING_20261021.receipt.json','CORRECTION_REVISION_STATIC_LOGICAL_SCOPES_20261021.receipt.json','INSTANCE_CLASSES_REVISED.receipt.json')],before_any_revised_research_rollout=True,all_six_requested_gates_passed=True,full_saved_nominal_actor_replays=32,eta0_existing_results_exact_reusable=True,posthoc_development=True)
print('CORRECTION_REVISION gate sealed; all6 requirements passed,32 full replays exact')
