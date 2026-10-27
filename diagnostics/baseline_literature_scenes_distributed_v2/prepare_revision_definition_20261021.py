"""Freeze requested Rc values and15-action menu before any revised research run."""
from experiment import *
from atomic_write import write
from states import config
from correction_revision_20261021 import safe_distance,contact_radius
import copy
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
rows=[]
for stage in ('single_integrator','paper_dynamics'):
 for scene in NAMES:
  c=config(scene);rows.append(dict(stage=stage,scene=scene,agent_radius=c.agent_radius,agent_collision_margin=c.agent_collision_margin,separation_buffer=.0001,inflated_radius=c.agent_radius+(.025 if stage=='paper_dynamics' and scene=='lit_doorway' else 0.),d_safe=safe_distance(c,stage),R_c=contact_radius(c,stage),R_c_hex=contact_radius(c,stage).hex(),goal_tolerance=c.goal_tolerance))
write(HERE/'CORRECTION_REVISION_PARAMETERS_20261021.json',dict(utc=utc(),rows=rows,R_c_multiplier=1.25,posthoc_user_decision=True,not_result_tuned=True))
menu=copy.deepcopy(read(HERE/'LEARNING_MENU.json'));fixed=read(HERE/'FIXED_ETA.json');assert len(menu)==12
menu.extend([dict(eta_a=fixed['10']['values'],eta_b=fixed['10']['values'],theta=None),dict(eta_a=fixed['13']['values'],eta_b=fixed['13']['values'],theta=None),dict(eta_a=fixed['13']['values'],eta_b=fixed['10']['values'],theta=.968)])
assert len(menu)==15 and menu[:12]==read(HERE/'LEARNING_MENU.json')
write(HERE/'LEARNING_MENU_REVISED.json',menu)
text='# 修订修正语义（2026-10-21，事后修订、开发证据）\n\n用户看过预注册结果后提出本修订。原语义与所有已封存结果保留，不能视为预注册检验。修订只将Brel按接触冲突开关置零，Bgoal/Bperp及相对向量逐对bound再平均的公式均不变。\n\n对每台机器人，存在邻居中心距≤Rc且二者没有同时到达时开关打开；已到达判据为当前位置距各自目标≤原goal_tolerance。Rc=1.25×安全层实际d_safe，不按结果调节。第二阶段门口用前视点膨胀半径；其他物理、安全QP、η、切换及6/7时限回退、跟踪和到达停止不变。\n\n| 阶段 | 场景 | 有效半径 | d_safe | Rc | 到达容差 |\n|---|---|---:|---:|---:|---:|\n'
for r in rows:text+='| '+ ' | '.join(map(str,(r['stage'],NAMES[r['scene']],r['inflated_radius'],r['d_safe'],r['R_c'],r['goal_tolerance'])))+' |\n'
text+='\n依据：配对行在滤波器平衡点起作用时，间距接近d_safe，接触开关覆盖这一机制；全部机器人已到达时相对分量为零，精确目标构型成为平衡点；目标附近徘徊终态开关关闭比例从已有保存证据计算并另行报告，不用于调Rc。集中式原OrthoFlow3三个分量由原_compute经compute取得，只将Brel对应行置零；分布式原local_basis全部邻居平均后只将Brel置零。\n\n单元门与静态检查将在CORRECTION_REVISION.receipt.json中封存；在该收据生成并验证之前禁止修订研究rollout。数值定义在本文件写出后不再修改，单元结果/静态证据写独立文件。\n'
assert not (HERE/'CORRECTION_REVISION.md').exists();(HERE/'CORRECTION_REVISION.md').write_text(text)
(HERE/'SAFETY_LAYERS_ADDENDUM_20261021.md').write_text('# 安全层通信假设补充\n\n本修订为看过预注册结果后的事后开发修订。每台机器人每步用自身当前位置和目标计算已到达标志，并广播一个是否已到达的比特；接触冲突只使用邻居位置、该标志和本机标志。它与原进度切换广播标量属于同一类通信假设。逐机器人投影仍不交换名义速度或安全速度，第一次/第二次投影的邻居、责任系数和约束完全不变。集中式保留原联合两次投影与原OrthoFlow3基，只开关相对分量。\n')
write(HERE/'CORRECTION_REVISION_HISTORY_20261021.json',dict(utc=utc(),reused_evidence=['COMPLETED_SCENE_NUMBERS.receipt.json','COMPLETED_DIAGNOSIS_NUMBERS.receipt.json','COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json','LOCAL_DIAGNOSIS_SCOPE.receipt.json','CBS_FLAT_CACHE_EQUIVALENCE.receipt.json'],compatibility='Nominal, physical states, projections and frozen eta unchanged. Eta0 execution expected exact and must pass32 saved replays. Nonzero revised Brel changes execution semantics and requires new controller identity/preflight; original results remain labeled preregistered.',missing='Revised nonzero fixed/menu executions and revised models/evaluation have not been run; instance taxonomy and wandering can be recomputed from old evidence',new_runs_authority='Explicit user corrections prompt, not implementation error retries',new_rollouts=0))
seal('CORRECTION_REVISION_DEFINITION_20261021.receipt.json',[HERE/n for n in ('prepare_revision_definition_20261021.py','CORRECTION_REVISION.md','SAFETY_LAYERS_ADDENDUM_20261021.md','CORRECTION_REVISION_PARAMETERS_20261021.json','LEARNING_MENU_REVISED.json','CORRECTION_REVISION_HISTORY_20261021.json')],before_any_revised_rollout=True,posthoc_development=True,fixed_Rc_multiplier=1.25)
print(rows)
