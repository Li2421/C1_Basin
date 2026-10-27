"""Delivery adapted only to the explicit post-freeze user instruction to stop D at107."""
from common import *
import numpy as np


def pct(x):return f'{100*x:.1f}%'


def main():
    stats=read(HERE/'STATISTICS.json');extrap=read(HERE/'EXTRAPOLATION_RESULTS.json');db=read(HERE/'DB_POSTFLIGHT.json')
    integrity=read(HERE/'INTEGRITY_COMPARE.json');assert db['passed']and integrity['passed']
    d=copy.deepcopy(stats['decisions']);d['D'].update(status='PARTIAL_USER_STOP',planned=120,completed=107,budget_censored=13);names={'PASSED':'通过','NOT_PASSED':'未通过'}
    first=f"甲：{names[d['A']['verdict']]}；乙：{names[d['B']['verdict']]}；丙：{names[d['C']['verdict']]}；丁：Gap1 N=10 外推107/120条完成，13条按用户要求停止，仅作不完整描述。\n\n"
    lines=[first,'结论范围仅为本修正族、本控制器集与本安全层下；不外推到未测拓扑或N=100。18个训练场景等权，场景内控制器类型等权，主标签数值unknown计入失败并另报上界；丁的计算截尾单列、不生成结果标签；无timeout→deadlock转换。\n\n']
    for code,title in [('A','留一场景：主模型−全局最佳单一修正'),('B','留一族测瓶颈：主模型−全局最佳单一修正'),('C','主模型相对交叉拟合最优的落后−低成本校准的落后')]:
        row=d[code];lines.append(f"{title}：{100*row['estimate']:+.2f}个百分点，95%区间[{100*row['ci'][0]:+.2f}, {100*row['ci'][1]:+.2f}]。\n\n")
    if d['C']['decision_rule']=='mean_difference_le_0.05':
        lines.append(f"丙按原文均值判定：平均额外落后不超过5个百分点；更严格的95%区间上界检验为{'通过'if d['C']['stricter_upper95_within5pp']else '未通过'}，单独报告而不替换原判定。\n\n")
    else:
        lines.append('丙按明确登记的区间规则判定：上述差值的95%区间上界不超过5个百分点；均值条件另报。\n\n')
    lines.append('未通过不等于已证明无效。区间为5000次场景→控制器类型→状态分层配对自助法，条件于冻结源模型选择；不包含重新采样完整训练场景后再拟合的全学习过程不确定性。丙规则来源与实现核对见C_CRITERION_CLARIFICATION.json和C_CRITERION_IMPLEMENTATION_CORRECTION.md。\n\n')
    lines.append('用户在已完成107条时明确要求停止余下慢请求。107条均保留；13条记为计算截尾，既不是数值失败，也不是timeout或deadlock。冻结方法、eta和甲乙丙均未改变；见USER_STOP_AMENDMENT.md与USER_EXTRAPOLATION_STOP_20261025.json。\n\n')
    lines.append('丁的已完成样本按计算耗时被选择，完成样本上的成功率可能有选择偏差。下表保留每种方法原计划2类控制器×12状态=24的类型等权分母，报告未完成项取0/1时的成功率界；这是识别界，不是完整样本成功率或95%置信区间。已完成碰撞数只描述已观测局。\n\n| 方法 | 已完成/计划 | 成功局数 | 原计划成功率下界—上界 | 已完成碰撞 | 停止条数 |\n|---|---:|---:|---:|---:|---:|\n')
    for name,row in extrap['methods'].items():
        lines.append(f"| {name} | {row['completed_n']}/{row['planned_n']} | {row['successes']} | {pct(row['success_lower_rate'])}—{pct(row['success_upper_rate'])} | {row['observed_collisions']} | {row['budget_censored_n']} |\n")
    lines.append('\n107条中25条成功、82条达到仿真时限、0条已观测碰撞、0条数值unknown。13条停止项只保留身份与保存点，不写rollout标签。两类控制器的完成分布、条件完成样本率、5000次固定方法/类型/状态重采样的端点敏感性区间见EXTRAPOLATION_RESULTS.json；这些区间不修复计算截尾偏差，也不替代原计划完整外推。\n\n')
    lines.append('\n逐场景方法表见METHOD_COMPARISON.md；全部所选eta、嵌套CV分数和unknown上界在LOSO_CHOICES/LOFO_CHOICES/STATISTICS.json。逐候选标签矩阵在LABEL_TABLES.json，五层证据见FIVE_LAYER.md与geometry/。\n\n')
    stages={}
    for stage in ('replay','capability','pilot','main','extrapolation'):
        row=read(HERE/f'{stage}_PREFLIGHT.receipt.json')
        if stage=='pilot'and(HERE/'PILOT_TAIL_COMPLETED.json').exists():
            original=Path(read(HERE/'PILOT_TAIL_COMPLETED.json')['original_queue_archive'])/'pilot_PREFLIGHT.receipt.json'
            row=read(original)
        stages[stage]=row
        if stage=='extrapolation':
            lines.append('extrapolation：原计划120请求，官方预检精确复用0、真正缺失120；107条完成后按用户要求停止其余13条，未完成部分未生成结果标签。\n\n')
        elif stage=='main':
            lines.append('main：计划60564请求；官方标准精确复用26481，另保留19条既有同UID数值unknown，总计复用26500，补跑34064。\n\n')
        else:
            lines.append(f"{stage}：计划{row['summary']['total_requested']}请求，官方预检后精确复用{row['reused_count']}、执行缺失请求{row['execution_count']}（能力门部分确定性种子1为已核验种子0轨迹的零步引用，不增加物理样本）。\n\n")
    if(HERE/'MAIN_PREFLIGHT_UNKNOWN_RETENTION.json').exists():
        lines.append('主标签官方preflight的标准可复用数为26481；另外19条为库中已有、同精确UID且未隔离的unknown。官方lookup_exact按numerical_failure=0过滤，故把这19条列入标准缺失；本轮按明确要求保留数值失败、不重跑，进入下界0/上界1，实际补跑34064而非34083。26500为26481条标准复用加19条unknown保留，不将其伪记为官方标准可复用记录。逐UID核对及未进入执行清单的证明见MAIN_PREFLIGHT_UNKNOWN_RETENTION.json。\n\n')
    outcome_rows=read(HERE/'MAIN_OUTCOME_ROWS.json')
    assert len(outcome_rows)==stages['main']['summary']['total_requested']
    unknown_uids={r['rollout_uid']for r in outcome_rows if r['unknown']}
    new_uids={p.stem for p in(HERE/'main/raw').glob('*.json')}
    new_unknown=unknown_uids&new_uids;reused_unknown=unknown_uids-new_uids
    assert len(reused_unknown)==read(HERE/'MAIN_PREFLIGHT_UNKNOWN_RETENTION.json')['retained_existing_exact_uid_unknowns']
    write(HERE/'MAIN_UNKNOWN_SUMMARY.json',dict(total_requests=len(outcome_rows),total_unknown=len(unknown_uids),
        reused_unknown=len(reused_unknown),new_unknown=len(new_unknown),
        reused_unknown_uids=sorted(reused_unknown),new_unknown_uids=sorted(new_unknown),
        numerical_failures_retained_without_retry=True,lower_label=0,upper_label=1))
    lines.append(f'主标签最终{len(outcome_rows)}条记录中，unknown共{len(unknown_uids)}条：既有{len(reused_unknown)}条、本轮新增{len(new_unknown)}条。全部保留且不重跑；每条主值按失败、敏感性上界按成功，分母不变。逐UID见MAIN_UNKNOWN_SUMMARY.json，具体错误与截至失败前的轨迹保留在原始记录中。\n\n')
    lines.append('旧Gap1本轮只复用地图和物理状态；新旧排序相关按用户修订记N/A（无共同精确eta），未补跑或近似配对。几何有效/单机器人有路不代表联合可解；正式deadlock解决证据未建立。\n\n')
    lines.append('重放10/10逐位通过；原场景8类×5场景×2key的80项接口兼容检查通过。历史26048个raw与轨迹事前哈希核验通过。数据库所有已完成请求身份、标签和轨迹哈希核对通过；13条停止请求的身份元数据已核对，未生成rollout标签，本轮无未合并journal；两个仓库按用户修订排除本轮新增status条目后的完整性核对通过。\n\n')
    lines.append('另对每个阶段清单的每次请求独立只读核对五类身份元数据，包含同UID的全部重复出现；不是仅检查每个UID的第一条。结果见METADATA_UID_CONTENT_AUDIT.json；登记事务的原有行逐表前后哈希同时保留。\n\n')
    lines.append('主标签声明的60564个场景×状态×控制器×候选格子逐项核对为完整且各出现一次，全部初始物理状态与所选控制器参数匹配声明。证明及清单哈希见MAIN_PLAN_COVERAGE_AUDIT.json；完整清单与覆盖证明在模型冻结时同时封存。\n\n')
    lines.append('核心导入依赖另有18文件清单（IMPORTED_SOURCE_AUDIT.json）。其中未在最初逐文件清单中的tracked依赖，以开工时已观测的Git提交0d102ab与空tracked diff还原起点字节哈希，和当前文件逐项一致；未将这一补充审计伪记为开工时的逐文件采集。\n\n')
    lines.append('工程过程保留：首次历史索引按物理执行身份处理跨批次raw位置；Slurm数组数字ID/显示ID不一致曾导致误释放运行中claim，已请求worker完成当前局后退出并修复，恢复前196条结果没有重复UID、没有unknown，已完成结果保留。完整事件见SCHEDULER_ARRAY_ID_REPAIR.json和SCHEDULER_EVENTS.jsonl。没有修改旧源码、环境、其他实验目录；控制器、安全层、物理步和成功判据未改。\n')
    prefix=read(HERE/'TASK_PREFIX_CORRECTION.json')
    lines.append(f"\n数量扫描生成器曾错误使用独立新种子；在主标签前改为固定UID排序N2源状态的episode_seed，N2→N4→N6的起点与目标前缀逐位一致。修正仅依据任务前缀约束，未按性能选择状态。原计划及{len(prefix['excluded_completed_requests'])}条已完成旧设计能力门证据均保留，该旧设计证据不进入修正后的能力门/训练/评估；见TASK_PREFIX_CORRECTION.json。\n")
    if(HERE/'PILOT_TAIL_COMPLETED.json').exists():
        lines.append('\npilot尾部将8个尚未启动的请求拆为独立任务，以避免排在两局长任务之后；没有重跑已完成请求。上方pilot计数取首次提交前收据，不能将尾部完成后的再次预检冒充首次零成本复用；详见PILOT_TAIL_REBALANCE.json及pilot_tail_history/。\n')
    if(HERE/'PILOT_STALL_RECOVERY.json').exists():
        lines.append('\npilot最后一条未完成请求触发30分钟无新结果规则，首次将Slurm时限从3小时缩至90分钟重提，后续仍遵循该规则。撤销尝试未产生科学标签；195条已完成结果先经官方merger入库，再由官方preflight确认只补缺失请求。全部撤销和重提事件见PILOT_STALL_RECOVERY.json及SCHEDULER_EVENTS.jsonl。用户再次确认保留这条请求，不作预算排除；见PILOT_REQUEST_RETAINED.json。\n')
    if(HERE/'CHECKPOINT_REPLAY_CHECK.json').exists():
        lines.append('\n为避免30分钟撤销时反复丢失同一长单局的进度，对该唯一精确UID启用经验证的操作性断点恢复。原runtime_new.py字节及登记的控制器身份保留；派生循环只插入完整环境/规划器/历史状态的保存恢复和计时累计，文本逆变换逐字恢复原循环。三个Gap数量场景在第37、83步分别终止进程并恢复，除墙钟计时外完整结果、动作、轨迹和安全证书逐位相同。断点不写入rollout表或充当科学结果，最终完整轨迹仍仅经官方journal/merger入库；补充源码、验证收据和断点哈希进入冻结及结束核对。详见CHECKPOINT_BUILD.json、CHECKPOINT_REPLAY_CHECK.json。\n')
    if(HERE/'CHECKPOINT_PLANNER_SCOPE_CHECK.json').exists():
        lines.append('\n之后对新场景的另一规划器CBS同样完成三个Gap场景的两次跨进程恢复逐位验证，并在全部新几何初始定义上核验两个规划器的序列化；据此将相同操作性包装用于新场景的PA/CBS。N=10外推仅核验初始状态与未运行actor的序列化，未调用多机器人action或读取外推结果；正式外推仍在MODEL_FROZEN后进行。见CHECKPOINT_PLANNER_SCOPE_CHECK.json。\n')
    if(HERE/'CHECKPOINT_STRICT_BITWISE_AUDIT.json').exists():
        lines.append('\n进一步复核已有PA/CBS六组跨进程验证产物，逐项比较dtype、shape和原始字节（包括正负零），除墙钟计时外6/6完全一致；没有新增仿真。原10局重放本身已采用同样的原始字节比较。证明及产物哈希见CHECKPOINT_STRICT_BITWISE_AUDIT.json，结束审计再次核对这些文件。\n')
    (HERE/'REPORT.md').write_text(''.join(lines))
    (HERE/'OLD_GAP1_COMPARISON.md').write_text('Gap1新旧候选排序相关：N/A。旧Gap1 v1/v2/v3的16点及v4全部128点与本轮文献线历史16点无共同精确float64 eta；遵照用户修订，不补跑、不近似配对。旧Gap1只使用地图配置与物理状态。\n')
    (HERE/'AXIS_TABLE.md').write_text(f'''| 泛化轴 | 检验 | 结果 | 范围 |
|---|---|---|---|
| 跨场景 | 18场景LOSO（甲） | {names[d['A']['verdict']]} | 文献6+对穿5+瓶颈7 |
| 跨场景 | 留一族测瓶颈（乙） | {names[d['B']['verdict']]} | 文献+对穿→瓶颈 |
| 跨场景 | 相对低成本校准（丙） | {names[d['C']['verdict']]} | 5个百分点非劣性界 |
| 跨因素·智能体数量外推 | Gap1 N10（丁） | 不完整描述：107/120完成、13停止，见REPORT | 同地图、半径、动力学；只评估冻结方法所选eta |
''')
    scenes=read(HERE/'SCENES.json');selected=read(HERE/'SELECTED_CONTROLLERS.json')
    layers=['| 场景 | 因素 | 几何有效 | 单机器人路径 | 联合可解 | 控制器类型通过数 | 主模型成功率（外推为截尾界） | 正式死锁解决 |\n|---|---|---|---|---|---:|---:|---|\n']
    for s in scenes:
        name=s['name'];nt=sum(v is not None for v in selected[name].values())
        if s['family']=='literature':valid='历史有效状态';paths='历史几何证据'
        else:
            g=read(HERE/'geometry'/f'{name}.json');valid=f"{sum(r['geometric_valid']for r in g['states'])}/{len(g['states'])}";paths=f"{sum(r['single_robot_paths']for r in g['states'])}/{len(g['states'])}"
        rate=stats['per_scene'][name]['ridge']['rate']if name in stats['per_scene']else extrap['methods']['ridge'].get('rate')
        rate_text=pct(rate)if rate is not None else 'N/A'
        if name not in stats['per_scene']:
            er=extrap['methods']['ridge'];rate_text=f"{pct(er['success_lower_rate'])}—{pct(er['success_upper_rate'])}（截尾界）"
        layers.append(f"| {name} | {s['factor']} | {valid} | {paths} | 未认证 | {nt} | {rate_text} | 未建立 |\n")
        detail=[f"# {name}：五层证据\n\n",f"族：{s['family']}；变化因素：{s['factor']}；状态数：{len(s['states'])}。\n\n",
            f"1. 几何有效：{valid}。机器人半径{s['config']['agent_radius']}m，限速{s['config']['max_speed']}m/s，dt={s['config']['dt']}s；时限{s['config']['max_steps']}步。没有缩小机器人或替换不合法状态。\n\n",
            f"2. 单机器人有路：{paths}。新场景逐状态路径见证见geometry/{name}.json；历史场景来源见HISTORY_CHECK.md。\n\n",
            "3. 联合可解：未认证。单机器人路径与能力门通过不构成联合可解证书。\n\n",
            f"4. 控制器成功：本场景选用{nt}类合格实例；场景内按类型等权。\n\n"]
        if name in stats['per_scene']:
            detail.append('| 方法 | 成功率 | 95%区间 |\n|---|---:|---:|\n')
            for method,row in stats['per_scene'][name].items():
                detail.append(f"| {method} | {pct(row['rate'])} | [{pct(row['ci'][0])}, {pct(row['ci'][1])}] |\n")
        else:
            detail.append('丁为用户停止后的不完整评估。保留每种方法24个原计划格子的类型等权分母；界为未完成项取0/1的敏感性界，未完成项没有科学标签。\n\n')
            detail.append('| 外推方法 | 已完成/计划 | 成功率下界—上界 | 已完成碰撞 | 停止条数 |\n|---|---:|---:|---:|---:|\n')
            for method,row in extrap['methods'].items():
                detail.append(f"| {method} | {row['completed_n']}/{row['planned_n']} | {pct(row['success_lower_rate'])}—{pct(row['success_upper_rate'])} | {row['observed_collisions']} | {row['budget_censored_n']} |\n")
        detail.append('\n5. 死锁解决：正式证据未建立。timeout、停滞和stalled_groups均未作为deadlock标签。碰撞与unknown保留在逐请求记录中。\n')
        destination=HERE/'five_layer'/f'{name}.md';destination.parent.mkdir(exist_ok=True);destination.write_text(''.join(detail))
    (HERE/'FIVE_LAYER.md').write_text(''.join(layers))
    (HERE/'GAPS.md').write_text('''# 实验完成后的适用范围与局限

用户修订已解决首次停止事项，历史记录保留在initial_stop/。用户最新修订后的工作已完成：甲乙丙使用完整主标签；丁107/120条完成、13条按用户要求停止，未完成原计划完整外推。判定与界见REPORT.md。

- 旧Gap1排序相关按修订为N/A，不是待补跑任务。
- 联合可解与正式死锁解决未认证；不把单机器人路径、排队、stalled_groups或timeout作为认证。
- 只在本修正族、本控制器集、本安全层与所测拓扑下报告；没有N100性能证据。
- 能力门失败的控制器按场景排除，详见CAPABILITY_TABLE；不存在跨场景强行补齐类型。
- unknown保留，主值按失败，固定策略另报上界；不重复数值失败。
- bootstrap区间条件于冻结的源拟合选择，不表示完整重新采样训练场景与重新拟合的总体不确定性。
- pilot只覆盖2状态×2候选，成本估计不是最坏运行时间保证。
- 丁107条按计算时间选择，完成样本可能有偏；13条计算截尾单列，不当失败、数值unknown或死锁，不删除原计划分母。
- 部分外推记账代码依据停跑修订在模型冻结后新增，属于事后描述性缺失处理；原75个封存文件、方法选择和甲乙丙统计不变。
''')
    if '\n## 完成核验\n'not in(HERE/'HISTORY_CHECK.md').read_text():
        with (HERE/'HISTORY_CHECK.md').open('a')as f:f.write('\n## 完成核验\n\n26048个历史标签raw/trace事前哈希核验通过，逐请求官方预检复用计数见各阶段PREFLIGHT.receipt.json；最终数据库与完整性检查通过，见DB_POSTFLIGHT.json、INTEGRITY_COMPARE.json。旧Gap1标签没有进入训练矩阵。丁按用户最新要求保留107条已完成结果，13条停止项仅核对身份与保存点、不生成rollout标签；见USER_STOP_AMENDMENT.md。\n')
    write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',scope='User-amended task; N10 is partial107/120 with13 user-stopped computations',extrapolation_planned=120,extrapolation_completed=107,extrapolation_budget_censored=13,decisions=d,reports={n:sha(HERE/n)for n in
        ('REPORT.md','AXIS_TABLE.md','METHOD_COMPARISON.md','HISTORY_CHECK.md','PILOT_COST.md','GAPS.md','FIVE_LAYER.md',
         'CAPABILITY_TABLE.md','OLD_GAP1_COMPARISON.md','DB_POSTFLIGHT.json','INTEGRITY_COMPARE.json',
         'EXTRAPOLATION_RESULTS.json','EXTRAPOLATION_CENSORED_CASES.json','CENSORED_CHECKPOINT_AUDIT.json',
         'USER_STOP_AMENDMENT.md','PARTIAL_FINALIZATION_SOURCES.json')}))
    print(first,flush=True)


if __name__=='__main__':main()
