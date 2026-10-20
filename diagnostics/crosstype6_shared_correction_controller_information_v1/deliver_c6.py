"""Readable delivery after sealed numeric work and closing integrity audit."""
from common import *
from menu_c6 import METHODS
from catalog_names import full_type
import csv

def cell(r):
    if not r or 'point' not in r:return 'N/A'
    lo,hi=r['CI95'];return f"{r['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]"

def rate(r):
    lo,hi=r['CI95'];return f"{r['point']:.2f}% [{lo:.2f}, {hi:.2f}]"

def main():
    assert (HERE/'NUMERIC_RESULTS.receipt.json').exists();audit=read(HERE/'INTEGRITY_COMPARISON.json');assert audit['passed'],'Inspect protected differences before delivery'
    r=read(HERE/'T1_T7_RESULTS.json');allr=r['groups']['类型等权合并'];info=read(HERE/'INFORMATION_VALUE_NUMERIC.json');oracle=read(HERE/'TYPE_INFORMATION_ORACLE_RESULTS.json');training=read(HERE/'training_BUDGET.json');evalbudget=read(HERE/'evaluation_BUDGET.json');traincounts=read(HERE/'training_COMPLETENESS.json');evalcounts=read(HERE/'evaluation_COMPLETENESS.json');probe_failures=read(HERE/'PROBE_FAILURES.json');gaps=[]
    gaps.append(f'全部19类型均是DEVELOPMENT；本轮没有进行最终3个完全未接触类型的检验。')
    gaps.append(f'700步离线探测失败{len(probe_failures)}条；原因、实际步数和controller/state/臂完整保存在PROBE_FAILURES.json，失败指示进入39维特征，没有均值冒充。')
    gaps.append('700步探测关闭任务终止，进度差停滞判据未排除机器人到达目标后的进度平台；早/晚停滞是按预注册公式得到的诊断分组，不能直接解释成发生死锁。严格冻结与在动但不前进特征另有未到目标的距离条件。本轮不据新结果更改特征、分组或触发规则。')
    gaps.append(f'训练请求unknown {traincounts["unknown"]}、新评估请求unknown {evalcounts["unknown"]}；一律保留不重跑，成功主值按下界，JSON另给上界及配对未知区间。')
    membership=read(HERE/'GROUP_MEMBERSHIP.json');gaps.append(f'重查后期η未实际实现而无法定义距离的controller-state有{len(membership["eta_distance_unresolved"])}项；不虚构η，分组只用有定义的距离，完整缺口见GROUP_MEMBERSHIP.json。')
    gaps.append('训练state完整1313候选菜单选择只对现有标签评分；未覆盖动作不算失败、不用预测补标签。逐类型条件成功率与未评分比例见FULL_MENU_DEVELOPMENT_VALIDATION.json；53槽内验证与源内全局触发校准另列，不能冒充完整菜单总体成功率。')
    for name in ('plots_OPTIONAL_GAP.json','feature_diagnosis_INFRASTRUCTURE_GAP.json','feature_diagnosis_NEEDS_IMPLEMENTATION_FIX.json'):
        p=HERE/name
        if p.exists():gaps.append('交付缺口：'+name+'；原始错误保留，主线数值未因此中断。')
    (HERE/'GAPS.md').write_text('# 缺口与适用范围\n\n'+'\n\n'.join(gaps)+'\n\n类型信息经验参照仅限已评估且两donor池共同支持的完全相同物理动作，不是对所有可能动作的理论上界；每个bootstrap中重新拟合donor选择，始终排除留出controller的全部副本。稀疏支持和回退比例见TYPE_INFORMATION_ORACLE_RESULTS.json。旧人定进度标签的阈值0.9681620884830543保持原身份；本轮人定对照和阈值消融使用精确0.968，8条逐位门验证的是实际切换步相同的病例，不宣称两个阈值全局等价。\n')
    with (HERE/'AXIS_TABLE.csv').open() as f:rows=list(csv.DictReader(f))
    lines=['# 全部19类型的开发检验','','本轮主方法 = '+METHODS[1]+'。每格成功率 % [95% CI]，所有122个合格controller均在24个新state、两个seed上评估；其选择模型整类留出该类型。','','| 类型 / 分组 | '+' | '.join(METHODS)+' |','|---|'+'---|'*len(METHODS)]
    for row in rows:lines.append('| '+row['类型 / 分组']+' | '+' | '.join(row[m] for m in METHODS)+' |')
    lines+=['','碰撞率、超时率、unknown、触发比例及条件平均触发步：EVENT_AND_CHOICES.json；伤害数与各分组配对差：T1_T7_RESULTS.json；逐controller：CONTROLLER_RESULTS.csv。选中阈值0只表示整局单η的无作用阈值，另给五个有效阈值的分布。']
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(lines)+'\n')
    lines=['# controller信息价值（开发证据）','','第一层的开局选择只给state；执行中按当前进度切换后期η，启用的切换重查也只看当时state。第二层新增最近W步的进度历史差p(t)−p(t−W)，用于实际触发“修不修、何时开始修”。第三层再给700步无成败离线探测特征，用于critic评价η、阈值和触发设置。下表是相邻层配对成功率差，单位百分点及95% CI；第一层相对只加安全投影。']
    blocks=[['类型等权合并','学习型','优化型','搜索型'],['只加安全投影 ≥ 98%','90–98%','只加安全投影 < 90%'],['探测中早停滞','探测中晚停滞','探测中不停滞'],['选中η对距人定较近半组','选中η对距人定较远半组']]
    for columns in blocks:
        lines+=['','| 信息层（方法全称） | '+' | '.join(columns)+' |','|---|'+'---|'*len(columns)]
        for method in METHODS[:3]:lines.append('| '+method+' | '+' | '.join(cell(info['groups'][g]['layers'][method]) for g in columns)+' |')
    support=read(HERE/'TYPE_INFORMATION_SUPPORT.json')['groups']
    lines+=['','| 分组 | 完美类型信息经验上界参照：同类其他实例选 − 其他类型选 | 共同支持动作占比 | 无共同donor支持回退比例 |','|---|---|---|---|']
    for group,row in oracle['groups'].items():
        coverage=support[group].get('type_equal_common_supported_action_fraction');lines.append('| '+full_type(group)+' | '+cell(row)+' | '+(f'{coverage:.1%}' if coverage is not None else 'N/A')+' | '+(f"{row['unscored_fallback_fraction']:.3%}" if 'unscored_fallback_fraction' in row else 'N/A')+' |')
    interpretation_groups=['学习型','优化型','搜索型','只加安全投影 ≥ 98%','90–98%','只加安全投影 < 90%','探测中早停滞','探测中晚停滞','探测中不停滞','选中η对距人定较近半组','选中η对距人定较远半组']
    def supported_groups(comparison,positive=True):
        chosen=[]
        for group in interpretation_groups:
            v=r['groups'][group]['comparisons'].get(comparison)
            if v is not None and (v['CI95'][0]>0 if positive else v['CI95'][1]<0):chosen.append(group)
        return '、'.join(chosen) if chosen else '无'
    lines+=['','在线进度信息的配对95% CI下界大于0的预设分组：'+supported_groups('T2_online')+'；离线探测特征对应分组：'+supported_groups('T3_offline')+'。','离线探测特征的配对95% CI上界小于0的预设分组：'+supported_groups('T3_offline',False)+'。其余组按各自区间解释，不能把区间跨0直接写成等效；这些是未经多重比较校正的开发分组证据。']
    for group in ('类型等权合并','学习型','优化型','搜索型','只加安全投影 ≥ 98%','只加安全投影 < 90%'):
        rr=r['groups'][group]['comparisons'];online=rr.get('T2_online');offline=rr.get('T3_offline')
        if online and offline:lines+=['',f"{group}：在线进度增益{cell(online)}；离线探测特征增益{cell(offline)}。"]
    lines+=['','类型信息参照按同一物理state、完整浮点η、准确阈值、触发与重查语义匹配；不用跨state提议序号充当同一动作。两个donor池共同支持才参加选择，始终排除留出controller；5000次配对分层bootstrap中重新计算donor动作选择。源目录固定为19个已知开发类型，donor controller/state重抽与目标组耦合。稀疏支持的经验参照不能解释为任意动作全集上的理论上界。']
    (HERE/'INFORMATION_VALUE.md').write_text('\n'.join(lines)+'\n')
    development=read(HERE/'FULL_MENU_DEVELOPMENT_VALIDATION.json')['folds'];devlines=['# 训练state完整菜单留一类型验证','','模型、checkpoint、混合权重与全局触发设置先冻结，再从完整1313候选选择，只给已有真实标签评分。下列成功率仅为有评分state上的controller等权条件成功率，不是全体训练state的pipeline成功率。53槽内排名与源内校准见DEVELOPMENT_OUTER_FOLDS.json。','','| 类型 | '+METHODS[0]+' 条件成功率 / 未评分 | '+METHODS[1]+' 条件成功率 / 未评分 | '+METHODS[2]+' 条件成功率 / 未评分 |','|---|---|---|---|']
    for t,rr in development.items():
        cells=[]
        for m in METHODS[:3]:
            v=rr['methods'][m];point=v['success_on_scored_states_controller_equal'];cells.append(('N/A' if point is None else f'{point*100:.2f}%')+f" / {v['unscored_fraction']*100:.2f}%")
        devlines.append('| '+full_type(t)+' | '+' | '.join(cells)+' |')
    (HERE/'DEVELOPMENT_VALIDATION.md').write_text('\n'.join(devlines)+'\n')
    freeze=r['freeze_prerequisite'];failures={m:[t for t,v in r['T6'].items() if not v[m]['point_at_least90']] for m in (METHODS[1],METHODS[2])}
    report=[
      '完整性审计通过。既有输入、源码、文档、PoC与官方仓库的保护核查见INTEGRITY_COMPARISON.json；其他diagnostics的并行变化只按stat单列，禁止目录未读内容。代码和输出只写本轮目录，共享DB仅官方登记、默认journal和串行merger；未做Git提交。',
      '修正本身不依赖controller：'+METHODS[0]+f"成功率{allr['methods'][METHODS[0]]['point']:.2f}%，相对{METHODS[7]}{cell(allr['comparisons']['T1_vs_Safety'])}，相对{METHODS[6]}{cell(allr['comparisons']['T1_vs_human'])}。逐类型和会卡分组结果见AXIS_TABLE.md及T1_T7_RESULTS.json；本轮只作开发证据。",
      f"在线层新增最近W步的进度历史差，用于修不修、何时开始修；第一层执行中已有按当前进度切换η的反馈；{METHODS[1]}相对第一层{cell(allr['comparisons']['T2_online'])}，高安全投影组目标{'达到' if r['T2']['high_goal_met'] else '未达到'}、低安全投影组目标{'达到' if r['T2']['low_goal_met'] else '未达到'}。离线700步探测特征额外用于critic选择η对、阈值和触发设置，相对在线层{cell(allr['comparisons']['T3_offline'])}；何时有用的完整分组与类型信息经验上界见INFORMATION_VALUE.md。",
      '每个学习型类型是否都≥90%：'+('达到冻结前提。' if freeze[METHODS[1]] else '在线停滞触发层尚未达到冻结前提，未达类型：'+'、'.join(failures[METHODS[1]])+'。')+('离线探测特征层全部达到。' if freeze[METHODS[2]] else '离线探测特征层未达类型：'+'、'.join(failures[METHODS[2]])+'。')+'每项点估计、95% CI和逐项判断见下表；本轮不做3个最终未见类型检验。',
      f"阈值学出相对固定0.968：{cell(allr['comparisons']['T4_threshold'])}；按state选触发设置相对源内选全局设置：{cell(allr['comparisons']['T5_state_trigger'])}；相对跨类型5冻结参照：{cell(allr['comparisons']['T7_vs_C5'])}。",
      f"训练新增请求52704，精确缓存{training['exact_DB_reuse']}，本地实际轨迹等价复用{training['local_trace_equivalence']}，实际补跑{traincounts['actual_physical']}。原44动作槽全部保留，新增9个q0槽；编码扩展不使旧物理标签失效。新评估唯一物理请求{evalbudget['unique_physical_requests']}，实际执行{evalcounts['actual_physical']}，本地等价{evalcounts['local_equivalence']}；9条探测失败和所有数值unknown保留。",
      'η来源按跨类型5未舍入覆盖结果封存：冻结8提议有序对始终保留，时间表生成器CI下界未过0而不启用，切换重查下界+0.0020312500000123812刚过0而启用。跨类型5参照保留其全部原始提议池和冻结权重，不重训。16条旧物理记录、72条新执行/64项逐位检查、32条跨19类型提前结束复用全部通过；跨类型5冻结critic三类六分支预测也逐位一致。',
      '全部19类型已被研究者使用。每个新state选择用没训练过该整个类型的留一类型模型，这与完全未见类型的确认性最终检验不同。新24state与DB全部可恢复物理state、UID及内容哈希去重；旧数据库缺少可恢复坐标的记录只保留去重限制，不夸大全局物理不重叠。',
    ]
    report[2]+=' 离线探测特征配对95% CI下界大于0的预设分组：'+supported_groups('T3_offline')+'；上界小于0的分组：'+supported_groups('T3_offline',False)+'。这仅是开发分组证据。'
    report[7]+=' 未达90%类型的η来源、触发与阈值证据见FAILURE_DIAGNOSIS.json；它区分未选择修正、从未触发、晚触发、已评估学习选择全败而人定成功等观察，不把这些标签夸称为控制变量后的因果归因。'
    feature_file=HERE/'FEATURE_LOTO_LINEAR_DIAGNOSIS.json'
    if (HERE/'FEATURE_DIAGNOSIS.receipt.json').exists():
        feature=read(feature_file)['versions'];report.append(f"离线特征诊断：基于训练state的只加安全投影成功率，在源类型内嵌套选择ridge正则、外层整类留出的线性预测中，旧24维特征类型等权R²为{feature['old24']['type_equal_R2']:.4f}，新39维为{feature['new39']['type_equal_R2']:.4f}。这是探测特征的预测诊断，不等同于修正pipeline成功率增益；后者以T3为准。全部相关系数、预测、源内曲线与此前10个缺失controller的32次/实例重试明细见FEATURE_CORRELATIONS.csv、FEATURE_LOTO_LINEAR_DIAGNOSIS.json、PREVIOUS_MISSING_CONTROLLERS_REATTEMPT.json。")
    prose_length=len(report)
    report += ['| 学习型类型 | '+METHODS[1]+' | 是否≥90% | '+METHODS[2]+' | 是否≥90% |','|---|---|---|---|---|']
    for t,v in r['T6'].items():report.append('| '+t+' | '+rate(v[METHODS[1]])+' | '+('是' if v[METHODS[1]]['point_at_least90'] else '否')+' | '+rate(v[METHODS[2]])+' | '+('是' if v[METHODS[2]]['point_at_least90'] else '否')+' |')
    (HERE/'REPORT.md').write_text('\n\n'.join(report[:prose_length])+'\n\n'+'\n'.join(report[prose_length:])+'\n')
    handoff='''# 跨类型6交接

以REPORT.md、INFORMATION_VALUE.md、AXIS_TABLE.md、GAPS.md、INTEGRITY_COMPARISON.json开始。全部19类型是开发证据，19折模型确实整类留出外层类型；没有做最终3个完全未接触类型检验。

本轮主方法 = 学习双 η·进度阈值学出·在线停滞触发。第一层的生成器和critic不看controller；在线层只加本局进度，离线层再加39维700步无成败探测。η来源按ETA_SOURCE_DECISION.json冻结为原8提议+实际late切换处隔离进程重查，不重训时间表生成器。进度阈值为五点网格，三种触发动作前判定。

复用入口：schedule_c6.py、actions_c6.py、execution_c6.py、reuse_c6.py。controller/物理/安全投影/事件/成功标准沿用经16条逐位复现的执行链；旧given/requery执行文件保持原字节，重查CPU helper防止导入冻结网络修改controller进程XLA_FLAGS。实际controller进程必须unset XLA_FLAGS。旧人定标签阈值保留完整精度，新人定对照为精确.968。

TRAINING_BUNDLE.npz的53槽保留原44槽全部q0及Safety q1，其余旧/新增未观测q1是结构性缺失。新编码19维，离线特征39维见PROBE_FEATURE_DEFINITION.json；失败指示和此前10缺失controller重试情况见PROBE_FAILURES.json与PREVIOUS_MISSING_CONTROLLERS_REATTEMPT.json。旧标签不因编码改变重跑。

CRITIC_TRAINING_PLAN.json是382个源集合/分支三seed拟合，critic_fits/保存曲线、概率、checkpoint；19个外层及最终模型在FROZEN_PIPELINE.json，checkpoint、预处理、概率混合、全局触发全在源类型内部选。NEW_EVAL_CHOICES.receipt.json早于新闭环，NEW_EVAL_CHOSEN_ACTIONS.json和evaluation_ACTION_LINKS.json追踪8个固定方法的真实UID。

本地等价仅*_ALL_LOCAL_EQUIVALENCE.json和evaluation/dynamic_aliases/，来源必须为实际轨迹，不写共享DB，不能作为新一层推断源。新评估先执行所选Safety，再官方merge后据其实际轨迹判未触发等价，节约补跑。unknown不重跑；timeout不当作deadlock。

T1_T7_RESULTS.json、TYPE_INFORMATION_ORACLE_RESULTS.json、INFORMATION_VALUE_NUMERIC.json、AXIS_TABLE.csv由NUMERIC_RESULTS.receipt.json封存，绘图只用单独只读.venv-c1-plots，图不作为主线前置。类型信息参照使用同state完整物理动作匹配，并在每个bootstrap重新选donor动作，留出controller的全部副本排除；稀疏共同支持和回退须继续报告。

coordinator_c6.py是前台受监督的恢复入口，未完成相同案例/种子可因基础设施原因重提最多3次，保留已完成/unknown；完成原始记录与journal之间中断可在finish_phase_c6.py恢复journal而不重跑物理。资源按当前18 CPU/2 GPU全用户总上限，每提交查squeue。其他目录/源码/docs/PoC/venv只读；禁止前缀只stat；不Git提交。用户要求非关键缺包、绘图和排版失败记缺口而继续主线，跨轮持续有效。

下一轮是否允许冻结，应先核对REPORT.md的每学习型类型≥90%逐项结果；达到之前不把本轮19类说成最终未见类型检验。可直接复用本轮执行器、探测特征、训练标签与冻结模型，无需重新训练或rollout能回答的问题一律从已有预测/checkpoint重算。
'''
    (HERE/'HANDOFF.md').write_text(handoff)
    seal('DELIVERY.receipt.json',[HERE/n for n in ('REPORT.md','HANDOFF.md','GAPS.md','AXIS_TABLE.md','INFORMATION_VALUE.md','DEVELOPMENT_VALIDATION.md','FULL_MENU_DEVELOPMENT_VALIDATION.receipt.json','NUMERIC_RESULTS.receipt.json','INTEGRITY_COMPARISON.receipt.json','FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json','FOLD_AND_FREEZE_AUDIT.receipt.json','TRAINING_HUMAN_ETA_EXCLUSION.receipt.json','FRESH_CANDIDATE_HUMAN_ETA_EXCLUSION.receipt.json','ITERATION_LOG.md')]+([HERE/'FEATURE_DIAGNOSIS.receipt.json'] if (HERE/'FEATURE_DIAGNOSIS.receipt.json').exists() else []),all_development=True,integrity_passed=True)
    print('Cross-type6 delivered',flush=True)

if __name__=='__main__':main()
