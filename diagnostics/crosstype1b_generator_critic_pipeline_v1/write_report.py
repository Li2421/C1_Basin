"""Final prose is rendered from frozen evidence, never manually supplied metrics."""
from core import *
from deliver_tables import TYPE_NAMES


def fmt(value):return 'N/A' if value is None else f'{value:.2f}'
def contrast(value):return f"{value['gain_pp']:+.2f}pp [95% CI {value['CI95_pp'][0]:+.2f}, {value['CI95_pp'][1]:+.2f}]"


def main():
    integrity=read(HERE/'INTEGRITY_COMPARISON.json');dev=read(HERE/'DEVELOPMENT_RESULTS_V2B.json')
    final=read(HERE/'CONFIRMATORY_RESULTS.json');space=read(HERE/'SELECTABLE_SPACE.json')
    p=final['summaries']['combined']['type_equal']['methods']['pipeline'];end=final['endpoints']
    curve_examples=[];selected_models=read(HERE/'SELECTED_MODELS_V2B.json')['outer']
    for controller_type in ('Flow','ring_ippo'):
        component=selected_models[controller_type]['full']['full'];history=read(Path(component['checkpoint_path']).parent/'HISTORY.json')
        at=next(r for r in history if r['step']==component['step'])['validation'][controller_type]
        last=history[-1]['validation'][controller_type]
        curve_examples.append(f"{TYPE_NAMES[controller_type]}在源内选中的第{component['step']}步，完整分支的外层NLL为{at['nll']:.3f}、同state排序准确率{100*at['ranking_accuracy']:.2f}%；到第2000步NLL为{last['nll']:.3f}")
    head=space['summaries']['combined']['type_equal'];coverage=final['endpoint5_generator_coverage_and_choices']['combined']['type_equal']
    counts={phase:read(HERE/f'{phase}_PREFLIGHT.receipt.json') for phase in ('pilot','remaining','new_evaluation')}
    gaps=[dict(item='训练state完整候选集的seed交叉拟合',reason='原设计仅为生成器动作采集seed0；锚点的seed1不能替代生成器seed1。用户明确选择保持原预算，训练交叉拟合标为缺口。',
        remedy='在精确preflight之后追加最多19840条32个旧评估state的生成器seed1，再版本化开发分析；不能将本次确认标签用于重新开发'),
      dict(item='历史物理坐标去重的有限证据',reason='部分历史DB身份没有可恢复的物理坐标；已核查全部UID及可恢复物理hash，分类计数见NEW_STATE_DEDUP.json。',remedy='恢复原始物理坐标后才能宣称对全部历史坐标绝对去重'),
      dict(item='隐式能量行为克隆策略',reason='按prompt仍保留缺口，本轮不处理',remedy='后续独立能力门与合格controller'),
      dict(item='新评估state旧默认固定参数的灾难数',reason='全程旧默认参数未列入本轮十三候选；训练state已复用历史精确标签计算，新state不追加未要求的仿真。',remedy='需要同新state/seed的旧默认参数标签才能计算新评估灾难数'),
      dict(item='未见类型表述的预训练范围',reason='旧critic初始化来源此前已见过原生Flow；本轮外层排除仅保证当前训练/归一化/选参不使用留出类型。',remedy='同时报告从头训练版本；对初始化版本的Flow折保留此限制')]
    write(HERE/'GAPS.json',gaps)
    (HERE/'GAPS.md').write_text('# 缺口与结论限制\n\n'+'\n\n'.join(f"- {g['item']}：{g['reason']} 补齐条件：{g['remedy']}" for g in gaps)+'\n')
    audit='通过' if integrity['passed'] else '未通过（变化明细见 INTEGRITY_COMPARISON.json）'
    e1=end['endpoint1'];passed=lambda value:'通过' if value else '未通过'
    intro=(f"完整性审计{audit}。训练state上按state样本内事后最佳候选相对一律按进度切换时间表（Sprog）的类型等权空间为"
      f"{head['state_in_sample_oracle_vs_progress']['gain_pp']:.2f}pp，按controller样本内空间为{head['controller_in_sample_oracle_vs_progress']['gain_pp']:.2f}pp；"
      '训练完整菜单交叉拟合空间不可识别，不能把样本内数值当作交叉拟合证据。'
      f"训练生成器覆盖的不差比例为{100*head['generator_coverage']['empirical_nonworse_fraction']:.2f}%，严格更好比例为{100*head['generator_coverage']['empirical_strict_better_fraction']:.2f}%。"
      f"开发主提升CI未取得正下界，训练交叉拟合目标缺标签，开发目标未全部达到，按预算内当前源类型嵌套选择的版本冻结。新state终点1{passed(e1['passed'])}："
      f"真正的生成器与看state和controller的critic（pipeline）相对一律按进度切换为{e1['gain_pp']:+.2f}pp "
      f"[95% CI {e1['CI95_pp'][0]:+.2f}, {e1['CI95_pp'][1]:+.2f}]；终点2{passed(end['endpoint2']['passed'])}，"
      f"交叉拟合平均遗憾{p['state_crossfit_regret_pp']:.2f}pp，可能伤害controller数{p['harm_possible']}。"
      f"终点3相对不看controller的critic：{contrast(end['endpoint3']['no_controller'])}，相对不看state的critic：{contrast(end['endpoint3']['no_state'])}；"
      f"终点4相对原生成器与原critic：{contrast(end['endpoint4'])}。")
    body=[intro,'',*sum(([paragraph,''] for paragraph in read(HERE/'FINAL_INTERPRETATION.json')['paragraphs']),[]),
      '本轮修正参数由冻结生成器按每个state产生五个提议，分别构成整局固定和按进度切换两个动作；新的critic预测十三候选在当前state、controller下的成功概率，按三seed集成均值取最大。仅安全投影、按第500步切换时间表和按进度切换时间表作为三个锚点。生成器不看被测controller，controller、物理、安全投影、事件与成功标准均沿用旧执行链。',
      '',f"新state类型等权成功率为{p['success_lower_pct']:.2f}% [95% CI {p['success_CI95_pct'][0]:.2f}, {p['success_CI95_pct'][1]:.2f}]。相对只加安全投影的配对提升为{contrast(final['summaries']['combined']['type_equal']['paired_comparisons']['pipeline__minus__all_safety_baseline'])}。碰撞{p['collision_pct']:.2f}%，超时{p['timeout_pct']:.2f}%，unknown {p['unknown_pct']:.2f}%。timeout仅作为事件，不直接称为死锁。",'',
      '动物园包含九类型、62个合格controller。环形交换场景流匹配策略（Flow）使用FF位置，其余八类型使用TT位置；Ring、H700、x64=False。冻结生成器为seed41第1900步，原critic为seed23第2700步。16条历史执行记录首动作、全位置轨迹、终止与步数逐位一致；16个Ring验证state的生成器提议与原critic分数逐位一致，最大误差0。', '',
      '48个训练state由16个旧调参state与32个旧评估state组成。24个新评估state的物理生成种子在开工预注册，候选与输入提前冻结，闭环仿真只在模型冻结和选择收据之后提交。旧锚点标签、探测特征和controller checkpoint直接复用。全部请求先官方登记和preflight，只跑genuinely_missing；unknown保留且不重跑，通过默认journal和单进程官方merger入库。', '',
      '资源最初为全局18CPU、2GPU shard；后按用户最新授权，在全队列没有其他任务时提高到24CPU，GPU仍2shards。每次提交重新检查其他任务；条件不满足时提交上限回到18CPU。没有训练新controller，没有安装包，没有git写操作。历史数据与源码边界见INTEGRITY_START/END及PREIMPORT_SOURCES。', '',
      'critic采用100维state分支96、33维环境分支32、10维动作分支32、24维controller分支32，共享128/64输出logit。源类型内填补与标准化；state归一化只用48个训练state。二项似然排除unknown，实际观测seed数上限16。AdamW学习率1e-3、权重衰减1e-4、裁剪5，seed17/23/41。新critic使用float32最高矩阵乘法精度，x64仍False；这不改变仿真或controller的精度。', '',
      '推理显式恢复h/c归一化参数为float32、controller归一化参数为float64；27组留出类型与模式的checkpoint恢复预测逐位一致，最大误差0。排查过程及原失败日志保留在ITERATION_LOG与PREPROCESSING_DTYPE_DIAGNOSIS。', '',
      '开发共两个版本。初版为单个完整critic；第二版将完整分支与不看controller的分支按概率加权，保留controller的权重仅从0.1、0.25、0.5、0.75、1中按源内NLL选择。权重1包含初版；各外折使用源内最优权重，而没有在外层结果之间挑最好的版本。每个分支沿用三个对应seed，组合后仍按集成均值argmax执行。主方法同时看state、controller和候选动作。保守选择规则额外报告，不替换主终点。', '',
      '为匹配第二版的结构，去state对照由看controller和动作的分支、只看动作的分支组合；后者新增92组三seed拟合，仍全部复用已有标签。不看controller时两个分支按相同源内规则选择同一个不看controller模型，因此合并为该模型。两版全部配置、结果与收据保留；初版见DEVELOPMENT_RESULTS.json，最终匹配版本见DEVELOPMENT_RESULTS_V2B.json。', '',
      '九次外层留一类型中，初始化与checkpoint只按其余八类内部的留一类型集成负对数似然选择；缓存完全相同的七源类型训练避免重复计算。每100步保存到2000步，外层曲线不用于选参。同state候选排序仅对经验成功差至少0.25的可辨识候选对计算，预测平局记0.5。从头训练、旧critic初始化和去controller/去state消融均单独报告。去state分支保留候选动作特征；候选本身随state改变，因此不能把它严格解释为每个controller始终执行同一候选。', '',
      '留出曲线显示后期拟合存在泛化风险：'+'；'.join(curve_examples)+'。这些外层数值仅用于报告，checkpoint仍按源内规则选择，没有改选各外层NLL最低点。', '',
      '新state使用没有参与本轮该模型训练和选参的留出类型模型；全部九类型模型仅作已见类型、未见state参照。旧critic初始化此前在原生Flow训练过，这一预训练范围不因外层排除而消失；从头训练对照用于呈现此限制。', '',
      '每个新state所有十三候选和nominal only均执行seed0–1。类型等权为主，另报controller等权。5000次配对bootstrap依次重抽类型、controller、state，两seed保留同簇，oracle在每次抽样中重选；unknown同时给上下界，同UID候选的不确定性在配对差抵消。oracle先按unknown下界规则选取，其上下界条件于这一选取规则，不是对所有潜在unknown取值重新最优化的界。交叉拟合按相反seed选择与评分，另报样本内和按controller的oracle。', '',
      '按controller的事后最佳以跨state固定候选编号定义；编号对应生成器动作时，η仍由各state的冻结生成器产生。交叉拟合只用一个seed选臂，估计有噪声，其他方法可能高于其点估计，表内相应遗憾保留负值。样本内oracle另列，帮助判断有限seed选择的乐观程度。', '',
      f"新state生成器十动作覆盖的不差比例为{100*coverage['empirical_nonworse_fraction']:.2f}%，严格更好比例为{100*coverage['empirical_strict_better_fraction']:.2f}%；主方法选中生成器动作比例{100*coverage['selected_generated_fraction']:.2f}%，锚点比例{100*coverage['selected_anchor_fraction']:.2f}%。有{100*coverage['controllers_with_state_varying_choice_fraction']:.2f}%的controller在不同state选择不同动作。", '',
      f"同一controller随机取两个不同state，选中候选编号不同的比例为{100*coverage['state_pair_candidate_index_disagreement_fraction']:.2f}%；按实际完整精度η及切换规则比较，动作身份不同的比例为{100*coverage['state_pair_full_eta_schedule_disagreement_fraction']:.2f}%。同一个生成器候选编号在不同state可对应不同η，逐state精确值已列入CONTROLLER_STATE_DETAILS.csv。", '',
      '每类型及两种合并权重的结果见AXIS_TABLE.md/CSV；事件、平均/最大遗憾与伤害数见AXIS_METRICS.json。CONTROLLER_STATE_DETAILS.csv逐候选列出每个controller/state的选择、两seed标签、成功数、完整精度η和遗憾。figures/含36张静态SVG：每类型选中动作分布、完整分支留出曲线、源内权重选择和同state三条执行轨迹；轨迹示例按观察到的相对安全投影增益排序，仅用于描述，不改变控制与评估标准。图例、unknown说明与预览核查见FIGURES_README.md及DELIVERY_CHECKS.json。', '',
      'DB规模（请求数/精确缺失数）：'+ '；'.join(f"{phase} {r['planned']}/{r['genuinely_missing']}" for phase,r in counts.items())+'。', '',
      '训练交叉拟合、历史坐标去重、隐式能量策略和旧默认参数灾难数的限制见GAPS.md。完整结果与收据分别为DEVELOPMENT_RESULTS_V2B、CONFIRMATORY_RESULTS、FROZEN_PIPELINE及NEW_EVAL_CHOICES。']
    (HERE/'REPORT.md').write_text('\n'.join(body)+'\n')
    write(HERE/'results.json',dict(utc=utc(),integrity_passed=integrity['passed'],development=dev,confirmatory=final,
        selectable_space=space,gaps=gaps,requests=counts,report=str(HERE/'REPORT.md')))
    handoff=['本轮已完成开发、冻结和一次性新state评估。结论与完整性审计见REPORT.md，数据限制见GAPS.md。',
      '下一条跨type2线可只读复用：critic_model.py与action_only_model.py（结构/源类型预处理）、train_critic.py与train_action_only.py（源集合缓存与GPU三seed训练）、select_models.py/select_models_v2.py/select_matched_v2b.py（源内嵌套选参、正controller权重收缩、匹配消融）、new_evaluation_choices.py（冻结推理与归一化dtype恢复）、statistics_pipeline.py（配对分层bootstrap）、CANDIDATES.json与condition_candidates_v2.py（冻结候选规则）、FROZEN_PIPELINE.json（九外层与全类型模型路径/sha）及NEW_EVAL_CHOICES.receipt.json（选择时间收据）。',
      '历史执行链和数据均保留原位；不得直接运行有旧目录写副作用的构造器。需要改变实现时复制到新轮目录并记录来源；必须再次按新prompt做源保护、历史证据核查和DB preflight。',
      '本轮训练生成器标签仅seed0，因此训练完整候选交叉拟合不可识别。新评估24state全部十四动作两seed齐全。不能利用确认标签重新调本轮模型后继续称原确认有效。',
      '资源最初18CPU/2GPU shards；用户后授权没有其他任务时24CPU，GPU仍2shards。resource_guard.py在提交前检查全队列并按条件执行24/18CPU上限。旧critic初始化有原生Flow预训练范围，跨类型表述需保留这项限制。',
      '典型轨迹图为仿真证据的静态读取，不是另一次实验。新评估逐候选/逐seed UID和数据包在NEW_EVAL_BUNDLE.npz，raw/traces位于new_evaluation/；共享DB官方journal仍在默认目录。']
    (HERE/'HANDOFF.md').write_text('\n\n'.join(handoff)+'\n')
    seal('FINAL_DELIVERY.receipt.json',[HERE/'REPORT.md',HERE/'HANDOFF.md',HERE/'GAPS.md',HERE/'GAPS.json',HERE/'results.json',
        HERE/'write_report.py',HERE/'INTEGRITY_COMPARISON.receipt.json',HERE/'FIGURES.receipt.json',HERE/'AXIS_TABLE.receipt.json',HERE/'CONFIRMATORY_RESULTS.receipt.json',HERE/'USER_TRAINING_ORACLE_DECISION.receipt.json',
        HERE/'PILOT_REPORT.md',HERE/'PILOT_REPORT.receipt.json',HERE/'SELECTABLE_SPACE.receipt.json',HERE/'FROZEN_PIPELINE.receipt.json',HERE/'NEW_EVAL_CHOICES.receipt.json',HERE/'USER_CPU_CAPACITY_DECISION.receipt.json',
        HERE/'ITERATION_LOG.md',HERE/'DEVELOPMENT_REVIEW.json',HERE/'DEVELOPMENT_RESULTS_V2B.receipt.json',HERE/'FROZEN_INFERENCE_TESTS.receipt.json',HERE/'FINAL_INTERPRETATION.json',HERE/'DELIVERY_CHECKS.receipt.json'])
    print('Final report, results and handoff sealed',flush=True)


if __name__=='__main__':main()
