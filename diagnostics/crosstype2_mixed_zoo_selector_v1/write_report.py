from core import *
from deliver_tables import TYPE_NAMES
from baselines import CLASSES
import csv

CLASS_NAMES={'no_correction':'不修正','parameter_correction':'修正参数','continuing_right_hand':'持续右手规则'}
def contrast(value):
    ci=value['CI95_pp'];return f"{value['gain_pp']:+.2f}pp [{ci[0]:+.2f}, {ci[1]:+.2f}]"
def metric(value):
    ci=value['success_CI95_pct'];return f"{value['success_lower_pct']:.2f}% [{ci[0]:.2f}, {ci[1]:.2f}]"
def passed(value):return '通过' if value else '未通过'


def main():
    audit=read(HERE/'INTEGRITY_COMPARISON.json');space=read(HERE/'SELECTABLE_SPACE.json');dev=read(HERE/'DEVELOPMENT_RESULTS.json');confirm=read(HERE/'CONFIRMATORY_RESULTS.json');end=confirm['endpoints'];results=confirm['results'];diagnosis=read(HERE/'TIMEOUT_DIAGNOSTICS.json')['groups'];gates=read(HERE/'LOCKED_CAPABILITY_UNBLINDED_AFTER_CONFIRMATION.json')['controllers']
    p=lambda group,key:results[group]['type_equal']['paired_comparisons']['pipeline__minus__'+key]
    locked=results['locked']['type_equal'];headroom={}
    for cohort in ('learned','nonlearned'):
        summary=space['results'][cohort]['type_equal'];headroom[cohort]=dict(state=summary['methods']['state_in_sample_oracle']['gain_vs_progress_pp'],controller=summary['methods']['controller_oracle']['gain_vs_progress_pp'],vs_fixed=summary['paired_comparisons']['state_in_sample_oracle__minus__source_fixed'])
    body=[f"完整性审计{passed(audit['passed'])}。训练状态中，学习型的按状态样本内可选空间相对一律按进度切换为{headroom['learned']['state']:.2f}pp，非学习型为{headroom['nonlearned']['state']:.2f}pp；开发目标{passed(dev['all_targets_met'])}。新状态上零样本修正流程是否胜过最强固定修法（主终点二）{passed(end['endpoint2']['passed'])}：锁定类型等权的pipeline相对三种固定方法中最好的一个为{contrast(end['endpoint2'])}。锁定类型pipeline成功率{metric(locked['methods']['pipeline'])}，相对只加安全投影为{contrast(p('locked','all_safety_baseline'))}。接近按状态交叉拟合事后最佳的检验（终点四）平均遗憾为{end['endpoint4']['regret_pp']:.2f}pp，可能伤害数{end['endpoint4']['harm_possible']}，{passed(end['endpoint4']['passed'])}。",'']
    endpoint1_summary='；'.join(f"{TYPE_NAMES[kind]} {contrast(value)}（{passed(value['passed'])}）" for kind,value in end['endpoint1'].items())
    endpoint5_summary='；'.join(f"{label}减不看控制器 {contrast(end['endpoint5'][group]['no_controller'])}、减不看状态 {contrast(end['endpoint5'][group]['no_state'])}" for group,label in [('locked','锁定类型'),('new_development','新增开发类型')])
    body[0]+=f" 各锁定类型相对只加安全投影（终点一）：{endpoint1_summary}。新增开发类型相对一律按进度切换、源类型多数票固定候选（终点三）分别为{contrast(end['endpoint3']['all_Sprog'])}、{contrast(end['endpoint3']['source_fixed'])}。控制器与状态信息消融（终点五）：{endpoint5_summary}。"
    if not audit['passed']:
        body[0]=body[0].replace('完整性审计未通过。','完整性审计未通过（唯一差异是运行期间外部新增未跟踪提示词docs/prompt_crosstype1c_selected_two_eta_schedule_20261018.md；8491个受保护源码与全部核查输入未变，两仓库HEAD及已有差异未变）。',1)
        body.append('原始完整性差异详见INTEGRITY_COMPARISON.json，逐项核查处置见INTEGRITY_DISPOSITION.json；保留原始未通过状态，不将其改写为完全无差异。该外部新增文档未被本轮读取或用于实验。\n')
    both=end['endpoint2']['passed'] and all(v['CI95_pp'][0]>0 for v in end['endpoint3'].values())
    body.append('主结论：'+('本轮在锁定新类型与留出的新增开发类型上都建立了相对固定修法的正配对置信区间。' if both else '本轮没有同时建立锁定新类型与留出的新增开发类型均胜过固定修法；不能宣称跨类型零样本选择的全部主张已经证明。'))
    body.extend(['','锁定类型上修正相对只加安全投影是否有效（终点一），按类型分别判断：'])
    protocol=read(HERE/'new_evaluation_protocol.json')
    no_repair_checks=read(HERE/'LOCKED_NO_REPAIR_CHECK.json')['types']
    for kind,value in end['endpoint1'].items():
        names=[n for n in gates if protocol['types'][n]==kind];unneeded=sum(gates[n]['no_repair_needed'] for n in names)
        check=no_repair_checks[kind];fraction=check['no_repair_subset_selected_safety_fraction'];selection=f'{fraction*100:.1f}%' if fraction is not None else 'N/A（没有该子集）'
        body.append(f"- {TYPE_NAMES[kind]}（{kind}）：pipeline减安全基线{contrast(value)}，{passed(value['passed'])}；先导门中{unneeded}/{len(names)}个实例只加安全投影已≥98%，全部保留。这些实例在新状态上选择不修正的比例为{selection}，可能伤害数{check['no_repair_subset_harm_possible']}、确定伤害数{check['no_repair_subset_harm_certain']}；逐实例见LOCKED_NO_REPAIR_CHECK.json。")
    body.extend(['','零样本修正流程是否胜过最强固定修法，采用锁定类型等权合并；每次bootstrap都重新取三个固定方法的最大值，并非事先按新结果挑一个固定方法后固定比较。三个固定方法为一律按进度切换、一律固定参数持续右手规则、源类型多数票固定候选。相对各固定方法分别为：'])
    for key,label in [('all_Sprog','一律按进度切换'),('all_RHR','固定参数持续右手规则'),('source_fixed','源类型多数票固定候选'),('source_mean_fixed','源类型等权平均最优固定候选')]:body.append(f"- {label}：{contrast(p('locked',key))}。")
    body.append('锁定类型各自的同口径固定修法比较为：')
    for kind in confirm['locked_types']:
        comparisons='；'.join(f"{label} {contrast(p(kind,key))}" for key,label in [('best_fixed','减三种固定方法中最好者'),('all_Sprog','减一律按进度切换'),('all_RHR','减固定参数持续右手规则'),('source_fixed','减源类型多数票固定候选')])
        body.append(f"- {TYPE_NAMES[kind]}（{kind}）：{comparisons}。")
    body.extend(['',f"留出的新增开发类型泛化检验（终点三）：相对按进度切换{contrast(end['endpoint3']['all_Sprog'])}，相对源类型多数票固定候选{contrast(end['endpoint3']['source_fixed'])}。该组包含三种模型预测控制、优先级时空规划和保守强化学习；每个控制器使用没有见过其所属类型的模型。",'',f"接近按状态交叉拟合事后最佳的检验要求锁定类型与新增开发类型各自满足平均遗憾≤3pp且伤害数为0，同时附合并口径；合并unknown遗憾上下界为{end['endpoint4']['regret_unknown_bounds_pp'][0]:.2f}至{end['endpoint4']['regret_unknown_bounds_pp'][1]:.2f}pp；可能伤害数{end['endpoint4']['harm_possible']}，确定伤害数{end['endpoint4']['harm_certain']}。按状态交叉拟合只用相反seed选臂并打分，另列样本内和按控制器oracle。有限seed的交叉拟合参照有噪声，可能被某个固定方法超过，不把负遗憾截成零。",'','控制器信息是否有用（终点五），完整修正流程的配对消融差：'])
    for group,label in [('locked','锁定类型'),('new_development','新增开发类型')]:body.append(f"- {label}：减不看控制器的critic为{contrast(end['endpoint5'][group]['no_controller'])}；减不看状态的critic为{contrast(end['endpoint5'][group]['no_state'])}。")
    for group,label in [('locked','锁定类型'),('new_development','新增开发类型')]:
        value=end['endpoint4']['by_group'][group]
        body.append(f"接近事后最佳检验，{label}单列：平均遗憾{value['regret_pp']:.2f}pp，unknown上下界{value['regret_unknown_bounds_pp']}，可能伤害数{value['harm_possible']}，{passed(value['passed'])}。")
    with (HERE/'CONTROLLER_STATE_DETAILS.csv').open() as stream:detail_rows=list(csv.DictReader(stream))
    drops=[]
    for name in sorted({r['controller'] for r in detail_rows}):
        rows=[r for r in detail_rows if r['controller']==name]
        gain=(sum(int(r['successes']) for r in rows if r['selected']=='True')-sum(int(r['successes']) for r in rows if r['candidate']=='10'))/48*100
        if gain<=-10:drops.append(f"{TYPE_NAMES[rows[0]['type']]}的{name}实例相对只加安全投影下降{-gain:.2f}pp")
    if drops:body.append('逐实例伤害核查：'+'；'.join(drops)+'（成功下界的配对点估计；unknown上下界与伤害确定性见主统计）。')
    body.append('不看状态的消融仍读取随状态变化的生成器候选，因此不能解释为移除了整条pipeline的所有状态信息。仅在相应消融配对区间支持时才声称controller信息有额外贡献，权重变大本身不是证明。')
    body.extend(['','训练状态的可选空间与开发检验均属于DEVELOPMENT。非学习型按状态样本内oracle相对源类型多数票固定候选为'+contrast(headroom['nonlearned']['vs_fixed'])+'；学习型为'+contrast(headroom['learned']['vs_fixed'])+'。按控制器样本内空间相对按进度切换，学习型为'+f"{headroom['learned']['controller']:.2f}pp，非学习型为{headroom['nonlearned']['controller']:.2f}pp。训练生成器只有seed0，未把这些数值冒充交叉拟合收益。",'','开发目标逐项为：'+json.dumps(dev['targets'],ensure_ascii=False)+'。当前结构包含单分支完整critic和按源内NLL选择的正权重概率收缩；没有按外层得分挑最好的初始化、checkpoint或controller权重。全部外层结果和两版选择保留，未达到目标也按预算冻结当前源内最优版本。'])
    folds=read(HERE/'DEVELOPMENT_OUTER_FOLDS.json');weights={f['held_type']:f['source_only_selections']['full']['selected']['full_weight'] for f in folds}
    body.extend(['','各外折源内选中的controller分支概率权重：'+json.dumps(weights,ensure_ascii=False)+'。前轮九折均为0.75；本轮同时给出每折源内NLL曲线和完整分支外层NLL、同状态候选排序曲线，外层曲线只用于诊断。',f"旧学习型九类型各两个实例的同新状态回归比较，相对前轮冻结九类型pipeline为{contrast(p('old_learning_regression','old_pipeline'))}。前轮模型保持原24维特征和13候选，直接恢复权重，没有重训。该参照在旧训练中见过原九种类型，而本轮这些实例使用排除目标类型的外折模型，因此这项回归差值不能单独归因于加入新类型。",'','新增类型的逐类描述（最佳修法为新状态按控制器样本内候选参照，不是部署时可用标签）：'])
    for kind in ('distributed_shooting','centralized_sampling','cross_entropy','prioritized_astar','ring_ippo_conservative','distributed_cross_entropy','centralized_mppi'):
        rows=[r for r in confirm['repair_class_selection'] if r['type']==kind];counts={c:sum(r['best_repair_class']==c for r in rows) for c in CLASSES};preferred=max(CLASSES,key=lambda c:counts[c]);chosen={c:float(np.mean([r['selected_class_fractions'][c] for r in rows])) for c in CLASSES};tail=diagnosis.get(kind)
        class_match=float(np.mean([sum(r['selected_class_fractions'][c] for c in r['optimal_repair_classes']) for r in rows]))
        motion=f"安全基线已知超时尾段中，严格冻结占未到达机器人步的{tail['strict_freeze_fraction']*100:.2f}%，在动但不前进占{tail['moving_no_progress_fraction']*100:.2f}%" if tail else '未观测到可用的安全基线已知超时尾段，不据此分类停滞'
        body.append(f"- {TYPE_NAMES[kind]}（{kind}）：{counts[preferred]}/{len(rows)}实例的样本内最佳属于{CLASS_NAMES[preferred]}；pipeline三类选择比例为"+'、'.join(f'{CLASS_NAMES[c]} {chosen[c]*100:.1f}%' for c in CLASSES)+f'；与按控制器样本内最优修法类的匹配率为{class_match*100:.1f}%（并列最优类均算匹配）；{motion}。这些描述不构成独立死锁验证。')
    body.extend(['','三族留一检验仅作开发期描述；各族的控制器类型均从训练与源内选参中排除：'])
    family_names={'learned':'学习型族','optimization':'优化型族','search':'搜索型族'}
    for family,summary in read(HERE/'LEAVE_ONE_FAMILY.json')['results'].items():
        body.append(f"- {family_names[family]}：修正流程成功率{metric(summary['methods']['pipeline'])}，相对一律按进度切换{contrast(summary['paired_comparisons']['pipeline__minus__all_Sprog'])}。")
    body.extend(['','固定结构规模曲线的类型等权平均遗憾如下，均相对共同seed0样本内事后最佳候选：'])
    for panel in read(HERE/'SCALING_PLOT_DATA.json'):
        points=panel['points'];label='源类型数' if panel['axis']=='m' else '源类型最佳修法类别覆盖数'
        values='、'.join(f"{r['x']}→{r['mean']:.2f}pp（{r['held_types']}个留出类型、{r['subsets']}组）" for r in points)
        descending=len(points)>=2 and all(b['mean']<=a['mean']+1e-12 for a,b in zip(points,points[1:]))
        body.append(f"- {label}：{values}；点估计"+('逐级不升。' if descending else '未呈现逐级下降。'))
    body.append('类别覆盖曲线若下降，只支持本轮固定设计下的描述性关联。源类型数、训练标签数量与类别覆盖相关，不同覆盖水平的留出类型集合也可能不同；误差线是跨留出类型标准差；不据此单独识别“见过更多卡法”造成改善的因果效应。')
    body.extend(['','本轮动物园为14个开发类型92个实例、2个实际锁定类型12个实例；可选隐式能量行为克隆策略未通过原两峰合成门，六个计划实例保留缺口，不替换。新评估范围是全部锁定12个、全部新增开发30个，以及原九学习型每类按原清单顺序两个，共60个。每个实例在24个新状态上完整执行16候选及只执行动作对照、seed0–1，合计48,960个请求。', '', '冻结生成器按状态提出五个η：变换均值和四个SHA确定性样本。每个η构成全程固定与按进度切到原完整精度η10两个动作；再加三个安全/时间表锚点和三组持续右手规则，共16候选。主critic逐状态预测并按三个seed集成均值argmax选择。候选不是事后从评估标签选的，选择收据早于第一个新评估rollout提交。', '', '三组各16条历史记录的首动作、完整位置、终止与步数全部逐位复现；72个历史状态的生成器提议和原critic分数逐位一致。A*补齐轴向、斜向、连续时空占用、重规划时刻与加速核等价测试；旧244条工程评估未用于调参。两种锁定优化器通过单机器人交叉熵首动作、匀速预测、零温度极限和同键逐位测试。', '', '保守强化学习的原训练环境逐字复制；float64等价位置误差3.11e-15、间距误差4.44e-15，float32终止类型与步数零不一致。旧组合观测检查把累计轨迹舍入误差混入编码误差，原失败报告保留；同物理状态1488项复测最大9.54e-7，通过未放宽的2e-6阈值。六个实例各20M机器人步，原课程、网络、种子、训练状态、预算和checkpoint规则不变，只改指定碰撞与接近奖励。', '', '训练状态补缺精确上限25,760条，实际按official preflight只跑缺失：旧学习型5,312、既有模型预测控制及修正A* 15,840、保守强化学习4,608。改变候选的开发追加预算使用0条。机制探测复用2,976条完整历史记录和192个200步前缀，补3,680条缺失探测，关闭终止、不读取成败。unknown全部保留，永久UID认领防止重跑。', '', 'critic输入为状态100维、环境33维、机制特征34维与动作16维；历史24维机制定义逐字保留。二项NLL只用已知标签，权重min(实际seed数,16)；AdamW学习率1e-3、权重衰减1e-4、梯度裁剪5，三个seed17/23/41，float32最高矩阵乘法精度。预处理只拟合源类型；冻结推理恢复h/c归一化参数float32、controller参数float64，全部外折恢复预测和旧pipeline复核逐位通过。', '', '十四折外层留一类型、源类型内部留一选初始化和checkpoint；三族留一只作描述。结构固定后按预注册随机源集合重训规模曲线，m=1由用户明确授权用唯一源类型内留一controller，且预处理排除该controller；m≥2仍源内留一类型。两种横轴数据、图与全部缓存源集合的配置、曲线、checkpoint均保留。类别覆盖与源类型数相关，曲线不单独识别因果效应。', '', '5000次配对bootstrap依次抽类型、控制器、状态，两个seed保持同簇；oracle与最强固定方法在每次重抽中重新选择。unknown同时报成功下界与上界，同UID不确定性在配对差抵消；oracle区间条件于按下界选定的臂，不是所有潜在unknown赋值的全局优化界。', '', '共享数据库123,513个历史状态UID/内容身份和116,441个可恢复物理hash已核查；4,607个不可恢复历史布局仍是去重证据缺口。新状态种子202610182001–202610182024，没有已知重复。冻结前没有对这24个新状态执行闭环rollout。锁定能力门只输出通过或排除；原始修法率在新评估完成后才解除盲态。', '', '物理、硬安全投影、事件和成功标准未改；环形交换流匹配策略保持FF，其余保持TT，OrthoFlow3基每步重算。timeout不直接称为deadlock。冻结原critic历史上已见原生Flow，源内排除不能抹去该预训练范围；从头训练版本一并报告。', '', '完整表格见AXIS_TABLE.md/CSV与AXIS_METRICS.json；逐控制器、状态、候选、seed身份及遗憾见CONTROLLER_STATE_DETAILS.csv；同一控制器、状态的逐seed交叉拟合参照及遗憾另见CONTROLLER_STATE_CROSSFIT_REGRET.csv。nominal到安全投影逐条转移、特征越界、超时尾段和选中动作变化见对应独立文件；figures/为仿真证据的静态读取。全部新代码与输出仅本目录，数据库仅官方追加登记、默认journal与单进程merger；没有Git写操作、包安装或读取禁止研究线内容。原prompt为18CPU、2GPU shard，用户随后明确重申空闲时允许24CPU；按用户授权空闲时用24CPU，有其他任务时保持其优先并退回18CPU总上限，GPU按用户后续授权，在没有其他GPU任务时可用全部8个shard，有其他GPU任务时退回原2个上限。授权收据见RESOURCE_AUTHORIZATION_24_CPU与RESOURCE_AUTHORIZATION_8_GPU，每次提交重新检查squeue。'])
    (HERE/'REPORT.md').write_text('\n'.join(body)+'\n')
    gaps=['# 缺口与结论限制','', '可选隐式能量行为克隆类型缺失：历史原两峰合成门未通过，没有合格Ring checkpoint，不替换类型；主确认按两个实际锁定类型判定。','历史物理布局4,607项无法恢复；全UID和可恢复坐标去重通过，不宣称对不可恢复布局绝对去重。','训练生成器完整菜单seed1缺失，训练可选空间只作样本内；新评估两seed完整。','新评估全程旧默认参数不在本轮16候选中，灾难数N/A；训练仅报可精确复用的历史F10参照，缺失不补造。','原critic暖启动历史预训练已见原生Flow，需与scratch版本并列解释。','规模曲线类别覆盖与源类型数量相关，仅支持描述性关联，未设计独立因果干预。','交叉拟合有限seed参照可能低于固定方法；unknown界条件于下界选臂，不等同对全部潜在标签赋值的极值界。']
    feature_gaps=read(HERE/'FEATURE_GAPS.json');gaps.append(f'机制探测有{len(feature_gaps)}个控制器存在不完整探测臂，逐项见FEATURE_GAPS.json；对应特征保留NaN并仅用源类型填补，不平均掉缺失探测，不重跑unknown。')
    if not audit['passed']:gaps.append('结束仓库状态审计因外部新增未跟踪提示词docs/prompt_crosstype1c_selected_two_eta_schedule_20261018.md而未通过；原始失败记录保留。8491个受保护源码和全部核查输入、两仓库HEAD与已有差异均未变。详见INTEGRITY_COMPARISON.json与INTEGRITY_DISPOSITION.json。')
    (HERE/'GAPS.md').write_text('\n\n'.join(gaps)+'\n')
    with (HERE/'ITERATION_LOG.md').open('a') as log:
        summary=dev['reports']['development']['type_equal']['methods'];single=summary['pipeline_single']['success_lower_pct'];final=summary['pipeline']['success_lower_pct']
        log.write(f'\n\n最终开发版本：单分支完整critic类型等权成功率{single:.4f}%，源内选择概率收缩完整pipeline为{final:.4f}%。两个版本、scratch/warm与匹配消融均完整保留；收缩权重由源内NLL选，不按外层最高分选版本。开发目标结果{json.dumps(dev["targets"],ensure_ascii=False)}。结构固定后完成两轴规模曲线，再冻结并只做一次全新状态确认；未用确认标签回调模型或候选。\n')
    handoff=['本轮混合黑盒动物园已完成开发、冻结、一次性新状态确认和完整性审计。主结论、终点与证据限制以REPORT.md和GAPS.md为准；不得利用本轮确认标签调整模型后继续把原确认称为有效。','下一轮跨类型第三轮可以只读复用：CANDIDATES.json与candidates.py（冻结生成器规则）；RHR_SHORTLIST.json（历史一次性短名单）；ZOO_MANIFEST.csv与各policies/实例checkpoint；PROBE_MANIFEST/PROBE_TRAJECTORY_INDEX、CONTROLLER_FEATURES.npz与可读CONTROLLER_FEATURES.csv（空白为缺失）；TRAINING_BUNDLE.npz及NEW_EVAL_BUNDLE.npz；FROZEN_PIPELINE.json与NEW_EVAL_CHOICES.receipt.json；SOURCE_AND_INPUT_RECEIPTS.json与INTEGRITY_COMPARISON.json。','当前有效入口：mixed_controllers.py、mixed_runtime.py、mixed_execution.py、zoo_design.py、train_critic.py（四种模式共用缓存）、select_mixed.py、single_source_selection.py、inference.py、new_choices.py、statistics_pipeline.py。保留的旧名称训练/选择文件主要是源码参考；新流程由advance.py协调，不能把参考旧入口的固定62/13维设定当成本轮入口。','所有构造器、core/common会将临时目录指向当前文件所在实验目录；下一轮不要直接运行本目录这些有写副作用的入口。按共同协议复制到新目录、记录源SHA/diff，并先复现。旧rollout标签不因critic表示变化失效；先DB preflight与原始RNG/物理/执行指纹核查，只补真正缺失，unknown不重跑。','规模曲线源集合与source-controller内层缓存路径在SCALING_TRAINING_PLAN.json、SCALING_DRAWS.json、SCALING_RESULTS.json；源集合完全相同才能复用对应fit，不应重复训练。m=1的controller内层例外得到用户明确授权，见USER_DESIGN_DECISIONS。','本轮新状态种子202610182001–202610182024；下一轮生成新状态时需再次对DB全体去重，包括本轮24个状态。实际锁定类型只有两种，可选隐式类型缺口不能由其他类型顶替。','冻结与模型推理必须恢复h/c统计float32、C统计float64，并使用float32 highest矩阵乘法。原生Flow预训练范围限制仍存在。所有新资源授权重新按最新prompt执行；本轮按用户后续授权，无其他任务时24CPU，有其他任务时其优先并回到18CPU总上限；GPU在没有其他GPU任务时可用全部8个shard，有其他GPU任务时回到2个上限。不得修改旧源码或旧目录、安装包或执行Git写操作。']
    handoff.insert(0,f'本轮全部相对路径均以 `{HERE}` 为根目录；原始任务规范为 `{ROOT}/docs/prompt_crosstype2_mixed_zoo_selector_20261018.md`。')
    handoff.append('当前资源配置为RESOURCE_LIMITS_ACTIVE.json；RESOURCE_LIMITS.json为CPU授权时的历史原件，为保持收据有效而保留。CPU24和GPU8授权事件见RESOURCE_AUTHORIZATION_24_CPU与RESOURCE_AUTHORIZATION_8_GPU的JSON及收据。调度器在提交前核查所有作业；模型、数据和统计设计不因扩容而改变。')
    (HERE/'HANDOFF.md').write_text('\n\n'.join(handoff)+'\n')
    write(HERE/'results.json',dict(utc=utc(),integrity=audit['passed'],development_targets=dev['targets'],endpoints=end,confirmatory_results_path=str(HERE/'CONFIRMATORY_RESULTS.json'),axis_table_path=str(HERE/'AXIS_TABLE.csv'),gaps_path=str(HERE/'GAPS.md')))
    seal('FINAL_DELIVERY.receipt.json',[HERE/n for n in ('REPORT.md','GAPS.md','HANDOFF.md','ITERATION_LOG.md','results.json','write_report.py','AXIS_TABLE.receipt.json','FIGURES.receipt.json','FIGURE_PREVIEWS.receipt.json','VISUAL_REVIEW.receipt.json','FEATURE_TABLE.receipt.json','ZOO_MANIFEST.receipt.json','CONFIRMATORY_RESULTS.receipt.json','DESCRIPTIVE_ANALYSIS.receipt.json','INTEGRITY_COMPARISON.receipt.json','INTEGRITY_DISPOSITION.receipt.json','DELIVERY_CHECKS.receipt.json','PREREGISTRATION.receipt.json','HISTORY_AUDIT.receipt.json','USER_DESIGN_DECISIONS.receipt.json','RESOURCE_AUTHORIZATION_24_CPU.receipt.json','RESOURCE_AUTHORIZATION_8_GPU.receipt.json','ENDPOINT4_SCOPE_DESIGN.receipt.json','SCALING_RESULTS.receipt.json','FROZEN_PIPELINE.receipt.json')],research_task_completed=True,confirmation_not_used_for_retuning=True)
    print('Final report, handoff and delivery receipt complete; primary:',passed(end['endpoint2']['passed']))


if __name__=='__main__':main()
