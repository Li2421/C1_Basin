"""Final delivery from completed, immutable scientific evidence and end audit."""
from common import *
from render_tables import TYPE_NAMES,METHODS
import csv


def ci(value):return f'[{value[0]:.2f}, {value[1]:.2f}]'
def yes(value):return '通过' if value else '未通过'


def main():
    required=('FROZEN_PIPELINE.receipt.json','SCALE_CURVE_COMPLETE.receipt.json',
        'DEVELOPMENT_FINAL_SELECTION.receipt.json','DEVELOPMENT_SELECTION_VALUE.receipt.json',
        'LOCKED_CHOICES.receipt.json','CONFIRMATORY_RESULTS.receipt.json',
        'LOCKED_TRAJECTORY_FIGURES.receipt.json','EXECUTION_BUDGET_FINAL.receipt.json','FINAL_EVIDENCE_COVERAGE.receipt.json','LOCKED_TIMEOUT_TRAJECTORY_DIAGNOSTICS.receipt.json')
    verified=[]
    for name in required:
        for path,digest in read(HERE/name)['files'].items():assert sha(path)==digest,path
        verified.append(name)
    audit=read(HERE/'INTEGRITY_FINAL_COMPARISON.json')
    frozen=read(HERE/'FROZEN_PIPELINE.json');iteration=frozen['iteration']
    dev=read(HERE/'iterations'/iteration/'RESULTS.json');scale=read(HERE/'SCALE_CURVE_RESULTS.json')
    lock=read(HERE/'CONFIRMATORY_RESULTS.json');ep=lock['endpoints']
    d=dev['summaries']['combined']['type_equal']['methods']['pipeline']
    dc=dev['summaries']['combined']['controller_equal']['methods']['pipeline']
    r5dev=dev['summaries']['combined']['type_equal']['methods']['r5_pca2_pipeline']
    l=lock['summaries']['combined']['type_equal']['methods']['pipeline']
    value=read(HERE/'DEVELOPMENT_SELECTION_VALUE.json')
    selected_value=next(r for r in value['iterations'] if r['iteration']==iteration)
    budget=read(HERE/'EXECUTION_BUDGET_FINAL.json')
    ranking=read(HERE/'DEVELOPMENT_FINAL_SELECTION.json')
    repair=ep['repair_vs_safety'].get('ring_ippo',{})
    ppo=lock['summaries'].get('ring_ippo',{}).get('type_equal',{}).get('methods',{})
    mechanism=lock['descriptive_mechanisms_and_preferences'].get('ring_ippo',{})
    target=dev['development_targets']
    scale_gain=scale['main_m7_vs_m1_paired_gain']
    primary=ep['primary_zero_shot'];near=ep['near_oracle_and_no_harm']
    phase_rows=budget['phases']
    timeout_diagnostic=read(HERE/'LOCKED_TIMEOUT_TRAJECTORY_DIAGNOSTICS.json')
    late=timeout_diagnostic['aggregate']['last100_steps']
    outside=[r['outside_development_range_fraction'] for r in lock['out_of_range'] if r['outside_development_range_fraction'] is not None]
    result=dict(utc=utc(),status='COMPLETED_WITH_REPORTED_SCIENTIFIC_GAPS',workflow_complete=True,
        integrity=audit,selected_iteration=iteration,development_targets=target,
        development_pipeline=d,development_controller_equal_pipeline=dc,
        source_only_best_fixed_comparison=selected_value,scale=scale,locked_endpoints=ep,
        missing_locked_types=lock['missing_locked_types'],execution_budget=budget,
        zero_shot_adaptive_selection_established=False,
        resource_limits=dict(cpu_shards=18,gpu_shards=5,existing_tasks_first=True),
        verified_receipts=verified,
        evidence=dict(development=str(HERE/'iterations'/iteration/'RESULTS.json'),
            confirmatory=str(HERE/'CONFIRMATORY_RESULTS.json'),axis_table=str(HERE/'AXIS_TABLE.csv'),
            zoo=str(HERE/'ZOO_MANIFEST.csv'),frozen_pipeline=str(HERE/'FROZEN_PIPELINE.json')))
    write(HERE/'results.json',result)
    opening=(f"完整性审计{yes(audit['passed'])}：{audit['protected_initial_count']:,} 个开工受保护 Python 文件、导入来源及两个仓库记录的比对见 `INTEGRITY_FINAL_COMPARISON.json`。"
        f"开发目标{yes(target['passed'])}：零样本选择器类型等权成功率 {d['success_lower_pct']:.2f}%，相对按比例切换时间表增加 {d['gain_vs_schedule_pp']:.2f} 个百分点，95% CI {ci(d['gain_CI95_pp'])}；优越性未通过。"
        f"规模曲线的点估计{'单调不增' if scale['monotonic_point_estimates'] else '不单调'}，七源类型相对单源类型的成功率增益为 {scale_gain['gain_pp']:.2f} 个百分点，CI {ci(scale_gain['CI95_pp'])}；不能据此宣称完整的类型多样性优势已成立。"
        f"锁定终点一在现有强化学习类型上{yes(repair.get('passed',False))}；终点二为 {primary['status']}，未通过完整主终点；终点三在现有类型上的数值目标{yes(near['available_cohort_numerical_target_met'])}，两类型完整判定仍缺失。"
        "参数共享独立近端策略优化强化学习策略（ring_ippo）六实例完成一次性确认；隐式能量行为克隆策略（implicit）未通过原两峰合成门，未训练 Ring 策略，保留缺口。")
    report=['# 跨 controller type 的实现：跨学习型类型第一轮','',opening,'',
        '本轮完成了可复查的动物园、开发比较、冻结选择器、规模曲线和现有锁定类型的一次性确认，但没有做出已获证据支持的跨类型自适应选臂。所有失败迭代与缺口均保留；开发结果为 DEVELOPMENT，锁定结果为 CONFIRMATORY，后者的完整两类型主终点不能由现有子集替代。','',
        '**动物园、训练和证据门。** 八个开发类型、56 个合格实例：环形交换场景流匹配策略（Flow）14 个；高斯行为克隆策略（gaussian）、确定性行为克隆策略（deterministic）、混合密度行为克隆策略（mixture）、扩散行为克隆策略（diffusion）、动作分块行为克隆策略（chunking）、循环网络行为克隆策略（recurrent）、条件变分自编码器行为克隆策略（cvae）各六个。历史38实例直接复用，新增18个开发实例训练并通过能力门。强化学习六实例各完成2,000万机器人步，无训练安全层，按32个训练留出state的只执行动作成功率选checkpoint；评估用原环境。清单和每实例checkpoint哈希见 `ZOO_MANIFEST.csv`，配置、曲线、原始训练记录见 `TRAINING_CONFIGS.json`、`IPPO_TRAINING_CONFIGS.json`、`policies/`、`figures/chunking_dev_loss.svg`、`figures/recurrent_dev_loss.svg`、`figures/cvae_dev_loss.svg`、`figures/ring_ippo_mean_reward.svg`及对应原始曲线JSON。','',
        '32/32条历史复现的首动作、终止、步数和完整位置轨迹逐位一致。六套坐标系和策略状态/随机键测试通过。训练环境float64位置最大误差3.11e-15、扫掠间距误差4.44e-15，两种精度各256条终止类型和步数均一致。原float32附加观测测试混入累计轨迹漂移，原失败保留，改用同一历史状态作观测检查而不放宽阈值，见 `TEST_HARNESS_AMENDMENT_20261018.json`。原冻结critic的66个历史选臂在CPU上全部一致，附加概率阈值的CPU失败保留；GPU重核通过原2e-6阈值，当前56实例三种比较方法的GPU选臂也与提前封存的CPU选择一致。','',
        '**开发结果。** 岭回归预测各臂相对按比例切换时间表（S500）的成功率差；特征填补、标准化和内层超参数选择均只拟合源类型。全部八折及内层误差在 `iterations/ridge_v1/OUTER_FOLDS.json`，闭式回归没有按训练步数挑checkpoint；历史模型早期过拟合问题通过源类型内嵌套调参检查。九次结构/特征比较逐次保存在 `ITERATION_LOG.md`，最终选择规则先于结果封存，精确并列取最早版本。','',
        f"所选模型交叉拟合事后最佳平均遗憾为 {d['crossfit_regret_pp']:.2f} 个百分点，最大controller遗憾 {d['max_controller_regret_pp']:.2f} 个百分点，伤害数 {d['harm_possible']}；没有留出类型比按比例时间表差超过3个百分点。Controller等权成功率 {dc['success_lower_pct']:.2f}%，相对按比例时间表增益 {dc['gain_vs_schedule_pp']:.2f} 个百分点，CI {ci(dc['gain_CI95_pp'])}。负交叉拟合遗憾来自小样本选臂噪声：交叉拟合的事后选择可能低于强固定臂，不表示超越真实oracle。",'',
        f"54个探测可用实例全部选择按进度切换时间表（Sprog），两个保留历史探测数值失败的实例回退按比例时间表。每折仅用源类型选择最佳固定臂也全部选按进度时间表；选择器相对该基线的增益为 {selected_value['paired_vs_source_only_best_fixed']['gain_pp']:.3f} 个百分点，CI {ci(selected_value['paired_vs_source_only_best_fixed']['CI95_pp'])}。统一按进度时间表成功率 {value['best_uniform_type_equal_success_pct']:.2f}%，样本内每controller最佳 {value['in_sample_controller_oracle_type_equal_success_pct']:.2f}%，样本内自适应空间仅 {value['observed_in_sample_adaptation_headroom_pp']:.3f} 个百分点。这是当前样本的描述性差距，不是总体不可提升的证明。不能把更好的统一修法当作行为特征已支持实例级选择。",'',
        f"只用行为指纹（R5-PCA2）的同结构模型开发成功率为 {r5dev['success_lower_pct']:.2f}%，机制特征模型只高 {d['success_lower_pct']-r5dev['success_lower_pct']:.3f} 个百分点。此次比较不支持把跨类型失败归因于指纹并宣称已靠换特征解决。原冻结扩展模型和原不看controller的critic仍为按比例时间表水平。",'',
        '| 迭代 | 特征版本 | 选择结构 | 相对按比例时间表增益 pp [95% CI] | 开发目标 |',
        '| --- | --- | --- | --- | --- |']
    structure={'ridge':'岭回归','ridge_margin':'保守回退岭回归','nearest_type':'最近源类型','shallow_tree':'共享浅树','armwise_tree':'每臂独立浅树','type_kernel':'类型内局部核回归'}
    for r in sorted(ranking['ranked_iterations'],key=lambda r:r['started']):
        report.append(f"| {r['iteration']} | {'均值机制特征24维' if r['feature_schema']=='mechanism_v1' else '机制及分布特征105维'} | {structure[r['strategy']]} | {r['gain_pp']:.3f} {ci(r['gain_CI95_pp'])} | {yes(r['targets']['passed'])} |")
    report+=['','**规模曲线。** 固定最终结构，每个留出类型抽取1、2、3、5、7个源类型，前四档各20组，最后一档一组；每次仅用抽到的源类型内部调参。单源类型无法再做源类型内留一，按预先规定使用固定正则1；其余档使用源类型内调参。因此规模变化也包含可用调参信息的变化，不能解释为隔离类型数的因果效应。规模区间条件于已记录的源抽样组与已拟合模型，在留出观察上重抽类型/controller/state，不在每次bootstrap中重训选择器或重抽源组。七类型预测与实际开发外折逐项相同。图为 `scale/scale_curve.svg`，每次训练的源实例/内层选择、原始预测、匹配对照及CI均在 `scale/`。','',
        '| 源类型数 | 交叉拟合遗憾 pp [95% CI] |','| --- | --- |']
    for r in scale['main_curve']:report.append(f"| {r['m']} | {r['regret_pp']:.3f} {ci(r['CI95_pp'])} |")
    fixed7=next(r for r in scale['fixed7_paired_gain_vs_single'] if r['m']==7)
    report+=['',f"固定七实例的受限对照中，七源类型相对单一流匹配源类型提高 {fixed7['gain_pp']:.2f} 个百分点，CI {ci(fixed7['CI95_pp'])}；该对照支持在此条件下增加源类型的收益，但其点估计并非逐档单调，且不能据此证明实例级自适应选臂。",'','主设计在三个及以上源类型时至少有18个源实例，单类型最多14个，完整等实例数对照客观不可构造。保留所有可行的精确数量匹配，并额外报告固定七个不同实例的对照；该补充仅限七个非流匹配留出类型，单类型供体只能是流匹配。它不能外推为覆盖全部类型的公平多样性结论。','',
        '**冻结及一次性锁定确认。** 模型、24维特征定义、缺失回退、同结构行为指纹对照、选择和终点分析代码在 `FROZEN_PIPELINE.json` 与收据封存。锁定选臂UTC为 '+lock['choices_utc']+'，第一批评估/调参标签提交UTC为 '+lock['first_label_submission_utc']+'。探测关闭终止并不读取成败；选臂收据早于第一批标签，冻结后没有据锁定结果重训、改特征或改阈值。','',
        f"现有强化学习类型：只加安全投影成功率 {ppo['all_safety_baseline']['success_lower_pct']:.2f}%，按比例时间表 {ppo['all_S500']['success_lower_pct']:.2f}%，冻结选择器 {l['success_lower_pct']:.2f}%。按比例时间表减安全基线为 {repair['point_pp']:.2f} 个百分点，CI {ci(repair['CI95_pp'])}；选择器减按比例时间表为 {l['gain_vs_schedule_pp']:.2f} 个百分点，CI {ci(l['gain_CI95_pp'])}；交叉拟合遗憾 {l['crossfit_regret_pp']:.2f} 个百分点、伤害数 {l['harm_possible']}。多数实例安全基线≥98%的例外规则{'触发' if ep['ppo_mostly_safety_at_least_98_percent'] else '未触发'}，完整主终点仍受隐式能量类型缺失限制。",'',
        f"现有强化学习类型的样本内最佳固定修法为 {', '.join(METHODS['all_'+a] for a in mechanism['in_sample_best_fixed_arms'])}。无成败安全探测的严格冻结比例均值为 {mechanism['mean_safety_probe_strict_freeze']:.4f}，在动但不前进比例均值为 {mechanism['mean_safety_probe_moving_no_progress']:.4f}。这些是描述性停滞诊断，timeout不直接称为deadlock。超出开发特征范围的每实例比例见 `CONFIRMATORY_RESULTS.json`；逐条只执行动作→安全投影转移见 `LOCKED_NOMINAL_TO_SAFETY_TRANSITIONS.csv`，汇总见对应SUMMARY。",'',
        f"实际安全基线的 {timeout_diagnostic['episodes']} 条超时轨迹另作描述性复核：最后100步中，未到达机器人步的严格冻结比例为 {late['strict_freeze_fraction_unfinished']*100:.3f}%，在动但不前进比例为 {late['moving_no_progress_fraction_unfinished']*100:.2f}%。这些超时尾段主要表现为在动但不前进；没有独立的死锁验证。完整逐条轨迹索引与分母在 `LOCKED_TIMEOUT_TRAJECTORY_DIAGNOSTICS.json`；这是确认后描述补充，没有新增假设检验、rollout或改变冻结终点。六个强化学习实例有 {min(outside)*100:.1f}%–{max(outside)*100:.1f}% 的机制特征超出开发实例的逐维范围。",'',
        '代表性轨迹按锁定前写定的中位controller、中位episode时长规则选择；同一state/seed的只执行动作、安全基线和选择器轨迹分别见 `figures/ring_ippo_nominal_only_trajectory.svg`、`figures/ring_ippo_safety_baseline_trajectory.svg`、`figures/ring_ippo_pipeline_trajectory.svg`。图仅读取哈希核验后的轨迹，不修改仿真。隐式能量类型没有有效Ring checkpoint，不能提供或伪造其轨迹。','',
        '**复用、统计及完整交付。** 旧探测复用1,216个可验证200步前缀，共243,132个既有物理步；终止前缀零物理复用与新增步骤单独记账。最终精确preflight后，按用户授权将旧类型新物理轨迹上限确定为18,846；实际记账 '+str(budget['old_type_new_physical_trajectory_attempts'])+' 次。开发额外6,000条标签预算使用0条。每批官方身份登记、preflight、默认journal、单进程merger及postflight保留；旧unknown不重跑，缺失数据不冒充unknown。','',
        '所有区间使用5,000次配对bootstrap：类型等权先抽类型，再抽controller、state簇，两个seed保留同簇；oracle在每次重抽内按原state奇偶重选。unknown按成功下界报告并保留上界；同一证据用于两个方法时在配对差中抵消。完整方法与每类型/两种合并方式见 `AXIS_TABLE.md`、`AXIS_TABLE.csv`、`AXIS_METRICS.csv`、`COMBINED_AGGREGATIONS.csv`，每实例最佳臂、选择、遗憾见 `CONTROLLER_SELECTION_DETAILS.csv`。原controller级critic选择菜单外臂的受影响单元格为N/A，不替换为本菜单臂；无专家参照不新增伪造专家。','',
        '边界记录：初始和接续的文件枚举曾显示禁读前缀的路径名，未读取内容、未用于实验；后续限定目标目录。禁止与并行目录审计只使用metadata，其变化与受保护源文件分开记录。没有git写操作、包安装或修改既有物理/安全层/成功标准。全局上限按用户最后决定为18 CPU、5 GPU shard，提交前逐次检查并保留 `RESOURCE_CHECKS.jsonl`。','']
    (HERE/'REPORT.md').write_text('\n'.join(report))
    gaps=['# 缺口、未达到的目标及补齐条件','',
        '- 隐式能量行为克隆策略：原ReLU两峰门失败；完整协方差采样恢复12/32，原门要求合计至少26/32且两峰各至少3。解析双峰能量用同采样器通过32/32，学习能量却偏向边界；三个平滑激活候选仍失败。原失败与全部候选保留，没有Ring训练或替换锁定类型。补齐需独立实现修正通过原门，再按原配方训练，作为下一轮新预注册的确认对象，不回填本轮主终点。',
        '- 开发优越性未达到：选择器相对按比例时间表CI下界≤0；九次比较及源类型最佳固定臂对照没有证明自适应选臂价值。下一轮需要有历史证据支持的偏好异质性或新controller类型/修正场景，不能靠重复同执行语义rollout解决。',
        '- 完整锁定主终点与两类型近oracle判定不完整：本轮只有六个强化学习实例，隐式能量类型缺失。保留现有类型的一次性确认结果，不以子集宣布完整通过。',
        '- 两个历史controller探测数值失败保留，特征不可用按冻结规则回退；全部任务unknown在统计中保留下界/上界，不重跑。',
        '- 完整等实例数单类型规模对照不可行：大样本档超过任何单一源类型的实例数。现有精确匹配点与固定七实例条件比较已交付；补齐需未来独立新增足够多同类型实例，不复制旧实例。',
        '- 原冻结controller级critic部分选择当前菜单外臂，受影响方法/类型标N/A；原因逐格见AXIS_TABLE_GAPS.json。补齐需要新的明确菜单和全类型相应证据，不改写原模型选臂。',
        '- 当前仅Ring H700、四机器人、集中安全投影及指定FF/TT位置；不宣称支持其他场景、分布式安全层、N=100或非学习型controller。','']
    (HERE/'GAPS.md').write_text('\n'.join(gaps))
    handoff=f'''# 跨type2接续输入

本轮工作流完成，开发优越性未达到；完整锁定主终点因隐式能量类型缺失而不完整。请先读 `REPORT.md`、`GAPS.md`、`results.json`，不要把本轮标为跨类型自适应选择成功。最终版本为 `{iteration}`，九次尝试全部保留。

直接复用的输入：

- 冻结模型：`FROZEN_PIPELINE.json`、`FROZEN_PIPELINE.receipt.json`；内部包括24维机制特征岭回归和同结构R5-PCA2模型。用 `selector.Regression.from_dict` 恢复模型，再 `choose(features)`，不要重新拟合预处理。`choose_locked.py` 是一次性本轮选臂入口，不应用它覆盖既有收据。
- 特征定义：`features.py`，顺序在冻结模型的feature_names。每controller为16state×三臂×300步无成败探测，初观测8键动作查询。`assemble.py`、`initial_actions.py` 给出提取入口；`DEVELOPMENT_FEATURES.npz`、`LOCKED_FEATURES.npz`、对应FEATURE_TABLE和PROBE_TRAJECTORY_INDEX为已封存输入。全部特征不可用则回退按比例时间表。
- 动物园：`ZOO_MANIFEST.csv` 每实例的类型、配方、checkpoint路径/哈希、能力门；新策略在 `policies/`，训练配置与收据、训练曲线均保留。共有56个开发实例、6个锁定强化学习实例；六个隐式能量占位是实现缺口，不能当作已训练或已排除controller。
- 完整标签：`DEVELOPMENT_BUNDLE.npz` 与 `LOCKED_BUNDLE.npz`，对应标签索引、DB登记/preflight/postflight及execution receipts。标签1成功、0已知非成功、-1unknown；原始碰撞/timeout单列。零样本方法不得把目标controller标签作为输入。
- 结果：`iterations/{iteration}/RESULTS.json`、`CONFIRMATORY_RESULTS.json`、`AXIS_TABLE.csv`、`AXIS_METRICS.csv`、`CONTROLLER_SELECTION_DETAILS.csv`、`SCALE_CURVE_RESULTS.json`及 `scale/`。`DEVELOPMENT_SELECTION_VALUE.json`区分自适应选择与统一好臂，样本内自适应空间约0.391pp。

锁定标签已经打开一次。下一轮不能继续把这些结果称为未见过的确认性证据；任何据此开发的修改须另设确认对象。隐式能量实现修复必须先通过原合成门，不放宽阈值、不替换锁定类型，不回写本轮结果。

所有Python执行使用 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`。新工作另开被授权目录；本轮冻结代码和数据不覆盖，不做git写操作，不装包。禁止目录按decentralized_/teacher_student_/multiscene_前缀剪枝。DB只能官方设计登记、默认journal、单进程merger；先搜历史和精确preflight，unknown不重跑。当前用户授权全局18 CPU、5 GPU shard，已有任务优先。用户允许最终精确preflight后调整旧类型预算，本轮已定18,846次，开发追加标签预算使用0。

最终审计见 `INTEGRITY_START.json`、`INTEGRITY_END.json`、`INTEGRITY_FINAL_COMPARISON.json`；最终交付校验见 `FINAL_DELIVERY.receipt.json`。没有需要继续等待的本轮训练或证据采集任务。
'''
    (HERE/'HANDOFF.md').write_text(handoff)
    paths=[HERE/name for name in ('REPORT.md','HANDOFF.md','GAPS.md','results.json','ZOO_MANIFEST.csv','ITERATION_LOG.md',
        'AXIS_TABLE.md','AXIS_TABLE.csv','AXIS_METRICS.csv','COMBINED_AGGREGATIONS.csv','CONTROLLER_SELECTION_DETAILS.csv',
        'INTEGRITY_FINAL_COMPARISON.json','DEVELOPMENT_FEATURE_TABLE.csv','LOCKED_FEATURE_TABLE.csv','render_final_report.py')]
    paths.extend(HERE/name for name in (*required,'FINAL_FIGURE_REVIEW.json','INTEGRITY_START.json','INTEGRITY_END.json','AUDIT_EXCLUSION_SET_CHECK.json'))
    seal('FINAL_DELIVERY.receipt.json',paths,completed_with_gaps=True,integrity_passed=audit['passed'],
        development_targets_met=target['passed'],complete_locked_primary_met=primary['passed'])
    print('Final report, handoff, gaps, results and delivery receipt written')


if __name__=='__main__':main()
