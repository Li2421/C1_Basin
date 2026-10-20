from core import *
from reporting import fmt
import csv,collections,subprocess
precision()
def main():
 prereg();audit=read(HERE/'INTEGRITY_COMPARISON.json');assert audit['passed'];a1=read(HERE/'analysis1_teacher_ceiling.json');a2=read(HERE/'analysis2_nominal_failure.json');a3=read(HERE/'analysis3_safety_stalls.json');a4=read(HERE/'analysis4_correction.json');a5=read(HERE/'analysis5_zero_controller.json');a6=read(HERE/'analysis6_relabeling_shared_noise.json');paper=read(HERE/'analysis7_paper_cells.json');secondary=read(HERE/'SECONDARY_ANALYSES.json');early=read(HERE/'PAIRED_EARLY_DIAGNOSTICS.json');tr=read(HERE/'MODE_OUTCOME_TRANSITIONS.json')['summary'];curves=read(HERE/'PLOT_DATA_FINAL.json');errors=read(HERE/'METRIC_ERROR_AUDIT.json');assert not errors['errors'];assert read(HERE/'UNIT_TESTS.json')['status']=='PASS' and read(HERE/'STATISTICS_TESTS.json')['status']=='PASS'
 # Validate every frozen receipt before final prose and delivery sealing.
 receipt_checks={}
 for n in ['PREREGISTRATION.receipt.json','EVIDENCE_RESOLVED.receipt.json','TESTS.receipt.json','METRICS_FROZEN.receipt.json','ANALYSIS_FROZEN.receipt.json','ANALYSIS_COMPLETE.receipt.json']:
  r=read(HERE/n);bad=[p for p,h in r['files'].items() if sha(p)!=h];assert not bad,(n,bad);receipt_checks[n]=True
 rate=lambda g,a,k='success':paper[g][a][k]['lower']['estimate']
 first='源码完整性审计通过，历史目录与指定源码的156,997个文件内容保持不变；共享数据库未打开、未写入。学习型控制器单独运行的主要可观测失败是碰撞几何边界、机器人间碰撞及无法完成目标推进；教师与三种学生的失败局中绕行方向不一致仅占2.75%–5.85%，不支持把多数失败归因于顺逆时针冲突。分布式安全层把原碰撞大多变成安全超时，主要表现为仍在运动但目标进展不足、约3.9台机器人未到达，存在连续低速停滞段的超时局仅0.12%–6.71%。前500步使用第13组、之后第10组修正参数的时间表（S500）增加早期目标进展并救回大量超时；被救回的教师/学生局中90.1%–97.1%在安全层下已绕行一致，因此不能把修正效果解释为提前统一绕行方向。零动作控制器加同一时间表仅28.1%成功，失败局中60.9%仍绕行一致；学习型控制器的贡献更符合持续目标推进与任务完成，而非仅选定初始绕行方向。闭环重标注学生（各自噪声；V1）与闭环重标注加共享随机种子学生（V3）没有提高单独成功率；教师自身只有8.37%成功，共享随机种子减少碰撞但增加超时，绕行不一致率下降的置信区间仍覆盖零。全部结论为DEVELOPMENT，timeout不直接称为deadlock。'
 text=first+'\n\n'
 text+='## 证据范围与执行\n\n完整任务原文与补充定义已在PREREGISTRATION.md中封存，UTC及SHA256见收据。解析29,024条角色记录，去重后为28,128条历史轨迹；只读使用六个历史目录，没有训练、没有controller/checkpoint查询、没有新增rollout。全程仅使用CPU，指标提取8个1CPU分片、聚合1CPU，提交前squeue无其他用户作业；无GPU申请。逐局指标见EPISODE_METRICS.csv，全部指标提取错误为0；动作与位置差分运动学残差最大为'+f"{errors['kinematic_max_abs_error']:.3g}"+'m。32条专家轨迹没有单列velocities，post-step速度由已有actions恢复，未执行环境；初始速度占位不用于post-step停滞判定。\n\n'+f"完整性比较允许并单列并行任务新增git条目：{json.dumps(audit['concurrent_new_git_entries_not_written_by_this_round'],ensure_ascii=False)}。这些文件不是本轮所写。两个仓库HEAD与diff哈希均保持不变。\n\n"
 text+='## 分析一：教师能力与学生表现的关联\n\n'
 text+=f"联合教师只跑时成功率{rate('TD','nominal only'):.2%}、碰撞率{rate('TD','nominal only','collision'):.2%}、超时率{rate('TD','nominal only','timeout'):.2%}；教师只加分布式安全层成功率仍为{rate('TD','safety baseline'):.2%}、碰撞率0%、超时率{rate('TD','safety baseline','timeout'):.2%}。此前文档的约29.5%碰撞、62.1%超时是两臂合并口径，不能当作“只跑教师”的失败分解。\n\n"
 for v in ['V0','V1','V3']:
  q=a1['comparisons'][v];text+=f"{FULL[v]}减去对应联合教师的成功率配对差为{fmt(q['student_minus_teacher']['lower'],True)}；14个教师成功率之间的Pearson相关为{q['Pearson']:.3f}、Spearman相关为{q['Spearman']:.3f}。"
 text+='\n\n10个global教师及其学生单独成功率均为零，因此global组内相关系数未定义；相关性主要反映少数local教师的差异，不能解读为广泛的预测能力。Pearson bootstrap存在零方差的未定义重抽，未用0填补；有效次数在JSON中列明。教师弱与学生弱相关，不是“学生能力不可能超过教师”的数学上界证明。\n\n'+(HERE/'ANALYSIS1_TEACHER_CEILING.md').read_text()+'\n![教师与学生成功率散点](figures/analysis1_teacher_student_scatter.png)\n\n'
 text+='## 分析二：单独运行的失败与方向标签\n\n'
 text+='专家32/32方向一致。四组学习型控制器只跑时，四机器人全部同向的比例仍为86.8%–93.9%；失败局不一致只有2.75%–5.85%。方向一致不是完成任务的充分条件。教师529次碰撞中，中心障碍271次、外环215次、机器人间43次，几何边界碰撞占91.9%。原学生的机器人间碰撞较多，但只有42.1%的机器人间碰撞满足预注册的迎面速度定义，且其中具有相反累计绕行方向的比例更低。\n\n'+(HERE/'ANALYSIS2_NOMINAL_FAILURE.md').read_text()+'\n![绕行标签和失败比例](figures/analysis2_nominal_modes.png)\n\n'
 text+='## 分析三：安全超时的位置与时长\n\n'
 text+='安全层将教师的529次原碰撞全部转为超时；原学生、闭环重标注学生、闭环重标注加共享随机种子学生相应为321、335、244局。超时局最后一次显著累计进展的均值约23.3–27.1秒（horizon35秒），不是普遍从早期就完全不动。严格低速停滞段与仍运动但无进展必须分开：后者的均值约48.5–51.5机器人·秒，前者只在少数局存在。\n\n'
 text+='原碰撞参与机器人与安全超时最终未到达机器人均可配对时，最终位置距离原碰撞位置≤0.5m的比例为教师15.3%、原学生25.1%、闭环重标注学生19.5%、共享随机种子学生20.6%；多数不是停在原碰撞点附近。0.25/1.0m敏感性、距离分位数及原碰撞同一时刻的位置差均保留。停滞开始时刻和逐段机器人索引见SECONDARY_ANALYSES.json与features/。\n\n'+(HERE/'ANALYSIS3_SAFETY_STALLS.md').read_text()+'\n![停滞时刻和最终位置](figures/analysis3_stall_locations_timing.png)\n\n'
 text+='## 分析四：修正的行为变化\n\n'
 for g in ['TD','V0','V1','V3']:
  e=early['early'][g]['S500']['100'];q=tr[g]['S500'];text+=f"{FULL[g]}：时间表相对只加安全层，前100步全队目标总距离额外减少{fmt(e['progress_difference'])}m，早期方向一致率差为{fmt(e['consistent_rate_difference'],True)}；被救回超时局中已有方向一致的比例为{fmt(q['already_consistent_among_rescued'],True)}。\n\n"
 text+='对原先不一致/静止的小子集，修正确实常使标签转为一致，但该小子集不能解释整体成功率的大幅提升。带时间表后仍可成功地采用不同绕行方向：联合教师249/753个成功局为不一致，原学生103/844、闭环重标注学生42/799、共享随机种子学生45/784。因此数据支持“更有效的目标推进”，不支持“全队必须更早统一方向”这一单一机制。\n\n'
 text+=(HERE/'ANALYSIS4_CORRECTION.md').read_text()+'\n![修正后的进展和到达分布](figures/analysis4_correction_progress_arrival.png)\n\n'
 text+='切换前后固定窗口为400–500步与500–600步；真实观察到600步的子集有存活者偏差。该子集中，闭环重标注学生的进展速度变化为'+fmt(a4['switch']['V1']['observed_600_post_minus_pre'])+'m/s，其他三组学习型控制器的CI均覆盖零；不能宣称切换普遍加速。全部局的到达分布在切换点附近出现集中完成，不能被仅分析600步仍未结束者的速度均值解释。固定第13组修正参数对照的差之差及吸收终止敏感性均在JSON中。3,648对全程第13组与时间表的进展和方向采样在500步以前完全一致，保证没有把切换前行为差异误归因于时间表。\n\n![切换窗口](figures/analysis4_switch_window.png)\n\n'
 text+='## 分析五：零动作控制器揭示的贡献\n\n'
 text+='零动作控制器加时间表只有28.1%成功；其失败局中60.9%方向一致，30.4%甚至四机器人全部同向。它在前100步的全队进展为5.68m，与原学生加时间表的5.64m相当，但至300步分别为5.95m与15.42m。学习型控制器对持续推进和完成任务有贡献，单纯“先选定绕行方向”解释不够。这仍不能精确分离网络的哪种内部特征产生贡献。\n\n'+(HERE/'ANALYSIS5_ZERO_CONTROLLER.md').read_text()+'\n![零动作与学习型控制器](figures/analysis5_zero_controller.png)\n\n'
 text+='已有round30d还记录了直奔目标控制器加分布式安全层及一直开启0.10m/s顺时针规则的32/32成功；本轮零动作消融不能据此宣称方法优于所有手工控制方案。\n\n'
 text+='## 分析六：闭环重标注与共享随机种子的局限\n\n'
 q=a6['paired']['V3_minus_V1'];text+=f"共享随机种子相对闭环重标注学生：成功率差{fmt(q['success']['lower'],True)}；碰撞率差{fmt(q['collision_rate_difference'],True)}；超时率差{fmt(q['timeout_rate_difference'],True)}。不一致局从41/896降到22/896，差值{fmt(q['mixed_rate_difference'],True)}，CI覆盖零，不能说已确认减少方向冲突。\n\n"
 text+='互斥碰撞计数显示，闭环重标注学生为机器人间35、中心216、外环83、多类型1；共享随机种子学生为25、163、56、0。碰撞减少主要来自几何边界接触减少，不能只归因于机器人间模式协调。\n\n'
 text+=f"全部配对局中，移动无进展时长差为{fmt(q['moving_no_progress_robot_seconds_difference'])}机器人·秒；碰撞减少使轨迹观察更久，存在暴露时长差。双方均超时、均观察到700步的428对中，差值为{fmt(q['moving_no_progress_difference_both_timeout'])}机器人·秒，CI覆盖零。可说失败从碰撞更多转向超时，不能断言同等观察时长下运动停滞强度显著增加。\n\n"+(HERE/'ANALYSIS6_RELABELING_SHARED_NOISE.md').read_text()+'\n![共享随机种子的失败转移](figures/analysis6_shared_noise_tradeoff.png)\n\n'
 text+='## 分析七：分布式学生线论文总表\n\n完整宽表为[PAPER_TABLE.md](PAPER_TABLE.md)，可复算长格式为[PAPER_TABLE.csv](PAPER_TABLE.csv)。每格包含成功率上下界与CI、碰撞率、超时率、unknown、样本数、来源轮次和DEVELOPMENT。缺失格不补造证据，不将原参数右手规则冒充公平调参规则。\n\n![论文总表热图](figures/analysis7_paper_heatmap.png)\n\n'
 text+='## 限定与偏差\n\n所有标签此前已打开，全部结果为DEVELOPMENT；没有把CI覆盖零解释为等效性证明。绕行标签只表示轨迹累计角位移，不等于控制器意图；几何相向pair同样只是诊断。转一致的条件分母、均超时子集与600步存活子集均有选择条件，已单列样本数。专家为特权参照，零动作控制器的两个seed逐位相同，只有32个独立state。相关性中10个global教师均零成功，组内相关未定义。\n\n额外计算了一项未预注册的“按观测时长归一化移动无进展”敏感性，已标为POST_HOC_DEVELOPMENT_SENSITIVITY并保留在附录数据，未用于正文机制结论或图表；见ANALYSIS_DEVIATION.json。正文关于暴露时长的判断仅使用原预注册的全局与双方均超时比较。\n\n本轮五个基础臂缺口及缺失的冻结critic选择均保持为缺；没有读取额外场景或推断未运行臂。结论限于已有四机器人Ring场景、14个冻结教师及其学生，不推广到其他场景或机器人数量。\n'
 (HERE/'REPORT.md').write_text(text)
 findings=dict(utc=utc(),status='COMPLETE_WITH_EXPLICIT_HISTORICAL_GAPS',integrity_passed=True,evidence_level='DEVELOPMENT',role_records=29024,unique_trajectories=28128,new_rollouts=0,training_steps=0,DB_opened=False,DB_writes=0,main_findings=dict(nominal_mode_conflict_not_dominant=True,teacher_nominal_success=rate('TD','nominal only'),teacher_nominal_collision=rate('TD','nominal only','collision'),teacher_nominal_timeout=rate('TD','nominal only','timeout'),safety_timeout_mostly_not_low_speed_immobility=True,correction_improves_early_goal_progress=True,correction_does_not_generally_increase_early_circulation_consistency=True,zero_controller_initial_direction_insufficient=True,shared_noise_collision_reduction=q['collision_rate_difference'],shared_noise_timeout_increase=q['timeout_rate_difference'],shared_noise_mixed_rate_change=q['mixed_rate_difference'],shared_noise_moving_stagnation_same_timeout_exposure=q['moving_no_progress_difference_both_timeout']),receipt_checks=receipt_checks,metric_audit=errors,source_integrity='INTEGRITY_COMPARISON.json',preregistered_primary_results='results.json',additional_analysis_deviation='ANALYSIS_DEVIATION.json');write(HERE/'CLOSEOUT_FINDINGS.json',findings)
 handoff='''本轮已完成分布式学生线的只读机制收尾。先读REPORT.md、CLOSEOUT_FINDINGS.json及PAPER_TABLE.csv。任务源为docs/prompt_round30g_student_line_mechanism_closeout_20261017.md；预注册和UTC/SHA256收据在本目录。全部DEVELOPMENT，不训练、不执行环境、不查询模型、不写共享DB；DB从未打开。

输入历史目录：decentralized_pipeline_v1、teacher_student_macflow_v1、teacher_student_macflow_v1b、teacher_student_macflow_v1c、teacher_student_macflow_v1d、teacher_student_macflow_v1f。开始/结束156,997文件的完整hash清单及Git状态、diff hash在INTEGRITY_START/END.json；比较见INTEGRITY_COMPARISON.json。并行任务新增git条目单列，非本轮所写。所有新增源码和输出限本目录，原目录不改。

核心入口：prepare.py只解析和配对已有标签/轨迹；metrics.py是纯轨迹函数；worker.py离线提取；analyze.py/stats.py做七项分簇统计；supplement.py导出到达分布、同碰撞时刻位置和逐教师配对差；paired_diagnostics.py做配对早期指标；transition_tables.py导出全模式×结果转移；reporting.py写七张表和论文表；plots.py画8组PNG/PDF；finalize.py封存交付。没有任何rollout或训练入口。不得重新执行历史runner作为分析捷径。

29,024条角色记录对应28,128条唯一轨迹，联合教师两行的只跑臂复用同一896条记录。MANIFEST.json逐行列出原始raw/npz路径、sha256、controller/state/seed与来源。features/shard*.jsonl保存唯一UID的全部逐局指标；EPISODE_METRICS.csv已展开为角色。分析1至7分别为analysis1_*至analysis7_*.json，results.json包含分析结果及链接；ANALYSIS_COMPLETE.receipt.json锁定原始聚合结果。最终综合判断在CLOSEOUT_FINDINGS.json，最终交付收据为DELIVERY.receipt.json。

重要结论：教师只跑碰撞59.0%，先前29.5%是与安全臂混合后的口径；名义失败大多已同向；安全超时多有运动和无进展，严格低速停滞少；修正提升早期目标进展，但不普遍提升早期同向率；零动作也能同向却推进不足；共享噪声减少碰撞、增加超时，但方向不一致率下降CI跨零，同等超时观察长度的运动无进展差CI也跨零。不能把timeout写成deadlock，不能把教师相关性写成数学上限。

所有CI=5000次controller再state簇bootstrap，seed2026101707，保留两seed/臂/变体配对；零动作两个seed逐位相同，不当64个独立state。unknown不重跑、上下界保留；缺失的集中式教师安全基线、教师公平右手规则及冻结critic等保持缺。额外未预注册归一化时长敏感性已单列ANALYSIS_DEVIATION.json并排除正文结论。

Python：/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；绘图用同父目录.venv-c1-plots/bin/python。PYTHONDONTWRITEBYTECODE=1，缓存/临时目录仅本目录。分析最多8CPU，无GPU。旧源码只读导入ring_exchange.expert.circulation_signature作32条专家参照测试，未调用规划或环境step。初始精度在全部导入后恢复x64=False；几何与统计使用numpy float64。专家缺少单列velocity，从已有actions恢复post-step速度；运动学误差上限2.22e-16m。

无需重跑任何轨迹。若要修改分析，先保存版本和新预注册，保留所有旧收据/原始证据；只读打开共享DB也必须file:...?mode=ro。任何后续源码修改仅限新获授权目录。
'''
 (HERE/'HANDOFF.md').write_text(handoff)
 # Source inventory for read-only project imports and each copied/reimplemented reference.
 imported={}
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and pathlib.Path(p).suffix=='.py' and pathlib.Path(p).is_file() and pathlib.Path(p).resolve().is_relative_to(ROOT) and not pathlib.Path(p).resolve().is_relative_to(HERE):imported[str(pathlib.Path(p).resolve())]=sha(p)
 write(HERE/'IMPORTED_SOURCE_AUDIT.json',dict(imported_project_sources=imported,read_source_references={str(ROOT/'ring_exchange/expert.py'):sha(ROOT/'ring_exchange/expert.py'),str(ROOT/'ring_exchange/environment.py'):sha(ROOT/'ring_exchange/environment.py')},copied_historical_files=[],reuse='Only read-only expert circulation_signature import; metric definitions reimplemented locally with original sources cited; no runtime function/global replacement'))
 write(HERE/'VALIDATION.json',dict(utc=utc(),frozen_receipts_valid=receipt_checks,features_complete=True,synthetic_tests=7,expert_trajectory_tests=32,statistic_edge_tests=4,kinematic_max_abs_error=errors['kinematic_max_abs_error'],prefix_consistency=secondary['prefix_consistency'],zero_seed_bitwise_checks=secondary['zero_deterministic_seed_checks'],required_tables_csv=[p.name for p in sorted(HERE.glob('ANALYSIS*.csv'))],figures_png=len(list((HERE/'figures').glob('*.png'))),figures_pdf=len(list((HERE/'figures').glob('*.pdf'))),source_integrity_passed=True,precision=precision(),DB_opened=False,new_rollouts=0,training_steps=0))
 seal('DELIVERY.receipt.json',[HERE/n for n in ['REPORT.md','HANDOFF.md','results.json','CLOSEOUT_FINDINGS.json','PAPER_TABLE.md','PAPER_TABLE.csv','INTEGRITY_COMPARISON.json','VALIDATION.json','PREREGISTRATION.receipt.json','ANALYSIS_DEVIATION.json']]+list((HERE/'figures').glob('*')),evidence_level='DEVELOPMENT',new_rollouts=0,training_steps=0,DB_writes=0);write(HERE/'DELIVERY_STATUS.json',dict(utc=utc(),status='COMPLETE_WITH_HISTORICAL_GAPS',receipt='DELIVERY.receipt.json',source_integrity_passed=True));print('delivery complete',utc(),flush=True)
if __name__=='__main__':main()
