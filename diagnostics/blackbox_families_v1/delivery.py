"""Final prose with full controller/method names and explicit evidence gaps."""
from common import *
from analysis import FULL_NAMES,METHOD_NAMES
import subprocess

def compare():
 begin=read(HERE/'INTEGRITY_BEGIN.json');end=read(HERE/'INTEGRITY_END.json');differences={}
 for section in ['basin','macflow']:
  a=begin[section].copy();b=end[section].copy()
  if section=='basin':
   for v in [a,b]:v['status']='\n'.join(line for line in v['status'].splitlines() if 'diagnostics/blackbox_families_v1' not in line)
  if a!=b:differences[section]=dict(begin=a,end=b)
 for section in ['poc_py','source_py']:
  a=begin[section];b=end[section];diff={k:dict(begin=a.get(k),end=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
  if diff:differences[section]=diff
 imports=read(HERE/'IMPORTED_SOURCES.json');supplemental=[x['path'] for x in imports['sources'] if x['round_begin_sha256'] is None]
 out=dict(utc=utc(),passed=not differences and not supplemental and not imports['uncovered'] and not imports['mismatches'],protected_start_end_snapshots_match=not differences,historical_supplemental_imports_match=not imports['mismatches'] and not imports['uncovered'],strict_round_start_import_coverage_gap=supplemental,differences=differences,excluded_round_directory=str(HERE),all_sampled_protected_old_files_unchanged=not differences);write(HERE/'INTEGRITY_COMPARISON.json',out);return out

def main():
 integrity=compare();r=read(HERE/'results.json');e1=r['endpoint1'];e2=r['endpoint2'];e3=r['endpoint3'];prefix='完整性审计通过' if integrity['passed'] else f"严格完整性审计未完全通过：{len(integrity['strict_round_start_import_coverage_gap'])} 项间接导入源码缺少本轮开工逐文件快照，仅与更早的不可变源码快照核对一致；已采样的保护源码、仓库状态与差异哈希起终一致" if integrity['protected_start_end_snapshots_match'] else '完整性审计未通过，保护文件差异见 INTEGRITY_COMPARISON.json';rg=e2['expanded_regret_pp'];p4=e3['pilot4_regret_pp'];lead=f"{prefix}。族终点一：{e1['capability_pass_families']} 个族的六个实例均通过观测能力门；其中 {e1.get('implementation_valid_families',e1['capability_pass_families'])} 个族满足实现要求并纳入确认性评估，{e1['supported_families']} 个族的按比例切换时间表相对只加安全投影基线的配对增益置信区间下界大于零。族终点二：零样本扩展选择模型的完整菜单交叉拟合平均遗憾的未知上下界为 {fmt(rg)} 个百分点，可能伤害数 {e2['harm_possible']}，条件一{status(e2['a'])}；条件二因前提未满足而{status(e2['b'])}。族终点三：四个先导状态（每个控制器八条闭环标签）选择的平均遗憾的未知上下界为 {fmt(p4)} 个百分点，{status(e3['status'])}。"
 summary=[]
 for family in FULL_NAMES:
  cap=r['families'][family].get('family_capability','GAP');pref=r['preferences'][family]['preference']
  if cap=='GAP_IMPLEMENTATION':summary.append(f"{FULL_NAMES[family]}：六个实例已通过观测能力门，实现单元测试失败，偏好为缺口。")
  elif pref=='未分出可靠偏好':summary.append(f"{FULL_NAMES[family]}：能力门通过，样本均值偏持续右手规则，但尚无可靠偏好证据。")
  else:summary.append(f"{FULL_NAMES[family]}：能力门通过，偏好{pref}。")
 lead+=''.join(summary)
 gaps=[];dedup=read(HERE/'STATE_DEDUP.json')
 if integrity['strict_round_start_import_coverage_gap']:gaps.append(f"起始导入来源审计覆盖缺口：{len(integrity['strict_round_start_import_coverage_gap'])} 项间接导入源码未在本轮开工快照单独采样。其历史不可变哈希与本轮记录一致，已采样源码起终一致，但不能追补当时快照。路径与对照见 IMPORTED_SOURCES.json；下一轮须在任何导入前覆盖 PoC 的 source、validation 和 vendor 依赖。无需因此重复训练或重跑本轮轨迹。")
 visibility=read(HERE/'STATE_DEDUP_VISIBILITY_REFINEMENT.json')['counts']
 if visibility['unrecoverable']:gaps.append(f"共享数据库中 {visibility['unrecoverable']} 个历史状态缺少可恢复的完整物理布局，无法证明与这些记录的物理去重；另有 {visibility['known_other_agent_count']} 个已知非四机器人状态，其数组布局与本轮不同，不属于重复风险。原始去重可见性计数过宽，细分证据见 STATE_DEDUP_VISIBILITY_REFINEMENT.json；全部可恢复且相容的状态及本轮状态彼此去重已通过。")
 for family,out in r['families'].items():
  if out.get('family_capability')=='GAP_IMPLEMENTATION':gaps.append(FULL_NAMES[family]+'：观测能力门六个实例通过，但初始直线单元测试只覆盖轴向路径，斜向空地图测试偏离超过一个栅格，违反实现要求；该族确认性评估与偏好为缺口。保留已认领的二百四十四条工程评估证据，二千八百二十八条未认领任务已跳过；没有取消在途任务、改控制器或重新选择。已另写工程修正参考及斜向路径、时空占用测试，位于 corrected_astar_reference.py 与 CORRECTED_ASTAR_ENGINEERING_TESTS.json；它从未用于本轮冻结执行，不能替代本轮证据。补齐仍需要通过完整单元测试，并在任何新的确认性标签之前重新预注册、探测、调参和冻结该族选择；不得用现有五族结果补选参数。')
  elif out.get('family_capability')!='PASS':gaps.append(FULL_NAMES[family]+'：通过能力门的实例少于四个或执行来源受阻，族结论不确定；不替换实例或调参。')
 if r['unknown_count']:gaps.append(f"五个有效族的确认性评估有 {r['unknown_count']} 条未知事件（按八臂表格计数，物理身份去重数另见审计）；全部保留并报告成功率上下界，没有重跑，不能据此断言已知失败。")
 if e2['b']=='INCONCLUSIVE':gaps.append('未满足至少一个新族的固定参数持续右手规则高于按比例切换时间表至少十个百分点的前提，因此零样本识别偏好族的条件二不确定。')
 for family,pref in r['preferences'].items():
  if pref['preference']=='未分出可靠偏好':gaps.append(FULL_NAMES[family]+'：固定参数持续右手规则与按比例切换时间表的配对差置信区间跨零，不能给出可靠偏好。')
 lines=[lead,'','全部新代码与输出位于本轮目录，数据库仅使用官方追加登记、提交前缓存检查、缺失项执行、默认 journal、单进程官方 merger 与完成后检查。没有 Git 写操作、包安装或访问被禁止的并行研究目录。所有选择收据早于主评估。超时不等同死锁；成功率相对只加安全投影基线的改善是本轮修正效果的指标。','','固定参数持续右手规则独立重算为按进度触发、四十步窗口、每秒零点三五米，三个历史组等权平均成功率下界为百分之三十四点八二一四。原冻结选择模型（A3）沿用历史预处理和参数，扩展选择模型新增持续右手规则动作；报告后续均使用完整名称。','','每族与全部方法的成功率及配对置信区间见 AXIS_TABLE.md 和 AXIS_TABLE.csv；事件、平均与最大遗憾、灾难数、伤害数、未知上下界见 METHOD_METRICS.json。逐条只执行控制器动作到只加安全投影的转移见 NOMINAL_TO_SAFETY_TRANSITIONS.json。合并结果仅含五个实现合格族。完整菜单按状态奇偶交叉拟合，另报样本内最佳臂。交叉拟合最佳臂由另一半状态选择，受有限样本影响，其评估值可低于预先固定的方法，遗憾因此可为负。四个 pilot 状态意味着每个控制器两臂各四次、合计八条用于选择的闭环标签；本轮能力门仍完整使用十六个先导状态，八条不是整个筛查与探测流程的总成本。各样本量数值与向量曲线见 PILOT_LABEL_COST_CURVE.csv 与 PILOT_LABEL_COST_CURVE.svg。','']
 for family,out in r['families'].items():
  if out.get('methods'):
   ci=out['paired']['schedule_minus_safety']['CI95_pp'];comp=out['paired']['right_hand_minus_schedule'];lines.append(f"{FULL_NAMES[family]}：按比例切换时间表减只加安全投影的配对差置信区间为 {ci[0]:.2f} 至 {ci[1]:.2f} 个百分点；固定参数持续右手规则减按比例切换时间表为 {fmt(comp['estimate_pp'])} 个百分点，置信区间 {fmt(comp['CI95_pp'])}；最佳菜单臂为{r['preferences'][family]['descriptive_best_menu_arm']}。")
 lines+=['','预注册的描述性成本曲线中，两个、四个、八个和十六个先导状态的合并平均遗憾未知上下界分别为零点八三至一点五一、零点八九至一点五六、零点八九至一点五六和零点八九至一点五六个百分点；所测最小规模两个状态、四条选择标签已达到平均三个百分点门槛，主分析仍为四个状态。曲线不是严格单调，也不保证逐控制器或逐族达标。扩展选择模型与原冻结控制器特征选择模型对三十个有效控制器均选择按比例切换时间表，加入持续右手规则菜单没有改变本次选择，因此不能据平均遗憾门槛通过而声称已证明识别出新族何时应选持续右手规则。扩展选择模型的最大控制器遗憾为二十一点八八个百分点，四个先导状态选择的最大控制器遗憾未知上下界为十八点七五至二十一点八八个百分点；平均门槛通过不保证每个控制器都接近最佳臂。','']
 lines+=['','缺口清单：']+['- '+x for x in gaps]+['','历史证据复用、训练来源与可复用性见 HISTORY_AUDIT.json；冻结的十八个学习实例配置、训练曲线、checkpoint 与收据位于 policies/。历史探测特征复用与新探测特征见 PROBE_FEATURE_REPORT.json。扩展模型的留一族检验只作 DEVELOPMENT（开发性）分析，见 LEAVE_ONE_FAMILY_DEVELOPMENT.json。','','执行记录保留了两项传输/登记修正：后续阶段固定使用原始实验登记身份；历史流匹配复现的场景别名修正仅重新登记并 journal 原轨迹，零物理重跑。优先级时空规划的直线路径单元测试不足由补充斜向测试发现；根据预注册的受阻族规则排除该族确认性分析，原封存选择和统计函数不变，排除未使用评估结果，详见 IMPLEMENTATION_GAP_ANALYSIS_AMENDMENT.json 与 ASTAR_OBLIQUE_UNIT_TEST.json。原分析脚本 analysis.py 保持封存；五个有效族的结果由 analysis_valid_families.py 调用同一份封存统计函数生成，封存选择未改变。原拟定的六组旧控制器探测未提交，已改为复用旧特征；对应旧设计保留为未执行记录。扩展选择模型的首次入口在训练前因原控制器与独立 checkpoint 特征身份不一致而退出，修正为原控制器的历史探测特征后才开始拟合，没有主评估标签参与。详见 REGISTRATION_IMPLEMENTATION_NOTE.json、FLOW_TRANSPORT_CORRECTION.json、PROBE_REUSE_PLAN_NOTE.json 与 EXPANDED_PREFIT_CORRECTION.json。提交过的入口版本逐字存档并与提交收据校验，见 SUBMITTED_REVISION_ARCHIVE.json。']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'GAPS.md').write_text('\n'.join('- '+x for x in gaps)+'\n');write(HERE/'GAPS.json',gaps)
 hand=[lead,'','工作根目录：/home/zhihan/research/Basin_C1。用户要求全部本轮写入位于 diagnostics/blackbox_families_v1/；其他既有源码只读。不得读取或写入 diagnostics/decentralized_blackbox_planners_v1/。不做 Git 写操作，不安装包。Python 为 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。','','设计依据为 docs/prompt_round38_more_blackbox_controllers_rhr_menu_20261017.md 和 docs/generalization_axes_protocol_20261015.md；PREREGISTRATION.md 与对应收据早于本轮新执行。固定参数持续右手规则、十八个学习实例、全部选择与公平调参规则均有冻结收据。所有选择先于主评估，不能用主评估重新选择方法或参数。','','数据库实验身份为 exp_blackbox_families_v1。各 phase 的 *_PLANNED_ROLLOUTS.json、*_CACHE_PREFLIGHT.json、*_PREFLIGHT.receipt.json、*_INSERTED_ROWS.json、*_REGISTRATION.receipt.json、*_MERGE.log 和 *_POSTFLIGHT.json 串起官方流程。未知直接身份仍在数据库，官方 planner 会把它们列为缺失；使用 *_POSTFLIGHT_UNKNOWN.json 证明已存在，不得重跑。旧场景别名的首次流匹配 journal 只被归为摘要；修正身份后原轨迹全部入库，原记录保留。','','控制器实现：集中式随机采样模型预测控制逐字复用 legacy_controllers.py；交叉熵方法在 controllers.py；优先级时空规划在 spacetime_astar.py；三种学习策略在 learned.py，训练入口 train_policies.py。持续右手规则的类逐字来自原实现，见 RIGHT_HAND_CLASS_SOURCE.json。按进度切换逐字复用 progress.py。episode.py 仅替换控制器原始动作来源和新增修正臂，物理、安全层、事件与成功标准均使用保护源码。时空搜索的等价编译加速在 fast_astar.py 与 astar_kernel.cpp 中，二十四条优先级路径与原实现逐位一致，这仅验证编译加速等价，并不证明原算法满足直线要求；搜索顺序、平局规则、几何及控制参数均未调整。最初 Python 作业继续完成，没有取消或重跑已认领任务。编译命令与版本见 FAST_ASTAR_BUILD.json。','','表格结果与逐条证据入口为 AXIS_TABLE.md、METHOD_METRICS.json、EVALUATION_LABELS.npz、EVALUATION_RAW_INDEX.json 和 NOMINAL_TO_SAFETY_TRANSITIONS.json；五千次配对控制器/状态簇 bootstrap 在 analysis_statistics.py。所有 unknown 保留下/上界；超时不是死锁。族终点一至三为前瞻 CONFIRMATORY（确认性）证据，留一族与训练标签分析为 DEVELOPMENT（开发性）证据。','','当前统计入口为 analysis_valid_families.py，原 analysis.py 与统计函数保留封存。优先级时空规划通过了上游观测能力门，但斜向直线路径单元测试失败，因此合并仅包含其余五族三十个控制器、一万五千三百六十条完整八臂评估；该族二百四十四条工程评估已保留，二千八百二十八条未认领身份按 MAIN_BLOCKED_UNCLAIMED.json 跳过。不要因原始 manifest 仍将这些身份列为缺失而补跑，也不得把它们伪装成数值未知。工程修正参考从未用于本轮执行，补齐该族需要新身份、预注册与选择冻结。','', '资源限制是用户全部任务合计十八个 CPU shard、两个 GPU shard；每次提交前保存 squeue，不可无条件启动新的十八路数组。没有视频交付。','','缺口与补齐条件：']+['- '+x for x in gaps]+['','完整性审计比较见 INTEGRITY_COMPARISON.json。补充导入来源的起始比对使用早于本轮的不可变源码快照，未在本轮初始快照重复采样，范围逐项列于 IMPORTED_SOURCES.json；最终逐路径、冻结收据与数据库覆盖核查见 FINAL_VERIFICATION.json。']
 (HERE/'HANDOFF.md').write_text('\n'.join(hand)+'\n');print(lead)

def fmt(x):
 if x is None:return '缺口'
 return f'{x[0]:.2f}' if x[0]==x[1] else f'{x[0]:.2f} 至 {x[1]:.2f}'
def status(x):return {'SUPPORTED':'通过','REJECTED':'未通过','INCONCLUSIVE':'不确定','PASS':'通过','GAP':'缺口','GAP_IMPLEMENTATION':'六个实例已观测通过，但实现单元测试失败、确认性评估为缺口'}.get(x,x)
if __name__=='__main__':main()
