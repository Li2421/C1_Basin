from support import *
from analyze import ARM_FULL
import collections
G=guard()
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:.2f}'
def span(x):return f'[{pp(x[0])}, {pp(x[1])}]'
def main():
 r=read(HERE/'results.json');a=read(HERE/'INTEGRITY_COMPARISON.json');parallel=read(HERE/'PARALLEL_TASK_DIRECTORIES.json');e=r['endpoints'];p=r['policies'];rules=r['source_rules'];branch=read(HERE/'NO_CORRECTION_BRANCH_FEASIBILITY.json');familytext='；'.join(FULL[f]+f"：主规则相对本族最佳固定类型准确率{pct(p['primary']['families'][f]['type_accuracy'])}，相对本族最佳固定臂{pp(p['primary']['families'][f]['difference_from_family_best'])}个百分点" for f in FAMILIES);secondary=e['secondary_minus_primary_DM'];opening=f"源码完整性审计{'通过' if a['passed'] else '未通过'}，官方共享数据库包与默认journal路径断言通过；开始识别{len(parallel['directories'])}个并行任务目录，其变化单列且本轮未导入或写入。确认一（主规则优于任何单一固定修法）{'通过' if e['confirmation1']['status']=='PASS' else '未通过'}：主规则成功下界{pct(p['primary']['success_lower'])}，最佳固定修法为{ARM_FULL[r['best_pooled_fixed_arm']]}，差{pp(e['confirmation1']['difference'])}个百分点，配对95%CI {span(e['confirmation1']['CI'])}；确认二（主规则优于一律时间表）{'通过' if e['confirmation2']['status']=='PASS' else '未通过'}，差{pp(e['confirmation2']['difference'])}个百分点，CI {span(e['confirmation2']['CI'])}。各族选择情况为{familytext}。探索性两级规则在分布式模型预测控制上相对主规则差{pp(secondary['difference'])}个百分点，CI {span(secondary['CI'])}，{'支持在下一轮以两级规则为主方法确认' if secondary['status']=='PASS' else '未证明解决右手规则配置选择问题'}。主终点为CONFIRMATORY，两级规则为EXPLORATORY。"
 resource_rows=[json.loads(x) for x in (HERE/'RESOURCE_AUDIT.jsonl').read_text().splitlines()];bad=[x for x in resource_rows if not x['within_cap']]
 resource_note=('资源硬性检查未通过：评估提交后，外部并行作业进入运行，使用户总CPU shard峰值达到'+str(max(x['cpu'] for x in resource_rows))+'，超过18；首次记录'+bad[0]['utc']+'，末次记录'+bad[-1]['utc']+'。Slurm拒绝scontrol挂起请求（账户权限不足），该次尝试未改变任何进程；之后另设仅对本轮作业发送STOP/CONT的尽力限流，保留进程/RNG/物理状态，不取消或重跑；STOPPED作业保留分配但不再消耗运行CPU计数；这不能抹去先前超限记录，资源总审计仍失败。资源事件与原始squeue见RESOURCE_INCIDENT、RESOURCE_AUDIT，信号记录见RESOURCE_THROTTLE（若有事件）。资源日志已早于实际干预记录超限，进度监测未及时处理报警是本轮流程缺陷。提交前检查通过不抵消运行期间的超限，完整交付不得宣称全部协议检查通过。' if bad else '资源记录未发现超过18 CPU shard或2 GPU shard。')
 source_note=('完整性整体失败的原因：起始已记录的受保护文件无变更，但另一个未列入起始例外的目录diagnostics/crosstype1_learned_zoo_selector_v1/在运行期间新增'+str(len(a['new_external_files']))+'个Python源码文件。未事后扩充例外清单；这些新增文件未被本轮导入或写入，最终import核验单列。Basin受检查HEAD/diff、MACFlow全部git快照及PoC/已记录历史文件均一致。' if not a['passed'] else '')
 opening+=' '+resource_note+' '+source_note
 lines=[opening,'','## 冻结流程与证据','','主规则只用round30h旧五族数据拟合，A='+str(rules['primary']['A'])+'、B='+str(rules['primary']['B'])+'米，右手分支为'+ARM_FULL[rules['primary']['right_hand_arm']]+'；探索性两级保持A/B不动，C='+str(rules['secondary']['C'])+'，冻结比例高于C用右手规则甲，否则用右手规则乙。目标为五个旧族等权成功下界最大，网格和最早平局规则完全继承。旧round30h的全仓审计有外部未导入文件变化，其实际使用源码与标签证据核验通过；本轮依据已保存的旧标签/诊断哈希拟合，未复用其外部文件。','','冻结顺序已按收据核验：预注册与两套规则→四个Gaussian训练完成并冻结58实例→928条安全诊断及全部R5特征冻结→主、次要和旧critic选择冻结→任务design、preflight、提交→任务标签→单进程官方merger→postflight→分析。三道门round28、round33、round30e各16条，位置/速度/动作dtype、shape、bytes及步数/终点全部相同。58个实例的确定性、限速、有限值和精度测试通过。','','58实例×32全新状态×2种子×6臂='+str(r['primary_six_arm_records'])+'条主菜单记录；加旧critic选中的全程η10或300步时间表对照，共'+str(r['total_task_records'])+'条任务。未知主菜单标签'+str(r['unknown_six_arm'])+'条，全部任务未知'+str(r['unknown_all_tasks'])+'条，均保留且未重跑，成功主结果按下界计，另报上界；unknown不是失败。诊断928条不读取任务成败，不写任务DB。旧critic复用42个实例已有R5，只为16个新实例补512条旧协议无标签R5。','','32状态用原round28生成器和LocalFrameConfig、新种子候选区间[2026101720000,2026101720999]，按顺序接受前32个；与全部DB uid/content及可解析物理签名去重。旧DB有4607个没有可解析物理状态的legacy记录，均不属于本轮精确Ring scenario uid；未知场景元数据不能被当成物理可比证据。STATE_DEDUP_AUDIT与STATE_UNRESOLVED_METADATA_AUDIT保留详细范围，不声称恢复了那些legacy物理状态。','','## 两个确认终点','','配对分层controller→state bootstrap共5000次，seed2026101710，两个seed始终成簇。每次重抽内重新选六固定臂中最好的一个。主合并每个controller等权，另报五族等权。','', '| 终点 | 平均差（百分点） | 配对95%CI（百分点） | unknown保守包络 | 结论 |','|---|---:|---:|---:|---|']
 for k,label in [('confirmation1','确认一：主规则减最佳单一固定修法'),('confirmation2','确认二：主规则减一律500步时间表')]:
  v=e[k];lines.append(f"|{label}|{pp(v['difference'])}|{span(v['CI'])}|{span(v['unknown_adversarial_CI'])}|{v['status']}|")
 lines.extend(['','| 五族等权补充 | 差（百分点） | 95%CI |','|---|---:|---:|'])
 for k in ['confirmation1','confirmation2']:lines.append(f"|{k}|{pp(e[k]['family_equal_difference'])}|{span(e[k]['family_equal_CI'])}|")
 methods=ARMS+['primary','secondary','old_critic','family_oracle','controller_oracle'];methodnames={**ARM_FULL,'primary':'冻结的三分支主规则','secondary':'冻结的两级规则（探索性）','old_critic':'冻结通用critic（仅原菜单）','family_oracle':'每族事后最优固定臂','controller_oracle':'每controller事后最优臂'};lines.extend(['','| 策略 | 成功下界—上界 | 层级95%CI | 对六臂controller事后最优的平均遗憾（百分点） |','|---|---:|---:|---:|'])
 for k in methods:lines.append(f"|{methodnames[k]}|{pct(p[k]['success_lower'])}—{pct(p[k]['success_upper'])}|{span(p[k]['CI'])}（百分点）|{pp(p[k]['oracle_regret'])}|")
 lines.extend(['','上述oracle仅含六臂；原通用critic可选全程η10和300步时间表，若其动作超出六臂，遗憾是相对六臂oracle的差，不必非负。原critic未重新训练；其14教师描述符沿用历史联合流探测协议，其他44实例为分布式协议，域差异明确保留，只作对照。','','## 每族是否选对','','| 黑盒controller族 | 只加安全层 | 全程η13 | 500步时间表 | 右手规则甲 | 右手规则乙 | 右手规则丙 | 主规则 | 两级规则 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
 for f in FAMILIES:lines.append('|'+FULL[f]+'|'+'|'.join(pct(p[k]['families'][f]['success_lower']) for k in ARMS+['primary','secondary'])+'|')
 lines.extend(['','| 黑盒controller族 | 最佳固定修法 | 主规则相对本族最佳固定类型准确率 | 允许并列最优类型准确率 | 主规则减族最佳固定臂（百分点） | 配对95%CI | 主规则灾难数 | 两级灾难数 |','|---|---|---:|---:|---:|---:|---:|---:|'])
 for f in FAMILIES:
  q=p['primary']['families'][f];b=p['secondary']['families'][f];lines.append(f"|{FULL[f]}|{ARM_FULL[q['best_fixed_arm']]}|{pct(q['type_accuracy'])}|{pct(q['type_accuracy_allow_tied_best'])}|{pp(q['difference_from_family_best'])}|{span(q['difference_from_family_best_CI'])}|{q['catastrophe_count']}|{b['catastrophe_count']}|")
 supplement=read(HERE/'SUPPLEMENTARY_DESCRIPTIVES.json')['families'];lines.extend(['','### 事后描述性口径澄清','','预注册类型准确率以本族最佳固定臂类型为参照；预注册灾难数以本族最佳固定臂的族平均为参照，因此即使某族全部选了同一最佳臂，难实例也可能计入灾难，不能解释为选择错误。为区分实例难度与选错，另报相对每个controller自身最佳臂的类型准确率，以及同一个controller上相对本族最佳固定臂低超过10个百分点的数量。以下仅事后描述，不更改冻结终点、原始结果或选择。','','| 黑盒controller族 | 主规则对逐实例最优类型准确率 | 允许逐实例并列类型 | 主规则同实例配对灾难数 | 两级同实例配对灾难数 |','|---|---:|---:|---:|---:|'])
 for f in FAMILIES:
  q=supplement[f]['policies']['primary'];b=supplement[f]['policies']['secondary'];lines.append(f"|{FULL[f]}|{pct(q['type_accuracy_vs_individual_best'])}|{pct(q['type_accuracy_allow_individual_ties'])}|{q['paired_controller_catastrophes']}|{b['paired_controller_catastrophes']}|")
 lines.extend(['','类型为不修正、时间表类（全程η13和500步时间表）、右手规则类。最佳类型平局按预注册菜单顺序，并另报允许任一并列类型的敏感性。灾难为某controller成功率比本族最佳固定臂族平均低严格超过10个百分点；unknown可能范围见results.json。类型准确不保证选对具体右手规则配置，必须同时看成功率与遗憾。','','## 探索性两级规则','','| 探索性比较 | 差（百分点） | 配对95%CI | 结论 |','|---|---:|---:|---|'])
 for k,label in [('secondary_minus_primary','全体：两级减主规则'),('secondary_minus_primary_DM','分布式模型预测控制：两级减主规则')]:lines.append(f"|{label}|{pp(e[k]['difference'])}|{span(e[k]['CI'])}|{e[k]['status']}|")
 changed_mpc=[n for n,a0 in p['primary']['families']['DM']['choices'].items() if a0!=p['secondary']['families']['DM']['choices'][n]]
 lines.extend(['','两级规则在分布式模型预测控制族改变了'+str(len(changed_mpc))+'个实例的右手配置（'+', '.join(changed_mpc)+'）；该族配对差和全部bootstrap抽样差均为0，改变配置未转化为本轮评估成功率提升。'])
 lines.extend(['','右手规则甲与乙不仅速度不同，触发方式及绕行方向参照也不同。这一比较检验两个既定配置的选择，不能单独归因于速度强弱。MPC差CI下界>0才按预注册称明显更好；即使通过也仅探索性，下一轮需要以两级为主规则重新确认。','','## 诊断与不修正分支','','固定探测集的平均初始全队目标距离为'+str(round(branch['mean_initial_team_goal_distance'],6))+'米。目标距离非负，所以平均300步累计进展的理论上限也是这个值，严格低于冻结阈值20米。因此不修正分支在本探测协议下不可达；这是几何与定义的直接推论，不是根据任务标签调整规则。主规则实际使用时间表与右手规则甲两个分支。本轮不得由成功终点通过推断“已经学会何时不修正”。','','| 黑盒controller族 | 冻结比例范围 | 进入右手分支实例数 | 不修正实例数 | 时间表实例数 | 冻结比例距0.01不超过0.02的实例数 |','|---|---:|---:|---:|---:|---:|'])
 for f in FAMILIES:
  d=r['diagnostic_distributions'][f]['freeze_proportion'];b=r['diagnostic_branch_counts'][f];lines.append(f"|{FULL[f]}|{pct(d['min'])}–{pct(d['max'])}|{b['right_hand']}|{b['no_correction']}|{b['schedule']}|{b['within_002_freeze_threshold']}|")
 lines.extend(['','新旧七特征分布见figures/new_old_diagnostic_distributions.*。冻结比例、运动无进展与进展只是无标签描述，不是deadlock标签；timeout不能称deadlock。对系统性选择失误，以下只解释已冻结分支与真实表现的关系，不作因果识别，也不改变阈值。',''])
 for f in FAMILIES:
  q=p['primary']['families'][f];b=r['diagnostic_branch_counts'][f];d=r['diagnostic_distributions'][f]['freeze_proportion'];best=q['best_fixed_arm']
  if q['difference_from_family_best'] < -1e-12:
   lines.append(FULL[f]+'：主规则低于本族最佳固定修法'+ARM_FULL[best]+'，差'+pp(q['difference_from_family_best'])+'个百分点。冻结比例范围'+pct(d['min'])+'–'+pct(d['max'])+'，阈值0.01将'+str(b['right_hand'])+'个实例送入右手规则甲、'+str(b['schedule'])+'个送入时间表；其中'+str(b['within_002_freeze_threshold'])+'个距阈值不超过0.02。'+('最佳固定臂同属右手规则，失配涉及具体右手配置，类型准确不足以保证配置正确。' if best in RULES else '该族的固定臂比较说明当前冻结特征分支与最优修法并非完全一致。'))
 lines.extend(['','## 新实例与训练记录','','本轮确认的是五个已见族内的新实例与新状态，不是未见controller族的泛化。教师14个为round14A原配方独立训练seed，学生28个为round30f最终第三轮固定20000步checkpoint；与round30h的权重sha均不同。闭环重标注学生与共享随机种子学生各14个在同一学生族内；这些学生此前在旧状态上有过评估，确认性来自冻结规则和全新32状态，不是声称此前完全没有任何学生标签。','','| 新高斯行为克隆实例 | 配方 | train seed | 固定训练步数 | dev选中步数 | dev最优NLL |','|---|---|---:|---:|---:|---:|'])
 for g in read(HERE/'GAUSSIAN_CONFIGS.json'):
  ss=read(HERE/'gaussian'/g['name']/'SUMMARY.json');lines.append(f"|共享观测高斯行为克隆策略（{g['name']}）|{g['recipe']}|{g['seed']}|{g['steps']}|{ss['best_step']}|{ss['best_dev_nll']:.6g}|")
 lines.extend(['','Gaussian严格复用原train/dev数据、normalization、训练函数与对应Flow训练步数，只用固定dev NLL选权重，没有以任何闭环成功率选checkpoint。全部train/dev哈希与训练曲线在gaussian/；没有打开expert test split。','','| 新模型预测控制实例 | 预测步长 | 样本数 | 避碰权重 |','|---|---:|---:|---:|'])
 ctrl=read(HERE/'CONTROLLERS.json')
 for i in range(6):
  v=ctrl['profiles'][f'DM{i}']['payload']['nominal_parameters'];lines.append(f'|分布式模型预测控制（DM{i}）|'+ '|'.join(map(str,v))+'|')
 lines.extend(['','| 新A*实例 | 栅格分辨率（米） | 重规划间隔（步） | 1.5米内邻居视为障碍 |','|---|---:|---:|---|'])
 for i in range(6):
  v=ctrl['profiles'][f'AS{i}']['payload']['nominal_parameters'];lines.append(f"|A*规划controller（AS{i}）|{v[0]}|{v[1]}|{'是' if v[2] else '否'}|")
 lines.extend(['','## 完整性、并行任务和流程差异','','起始全树范围见INTEGRITY_BEGIN与integrity.py的HIST；PoC、实际复用结果/训练目录及所有非并行Basin已有Python源码均受保护。旧critic的物理normalizer所在stage另在首次读取前完成NORMALIZER_STAGE_BEGIN，结束逐文件核对。准备阶段保留INTEGRITY_PREPARATORY，正式使用的BEGIN由INITIAL_SUBMISSION_PREFLIGHT按哈希封存，没有以新快照覆盖旧收据。实际import的全部源码单独核对；并行目录不允许被本轮import或写入。',''])
 for q in parallel['directories']:lines.append('- '+q['path']+'：'+q['reason'])
 lines.extend(['','并行目录变化见INTEGRITY_COMPARISON.json；Basin与MACFlow_Official的起止HEAD、status、diff哈希保留。未commit、stash、checkout、add或恢复外部文件；旧源码、物理、事件、成功标准、安全和右手规则实现均未改，COPY_AUDIT与COPY_DIFFS保存逐字复制/路径差异。所有缓存、GPU缓存、临时文件、图表均在本轮目录，官方DB流程是唯一外部写入。','','绘图环境是已有Python 3.12的NumPy/matplotlib环境，不包含JAX；原封存plots.py保留未改，复制为plots_render_cpu.py，仅移除绘图所不需要的JAX导入和自重启。PLOTTING_AMENDMENT与patch保留差异，父进程按协议恢复x64=False，子进程不执行JAX运算、控制器或统计。进程暂停只改变elapsed_seconds中的墙钟等待，不改变以仿真步计的物理/事件/成功标准。过程记录：唯一的单实例测试作业的squeue记录在提交后而非紧邻提交前采集，详见UNIT_SUBMISSION_AUDIT；其仅一步动作查询、零仿真步，使用1 CPU。其他训练/复现/探测/评估提交前均有squeue与资源预留收据。诊断初始作业每片8G，后续6G，均不影响数值语义。准备阶段额外扩展全仓快照的冗余进程已停止，原封存BEGIN未被替换；normalizer依赖在首次使用前单独完整记录。无重跑unknown。资源检查情况见首段与RESOURCE_INCIDENT，不能宣称运行期间始终满足资源上限。','','DB按design追加新身份、逐旧行哈希/rollout计数不变、官方preflight、只跑missing、默认官方journal、ps/squeue核验后单进程官方merger与postflight执行。保存全部插入行与sha；不直接SQL插入rollout，不UPDATE/DELETE已有行。','','## 结论与下一步','','两个确认终点的统计判定与完整协议验收必须分开：资源总占用超限，且未分类外部新增源码使整体完整性检查失败。本轮不签发passed=True的完整交付收据；所有结果保留并作限制说明，不重跑、不修改起始审计、不恢复他人文件。'])
 if r['conclusion']=='BOTH_CONFIRMATIONS_PASS':lines.append('两个预注册确认终点均通过，支持在规定五族新实例和新状态上，用一段不看成败的诊断零样本选择既定修法，作为分布式部分主方法的前瞻证据。该支持不包含“不修正”分支能力，因为它在本协议下不可达。')
 elif r['conclusion']=='ONLY_DEFAULT_BEATEN':lines.append('确认二通过、确认一未通过：主规则优于此前一律500步时间表的默认做法，但未证明优于事后最佳单一固定修法'+ARM_FULL[r['best_pooled_fixed_arm']]+'。应按每族表与阈值关系解释，不能把探索性两级结果替代主终点。')
 else:lines.append('主规则未获得预注册的前瞻支持。各族表现和阈值邻近统计显示哪些实例系统性选错，应据此提出下一轮预注册；本轮没有按新结果更改阈值、菜单或controller。')
 if secondary['status']=='PASS':lines.append('两级规则在分布式模型预测控制上明显更好，但这仍是探索性发现；下一轮应先冻结两级规则为主选择器再确认。')
 lines.extend(['','结论限固定Ring场景、4机器人、58个指定实例与两个future seeds；不外推任意新族、新地图或N=100。学生/教师谱系相关，右手菜单带有历史族调参信息，bootstrap条件于冻结规则，未把这些关联当作消失。','','## 交付索引','','PREREGISTRATION及收据；FROZEN_RULES与旧数据网格；ALL_CONTROLLERS_FROZEN；48条REPLAY_GATE；UNIT_TESTS；DIAGNOSTIC_FEATURES与R5及DIAGNOSTICS_FROZEN；ALL_CHOICES_FROZEN；main_CACHE_PREFLIGHT/POSTFLIGHT、registration/main与main_MERGE_RECEIPT；CONTROLLER_SIX_ARM_TABLE.csv；results.json与BOOTSTRAP_DRAWS；figures/；完整性、并行和import审计；FINAL_VERIFICATION与DELIVERY.receipt。'])
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');handoff='''# 修法选择前瞻确认交接

规范为 docs/prompt_round30i_fix_selection_prospective_20261017.md，全部新代码/证据在本目录；历史、PoC、官方MACFlow及Python环境只读，DB仅官方流程追加。先读REPORT.md首段，再看PREREGISTRATION、FROZEN_RULES、results.json与FINAL_VERIFICATION。

'''+opening+'''

58实例：独立seed联合教师14；闭环重标注学生14与共享随机种子学生14（合为学生族）；新Gaussian4；新MPC6；新A*6。checkpoints与参数在CONTROLLERS，学生/教师原名映射在profiles与TEACHERS。训练仅四个Gaussian，原始train/dev和固定dev选权重，旧controller不重训。主阈值0.01、20米、右手规则甲；两级附加阈值0.05（高于用甲，否则乙），只拟合旧round30h。

32新state生成种子区间[2026101720000,2026101720999]；DB全部uid/content和可解析物理签名去重，旧legacy缺物理元数据范围明确记录，精确目标Ring场景无未解析项。16诊断probe沿用round30h，不读取成败；928条安全300步、额外512条旧critic无标签R5；42实例R5复用。全部特征、所有选择先于23360条任务：22272条六臂加1088条旧critic选中的F10/S300对照。该额外对照不扩充主菜单或oracle。

程序顺序：integrity BEGIN → preregister → launch_initial（GPU训练及三门各16）→ finalize_controllers → test_instances → probe_design → submit diagnostic/probe → freeze_features_choices → design/preflight → submit main → postflight官方merger → analyze → plots → extra_dependency END及integrity END → report → verify。coordinator协调后半流程，不应重复提交已完成任务。unknown保留，不重跑；timeout不当deadlock。复制的worker/pipeline/runtime/fair/student/astar等核心逐字保留；support仅预载不同冻结实例。

本轮阈值使不修正分支不可达：平均初始全队目标距离19.9192646米就是平均进展理论上限，小于20米。不得把成功终点解释为已确认“不修正”能力。旧critic的14教师R5协议与D-local不同，原菜单对照单独解释。两级只探索性，不替代主确认。原学生/教师谱系相关，结论不外推新场景/机器人数量。

源代码完整性：主BEGIN按初始提交收据哈希保留；normalizer stage首次读取前另做完整起始审计；并行目录由初始squeue/24h/untracked识别，未导入或写入，变化单列。UNIT_SUBMISSION_AUDIT披露唯一单实例测试作业squeue在提交后采集的流程差异。官方DB路径在所有入口/worker导入后断言；JAX x64=False。所有plot仅读证据；无视频要求；无git写命令。

最终资源检查因历史超限失败，整体源码审计因51个未列入起始例外的外部新增源码失败；已记录旧文件与实际导入文件另核对，不能把统计终点通过写成全部协议通过；见RESOURCE_INCIDENT、RESOURCE_SUMMARY、RESOURCE_THROTTLE与QUALIFIED_DELIVERY.receipt（如已生成）。原verify.py保留严格失败门槛，不通过资源检查就不会生成passed=True的DELIVERY。绘图入口修复范围见PLOTTING_AMENDMENT及patch，原封存plots.py与分析不改。只读核验已有收据不得重新提交实验。需要新实验必须另行预注册、查DB、只跑缺失，不能通过重跑消除unknown或修改历史源文件来通过审计。全部原始case/raw/trace/UID/hash在ANALYSIS_MANIFEST；controller×六臂总表标明上下界与选择；完整统计在results和5000 paired bootstrap draws。
''';(HERE/'HANDOFF.md').write_text(handoff);print('REPORT / HANDOFF written',flush=True)
if __name__=='__main__':main()
