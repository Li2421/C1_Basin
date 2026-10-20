# 修法选择前瞻确认交接

规范为 docs/prompt_round30i_fix_selection_prospective_20261017.md，全部新代码/证据在本目录；历史、PoC、官方MACFlow及Python环境只读，DB仅官方流程追加。先读REPORT.md首段，再看PREREGISTRATION、FROZEN_RULES、results.json与FINAL_VERIFICATION。

源码完整性审计未通过，官方共享数据库包与默认journal路径断言通过；开始识别5个并行任务目录，其变化单列且本轮未导入或写入。确认一（主规则优于任何单一固定修法）通过：主规则成功下界86.64%，最佳固定修法为前500步η13、之后η10的时间表，差11.75个百分点，配对95%CI [7.73, 15.25]；确认二（主规则优于一律时间表）通过，差11.75个百分点，CI [7.73, 15.25]。各族选择情况为闭环重标注及共享随机种子分布式学生：主规则相对本族最佳固定类型准确率100.00%，相对本族最佳固定臂0.00个百分点；独立训练种子联合MACFlow教师（分布式安全层）：主规则相对本族最佳固定类型准确率100.00%，相对本族最佳固定臂0.00个百分点；新配方共享观测高斯行为克隆策略：主规则相对本族最佳固定类型准确率100.00%，相对本族最佳固定臂0.00个百分点；新参数分布式模型预测控制：主规则相对本族最佳固定类型准确率83.33%，相对本族最佳固定臂-12.76个百分点；新设置A*规划controller：主规则相对本族最佳固定类型准确率66.67%，相对本族最佳固定臂-11.46个百分点。探索性两级规则在分布式模型预测控制上相对主规则差0.00个百分点，CI [0.00, 0.00]，未证明解决右手规则配置选择问题。主终点为CONFIRMATORY，两级规则为EXPLORATORY。 资源硬性检查未通过：评估提交后，外部并行作业进入运行，使用户总CPU shard峰值达到24，超过18；首次记录2026-10-17T16:23:19.598441+00:00，末次记录2026-10-17T18:30:59.998344+00:00。Slurm拒绝scontrol挂起请求（账户权限不足），该次尝试未改变任何进程；之后另设仅对本轮作业发送STOP/CONT的尽力限流，保留进程/RNG/物理状态，不取消或重跑；STOPPED作业保留分配但不再消耗运行CPU计数；这不能抹去先前超限记录，资源总审计仍失败。资源事件与原始squeue见RESOURCE_INCIDENT、RESOURCE_AUDIT，信号记录见RESOURCE_THROTTLE（若有事件）。资源日志已早于实际干预记录超限，进度监测未及时处理报警是本轮流程缺陷。提交前检查通过不抵消运行期间的超限，完整交付不得宣称全部协议检查通过。 完整性整体失败的原因：起始已记录的受保护文件无变更，但另一个未列入起始例外的目录diagnostics/crosstype1_learned_zoo_selector_v1/在运行期间新增51个Python源码文件。未事后扩充例外清单；这些新增文件未被本轮导入或写入，最终import核验单列。Basin受检查HEAD/diff、MACFlow全部git快照及PoC/已记录历史文件均一致。

58实例：独立seed联合教师14；闭环重标注学生14与共享随机种子学生14（合为学生族）；新Gaussian4；新MPC6；新A*6。checkpoints与参数在CONTROLLERS，学生/教师原名映射在profiles与TEACHERS。训练仅四个Gaussian，原始train/dev和固定dev选权重，旧controller不重训。主阈值0.01、20米、右手规则甲；两级附加阈值0.05（高于用甲，否则乙），只拟合旧round30h。

32新state生成种子区间[2026101720000,2026101720999]；DB全部uid/content和可解析物理签名去重，旧legacy缺物理元数据范围明确记录，精确目标Ring场景无未解析项。16诊断probe沿用round30h，不读取成败；928条安全300步、额外512条旧critic无标签R5；42实例R5复用。全部特征、所有选择先于23360条任务：22272条六臂加1088条旧critic选中的F10/S300对照。该额外对照不扩充主菜单或oracle。

程序顺序：integrity BEGIN → preregister → launch_initial（GPU训练及三门各16）→ finalize_controllers → test_instances → probe_design → submit diagnostic/probe → freeze_features_choices → design/preflight → submit main → postflight官方merger → analyze → plots → extra_dependency END及integrity END → report → verify。coordinator协调后半流程，不应重复提交已完成任务。unknown保留，不重跑；timeout不当deadlock。复制的worker/pipeline/runtime/fair/student/astar等核心逐字保留；support仅预载不同冻结实例。

本轮阈值使不修正分支不可达：平均初始全队目标距离19.9192646米就是平均进展理论上限，小于20米。不得把成功终点解释为已确认“不修正”能力。旧critic的14教师R5协议与D-local不同，原菜单对照单独解释。两级只探索性，不替代主确认。原学生/教师谱系相关，结论不外推新场景/机器人数量。

源代码完整性：主BEGIN按初始提交收据哈希保留；normalizer stage首次读取前另做完整起始审计；并行目录由初始squeue/24h/untracked识别，未导入或写入，变化单列。UNIT_SUBMISSION_AUDIT披露唯一单实例测试作业squeue在提交后采集的流程差异。官方DB路径在所有入口/worker导入后断言；JAX x64=False。所有plot仅读证据；无视频要求；无git写命令。

最终资源检查因历史超限失败，整体源码审计因51个未列入起始例外的外部新增源码失败；已记录旧文件与实际导入文件另核对，不能把统计终点通过写成全部协议通过；见RESOURCE_INCIDENT、RESOURCE_SUMMARY、RESOURCE_THROTTLE与QUALIFIED_DELIVERY.receipt（如已生成）。原verify.py保留严格失败门槛，不通过资源检查就不会生成passed=True的DELIVERY。绘图入口修复范围见PLOTTING_AMENDMENT及patch，原封存plots.py与分析不改。只读核验已有收据不得重新提交实验。需要新实验必须另行预注册、查DB、只跑缺失，不能通过重跑消除unknown或修改历史源文件来通过审计。全部原始case/raw/trace/UID/hash在ANALYSIS_MANIFEST；controller×六臂总表标明上下界与选择；完整统计在results和5000 paired bootstrap draws。
