# v1最终交接：按用户决定收尾、转入分布式v2

本节是最终状态，优先于下方保留的历史运行说明。v1不再提交任何新批次，COORDINATOR_STOP永久保留；不得重启协调器或恢复圆形流程版claims。学习型controller与扩展均未启动。

流程版结论仅包含窄通道、二维拥挤随机、门口三个完成场景；合并结果在MIGRATION_RESULTS.json的three_scene_fixed_combined中。只有窄通道与门口完成场景内学习的独立评估，二维拥挤随机因8机/14机几何缺口没有学习结果；three_scene_learned_combined为null。四场景合并未计算。

圆形流程版：保留682条已落盘raw、trace与journal，以及48000条完成的单机能力筛选；8918条未落盘请求直接丢弃，不记unknown、不恢复。部分样本不用于估计迁移效果。见CIRCLE_FLOW_USER_STOP.json。

paper_bvc_v2仅是v1内论文版缓冲Voronoi单元修正分支的名字；两个原worker保持原样直到完成。它不等于即将开始的分布式流程v2。论文版四场景的已有记录均保留，完整比较见PAPER_REPRODUCTION。

最终入口：REPORT.md、GAPS.md、AXIS_TABLE.md、DB_POSTFLIGHT.json、INTEGRITY_COMPARISON.json、FINAL_NUMERIC_RESULTS.receipt.json、DELIVERY_VALIDATION.receipt.json和V1_FREEZE.receipt.json。严格全量开工审计缺少两个非Python配置的整文件开工hash，不能补造；已使用参数视图与保护源码的比较单列。

下一阶段执行docs/prompt_baseline_literature_scenes_distributed_v2_20261020.md，v1目录在v2中只读。特别注意：v1的experiment.py导入会把TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、CUDA_CACHE_PATH指向v1；v2不得直接导入后让缓存写回v1，应按新prompt的源码复用规则处理。仅读取旧raw/trace/checkpoint或复制带来源hash的新文件。

最新用户资源授权：若没有其他任务占用，全用户合计最多24个CPU份额、5个GPU份额；每次提交前检查squeue并计入所有现有任务。v1收尾没有利用新增额度启动任何研究批次。

下方仅保留构造接口、数据路径、执行语义及历史排障信息，原恢复命令不是继续v1的授权。

---
# 四个文献场景流程迁移交接

本目录是`docs/prompt_baseline2_four_literature_scenes_pipeline_migration_20261020.md`对应的baseline2开发实验。所有新增源码、独立依赖、轨迹、模型和报告都在本目录；唯一外部写入是官方共享DB的追加登记、默认journal与单进程merger。不做Git commit。查看`STATUS.md`/`COORDINATOR_STATE.json`区分实际完成、精确复用和待计算；最终结果入口是`REPORT.md`，不能将预注册计划数当完成数。

## 构造与代码

|场景|构造|论文版|流程版差异|
|---|---|---|---|
|窄通道场景（lit_narrow_passage）|三机器人、无墙，state0由400dpi图8首帧读数；63个±.05米起点扰动|`paper.impc`；`paper_bvc_corrected.bvc_corrected`|单积分器、默认集中投影；保留已到达机器人；图中1.8秒不等于真正t0|
|二维拥挤随机场景（lit_crowded_random）|`official_sampler.py`保留GUI两个采样方法原函数体；每N100组|同上，N=2/4/6/8/10/12/14|2米方区硬墙，700组中28组有效；不补抽、不缩半径|
|圆形围场五机随机场景（lit_circle_five）|五个半径2米圆盘，中心面积均匀拒绝采样于半径9米圆盘|`paper.circle`；`paper_secondary.circle`|外切128边形硬墙；原论文为软圆约束；速度界.37365109585434214米/秒|
|门口场景（lit_doorway）|`states.walls`的两段紧包矩形，64组起点|`paper.doorway`调用原MPC类|单积分器、真实机器人联合避碰；无限外侧绕行；40.4秒|

`states.config(scene,N)`、`states.walls(scene)`与`flow_scene.LiteratureEnv(state,config)`构造流程环境。原始状态在`PAPER_STATES.json`、`FLOW_STATES_UNCHECKED.json`，有效集合在`FLOW_STATES_<scene>.json`；几何、单机栅格路径与容量下界在`GEOMETRY_VALIDATION_<scene>.json`。联合可解未形式认证；规划器粗网格无终点cell不等于物理几何不可达。

`flow_runtime.py`保存规范执行语义：名义控制→速度界/到达停止→原认证投影→OrthoFlow3→同一投影→原物理执行。`flow_runtime_certified.py`使用精确缓存与原网格边谓词无入边证书，语义相同。`GRID_CERTIFICATE_PROOF.json`和严格字节回放收据说明加速范围。`controllers.py`、`certified_planning.py`适配八类、48个实例；参数在`CONTROLLER_MANIFEST.json`。不修改已有controller或保护源码。

论文版使用本目录`.venv-ext/`，官方代码commit与全部文件hash见`EXTERNAL_SOURCES.json`。没有修改外部算法；`external_patches/`保留为空表示不需要接口补丁。缓冲Voronoi单元首版误加末速度0的701条作为变体保留；修正分支为不同语义，见勘误二。门口与文献交叉口的模拟器在两次3步调用间不重置，实际50轮30秒；原raw的15秒/150步字段错误，必须使用`TIME_METADATA_CORRECTION.json`，不要改写或重跑原记录。

官方IMPC的`uav.get_dynamic`先覆盖连续A，再在计算B时使用已经离散的A，实际输入矩阵不等于正确Tustin，也不同于论文Euler。`OFFICIAL_DYNAMICS_AUDIT.json`直接验证初始化矩阵，没有生成控制轨迹。官方ρ=.2、r_min=.335和η更新也与论文参数说明不同；全部保留原代码，在参数/复现报告中明确差异，不能宣称严格按论文方程复刻。

## 学习、统计与背景

`LEARNING_MENU.json`在标签之前密封。`learning_design.py`生成32训练/32评估不相交状态、每类清单前两个实例与十二动作请求；`learn_critic.py`沿用原不看controller的架构与训练规则，八类留一，内层仅用源类型选择checkpoint。`freeze_scene_choices.py`在评估preflight之前密封逐状态动作。`result_inventory.logical`按精确UID解析复用结果；同一执行语义的旧标签不会因critic输入变化失效。

冻结零样本流程的四机圆环观测/上下文与RingEnv恢复在四个新场景均无定义，记不适用。二维拥挤随机场景预定8机/14机学习状态均无效，保留缺口。确定性行为克隆策略、混合密度行为克隆策略各两seed是第二优先级新试点，代码在`pilot_*`；只复用既有成功冲突搜索多机规划训练轨迹。若通过95%单机门，`learning_extended.py`/`learn_critic_extended.py`创建独立十类型补充，不覆盖八类数据或冻结选择。

`failure_analysis.py`只读NPZ分类与重算九组敏感性；`analyze_baseline.py`逐场景生成基线与描述性几何普查。`migration_statistics.py`做5000次类型→controller→state配对bootstrap，seed0/1保持同簇；unknown给下界/上界。`result_inventory.py`逐场景汇总，`aggregate_results.py`汇总四场景固定修正及有学习数据的场景子集。完整四场景学习数据不足时留空，不把子集合并冒充四场景。`background_results.py`报告名义输入到安全投影的逐局转移，白盒参照独立列出。

数学安全的条件、投影平衡锥与充分进展条件在`THEORY_CONDITIONS.md`。本轮critic未强制统一任务进展裕量；不声称普遍死锁解除或N=100能力。

## 运行与恢复

流程Python只读使用`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`；论文Python使用本目录`.venv-ext/bin/python -B`；图最后使用指定`.venv-c1-plots/bin/python -B`。所有入口先导入`experiment.py`，关闭bytecode并把缓存/TMP放入本目录，JAX x64=False。不要在已有环境安装包。

共享DB实验ID为`exp_baseline2_literature_scenes_v1`。每批`*_CASES.json`、`*_PLANNED_ROLLOUTS.json`、`*_CACHE_PREFLIGHT.json`与`*_PREFLIGHT.receipt.json`共同界定身份和缺失集合。`registry.py`只追加登记，前后哈希验证旧登记行不变；worker只写默认journal，不直接插入rollout。`db_merge.py`有本轮单进程锁，调用官方merger。

对既有未完成批次，先查看作业日志并区分基础设施、实现错误与数值unknown；只对基础设施留下的缺失请求执行`stream_queue.py recover <stage>`后再`submit.py <stage> flow_certified.sbatch <CPU数>`。完整raw与unknown始终跳过。`stream_queue.py build <stage>`仅在尚无队列收据时执行。新请求登记前先调用`db_merge.py`让已完成journal对preflight可见。不可重复运行已封存的design脚本覆盖收据。

`coordinator.py`是本轮单一调度入口，按依赖推进且调用`submit.py`。提交锁重新查询全部用户作业，最多16个Slurm CPU份额、2个GPU份额，为本地协调留2个CPU；总量保持用户18/2上限。`COORDINATOR_STOP`只停止调度器，不取消worker。恢复前检查是否已有协调器，避免启动第二个训练调度进程。状态至少每15分钟写一次；所有研究unknown不因重启而重试。

原始大数据、checkpoint、SQLite与来源不明历史证据均保留。最终审计与postflight只能在worker/登记工作结束后进行；报告中注明未实现项、未复现论文证据和审计覆盖限制。视频未请求，不交付视频。


### 圆形场景的等价计算捷径

`controllers_interior.py`是本轮`controllers.py`的有来源复制，只增加NumPy float64整批预测点在保守内圆时的精确零墙代价返回，随机抽样、代价的其余项、参数与算法不变。`interior_nominal.py`组合原证书规划器，`flow_runtime_interior.py`保持规范执行链。16条五机完整旧轨迹覆盖8类全部关键数组逐字节一致，见`CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json`；同一旧位置上的配对计时及部署收据另存。原canonical DB身份不变，实际backend写入raw。调度器v6只在部署收据存在时对圆形场景切换`flow_interior.sbatch`，已完成标签不重跑，未落盘中断与claims恢复均留记录。`coordinator.py`优先提交新依赖阶段，之后才把空闲CPU补给已有原子队列；仍以全部用户squeue限制16个Slurm CPU并预留2个本地份额。
