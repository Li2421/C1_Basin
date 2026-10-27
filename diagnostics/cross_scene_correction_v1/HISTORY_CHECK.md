# 历史检索与复用登记

首次核查阶段未训练、未生成状态或 rollout、未 import 历史模块、未提交 Slurm、未写数据库；用户修订后已恢复，当前进度见 REPORT.md 和 PROGRESS.json。以下“可复用”表示历史证据的适用范围；尚未逐请求运行官方 preflight，不宣称已完成精确标签复用。机器可读摘录见 `HISTORY_FACTS.json`，审计输入哈希见 `EVIDENCE_SHA256_START.json`。

| 历史路径 | 与本轮执行语义 | 可复用范围 | 缺口或冲突 |
|---|---|---|---|
| `diagnostics/axis3_cross_scene_v1/REPORT.md` | 不同：Flow、集中式安全层与集中式修正；报告另记 Toy 精度复现缺口 | 历史跨场景结论及局限，不重做旧训练 | 不能充当本轮文献线标签；原报告 XS1 是四场景 INCONCLUSIVE、可评估三场景子集 REJECTED，不能简写为四场景均已检验失败 |
| `diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/FINAL_REPORT.md` | 目标语义来源，包含静态/移动 user-fix 分支与圆形旧场景分支 | 已有候选、训练状态、控制器、raw/trace 和候选选择研究结论 | 未做本轮逐位重放；未把汇总报告等同于逐请求缓存核验 |
| 同目录 `ACTIONS.json`、`ceiling_common.py`、`ceiling_worker.py`、`ceiling_replay_circle.py` | 固定动作可精确定义；逐位兼容仍待验证 | 四场景均含所需 49 个固定候选，历史 16 的 float64 与 protocol 一致 | `register2` 的单进程metadata INSERT流程现已按用户修订准许；原有行逐表哈希保持一致 |
| `diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/`：`SCREEN_SELECTION_PRIMARY_20261023.json`、`old_fixed_train_CASES.json`、`REPLAY_CASES.json` | 窄通道/拥挤随机/圆形历史标签候选来源；静态/移动须沿用修订语义 | 六个场景各 16 个训练状态；历史 raw/trace 作为后续重放来源 | 窄通道和拥挤随机上一轮训练选择含 `centralized_mppi0`；圆形与三个八机才为 7 类。prompt 的统一 7 实例说明不符合历史选择 |
| `diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_*.json` | 物理状态和训练池可读；不能混用未修订的控制链标签 | 已检索原版/REVISED 划分及 training_controllers（每类多个候选实例） | 旧场景MPPI取舍已按用户修订固定；新场景仍需逐实例能力门 |
| `datasets/gap1_eta_scaling_v1/` | Flow + 集中式安全/修正，与本轮不同 | 本轮仅使用 design_manifest 地图配置和物理状态，旧标签不使用 | N=2/10/50；16 个旧候选与本轮文献线历史 16 精确交集为 0 |
| `datasets/gap1_eta_scaling_v2/` | 同上 | 本轮仅用地图配置和物理状态；旧标签不使用、不重复旧训练 | N=2/10/50；16 个候选精确交集为 0 |
| `datasets/gap1_eta_scaling_v3_n20/` | 同上 | 本轮仅用 N=2/10/20 地图配置和物理状态，旧标签不使用 | 16 个候选精确交集为 0 |
| `datasets/gap1_eta_scaling_v4_global128/` | 同上 | N=2/10/20 各 32 train 状态；可按 uid 排序读 N2 前8、N10 前12；Gap config 开口 .62m、r=.16、v=.5、dt=.05 | 全部128个候选与本轮历史16精确交集仍为0；不能按索引配对不同 eta 算排序相关 |
| `bottleneck_family/scenario.py`、`bottleneck_family/__main__.py` | 母版几何接口存在，尚未接入文献线执行 | `Config`、`Geometry`、`build_instance`、`preset`，墙与单机器人路径见证 | 五层新场景报告尚未执行；不将单机器人有路或 joint_feasibility=not_certified 当作联合可解/死锁解决；母版 max_steps 默认10000与指定数量时限需执行时明确登记 |
| `diagnostics/axis4b_agent_count_v1/REPORT.md` | 原 Gap 控制链，不同于本轮 | N=2/10/20/50 时限2000/8000/16000/40000、能力与五层证据的限制 | 原三臂能力门未通过不代表本轮文献线单机器人能力门失败；不重训旧 Flow、不把 timeout 当 deadlock |
| `shared_rollout_db/README.md`、`src/{planner,cache_writer,merger,historical_ingest,integration,cache_hook}.py` | 缓存规范必须遵守 | 官方 preflight、exact float64 eta、默认 journal 与单进程 merger 接口存在 | 精确新身份由单进程官方connect/transaction预登记；rollout仍只走默认journal与官方merger；禁止通用推断分支冒充MAC_ONLY |

旧执行环境 `experiment.py` 在 import 时会 mkdir 并设置旧目录的 TMPDIR/XDG/CUDA 等路径；原生 CBS 缓存头文件还硬编码旧 `runtime_cbs_query_cache`。已仅阅读源码，未 import 或调用；若后续解除停止条件，须在本轮目录创建隔离副本、记录来源 SHA，先通过逐位重放，不能直接调用旧 worker/main。

最近24小时有写入的其他目录为 `diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/`（扫描记录392475个文件）；见 OTHER_ACTIVITY_START.json。Slurm 开工记录显示其他用户 shaowen 的 RUNNING 作业28474目录 `/home/shaowen/nuPlan`，另有依赖等待作业28475。本用户无 Slurm 作业；见 SQUEUE_START.txt。上述历史目录仅文件读取，未 import、未写入。

## 用户修订后恢复

原有停止原因已由 PROTOCOL_AMENDMENT.md 明确解决：新身份登记沿用既有单进程INSERT流程；Gap旧标签不作训练或描述性排序；窄通道和拥挤随机保留8类；status按本轮目录排除规则核对；执行入口使用隔离副本。此前检查结论保留为历史过程。

## 固定地图任务前缀校核

按用户指定UID顺序的N2前8个状态，使用其原episode_seed与train分割调用build_instance，N2起点和目标均可逐位恢复；N4、N6采用这些相同episode_seed后，N2→N4→N6前缀逐位一致，地图、机器人半径和动力学不变。初版独立新种子的计划保留在design_correction_history/task_prefix_v1/；已完成的1条旧N4能力门证据保留并排除，不用于新计划。其余10110条能力门结果经官方合库/预检精确复用，只有16258个当前请求缺失。详见TASK_PREFIX_CORRECTION.json。

## 完成核验

26048个历史标签raw/trace事前哈希核验通过，逐请求官方预检复用计数见各阶段PREFLIGHT.receipt.json；最终数据库与完整性检查通过，见DB_POSTFLIGHT.json、INTEGRITY_COMPARE.json。旧Gap1标签没有进入训练矩阵。丁按用户最新要求保留107条已完成结果，13条停止项仅核对身份与保存点、不生成rollout标签；见USER_STOP_AMENDMENT.md。
