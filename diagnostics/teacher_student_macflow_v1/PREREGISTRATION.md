# Round30 预注册

UTC: 2026-10-15T06:40:16.268228+00:00

本次续跑以当前任务文档为规范；旧版阻塞产物原样保存在 attempt1_blocked/，归档 hash 已校验。当前尚无学生训练、能力门、probe、评估标签。教师为 DEVELOPMENT 参照；学生新标签满足流程时 TS1–TS3 为 CONFIRMATORY。

首先执行噪声一致性门：按 round28 controller_order，对每个教师的实际训练 manifest 顺序中的前 72 个 train rollout，取第 0 个数据 state，key=fold_in(PRNGKey(2026103030+teacher_index),sample_index)，合计 1,008 个。检查 traced nominal 内部断言、独立噪声逐位一致、native 经原 to_world 与再限速和非 traced nominal 逐位一致。任何一项失败立即停下报告，不改容差、不绕过、不训练。此门只采样动作，不执行环境 step，因而不产生 rollout 标签。

14 个教师与实际训练数据路径、checkpoint/config/manifest hash 和训练 seed 见 TEACHERS.json。所有 manifest hash 与各教师训练配置匹配。全部训练参数见 TRAINING_CONFIG.json，训练 80,000 步、batch 256、每 500 步验证固定 8 批，以 dev 蒸馏均方误差选 checkpoint。每个教师一个学生，一个 seed，不重训、不替换。每个 state 的速度由初始速度与后续保存位置差/dt 重建，对比已存 actions/observations 并报告差异；教师观测始终由 runtime.observation 重新计算。浮点重建若无法满足原 reset 接口则停止报告，不裁剪或改变物理 state。

网络采用官方 ActorVectorField（512,512,512,512，actor_layer_norm=False，lr=3e-4），输入 own local_observation 行、id one-hot、own 2D noise。共享参数。不加 critic 或 Q 项。教师冻结事后蒸馏，区别于官方联合训练；教师为展平联合 flow，区别于官方逐机器人处理。学生动作与教师 raw 同空间、同物理单位；按各教师实际 frame 走原 to_world，再限速。当前文档将 raw 统称局部坐标，但全局教师实际 to_world 为恒等映射，按原 runtime 保持语义。

学生部署不通信，仅知道自身目标；噪声以(seed,robot_id)派生独立流，时间表只用机器人自身时钟。只读调用 round28 D-local 安全实现；RHR 参数冻结为 .01m/s、40 连续步、.15m/s 顺时针扰动。学生七臂 nominal only、safety baseline、F10、F13、S300、S500、B-RHR；教师只补 nominal only 与 D-local safety baseline 的精确缺失，其余复用 round28，不运行专家。

评估固定 round28 EVAL_STATES.json 的 32 state，seed 0–1。pilot 候选范围 202610300000–202610300999，先与 DB 和 round28 保留范围/物理 state 去重取前16，封存去重清单后才运行。能力门 safety baseline/S500，seed0；两臂均<15%或均>98%排除；少于8个学生整轮 INCONCLUSIVE，不替换、调参、重训。

通过能力门者运行 R5：历史固定16 probe states × η10/η13 ×200步，不提前终止。采用 round16 冻结预处理/PCA/A3/A0，完整菜单交集=[F10,F13,S300,S500]，固定菜单=[F10,F13]。记录 G_E、always-S500、always-F10。选择和收据必须在主评估标签前封存。不得直接运行含 monkey patch 的 round27 脚本；只能在新目录重写，依赖只读。

缓存与 DB：每阶段先历史审计、manifest、官方 preflight、封存，再提交缺失任务；unknown 保留不重跑。workers 使用默认 cache_writer.append_journal 路径；官方 merger 单进程合并；不改任何模块全局变量。源码完整性除两个仓库 git 状态/diff 外，还包含 PoC poc/全部.py hash 及每个 import/复制源码 hash。全部 JAX 导入后显式恢复 x64=False。资源合计上限18 CPU shard、2 GPU shard；本轮不要求视频。

统计：先 controller、再 state 成簇配对 bootstrap 5,000 次。unknown 下界记0，另报上界；不称为已知失败。TS1：S500−safety baseline 的均值 CI 下界>0，另报 nominal 碰撞到 safety 成功/timeout/unknown 转移。TS2：最佳时间表−最佳固定均值 CI 下界>0，且≥70%学生满足最佳时间表≥最佳固定−1pp。TS3a：A3 相对四臂 oracle 平均遗憾≤3pp且 A3−safety baseline CI 下界>0；报告与 A0/G_E/S500/F10 配对差。TS3b：至少3个学生 F10−F13≥10pp且至少3个反向≥10pp，否则 INCONCLUSIVE；满足前提则 A3遗憾−A0遗憾 CI 上界<0 判通过，并报最佳固定一致率。每次 bootstrap 内重新求最佳固定/时间表及 oracle。TS4：教师学生最佳臂一致率、四臂 Spearman 与η类型预测。TS5：逐臂学生对教师 D-local/集中TT 配对差、安全与停滞诊断、单核预热后学生逐机器人/教师10步/QP耗时；专家引用32/32与.075s特权参照。

结论映射严格采用任务第六节；任一安全层臂碰撞优先报告并为所有成功率结论附前提。timeout 不是 deadlock，停滞仅诊断。未达到前置门则不得给 TS1–TS3 统计结论。完整任务规范的版本 SHA256 见收据。
