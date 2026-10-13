# 3D 无人机 pipeline 首轮移植结果

已接通 **真实 MACFlow → 三维硬安全投影 → 三系数 η 修正 → 真实 rollout 标签 → Q 选 η**，并完成一个分离 train/dev/test 初态的小型验证。结论分两部分：纯 Flow 的首次移植未成功；显式加入近目标位置伺服的 TT 组合控制器完成了闭环验证。不能把组合控制器的成功归为纯 Flow 成功。

测试初态 6 个，Q 选参成功 6/6，η=0 成功 0/6；但 Q 全部选择同一个 η，固定该 η 也成功 6/6，**尚无按状态选参优于固定参数的证据**。这是固定地图、N=2、理想速度跟踪的工程结果，不是 B15、飞行动力学、规模或跨控制器泛化结论。

## 从哪些旧证据继续

设计、几何可解性和联网原始文献见 [首轮设计记录](drone_3d_deadlock_design_20261013.md)。本轮沿用该地图、半径、动力学、碰撞检查、目标容差和死锁判据，未放宽成功标准。

| 旧证据/代码 | 本轮处理 | 新证据的必要性 |
|---|---|---|
| `diagnostics/drone_3d_pilot_v3/`，三个成功 teacher NPZ、三个死锁基线 | 原始 trace/hash 和基线结果直接复用；不重复生成示范 | 没有对应的三维已训练 Flow 闭环成绩 |
| `new_benchmark_common/macflow.py` | 直接复用 `JointMACFlowAgent`、官方 Flow 网络、训练及 checkpoint 接口 | 旧二维权重的动作/观测/环境不兼容；训练一个小型三维模型 |
| `shared_control/basis_families.py` | 保留 goal/perpendicular/relation 三列含义和原缩放，显式升维到 xyz | 退化共线状态无法凭空产生 y/z；本轮由学习 Flow 的侧向分量提供方向 |
| `Basin_C1_flow_field_poc_20261004/poc/field_controller.py` | 核对 FF 的冻结物理状态、固定 δ、十步场内投影语义 | FF 不能冒充 TT 的一次末端投影 |
| `diagnostics/physical_state_critic_formal_v1/train_ring_ff.py` | 复用 `StateCritic` 两层 64 宽网络，输入改为 24 维物理状态与 3 维 η | 二维 Q 标签不对应本控制器；后续仅改 Q 编码必须复用本轮标签 |
| 首轮 16 条纯 Flow TT rollout | 读取已存 NPZ 定位越过瓶颈后偏离目标；不重复跑旧 η | 明确缺少近目标/已到达状态监督，补 12 条真实末段 teacher 轨迹 |
| 第二个 Flow 的 8 条闭环测试与原模型的 3 条 FF 测试 | 全部保留失败结果 | 补数据后仍未成功，另建显式 terminal-servo controller 身份 |

每个阶段的 `protocol.json`、`history_audit.json`（末段 teacher 的审查在 protocol 内）、`planned_rollouts.json` 和 preflight 均保留。worker 写 journal 后单进程 merge；没有通过换 critic 编码使旧标签失效。

## 实际实现

- [pipeline.py](../drone_3d/pipeline.py)：每机观测 `[p, v, g-p, p_other-p]`，形状 `[2,12]`；动作 `[2,3]`。`v` 为上一步实际速度，单积分运动学不另加加速度状态。观测没有 teacher phase、计时器或 waypoint。
- [train_flow.py](../drone_3d/train_flow.py)：缓存示范训练真实 MACFlow，十步 Euler、标准 CFM 及 endpoint loss；训练/验证按完整示范分组。仅移除依赖 teacher 计时器的静止等待监督。
- [pilot.py](../drone_3d/pilot.py)：复用已验证的三维硬投影与 simulator。连续线段—实体盒、相对线段—球/椭球检查均由 simulator 执行。
- `[B_goal, B_perp, B_relation]` 形状 `[2,3,3]`，`δ=Bη`。正交列保留 `3.303687238760696` 缩放；共线时退化行为有测试，没有人为添加避让路线。
- **TT**：`u=P(P(u_Flow)+Bη)`。**FF**：冻结物理状态及 δ，在十步 Flow 积分内反复投影，最后只验收，不再加终端投影或裁剪。TT/FF 各自注册控制器身份。
- [terminal.py](../drone_3d/terminal.py)：单列 `terminal_servo_v1`，仅支持 TT。某机距目标小于 0.6 m、目标及直达线段通过墙面几何检查时，用 `bounded(0.8(g-p))` 替换该机 nominal；随后仍经相同安全投影、basis、η 和末次投影。没有瓶颈路线、让行 phase 或死锁触发 fallback。机间安全由同一联合投影及连续检查处理。
- [critic.py](../drone_3d/critic.py)：固定一个已注册 controller 的 `Q(s,η)`，控制器 UID 作为模型适用性条件，不能跨 controller 使用。输入为物理状态和 η；在 t=0 选 η 后整条轨迹保持不变。只读取真实闭环二值成功标签，数值错误不作负例；无正负对比时禁止训练。

本轮不修改二维共享控制器和四场景运行流程。三维环境 fingerprint、执行源码 hash、checkpoint hash、状态、精确 η 和 future seed 都进入缓存身份。

## 训练和失败结果

| 阶段 | 实际工作 | 结果 |
|---|---|---|
| `flow/` | 复用 seed17/23 的 852 个示范 transition；seed41 的 426 个用于验证；宽64、6,598参数、4,000步 | 训练约7.36 CPU秒，最佳验证 normalized endpoint MSE 0.003364 |
| `smoke_tt/` | 原 Flow，seed17，8 个 η | 8 timeout，0 success |
| `eta_refine/` | 原 Flow，seed17，再检查8个不同 η，不重训 | 8 timeout，0 success；多个已越过瓶颈但末段不达标 |
| `terminal_teacher/` | 明确末段数据缺口后，新增12条真实 goal+CBF 小轨迹 | 全部安全成功，共233个 transition；约0.147秒，不属于 Flow 成绩 |
| `flow_terminal/` | 原示范加末段监督；宽128、21,382参数、8,000步 | 约22.37 CPU秒；验证最佳在5,500步，MSE 0.002849 |
| `terminal_smoke/` | 第二个 Flow，seed17，8个 η | 6 timeout、2 deadlock，0 success；监督误差降低未转化为闭环成功 |
| `smoke_ff/` | 原 Flow，seed17，3个 η | 3 timeout，0 success |
| `hybrid_smoke/` | 原 Flow 加显式目标伺服，seed17，3个 η | η=0 timeout；`[0,.5,0]` 36.3秒成功；`[.5,.5,0]` 22.8秒成功 |

纯 Flow 共27条：25 timeout、2 deadlock、0 success。所有失败保留原终止原因，无碰撞或数值中止。并未在全失败标签上拟合一个虚假的成功 Q。

各 smoke manifest 是运行上限；实际只执行第一个示范初态，随后提前停止或转入独立新初态验证。未运行的60个请求仍标为缺失，不计入结果；原因见各目录 `execution_scope.json`。它们不是已完成的大 batch。

## Q 的独立初态验证

冻结原 Flow checkpoint `flow/best.pkl`，SHA256 前缀 `d1a6d6108768`，使用 `terminal_servo_v1 + TT`。三个候选 η 固定为 `[0,0,0]`、`[0,.5,0]`、`[.5,.5,0]`。

24个新初态各自独立扰动两机位置，扰动半幅为 x=0.08、y=z=0.015 m，目标交换。12个 train、6个 dev、6个 test；不复用示范初态作独立测试。每个 state–η 使用同一个 Flow future seed=0，η 间采用相同随机数。共72条真实 rollout，非72个独立初态，也不构成随机鲁棒率估计。

| η / selector | train，12初态 | dev，6初态 | test，6初态 |
|---|---:|---:|---:|
| `[0,0,0]` | 1/12 | 0/6 | 0/6 |
| `[0,.5,0]` | 12/12 | 6/6 | 5/6 |
| `[.5,.5,0]` | 12/12 | 6/6 | 6/6 |
| Q选参 | — | 6/6 | 6/6 |

Q仅用36条 train、18条 dev 标签训练1,200步。测试预测在测试 rollout 运行前落盘；runner 对 test 设置预测存在性及 controller 匹配门槛。测试选择结果通过已有精确 rollout 读取评估，不重复生成同一条选中轨迹。

Q在六个测试初态全部选择 `[.5,.5,0]`。按预定“训练成功率相同取首项”的固定基线是 `[0,.5,0]`，得到5/6；另一项训练并列最优的固定参数已经是6/6。因此不能只报告6/6对5/6后宣称自适应Q有效，也不能把η=0的六次timeout说成六次已认证死锁。

六条 Q 选中轨迹全部达到原 `0.04 m` 目标容差。最小墙面物理净距 `0.0201249 m`，最小机间排除椭球余量 `0.00900648`（无量纲）；所有18条测试轨迹均通过连续碰撞检查。

可复查入口：[Q评估](../diagnostics/drone_3d_pipeline_v1/q_panel/critic/evaluation.json)、[冻结预测](../diagnostics/drone_3d_pipeline_v1/q_panel/critic/test_predictions.json)、[训练记录](../diagnostics/drone_3d_pipeline_v1/q_panel/critic/fit.json)、[缓存回查](../diagnostics/drone_3d_pipeline_v1/q_panel/cache_postflight.json)。72/72均可精确复用，后续改 Q 表示无需新 rollout。

最终[证据审计](../diagnostics/drone_3d_pipeline_v1/evidence_audit.json)核对了114条新增记录的trace哈希、逐步运动学、记录中的连续安全余量、成功终态及死锁判据。预测文件比首条test记录早59.10秒。另在 `cache_view_check/` 验证异目录首次读取和再次读取均复用同三条原证据，两个调用均新增0条rollout；这些索引副本不算新的样本。原Q训练脚本快照与fit记录中的hash一致，后续缓存/报告接口修改没有覆盖原训练证据。

## 完整仿真视频与资源

- [同初态死锁基线与学习组合控制器，seed17](../diagnostics/drone_3d_pipeline_v1/videos/deadlock_show/seed17_full_comparison.mp4)：229帧；左边目标控制器13.7秒满足死锁判据，右边 Flow+η+位置伺服22.8秒成功。右侧η由开发测试指定，不冒称来自未见状态的Q选择。完整[manifest](../diagnostics/drone_3d_pipeline_v1/videos/deadlock_show/seed17_video_manifest.json)。
- [独立测试初态：η=0 与 Q选择，seed2026101318](../diagnostics/drone_3d_pipeline_v1/videos/q_selected/seed2026101318_full_comparison.mp4)：801帧，保留基线全部80秒直到timeout，并保留选中轨迹完整过程；终止后仅停留末状态。[manifest](../diagnostics/drone_3d_pipeline_v1/videos/q_selected/seed2026101318_video_manifest.json)。

视频只读取原 NPZ，逐物理步播放；没有合成、跳帧或改标签，画面坐标轴比例明确标注不等。两个视频属于单场景开发交付。该实验不是旧四场景大型 batch；Q部分也适用仓库的纯Q视频豁免。正式三维大型 batch 的统一视频 schema 仍需扩展。

用户最新预算为 **5 CPU、2 GPU shards**。实际每个 worker 固定1 CPU，Slurm任务不申请GPU，训练也使用CPU。纯/组合控制器102条 rollout 合计约86.76进程CPU秒；两次Flow训练合计29.73 CPU秒；Q训练循环约0.31墙钟秒，末段12条teacher约0.147秒。主要数值工作约2 CPU分钟，不含进程启动、审查、测试和视频编码；不把它写成整个研究过程的总耗时。

本轮新增114条真实记录：102条学习控制器评估、12条末段teacher；原v3三条示范及基线直接复用。九项机制/几何检查通过，日志见 [tests.log](../diagnostics/drone_3d_pipeline_v1/tests.log)。JAX在无GPU的CPU作业中打印CUDA插件探测异常，但CPU后端训练、测试均完成；日志保留。

## 下一步证据缺口

优先解决纯 Flow 的闭环末段误差及示范支持不足；当前额外末段监督已经试过且失败，不应重复同一训练。其次构造会改变最佳 η 的物理初态/几何，才能检验Q是否优于固定参数。升级双积分/四旋翼动力学需要重建安全层和连续碰撞验收；当前成绩不代表真实无人机安全飞行。

运行和缓存复用命令见 [drone_3d README](../drone_3d/README.md)。本轮代码与关键证据约17 MB；未删除旧结果，也未改动其他正在运行的研究作业。
