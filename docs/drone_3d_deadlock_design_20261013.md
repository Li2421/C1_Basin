# 三维无人机死锁场景与低算力移植验证

> 本文保存首轮场景设计时的证据状态；后续已实施三维 Flow/TT/FF/Q 移植，最新结果见 [pipeline 移植记录](drone_3d_pipeline_port_20261013.md)。下文“尚未移植”描述的是场景 pilot 阶段。

采用**低顶通道迎面交换＋侧上方避让舱**作为首个无人机导航场景。两机沿狭窄通道相向飞行，安全层使它们停在对方前方；左机必须后退、上升、侧移，让右机通过后再回到主通道。场景含真实三维几何和三轴运动，适合先检验 Basin C1 的安全投影、干预可达性和 Q 标签语义。

2026-10-13 完成联网调研和 CPU 工程 pilot：3 个保持迎面对称的初态全部出现基线互相阻塞，同初态干预全部成功。最终 6 条 rollout 用单核 CPU 约 **0.74 秒**，峰值内存 **53.5 MiB**，GPU 为零。当前控制器是目标吸引基线和显式路径证人；尚未迁移或训练三维 Flow/Q，也没有飞行动力学或随机鲁棒性结论。

## 研究依据

以下是原论文或作者维护的代码，检索日期为 2026-10-13。文献方法没有在本轮重跑，也没有把其保证归给本 prototype。

| 工作 | 别人如何处理 | 对当前移植的启发 |
|---|---|---|
| [Chen 等，障碍密集环境中的轨迹规划与死锁解决](https://arxiv.org/abs/2210.04231)，2023 修订版 | 在线安全走廊保证避障；动态优先级配合右手规则处理静态障碍造成的死锁；包含最多 8 架微型四旋翼实验 | 窄通道及让行空间是有依据的无人机任务；单靠局部安全约束并不能完成协调 |
| [Park 等，Decentralized Deadlock-free Trajectory Planning](https://arxiv.org/abs/2209.09447)，2023 修订版 | 栅格多智能体路径规划提供 waypoint，subgoal 优化处理收敛，LSC 负责安全；含 10 机硬件展示 | 将几何可解的路径证人与待学习局部控制分开比较；不能把 teacher 的成绩算成 Flow 的成绩 |
| [Lee 与 Park，MC-Swarm](https://arxiv.org/html/2505.08593v1)，2025 | 分离 coordination state update 与 trajectory optimization；采用三维双积分动力学，并考虑异步、低通信协调 | 后续再加入速度、加速度与异步执行；当前单积分只验证导航层机制 |
| [RVO2-3D 作者代码](https://github.com/snape/RVO2-3D) | C++ 三维 ORCA 实现 | 可作为后续轻量局部避碰 comparator；本轮实际运行的是 CBF，未声称 ORCA 复现 |
| [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | 单机/多机四旋翼仿真；已有 PID、速度跟踪及 downwash 示例 | 几何机制成立后，再以 headless 模式复查少量轨迹的跟踪与气动影响 |

基于这些文献，当前选择是：先用三维运动学定位死锁/解锁机制，然后升级动力学。论文中的理论保证各有自己的地图、通信和模型假设，不能直接覆盖本场景。

## 实验前历史证据审查

机器可读记录见 [history_audit.json](../diagnostics/drone_3d_pilot_v3/history_audit.json)；精确请求及复用结果见 [planned_rollouts.json](../diagnostics/drone_3d_pilot_v3/planned_rollouts.json)、[cache_preflight.json](../diagnostics/drone_3d_pilot_v3/cache_preflight.json)。

| 已检查证据 | 可复用内容 | 兼容性与缺口 |
|---|---|---|
| `shared_control/hard_projection.py`、`basis_families.py` | 硬欧氏投影结构、KKT/可行性验收原则、三系数干预接口设计 | 实现显式要求 `[N,2]`，墙为二维有限线段；需新的 `[N,3]` 几何，不能直接调用 |
| `docs/part1_implementation_theory_audit_20261012.md` | TT/FT/TF/FF 执行语义及理论适用范围 | 三维动作不改变抽象投影定理；FF 仍不是一次终端投影 |
| `bottleneck_family/README.md`、既有 Gap 历史研究 | 共享资源竞争、排队/死锁区分、几何与人数分开测试 | 当前三维是独立工程 pilot；不替换二维规模母版，不宣称 N=100 能力 |
| `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_v2/assets/{toy_ckpt.pkl,db_ckpt.pkl}` | 已有二维权重作为未来部分初始化候选；本轮核验存在性与 hash | 动作/观测及环境语义不匹配，不能冒称已迁移的三维 checkpoint |
| `diagnostics/controller_loco_real_flow_v1/REPORT.md`、`controller_flow_blend_v1/` | controller 条件及 coverage 问题的历史证据 | 不能用旧二维 Q 标签回答新三维执行效果 |
| `shared_rollout_db/rollout.sqlite`、仓库报告关键词检索 | 既有四场景、宽度族、Gap 的状态/控制器索引 | 初次检索未找到可匹配的无人机三维配置、预测、checkpoint 或 rollout；UNKNOWN 记录不自动复用 |

新增的理由是**物理状态、动作维度、障碍、安全约束和 controller 执行发生变化**，而非 critic 输入编码变化。之后仅改变 critic 表示时，必须复用本场景兼容的原 rollout 标签。

## 固定场景

单位均为米。物理球半径固定为 `r=0.18`，速度上限 `0.5 m/s`，`dt=0.1 s`，最多 `800` 步。主要参数是研究设计值，不是特定机型的实测参数。

- 主通道：`x ∈ [-3,3]`，`y ∈ [-0.3,0.3]`，`z ∈ [0,0.8]`。
- 避让舱：位于 `x ∈ [-1.5,-0.2]`，向侧方扩展到 `y=0.9`，顶棚 `z=1.1`；侧方 `y≥0.3` 区域下方有高 `0.55` 的实体台座。
- 两机初态约为 `(-2.3,0,0.4)` 和 `(2.3,0,0.4)`，目标互换。seed 17/23/41 仅对两机共同的 y/z 平移作 ±0.012 扰动，保留迎面对称。
- 机间排除区：`(Δx/0.5)²+(Δy/0.5)²+(Δz/0.8)² ≥ 1`；控制器另加 0.005 的无量纲尺度裕量。垂直距离更保守，只是 downwash 风险的几何代理，**不是下洗气流模拟或经过标定的飞行安全距离**。
- 墙体使用真实球—实体盒净距，控制器额外保留 `0.02 m`。所有机器人及墙尺寸均未为了通过测试而缩小。

主通道在同 x 截面，球心最大横向/高度差分别为 `0.24`、`0.44`：

`(0.24/0.5)²+(0.44/0.8)² = 0.5329 < 1`。

因此在规定机间排除区下，主通道不能错身。全场高度为 1.1，球心最大高度差仅 `1.1−2×0.18=0.74<0.8`，**保持同 y 时，单纯升降也不能错身**。侧方空间的低处被实体台座占据，又迫使侧移前先升高。该结论针对包含机间安全排除区的导航问题；不能把它称为仅由物理机体接触约束证明的不可通过性。

在 pilot 中，路径证人为左机设置阶段目标：后退至 `(-0.85,0,0.4)` → 升至 `(-0.85,0,0.87)` → 侧移至 `(-0.85,0.60,0.87)` → 等右机通过 → 侧移回主通道 → 下降 → 到最终目标。每步仍调用相同硬安全投影和同一个 simulator，不直接写入目标位置。完整轨迹的空间秩为 3，确实包含非共面的三轴运动。

## 实际运行的控制链与验收

导航模型为 `p⁺=p+dt·u`，每个物理步内 u 恒定；不建模姿态、推力、电机、加速度上限或速度跟踪误差。

基线 `goal_cbf`：`u_ref=bounded(0.8·(g−p))`，随后解硬投影 `min ½||u−u_ref||²`，含机间/墙面 CBF 和每机三维速度球。求解器为 Clarabel，每个速度球是 4 维二阶锥；无软约束、无越界后的限幅。

干预 `route_witness`：触发同一死锁监视器之前逐步执行基线；触发后切换上述显式 waypoint 序列。它具有地图、指定让行机和阶段信息，是**联合可解的构造性证人**，不能归入学习控制器或旧 η 修正成功率。

死锁判据 `mutual_block_v1` 同时要求：两机都未完成；连续 5 秒最大速度 `<0.005 m/s`；各机窗口累计路径 `<0.02 m`；归一化机间距离减一 `<0.02`；原目标速度仍要求相向接近。timeout 单列，不改名为 deadlock。这里得到的是有明确判据的有限时窗持续互阻证据，不是任意扰动下无限时间死锁的数学证明。

连续运动验收在 simulator 内逐 transition 计算：相对线段到原点的解析最小距离检查机间物理球和排除椭球；分段二次函数的解析最小值检查线段到实体盒距离；凸边界按端点检查。不依赖渲染帧、抽样点间距或最终截图。

## 数值结果与资源

最终版本：[`diagnostics/drone_3d_pilot_v3/summary.json`](../diagnostics/drone_3d_pilot_v3/summary.json)。

| 初态 seed | 基线 | 基线时刻 | 同初态干预 | 完成时刻 | 干预前基线前缀 |
|---|---|---:|---|---:|---|
| 17 | deadlock | 13.7 s | success | 49.0 s | 逐点完全一致 |
| 23 | deadlock | 13.7 s | success | 49.0 s | 逐点完全一致 |
| 41 | deadlock | 13.7 s | success | 49.0 s | 逐点完全一致 |

- 6 条 rollout 的墙钟时间合计 `0.738 s`，进程 CPU 时间 `0.716 s`；单核 affinity、BLAS/求解器单线程，GPU 为零。此时间不包括调研、进程启动、文档和视频编码，不外推为学习模型或飞行动力学速度。
- 当前进程 `/proc/self/status` 的内存高水位 `53.52 MiB`。`getrusage` 的 `565.44 MiB` 也原样保留，因其与当前映像的 VmHWM 不一致，不把它当作这个仿真器自身的内存测量。
- 全程最小墙面物理净距 `0.0200058 m`，两机物理球净距 `0.1425359 m`，归一化机间排除区余量 `0.0050719`。三个量单位不同，不混合报告为米。
- 5 个单元检查通过：单机直线路径、无法错身的截面条件、帧间对穿、帧间穿墙、活跃约束投影。6 条轨迹已 journal/merge 入共享库；再次运行全部复用，新增 rollout 为零。
- 上述 3/3 仅是 3 个对称初态的工程结果，不能称为 B15、未见几何泛化或带噪声鲁棒成功率。

保留的探索记录：`drone_3d_pilot_v1` 的 6 条短轨迹因过严的 `Solved` 状态检查中止，记为 numerical_failure；`v2` 在 `Solved/AlmostSolved` 都通过同样可行性及 KKT 残差验收时接受解，得到 6 条有效轨迹，但墙面最小净距约 0.11 mm；`v3` 显式加入 2 cm 墙面裕量并拆分米制/无量纲 buffer，补跑 6 条。**本次总共新建 18 条记录，其中 6 条数值中止、12 条完整轨迹**；不是只报告成功的最终 6 条。v1/v2 源码快照、原记录、配置与失败均保留。

## Pipeline 移植方案

目标接口保持 `state + controller + η → true rollout → outcome/Q`。先完成可解释的 terminal correction，再按 [Part 1 审查](part1_implementation_theory_audit_20261012.md) 的真实语义引入 Flow field placement。

| 模块 | 当前 pilot | 正式移植需要补齐 |
|---|---|---|
| 场景与状态 | `[N,3]` 位置、目标；三维实体盒；N=2 | 扩展 snapshot/observer；双积分时显式加入速度；有 mode/phase 的 controller 必须把内部状态纳入缓存和 critic 条件 |
| nominal/Flow | 解析 goal controller；路径证人单列 | `2N→3N` 动作输出、3D observation 和重新标定的 normalization；旧 checkpoint 只能经兼容性检查后部分初始化 |
| 硬安全层 | 三维 CBF＋3D 速度球＋连续碰撞 | 与共享 FilterResult/snapshot 契约接通；保留二维回归。双积分时使用相对阶二的屏障或短 MPC，不能直接把速度 CBF 当加速度 CBF |
| 低维修正 | 本轮无 η 干预，缓存 η=0 只是占位 | 给 basis 新版本；验证后退、垂直、侧向方向在需要的状态存在且不全被投影消除 |
| Q/critic | 未训练 | 从真正执行的三维 correction pipeline 得到标签；学习 `Q(s,C,η)`，保持 state/controller/geometry 分组验证 |
| 视频与证据 | 完整三维 trace＋独立开发视频 | 当前统一四场景入口显式是 `[T+1,N,2]`，正式三维 batch 前需扩展其 schema/renderer/validator，不能把 z 丢掉后称完成验收 |

**不能直接把旧 OrthoFlow3 向量补一个零坐标。** 在理想的共线迎面状态，目标方向、安全动作和相对位置都沿 x；正交残差退化为零。因此简单维度延拓后的 `Im B` 可能仍只有 x 方向，有限维 η 无法凭空生成去避让舱的 y/z 分量。这是结构性检查，应在训练前解决。

建议首个新 basis 试验保留三个系数，但为每个列场写明三维语义及 scale，例如“任务方向”“指定让行侧的回退方向”“通往避让舱的三维导航场”。第三列可包含不同位置上的升高/侧移，须使用同一张公开几何地图构建，明确其是否包含 route/phase；绝不能把 teacher 阶段隐藏在 critic 看不到的输入里。该方案本轮尚未实现或验证。一个 stateless 3D 导航场也可能在入口陷入局部极小，应先做正交方向及投影响应检查。

3D 物理空间与 η 的维数是两个问题。保留三系数是优先尝试，不是三维空间必然需要/只需要三个干预系数的结论。

## 后续探索预算与停止条件

1. **先补非对称扰动与 correction 表达能力。** 固定 N=2、当前地图和动力学，冻结新的 basis/controller 后，以 24 个位置/目标状态 × 8 个 η 为上限，最多 192 个确定性 rollout。seed 只用于生成初态；相同确定性 controller 不靠重复 continuation seed 伪造独立试验。必须先 preflight，只跑缺项，首批 16 条测算真实耗时；单核预算先设 10 CPU 分钟。
2. **训练门槛。** 若不同 η 没有实质不同的成功/失败结果，或缺少可用的升高/侧向动作，就停止训练并审查 basis；不能让 Q 学一个全失败表。必要时先少量 teacher 数据做 3D 小型 Flow-BC，再冻结它收集 Q 标签；不默认用 GPU。模型可从两层宽 64 的 CPU 小网络试起，运行预算 10 分钟、按完整初态分组验证。
3. **做有区分力的对照。** 比较 goal+CBF、固定让行规则、显式路径证人、冻结 Flow+CBF、固定 η、Q 选 η，并做去掉 z 列/去掉 y 列的消融。优先报告 Q 是否超过简单固定规则，而不是只和必定卡住的对称基线比较。
4. **再升级动力学。** 首先用三维双积分状态 `(p,v)`、加速度限制和重建的安全层，只复查 6–12 个有代表性的成功/失败初态。此时连续轨迹是抛物线，当前线段安全检查不再适用，必须增加对应连续检查或保守界。再通过 gym-pybullet-drones 的低层跟踪器做少量 headless 回放，记录实际跟踪误差和连续安全边界。单积分结果不能替代这步。
5. **最后扩大 N 与地图族。** N=4 时让新增机器人加入同一通道资源竞争，维持半径/动力学不变，检查实际排队容量；地图变化单独开展。只有当前资源容量允许时才继续 N=10，避免无依据地扩成 N=100。正式评估继承仓库的视频与缓存要求。

## 交付与复现

- 代码：[drone_3d/pilot.py](../drone_3d/pilot.py)、[runner](../drone_3d/run_pilot.py)、[renderer](../drone_3d/render.py)。
- [完整 49.1 秒对照视频](../diagnostics/drone_3d_pilot_v3/seed17_full_comparison.mp4)：491 帧、10 fps，含两条完整初态到终态轨迹，短基线终止后保持终态；无裁剪、跳帧或插值轨迹。
- [轨迹概览](../diagnostics/drone_3d_pilot_v3/seed17_overview.png)、[视频 manifest](../diagnostics/drone_3d_pilot_v3/seed17_video_manifest.json)。这是一份新场景开发视频；本轮未运行四场景大 batch，不以它替代正式四场景视频交付。
- [运行说明](../drone_3d/README.md)。复跑 final pilot 将使用缓存；原始 NPZ/JSON/MP4 在 diagnostics 中，沿用仓库的大文件忽略规则。
