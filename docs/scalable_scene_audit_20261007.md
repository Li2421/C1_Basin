# 可扩展死锁场景母版审查（2026-10-07）

## 结论

四个现有场景的**当前实现都不能直接支持 2 → 10 → 20 → 50 → 100 个机器人**。
其中 Double-Bottleneck 的共享通道/相向交通最适合作为母版的概念基础。
推荐以文献中的 **Gap 双向窄门交换**为主配置，保留双瓶颈为变体：
`bottleneck_family/`。这属于对双瓶颈思路的泛化；旧四场景和历史实验语义保持原样。

已实现可变 N 的物理母版、障碍参数、实例验证和可视化，并新增 N=2/10/50 的
联合可解参考轨迹及独立 Flow Matching 训练流程。**尚未证明纯 Flow-BC/eta
控制器能够稳定完成任务，也没有完成百机器人死锁解决实验。**

## 现有四场景审查

| 场景 | 当前数量与耦合限制 | 已有障碍变化能力 | 死锁评价问题 | 母版判断 |
|---|---|---|---|---|
| Toy Give-Way | 两机器人；plant/CBF 多处 `[2,2]` 固定 | 走廊宽、侧湾、长度、整体布局尺度；拓扑固定 | 全体低速/低进展判据，不能直接覆盖大群体局部停滞 | 保留最小机制实验 |
| Double-Bottleneck | 四机器人；固定起终点、6 对约束、专家双向分组 | 门宽、门长、中央腔室及外侧空间尺寸；固定双门拓扑 | 全体 max-speed/max-progress 判据会被远处运动机器人掩盖 | **最接近，适合泛化** |
| Four-Way Intersection | 四机器人、`[4,18]` observation；专家最多搜索 24 个通行顺序 | 外边界和初态扰动；中心本身没有内部障碍接口 | plant 终止只有 collision/success/timeout，没有独立 deadlock 事件 | 保留开放交叉冲突控制组 |
| Ring Exchange | 四机器人、`[4,23]`，local frame 和采样都固定 4 | 中心圆和外圆半径；固定单障碍环域 | plant 没有独立 deadlock 事件；放大圆周可能使局部交互变稀疏 | 保留绕行方向一致性控制组 |

关键源码依据：

- `single_integrator/environment.py:67`、`single_integrator/cbf.py:53`：两机器人形状限制；
  `single_integrator/environment.py:207`：全局停滞条件。
- `double_bottleneck/environment.py:148`、`double_bottleneck/scenario.py:11`：固定数量和分组；
  `double_bottleneck/environment.py:402`：全局 max 条件。
- `four_way_intersection/environment.py:69`、`four_way_intersection/expert.py:22`：固定四机器人和排列。
- `ring_exchange/environment.py:149`、`ring_exchange/local_frame.py:40`：四机器人和固定局部观测。
- `new_benchmark_common/macflow.py:128`：建模时可以指定 N，但网络展平到固定 joint dimension。
  **能为不同 N 建不同模型，不等于一个四机器人 checkpoint 可直接运行 N=100。**
- `shared_control/hard_projection.py:62` 已按 N 生成全对约束，可以复用；N=100 有 4,950 对，
  还要加 N×墙段约束，不能把维度兼容当作实时性能证明。
- `shared_control/diagnostic_corrector.py:21` 的邻居方向全局平均会随 N 改变；3 个全局 eta
  是否足以处理多个同时发生的局部冲突，需要单独验证。

## 优先采用已有死锁测试设计的依据

1. **Dergachev & Yakovlev，CASE 2021**：使用 Gaps 1 / Gaps 3，两厅之间有一/三个
   窄口，两组机器人交换侧别；另有 Rooms 和 Warehouse。实验数量为 5–40。
   它直接研究局部死锁检测和 MAPF 解锁，最适合作为主场景来源。
   [论文](https://arxiv.org/abs/2107.00246)，[作者代码](https://github.com/PathPlanning/ORCA-algorithm)。
2. **Cooperative-ORCA*，2026 预印本**：继续使用 Gap，并加入 Random/Room/Warehouse；
   数量为 10–200，每档 50 个实例。这说明同类场景已用于更大的数量测试，
   不是本仓库控制器可扩展的证据。[原文 Benchmarking](https://arxiv.org/html/2606.22757v1#S5.SS1)。
3. **Grover、Liu、Sycara 的 CBF 死锁研究**：两/三机器人案例适合核验死锁机制，
   不足以单独支持百机器人、可变障碍母版。
   [Deadlock Analysis and Resolution](https://arxiv.org/abs/1911.09146)。
4. **LIVEPOINT，2025**：Doorway/Intersection 可作为小规模安全与活性对照，项目有视频；
   此处没有把其结果外推到 100 个机器人。[项目](https://livepoint-uva.github.io/)，
   [论文](https://arxiv.org/abs/2503.13098)。
5. **MovingAI** 提供可复用 Rooms/Warehouse/Random 等地图和实例，可作为母版之后的外部
   泛化测试；其离散 MAPF 结果不能直接作为连续圆盘机器人的成功证据。
   [官方地图](https://www.movingai.com/benchmarks/mapf/index.html)。

本次写的是**受这些场景启发的连续空间变体**，没有复制作者实现，也没有声称复现
论文原始地图、物理参数、基线或成功率。并列三门对应 Gaps 3 的设计；串联双门是
结合本仓库 Double-Bottleneck 的扩展，两者不能混为一谈。

## “增加机器人有意义”的实验设计

主实验保持同一地图、半径、速度、门宽和动力学，使用 N=2/4/10/20/50/100。
同一 geometry/seed 下任务序列是嵌套的：新增机器人不改已有机器人的起终点。
所有机器人必须从一侧前往另一侧，经过共同隔墙。单门主配置的潜在相向竞争对数为
1、4、25、100、625、2500（分别对应上述 N），并不是把互不相干的双机器人案例平铺。
这里统计的是**共享通行资源的潜在竞争对**，不是已经观测到的碰撞/死锁边数。

两类数量实验必须分开：

- **固定地图压力测试**：只增加 N，反映排队、瓶颈资源争用和拥堵传播。
- **近似固定密度扩展测试（待实验设计/实现）**：增加 staging/room 面积，保持局部
  门宽和机器人半径不变，同时记录瓶颈数量/通量。不能把整体等比放大与单纯增加 N
  混为一个结论；也不能只增加彼此独立的房间宣称解决了更大规模协调。

环境维度单独控制：门宽（单列/可并行）、并列门数、串联门数、门的纵向偏移、
障碍位置/尺寸/数量。先一次只改一项，再测组合 OOD；禁止通过缩小机器人增加容量。
新增障碍要保留可通行性证据。单机器人有路是必要条件，联合无碰撞可达仍需要参考
规划器提供实际轨迹证书；找不到参考解不自动等于不可解。

## 本次已落地与未完成部分

| 层次 | 当前状态 |
|---|---|
| N=2/10/20/50/100，六种几何配置 | 已生成并检查共 30 个实例；另有 N=4 单元测试 |
| 静态障碍变化 | 单门/双门/错位门/宽门/三门/矩形障碍；可用 JSON 改参数 |
| 连续 SI 动力学与碰撞 | `p += dt*u`；精确扫掠墙/机器人碰撞检查、速度限幅验证 |
| 初态/目标合法、单机器人路径证据 | 已实现保守可见图，拒绝堵死通道的实例；保守方法可能拒绝实际有路的复杂地图 |
| 局部停滞候选 | 已实现，其他机器人运动不会掩盖局部簇；不会自动算 certified deadlock |
| N-agent 共享安全层接口 | N=100 零输入可行性/形状测试通过；不是冲突中的求解速度或安全成功率测试 |
| N-agent 视频绘制 | 已支持超过四个机器人，100 个对象编码 smoke 通过；仍须真实完整轨迹 |
| 联合可解参考轨迹 | N=2/10/50 的单门配置已用单机器人轮流通行 A* 完整零碰撞轨迹验证；这不是学习策略成功率 |
| 可变 N Flow Matching 训练 | N=2/10 已独立训练并闭环评估，N=50 正在训练；每档使用独立展平联合动作模型，尚不支持一个 checkpoint 跨 N |
| 死锁解决控制器与正式死锁基线 | 尚未验证；路径监督组合控制器可完成部分/全部开发集，但不能由成功率直接推断死锁解除 |
| 动态障碍、任意多边形、MovingAI 原地图导入 | **尚未实现**，当前只支持静态轴对齐矩形 |

本轮先做 N=2/10/50 的闭环 pilot，之后再推进 20/100 和障碍变化。
所有控制器使用相同单机器人全局路径/waypoint 语义，避免把不会绕障的直线目标控制器
撞上墙所造成的停滞包装成多机器人死锁。现有 Flow-BC checkpoint 不能直接接到新 set
observation；需独立训练，或设计带 mask 的共享/集合模型并做迁移消融。

评价必须同时报告：全队成功率和个体完成率、扫掠碰撞、到达时间/吞吐、正常排队时间、
局部停滞簇大小/持续时间、经判定的死锁解锁率，以及每步耗时/内存。
局部低速可由正常让行造成；只有停滞和邻近关系不构成正式死锁证据。全体均值又可能
掩盖局部互相阻塞，故需审查 blocking dependency / 进展窗口及参考可行解。
N 增大时，任务期限要依据路径长度和瓶颈通量制定并冻结，不能把合理排队超时当死锁。

正式实验继续用共享 rollout 数据库的兼容性与 B15/seed 协议；checkpoint、geometry、N、
状态、控制器、判据等变更必须隔离指纹。规模与障碍 OOD 应独立留出，不能只改变 seed。
大 batch 的原四场景视频要求继续有效；母版参与时另提供其完整视频与单独的规模标记。

## 复现与验证

- 入口与 API：[bottleneck_family/README.md](../bottleneck_family/README.md)。
- 训练与闭环结果：[Gap 1 规模试验](gap_flow_scaling_20261007.md)。
- 30 个几何检查结果：`diagnostics/bottleneck_family_scaffold/audit/validation.json`。
- [规模与障碍预览](bottleneck_master_overview.svg)只展示初态/目标，**不是仿真视频**。
- 母版 13 项测试通过；视频工具 13 项测试通过。测试包括扫掠穿墙/穿越机器人碰撞、
  局部停滞、闭门拒绝、嵌套规模任务、N=100 安全接口与视频编码。
- N=2/10/50 的 Flow 数据与闭环结果保存在 `diagnostics/gap_flow_v1/`；这是开发 pilot，
  不满足旧四场景的正式视频交付条件，也没有被命名为正式死锁解除 batch。
