# C1 B5 Correction Family：生产接口、有限 Oracle 与独立回测

实验日期：2026-10-27。分支 `research/b5-oracle-20261027`；起点提交 `fe3ec0b96901dcc735a7713c7ae14e6746673bf4`。交付提交见本分支 `git log -1`。代码与结果目录为 `diagnostics/b5_oracle_20261027/`。

**实际证据支持增加主动侧向方向，但不支持将完整 B5 直接替换 B3。** 开发队列 B3@168 解决 50/64，固定 T-goal 的 ADD 候选池再救回 12/14；独立冻结配置后 ADD-goal 仅 10/32，B3 为 18/32。较好的新配置为三列 G,R,T-cluster（20/32），绕中心右手规则为 29/32。旧四场景中，从历史结果选出的原 B3 为 14/16，新固定 ADD 为 0/16、BLEND 为 6/16。开发 Oracle 的成功不能代替泛化证据。

共完成 **13,472 次新闭环仿真**（当前生产 13,312，旧 checkpoint 160），复用历史精确匹配标签 1,024 条、零系数等价结果 198 条；总计 14,694 条逻辑候选记录、13,472 份新轨迹。所有实验限制在 CPU 0–17，工作进程单线程 BLAS；未使用 GPU，未训练 Generator、Critic 或 nominal。累计 worker 仿真墙钟时间 24723.6 秒，不能当作整项工作的耗时或测量的 CPU 时间。

## 1. 历史证据与真实接口

先检查 `shared_rollout_db/README.md`、原共享控制模块、`diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/` 的 runtime、后端、场景、控制器冻结配置和 51,744 条 `hetset/fresh.jsonl`。快照与当时存活的 harness Python 文件无差异，证据哈希见 [evidence_audit.json](evidence_audit.json)。原菜单16在相同物理初态、nominal、seed=0 下的合法标签复用；历史 sE 中超出本次参数盒的条目没有混入 B3 成功数。新增方向、Sobol 参数、失败状态的完整轨迹和独立种子是补跑的证据缺口。

**参考包缺失：**递归检索 `/home/zhihan`、`/tmp`、`/mnt` 未找到 `C1_B5_validation_v0_20261027.zip` 或 `IMPLEMENTATION_SPEC_CN.md`，因此没有读取该规范，也没有运行其 `run_checks.py`。下述自有回归检查不能替代参考包验收。

当前生产为 `scene_search/harness/runtime_new.py`：nominal → 原第一次安全投影 a → 原 G/P/R 修正 → 原第二次安全投影 → 原物理环境。两个 nominal 为 `centralized_sampling`、`gradient_mpc`，均使用原冻结优化配置，当前这条流水线没有神经网络 checkpoint。

G 为限速目标残差；P 为 Tikhonov 正交 flow 残差，尺度 3.303687238760696，epsilon=1e-6 乘速度尺度；R 为逐对限幅 separation 的平均，并保留 20261021 后端的接触冲突门控（安全直径的1.25倍）。所有均已到目标的 pair 不参与该门控。旧共享 OrthoFlow3 的无门控 R 与此不同，未将旧实现替换进当前生产。

当前联合可行域是 **各机器人原二维可行域的笛卡尔积**，不是集中式相对速度联合 QP：neighbor radius=.5，pair responsibility=.5，gamma/gamma_wall=1，原径向速度锥、Clarabel、重试及已有 `speed_only` / `safety_minimax` restoration 均保留。只有静态 support 诊断另解线性目标 SOCP；闭环始终调用原 safety 实现。静态投影也使用生产的 `project_velocity`，在原约束上检查、不调用 restoration。

当前碰撞用原 swept pair/wall 判据；全部目标距离低于原 tolerance 才成功；用尽 horizon 是 timeout。当前外部场景 runtime 没有严格数学 deadlock 终止判据。旧 Toy / Double-Bottleneck 的原 monitor 可以输出 `deadlock`，这些标签单独保存，不与 timeout 合并。原有两阶段 η 切换（进度 theta 或 6/7 horizon）已审计并用于 D。

原四场景 checkpoint 在活动路径缺失，随后在 `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/` 找回，见第7节。没有替换、重训 checkpoint，没有改变旧安全层或成功判据。

## 2. 实现与回归

新增可选模块 [correction_b5.py](../../shared_control/correction_b5.py)，原 G/P/R 由调用方提供。生产适配 [runtime.py](runtime.py) 只在原第二次投影前插入 target hook；源代码生成的其余运行语句保持原样，controller UID 包含完整 recipe、η5、角色、waypoint 菜单、调度和执行源码哈希。所有旧生产文件未修改。

- T-goal = JG。
- T-cluster 在当前接触图连通组内，使用共同尺度 `speed / max(speed, max_radius)` 乘 J(p−组中心)；单点组为零。该尺度不放大小组半径，所以不同 T 的比较仍含尺度差异，不能把弱结果直接解释为方向无用。
- T-wall 旋转实际原墙体 CBF 行中下界最紧的法向，限幅到原速度尺度，同下界按原行顺序选取。无墙场景该列为零。
- 每个有序 yielder/passer 支持13配置：HOLD，后/左/右/前 × 0.5/1/2 安全直径。局部前向为初始 yielder→passer；轴和 waypoint 在激活时冻结。clear 使用原目标残差限幅。

ADD 为 `a + B3 eta[:3] + etaT T + etaY m(clear-a)`；BLEND 将让行者整个投影前 corrected input 与 clear 插值，其他机器人不变。因此 ADD 在固定状态、角色和 waypoint 下对 η 仿射，BLEND 不是固定 B 的五维线性族。HOLD 仅给出零期望速度，没有额外硬约束，也没有新增到达冻结。

主参数盒为 `[.5,1.25] × [-.5,.5] × [0,.75] × [-1,1] × [0,1]`。ηT=ηY=0 时保持原 G/P/R 运算顺序：两条真实 smoke 轨迹上 **2,166 次逐位等价检查通过**，原 safety 动作76次检查最大误差0；旧四场景另有 **121,561 次逐位检查通过**。[regression.json](regression.json)。原 `test_safety_eta3.py` 的5个测试函数直接调用全部通过；环境无 pytest，未声称 pytest suite 通过。

## 3. 真实状态机制诊断

从实际 B3 失败轨迹后2/3中选取最静止且未完成的状态，不将其自动命名为数学死锁。初始7个状态包含对径交换、十字口阻塞、拥挤方形、墙附近局部极值及已到目标机器人；另外分析旧 Toy、Ring、Double-Bottleneck、Four-Way 各1个状态及开发最后残余案例。

| 真实失败轨迹状态 | rank B3 | 最大 rank B5 | G/P/R 范数 | 可行域进度 support | B3顶点最大进度 |
| --- | --- | --- | --- | --- | --- |
| xsc_hs_plus_w025/centralized_sampling | 3 | 5 | 1.000/0.001/0.189 | 0.000 | 0.000 |
| xsc_hs_box_n09_r014_f25/centralized_sampling | 3 | 5 | 0.508/0.108/0.362 | 0.436 | 0.233 |
| xsc_hs_box_n09_r014_f25/gradient_mpc | 2 | 4 | 0.503/0.016/0.000 | 0.067 | 0.067 |
| xsc_hs_swap_n07/centralized_sampling | 3 | 5 | 1.323/0.061/0.364 | 0.044 | 0.040 |
| xsc_hs_swap_n07/gradient_mpc | 3 | 5 | 1.323/0.041/0.297 | 0.034 | 0.033 |
| hs_obst_n08_R140_r045/centralized_sampling | 1 | 3 | 0.500/0.000/0.000 | 0.000 | 0.000 |
| hs_obst_n08_R140_r045/gradient_mpc | 2 | 4 | 0.500/0.005/0.000 | 0.002 | 0.000 |

表中进度是未完成机器人朝最终目标的瞬时速度之和，不是完整脱困证书。详细奇异值、两两角度、投影后的运动、原活跃约束、原速度可行 witness 和行残差均在 [mechanisms/](mechanisms/)。例如障碍场景的 centralized_sampling 状态 G/P/R 范数约 `.5 / 9.53e-13 / 0`，rank=1，P 的流方向依赖和 R 门控退化明显；T/Y 可将最大 rank 提升到3。十字口 G 与 R 夹角约150.29°，P 范数仅5.64e-4；增加代数 rank 并不保证投影后保留运动。

静态机制检查把该失败状态视为一次新的修正激活点，在此处冻结角色轴/waypoint；它不伪装为原整局初态所冻结的同一waypoint。每个状态检查3 recipes × 最近 pair 的两个有序角色 ×13菜单 ×32盒顶点，共2,496个顶点。12个真实状态合计 **29,952 次 ADD 顶点检查**。用 NNLS 估计投影前向量到原点法锥的残差，活跃行容差1e-6，并核验0是否在原可行域。速度球在0处不活跃；当前分布式域按每个 local set 检查，旧四场景按原联合域检查。**这些是浮点数值估计，不是严格数学证书**；BLEND 不适用此仿射顶点论证。没有把一次有限搜索失败或小残差写成不可能定理。

特别案例 `hs_obst_n08_R140_r045 / centralized_sampling / state12`：固定 B3/B5 池均未成功。选中状态最大速度约.0198735，GPR rank=3；原域朝目标进度 support=0.036932447878，B3盒顶点已达0.036932447838。后来释放 T/Y 的阶段调度成功。因此该例支持“整局固定参数不足”，不支持“缺少可行瞬时方向”的结论。数据为 `mechanisms/hs_obst_n08_R140_r045__centralized_sampling__residual_state12.json`。

文献只用于提出机制类别：参考 [Wang 2017 作者稿](https://liwanggt.github.io/files/A2_Safe_Swarm.pdf)、[Grover IJRR 2023](https://arxiv.org/html/2206.01781)、[Zhang T-RO 2025 作者实现](https://github.com/Parker-Zhang/Shaping-CLF-MCBF)（DOI 10.1109/TRO.2025.3600159）。这些工作的系统和约束假设不能直接当作本项目定理。已覆盖真实双机器人 Toy 与群体阻塞；**尚未完成专门的严格共线两机器人、孤立三机器人约束链/闭环、几何认证的局部围堵对照**。也没有证明所有场景几何联合可解。

## 4. A/B/C：逐级 Oracle 搜索

先在四开发场景的 state8、两个 nominal 上跑小规模 A；随后将开发队列扩大到 states8…15，共64 scene/controller/state 案例，future seed=0。A40 为16原合法菜单 +16 scrambled Sobol +8盒顶点；失败才递增到168。小规模最难的对径案例还额外做到 B3@296。所有预算是候选池上限，不是连续空间全局最优；候选成本和身份逐条保存在 JSONL。

B3 预算曲线：8→45/64，16→47/64，24→48/64，40→49/64，72→50/64，168→50/64。增加搜索本身救回了至少1个 A40 失败，不能归功于 B5。14个残余全部来自 centralized_sampling；gradient_mpc 在该开发队列已经全部可由合法 B3 候选解决。

每个 ADD/BLEND recipe 向同一 B3 基础池加入96候选（2个 T-only、26角色菜单×3个 T系数、16 Sobol η5）。主曲线中最大预算为168+96=264，**原 B3 候选保留在每个 B5 池中**；历史已成功案例直接保留，不重复 rollout。角色搜索仅覆盖初始最近 pair 的两个次序，未穷举所有 n(n−1) pair。

| 独立固定 T recipe | 新增候选/困难队列 | 成功候选 | 救回/14 | 全64队列覆盖 | 碰撞候选 |
| --- | --- | --- | --- | --- | --- |
| add_goal | 1344 | 67 | 12 | 62 | 459 |
| add_cluster | 1344 | 15 | 8 | 58 | 241 |
| add_wall | 1344 | 18 | 8 | 58 | 307 |
| blend_goal | 1344 | 38 | 10 | 60 | 397 |
| blend_cluster | 1344 | 0 | 0 | 50 | 31 |
| blend_wall | 1344 | 8 | 5 | 55 | 195 |

ADD 三 recipe 的更大池合并救回13/14，BLEND合并也救回13/14；这分别是最大168+288=456候选的 **recipe-union Oracle**，绝不是自动选 recipe 的单一固定方法。小规模 B3@296 的一个额外深搜记录单列，不并入所有64例的168预算线。候选碰撞始终计失败，不能用已到目标比例代替成功。

T-goal-only 在剩余6个对径交换案例均有成功，说明无需让行列也能补足原 flow-perp 的主动转向。ηT=0 的 Y-only 在5个困难案例有成功（1障碍、3十字口、1对径），说明角色化 clear 能在部分阻塞场景提供作用。T-wall-only 仅在1个困难十字案例成功；T-cluster-only 在该困难队列没有成功。完整 recipe 的成功可能来自 Y，不能把全部收益归于 T。原始 attribution 集合见 [summary.json](summary.json)。

## 5. D/E、参数盒与维数

| 阶段机制 | 候选数 | 成功候选 | 救回/14 |
| --- | --- | --- | --- |
| b3 | 224 | 0 | 0 |
| add | 51 | 29 | 14 |
| blend | 51 | 25 | 12 |

B3两阶段224个候选未救回这14例；ADD释放调度在 recipe-union 下覆盖14/14，包含固定池最后残余 state12（`release_add_goal_sob14`、`release_add_wall_sob6`）。BLEND调度自身覆盖12/14；与其固定池合并仍覆盖13/14。各候选释放规则固定为 horizon/3 后切回 B3，未在 rollout 中由 Oracle 挑选切换时刻。调度候选来自开发成功/低残差候选，属于自适应开发搜索；不能与固定96候选假装等成本。

只放宽 Goal 下界到0的小规模配对消融：B3新增24候选0成功，ADD新增288候选5成功，但没有新救回案例。原盒已有同一对径案例成功，现有证据不支持把收益归因于允许 Goal=0。该范围消融尚未扩大到全部64案例。

以下维数消融只统计原盒、固定调度、ADD；对实际候选删除列，保留其余系数，没有针对每个低维族重新全局优化：

| 保留列 | 实际检查候选 | 成功候选 | 救回困难案例/14 |
| --- | --- | --- | --- |
| GRTY | 123 | 33 | 13 |
| GPTY | 123 | 30 | 13 |
| GTR | 123 | 78 | 12 |
| GTY | 123 | 31 | 13 |
| noT | 28 | 10 | 5 |
| noY | 28 | 10 | 10 |

这里是跨 recipe 的有限候选/删除 witness 合并。四列 GRTY、GPTY 与三列 GTY 均覆盖完整固定 ADD union 的13个困难成功案例；GTR覆盖12个。**未找到必须同时启用五列才能成功的不可替代样本。** 这不证明任意状态都可压缩，也不证明删列失败意味着该低维族不可解。`noT`/`noY` 只在所选成功 witness 上做配对删除，池较小，不能与前四行当作等预算排名。

## 6. 冻结后的独立测试

独立设计在扩大开发搜索前由初始 state8 pilot 冻结：[INDEPENDENT_FROZEN.json](INDEPENDENT_FROZEN.json)。四个新场景定义，states16/17，future seeds1/2，两个原 nominal，每方法32次。B3按开发场景族/nominal 选一个候选；B5每 recipe 用一个全局 pilot 候选迁移到两个 nominal，角色只按初始最近 pair 映射，测试结果不参与选 η、recipe 或角色。该比较衡量这些冻结策略的迁移，不是独立测试集上的 Oracle，也不代表已经充分调优 B5 的部署策略。

**标签陷阱已明确纠正：**`blend_cluster` 冻结 η=`[.875,0,.375,.75,0]`，P=Y=0，实际只有 G,R,T-cluster，ADD/BLEND在此完全相同。它的20/32不能作为非线性 BLEND 或完整五维的胜利。其余冻结参数、HOLD与waypoint均可由该文件直接复核。

| 固定方法 | 成功/32 | timeout | 碰撞 | 成功条件时间 s | 全队列截断时间 s | 计算 ms/step |
| --- | --- | --- | --- | --- | --- | --- |
| safety | 17 | 15 | 0 | 11.174 | 14.348 | 5.388 |
| B3 | 18 | 14 | 0 | 9.108 | 13.245 | 7.013 |
| G,P,R + T-goal，Y=0 | 16 | 8 | 8 | 6.744 | 13.247 | 6.889 |
| add_goal | 10 | 14 | 8 | 6.670 | 14.509 | 7.782 |
| add_cluster | 18 | 14 | 0 | 10.694 | 14.003 | 7.548 |
| add_wall | 0 | 32 | 0 | — | 16.538 | 6.954 |
| blend_goal | 8 | 16 | 8 | 6.175 | 14.819 | 6.935 |
| G,R,T-cluster（原标签 blend_cluster） | 20 | 12 | 0 | 8.945 | 12.897 | 7.580 |
| blend_wall | 0 | 32 | 0 | — | 16.538 | 6.466 |
| GRTY | 10 | 14 | 8 | 6.670 | 14.509 | 6.772 |
| GPTY | 10 | 14 | 8 | 7.390 | 14.734 | 6.791 |
| GTR | 16 | 8 | 8 | 6.744 | 13.247 | 6.915 |
| GTY | 10 | 14 | 8 | 7.390 | 14.734 | 6.693 |
| add_goal_release | 14 | 10 | 8 | 6.964 | 13.772 | 6.716 |
| RHR_A | 23 | 8 | 1 | 10.739 | 13.919 | 5.324 |
| RHR_B | 29 | 3 | 0 | 12.005 | 12.723 | 5.409 |

同一物理初态/nominal 下各方法使用 matched seed；两个 future seeds 中11–16/16轨迹对逐位相同（随方法而异），不能按32个独立随机样本解释。CI 用8个 scene/state 物理块重采样10,000次，保留块内两个 nominal 和两个 seed，是小样本描述性区间，不是部署概率认证。零成功的 bootstrap 区间退化为[0,0]，**不表示真实成功概率严格为0**。不做 B15/Q16 等稳健性声明。

B3成功率 CI=[.25,.875]；三列 GRT-cluster CI=[.25,.875]；RHR B CI=[.78125,1]。与 B3 的配对比较：

| 方法 | 配对成功数 | 方法 / B3 时间 s | 救回 / 损失 | 成功率差 95% CI |
| --- | --- | --- | --- | --- |
| add_goal | 4 | 7.525 / 9.375 | 6 / 14 | [-0.75, 0.25] |
| add_cluster | 14 | 9.200 / 8.761 | 4 / 4 | [-0.3125, 0.25] |
| blend_goal | 2 | 6.050 / 11.300 | 6 / 16 | [-0.8125, 0.1875] |
| G,R,T-cluster（原标签 blend_cluster） | 18 | 8.567 / 9.108 | 2 / 0 | [0.0, 0.1875] |
| add_goal_release | 8 | 8.163 / 8.125 | 6 / 10 | [-0.5625, 0.3125] |
| RHR_A | 17 | 10.594 / 8.915 | 6 / 1 | [-0.0625, 0.4375] |
| RHR_B | 17 | 10.588 / 9.194 | 12 / 1 | [0.03125, 0.65625] |

任务时间分成：成功样本条件均值、双方同时成功的配对时间，以及全队列截断均值（全部非成功，包括碰撞/数值未知，赋原 horizon）。例如 RHR B 条件完成时间12.005s较长，但截断时间12.723s优于 B3 的13.245s；在17个配对成功样本中 RHR B反而慢1.394s。不能以不同成功子集的条件均值宣称加速。计算开销是同一CPU约束下每 episode 实测墙钟/步数，不含进程导入；完整 elapsed_seconds 逐候选保留，未据此声称硬实时保证。

按场景和 nominal 拆分，保留所有不利结果：

| 独立场景（每项8次） | B3 | ADD-goal | ADD-cluster | GRT-cluster | RHR A | RHR B |
| --- | --- | --- | --- | --- | --- | --- |
| hs_obst_n08_R142_r050_te | 0 | 0 | 2 | 0 | 0 | 6 |
| xsc_hs_box_n09_r013_f25_te | 8 | 2 | 8 | 8 | 8 | 7 |
| xsc_hs_plus_w030_te | 8 | 0 | 4 | 8 | 7 | 8 |
| xsc_hs_swap_n06_te | 2 | 8 | 4 | 4 | 8 | 8 |

| nominal（每项16次） | B3 | ADD-goal | BLEND-goal | GRT-cluster | RHR B |
| --- | --- | --- | --- | --- | --- |
| centralized_sampling | 8 | 4 | 4 | 10 | 15 |
| gradient_mpc | 10 | 6 | 4 | 10 | 14 |

局部脱困使用独立轨迹诊断：40步内目标距离进展<.05m、均速<.025m/s且尚未到目标，记首次停滞；该组所有机器人各自再减少.05m目标距离或到达时记脱困。它可能标记正常排队，且可能漏掉需要暂时远离目标的脱困，所以不是数学 deadlock 判据。未脱困数量也列出，不能只看条件时间。

| 方法 | 有停滞诊断 /32 | 其中脱困 | 脱困条件时间 s |
| --- | --- | --- | --- |
| safety | 24 | 15 | 3.470 |
| B3 | 22 | 10 | 3.930 |
| G,P,R + T-goal，Y=0 | 12 | 0 | — |
| add_goal | 18 | 4 | 1.700 |
| add_cluster | 24 | 14 | 4.600 |
| add_wall | 32 | 9 | 2.939 |
| blend_goal | 20 | 8 | 2.650 |
| G,R,T-cluster（原标签 blend_cluster） | 20 | 10 | 4.520 |
| blend_wall | 32 | 2 | 3.300 |
| GRTY | 18 | 4 | 1.700 |
| GPTY | 20 | 6 | 1.750 |
| GTR | 12 | 0 | — |
| GTY | 20 | 6 | 1.750 |
| add_goal_release | 12 | 8 | 2.013 |
| RHR_A | 22 | 18 | 2.183 |
| RHR_B | 22 | 22 | 2.336 |

全量 CI、配对时间和各场景/nominal 指标见 [summary.json](summary.json)、[independent_metrics.csv](independent_metrics.csv)；逐轨迹局部事件见 `local_escape_metrics.json`。

## 7. 原 Toy / Ring / Double-Bottleneck / Four-Way 回测

使用找回的原 `validation_v2/adapters.py`（OtherRuntime、FourRingRuntime）及 native checkpoint；安全层仍为这些旧场景的原联合 solver，未改成当前外部场景的分布式 solver。原 GPR 无新接触门控，Toy保持原 x64、其他保持原x32。环境、CBF、安全投影、basis等7个关键源文件逐字节对照通过。旧 RHR在原联合安全域中投影活动机器人扰动；不将其伪称为当前分布式每机器人投影的同一实现。

| 场景 | checkpoint（相对找回目录） | SHA256 |
| --- | --- | --- |
| Toy | validation_v2/assets/toy_ckpt.pkl | 8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32 |
| Double-Bottleneck | validation_v2/assets/db_ckpt.pkl | 6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd |
| Ring | assets 下 base_u_v10_local_macflow/best.pkl | 11ea8860e05ea19aef5c10ec12041c592d69460260e28a4f726af4533d19bcda |
| Four-Way | assets 下 base_u_v13_broad_global_source_balanced_macflow/best.pkl | 643270b8376bcd2a77b31bf849c5f259814f1ea015b0d5fd131c11ca581cce1e |

精确绝对路径、源码/配置哈希、初态与随机流见 [LEGACY_FROZEN.json](LEGACY_FROZEN.json)。每场景原协议前2个初态×新未来种子16/17，共16次/方法。为避免把弱中点B3作为唯一基线，又从历史 seeds0…15结果中冻结原 generator_location/sample0 的最佳合法非零候选，在相同16/17上测试 `B3_archived`：[LEGACY_ARCHIVED_B3_FROZEN.json](LEGACY_ARCHIVED_B3_FROZEN.json)。这是预先有历史证据的强基线，未用新种子挑选。

| 旧场景（每项4次） | B3_archived | safety | B3 | T_only | ADD5 | BLEND5 | RHR_A | RHR_B | FF_B3 | FF_ADD5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| double_bottleneck | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| four_way_intersection | 4 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| ring_exchange | 2 | 2 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| toy_give_way | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 |

| 旧场景方法 | 成功/16 | deadlock / timeout | 成功条件时间 s | 截断时间 s | 计算 ms/step |
| --- | --- | --- | --- | --- | --- |
| B3_archived | 14 | 0 / 2 | 23.646 | 25.066 | 2.942 |
| safety | 10 | 0 / 6 | 34.335 | 38.022 | 1.723 |
| B3 | 0 | 4 / 12 | — | 43.750 | 1.874 |
| T_only | 0 | 4 / 12 | — | 43.750 | 2.037 |
| ADD5 | 0 | 4 / 12 | — | 43.750 | 1.957 |
| BLEND5 | 6 | 4 / 6 | 22.442 | 36.541 | 2.054 |
| RHR_A | 0 | 0 / 16 | — | 43.750 | 1.923 |
| RHR_B | 1 | 0 / 15 | 24.800 | 42.644 | 1.972 |
| FF_B3 | 4 | 1 / 11 | 18.188 | 37.672 | 5.992 |
| FF_ADD5 | 0 | 0 / 16 | — | 43.750 | 6.886 |

B3_archived 14/16，bootstrap CI=[.625,1]。BLEND 6/16，其中相对强B3救回2次、损失10次；4个双方成功样本中 BLEND 时间26.088s vs B3 16.150s。不能用中点B3的0/16宣称 BLEND普遍优越。样本仅8个物理状态，未完成旧场景全参数B5 Oracle，结果是冻结策略回归。

TT为终端修正；FF_ADD 用常量物理修正的原 field lifting/integrator，保留原 flow key。FF_B3 4/16、FF_ADD 0/16；没有把 FF 中未定义的 BLEND冒充与终端公式相同的五维方法。所有旧场景碰撞和数值失败均为0，原 monitor 的 deadlock 与 timeout如表分列。

## 8. 安全层发现、失败归因与八个问题

**确认了原生产墙体几何不一致，未修改。** 十字口物理墙碰撞半径含几何默认 `wall_radius=4/600`，而 LiteratureConfig 中 CBF `wall_radius=0`；物理墙间距.0616667，CBF间距.0551，相差.0065667。独立 ADD-goal 的8个十字口样本均墙碰撞，而原 solver constraint_violation_steps=0、restoration_steps=0。例如 `roll_adcf7084f5d47a2d1156b6918dde0de198c1b6dc60eb694c6ffba531ee8235c7` 的 swept物理clearance=-.000287924，CBF间距clearance仍为+.006278742，原线性约束残差约1.09e-14。证据含具体机器人、墙段、步数、动作与几何，见 [safety_geometry_audit.json](safety_geometry_audit.json)。拥挤方形另有更保守的半径配置差异，已逐场景记录，不能笼统认为所有场景都欠保守。

所有原碰撞判据仍生效，这些轨迹均失败；没有通过放松阈值或修改 CBF 增加成功率。修复该既有不一致需要单独实验版本，不能把修复前后的结果混作本次 Basis 收益。本次闭环 numerical failure为0；静态 solver unknown也单独记录。失败并非都由这个问题导致，墙外对径交换和旧四场景的 timeout同样保留。

1. **B3解决什么、失败什么？** 开发64例解决50例，包含拥挤、十字口、障碍及全部 gradient_mpc 开发案例；仍有 sampling 的6个对径、6个十字口、2个障碍案例失败。静态证据看到 P 随 flow变小、R门控为0、原投影抵消等机制，但这些不是每个失败的充分分类定理。旧强B3成功14/16，已有能力不能忽略。
2. **T/Y分别贡献什么？** 主动 T-goal 能在 P极弱时提供横向运动，并救回6个困难对径案例；Y-only救回5个困难案例，提供角色化让路。让行可能拖住后续完成，释放时机很重要。HOLD在ADD仅抵消a，不能消掉GPR；BLEND才能将整个输入插值到clear，但效果取决于ηY和后续释放。
3. **哪个T最有价值、是否冗余？** 开发困难池 T-goal最好；独立测试保留能力较好的是小幅 GRT-cluster。T-wall没有普遍优势，无墙时恒零；cluster的共同缩放与group定义仍有尺度混杂。不能挑每个状态最优T后宣称存在统一最优recipe。
4. **B5明显优于B3 Oracle吗？** 有限开发池的覆盖明显提高（固定ADD-goal 62/64 vs B3@168 50/64），但预算264 vs168不等，且B3全连续空间未排除。独立冻结与旧四场景没有验证普遍优势。当前证据不支持直接部署完整B5替换B3。
5. **五维不可替代吗？** 没有发现。固定ADD union的13个救回案例均存在GRTY、GPTY或GTY的有限成功witness；独立较好配置本身是三列。保留B5作为研究超集合理，最终压缩应由更多冻结独立数据决定。
6. **ADD还是BLEND、是否依赖阶段？** 当前开发固定T-goal ADD救12例、BLEND救10例，且各recipe结果不一致；旧Ring/Four中BLEND有收益。释放调度让最后困难案例成功，独立ADD-goal从10/32提高到14/32，但仍低于B3。没有统一优胜者，不能把Y=0的20/32算BLEND收益。
7. **表达、搜索还是时序？** B3从40到168多救1例是搜索证据；T-only救回B3@296仍失败的pilot是新方向的实证支持，但不是B3不可解证明；最后state12在原域已有B3正进度且释放成功，属于固定时序不足证据；十字口碰撞需归入几何/安全模型不一致；0个数值失败不支持把本次闭环失败归因于solver异常。
8. **是否有安全允许脱困但所有当前B都不行的典型状态？** 尚未建立这样的证书。固定全部recipe残余state12后来被阶段法解决，扩大开发队列没有留下“所有已测固定+阶段候选都失败”的案例；独立/旧场景仍有大量冻结策略失败，但未做其完整Oracle，不能据此宣称整个修正族不可能。瞬时可行正进度也不等于存在完整安全脱困路径。

## 9. 原始数据、缓存、复现与未完成范围

[预算曲线](budget_curves.csv)、[汇总图](results.png)、[可导出PDF](results.pdf)。主逐候选数据 `candidates.jsonl`，旧场景 `legacy_candidates.jsonl` / `legacy_prior_candidates.jsonl`；规范化全量导出 `candidates_verified.jsonl` 含全部14,694条记录、η5、recipe、角色、显式冻结坐标/waypoint、seed、UID、成本、终止原因和 trace。`traces/`、`legacy_traces/` 保留位置/动作/随机 key，当前生产还保留原约束几何和投影证据。各阶段 `*_planned_rollouts.json`、`*_cache_preflight.json`、`*_execution.json` 保留候选顺序、配置、历史复用与预算。

**缓存登记修复披露：**最初本实验 state INSERT OR IGNORE 漏了必填 provenance_json，使state登记被静默跳过。各阶段preflight确实运行，但原首轮merger将journal标为ambiguous，不能宣称首次入库即正确。最终从原计划的 exact_configs 重建48个缺失state元数据，增加写后读取断言，保留原journal，通过官方merger重新导入已有证据，未重跑物理轨迹。1024条历史复用行补齐缺失eta字段；原spec/eta_uid始终正确。见 [cache_registration_repair.json](cache_registration_repair.json)、`candidate_metadata_amendments.json`。最终核验：**14,694条精确cache记录，0重复UID、0冲突、全部轨迹存在**。[verification.json](verification.json)。legacy.py的交付后半段仅修复登记/恢复逻辑，原冻结任务仍记录仿真时源码hash，未重写为当前文件hash。

恢复入口（不会主动覆盖已完成结果；仍依赖本地原快照、checkpoint和数据）：

```bash
bash diagnostics/b5_oracle_20261027/resume.sh
```

脚本使用18CPU affinity、单线程BLAS；历史/零列结果通过cache复用。只重算报告与图，可直接运行 `analyze.py`、`plot.py`、`verify.py`、`write_report.py`，Python环境路径已写入 resume.sh。原始JSONL/NPZ/SQLite留在本机研究目录，体量较大，Git只提交代码、文档、冻结配置与小型结果；**Git commit不等于这些原始证据已备份**。

本阶段完成生产可选接口、等价回归、A–E有限实验、机制数值诊断、512条独立测试及160条旧checkpoint回测。仍未完成：缺失参考包验收、严格法锥数学证书、所有角色/waypoint的充分连续搜索、专门三机器人/局部围堵对照、完整Goal下界0全队列消融、每个低维族独立优化、旧四场景B5 Oracle、大样本独立验证和原墙体模型不一致的另版本修复。未训练Generator/Critic，也没有声称支持N=100或证明联合可解。这些缺口不改变已记录结果，但限制了生产推广结论。
