# Double-Bottleneck：4-agent eta / interaction compatibility 审计

日期：2026-09-23。本文是只读 source-level 审计；除本文档外没有修改控制器、环境、投影、rollout 或实验结果。结论针对首个 Double-Bottleneck smoke implementation，目标是保留现有 3D episode-level diagnostic eta，而不是设计新的 eta 或训练 `G_phi`。

## 1. 权威调用链与源文件

当前 success-basin rollout 并不直接使用任意可导旧 C1 rollout，而是执行以下链：

1. `diagnostics/success_basin_multimodality/run.py:6-16` 将 `/home/zhihan/research/02_C1_Toy_GiveWay` 放在 import path 首位；
2. `run.py:24-25` 检查实际 `project_velocity` 的路径和冻结 hash；
3. `run.py:41,45-46` 用 `(seed, pair_id, absolute_step)` 构造 Flow key 并逐物理步重新采样；
4. `run.py:49-55` 依次执行 Flow speed bound、第一硬投影、`G_eta`、第二硬投影、环境 step；
5. `run.py:39` 每条完整 continuation 只构造一个 `DiagnosticCorrector(DiagnosticPhi(*eta))`，所以同一 eta 持续到 first terminal event；`run.py:52` 每个 timestep 从当前 observation 和当前 `u_safe` 重算 correction。

冻结 success-basin 实验使用外部 Toy Give-Way 的 Clarabel exact-SOCP `single_integrator/cbf.py`（SHA256 `841a2dbb...`），而本 checkout 的同名文件当前是早期 SLSQP backend（SHA256 `f127d09c...`）。二者在本审计涉及的 2-agent barrier 几何和可行集上相同，但数值 backend 不同。Double-Bottleneck 必须明确选择一个权威 backend，不能因 import 顺序静默切换。已有 success-basin 语义的权威选择是前者。`diagnostics/success_basin_multimodality/exact_projector.py:1-5,84-95` 的 retry 也声明并实现相同目标、约束和可行集，而不是 relaxed fallback。

## 2. 当前 2-agent eta 的精确定义

### 2.1 Observation 与 joint layout

`single_integrator/environment.py:128-133` 对 agent `i in {0,1}`、唯一另一 agent `j=1-i` 产生

\[
o_i=[p_i,\;v_i,\;q_i-p_i,\;p_j-p_i,\;v_j-v_i]\in\mathbb R^{10}.
\]

`flowbc/giveway_flowbc_agent.py:23-27,43-53,104-123` 将 `[B,2,10]` flatten 为 20 维条件量，从四维 Gaussian noise 生成 `[B,4]` joint action，再 reshape 为 `[B,2,2]`。checkpoint 因而是严格的 20-input/4-output **two-agent joint policy**，不是 agent-wise 可直接扩展的共享策略。

### 2.2 Diagnostic corrector

定义现有逐行径向限幅（代码为 `diagnostics/cl_fhcb/closed_loop.py:65-69`）：

\[
\operatorname{bound}_{v_{\max}}(x)
=x\min\!\left(1,\frac{v_{\max}}{\max(\lVert x\rVert_2,10^{-30})}\right).
\]

由 `closed_loop.py:21-41,71-84`，当前实现精确为

\[
\begin{aligned}
b^{\rm goal}_i(z)&=\operatorname{bound}_{v_{\max}}(q_i-p_i),\\
b^{\rm rel}_i(z)&=\operatorname{bound}_{v_{\max}}(p_i-p_j),\\
g_i(z,u_{\rm safe};\eta)&=
\eta_1 b^{\rm goal}_i+\eta_2u_{{\rm safe},i}+\eta_3b^{\rm rel}_i.
\end{aligned}
\]

代码中的具名变量依次是 `goal_feedback`、`safe_feedback`、`relative_feedback`（`closed_loop.py:25-27`），局部变量是 `a,b,c`（第 82-84 行）。`observation[:,6:8]` 是 other-minus-self，代码在第 79-81 行先取负，因此上述 `p_i-p_j` 符号是确定的。

维数为：每个 basis `[2,2]`，joint correction `[2,2]`，flatten 后 4 维。三个 eta 都是同时作用于两个 agent 的共享标量，而不是 per-agent 参数。输出是 deterministic；随机性仅来自 Flow。

物理上三个 basis 都被当成 velocity-like action，最终 `g` 的单位是 m/s，eta 按实验惯例为无量纲数。严格地说，`goal_basis` 和 `relative_basis` 的输入是米、却直接与 `max_speed` 的数值比较；因此代码隐含了 `1/s` 的 position-to-velocity gain，而不是一个显式带单位的变换。这是既有语义，首个 4-agent 实现不应趁机“修正”它。

各分量的具体作用是：

- 增大 `eta1`：增强所有 agent 指向各自 goal 的 correction；减小或取负则削弱/反向。
- 增大 `eta2`：使 projection 前 candidate 中的 safe-action 系数成为 `1+eta2`；负值抑制当前 `u_safe`，正值放大它。
- 增大 `eta3`：增强 agent 彼此分离方向；负值变成相对吸引。

eta 本身只检查 finite（`closed_loop.py:30-33`），没有 clipping、normalization、gate 或 nonlinear transform。三个 basis 求和后也没有 raw clipping；完整 action 是

\[
u_{\rm exec}=\Pi_{\cal U}(u_{\rm safe}+g),
\]

所以最终 nonlinear/clipping effect 只来自第二次精确硬投影（`closed_loop.py:229-238`）。

### 2.3 现有 eta grid 与 basin 语义

当前 SBMA Phase-A 的预声明 grid 在 `diagnostics/success_basin_multimodality/setup.py:25-28,59-66,82-88`：

- `eta1/goal`: `{0.5, 0.75, 1.0, 1.25}`；
- `eta2/safe`: `{-0.5, -0.25, 0, 0.25, 0.5}`；
- `eta3/relative`: `{0, 0.25, 0.5, 0.75}`。

这是一个 80-cell 搜索设计，不是 corrector 参数的合法范围。grid 中每个 eta 是一条完整闭环 policy，固定贯穿整个 episode；不是一步 action，也不是短 open-loop chunk。Double-Bottleneck 可沿用这种搜索语义，但不应把上述数值 box 自动声明为新场景的充分搜索范围。

## 3. 2 → 4 agents 时会直接断裂的接口

以下不是推测，而是当前源码中的 shape/索引约束：

| 部件 | 当前硬编码 | 4-agent 后所需 |
|---|---:|---:|
| Environment observation | `[2,10]`；唯一 `j=1-i` (`environment.py:128-133`) | 至少一个明确的 4-agent observation/wrapper |
| FlowBC checkpoint | joint 20 → 4 (`giveway_flowbc_agent.py:23-27,104-123`) | nominal joint action 8；checkpoint 不能直接载入 8-D head |
| `bounded_nominal` | 只接受 `[2,2]` (`environment.py:65-71`) | `[4,2]` |
| Diagnostic corrector | 显式拒绝非 `[2,10]`,`[2,2]` (`closed_loop.py:71-84`) | 产生 `[4,2]`，但仍只用 3 个 eta |
| Pair CBF | 只构造 pair `(0,1)` (`cbf.py:52-80`) | 六个 unordered pairs |
| CBF matrix/action | `A[:,4]`、`reshape(4)`、两个 speed balls (`cbf.py:83-94,138-181`) | `A[:,8]`、`reshape(8)`、四个 speed balls |
| Rollout diagnostics | `reshape(4)`、两 agent speed signature (`geometry_rollout.py:46-50,114-115`) | joint 8-D signature |
| Monitor history | goal errors/progress 长度 2 (`environment.py:157,192-209`) | 长度 4；必须保留原 predicate 含义 |

因此，“不改 checkpoint 权重”是可行的；“把现有 FlowBC 原封不动当作一个 4-agent joint policy”则在数学和 tensor 维数上都不成立。

## 4. 推荐的最小 3D eta 扩展

### 4.1 先 pairwise bound，再做 arithmetic mean

对 `N=4`，推荐保留同一个 episode-level
\(\eta=(\eta_1,\eta_2,\eta_3)\)，定义

\[
\begin{aligned}
b^{\rm goal}_i(z)
  &=\operatorname{bound}_{v_{\max}}(q_i-p_i),\\
b^{\rm rel}_i(z)
  &=\frac{1}{N-1}\sum_{j\ne i}
    \operatorname{bound}_{v_{\max}}(p_i-p_j),\\
g_i(z,u_{\rm safe};\eta)
  &=\eta_1b^{\rm goal}_i(z)+\eta_2u_{{\rm safe},i}
    +\eta_3b^{\rm rel}_i(z),\qquad i=0,\ldots,3.
\end{aligned}
\]

这里的顺序是关键：**每个 `p_i-p_j` 分别经过旧的 `bound`，然后对三项取算术平均**。不是先平均 raw displacements 再 bound，也不是求和后再 bound。

此定义有四个直接性质：

1. **`N=2` 精确退化**：此时 `1/(N-1)=1`，且和中只有唯一 `j != i`，所以 relation basis 就是唯一一项 `bound(p_i-p_j)`，逐 bit-level 运算顺序外的数学表达与旧定义完全一致。
2. **全 agent permutation equivariance**：同时重排 position、goal、safe-action 与输出 agent blocks，只会重排求和索引，不改变每个重排后 block 的值。
3. **eta 尺度基本保持**：由三角不等式，\(\lVert b_i^{\rm rel}\rVert\le v_{\max}\)。若不除以 3，同一个 `eta3` 在 4-agent 场景可产生旧场景三倍的 basis scale，混淆“环境变复杂”和“参数标度改变”。
4. **不引入净平移偏置**：`bound` 是奇函数，每个 unordered pair 的两项相消，故 \(\sum_i b_i^{\rm rel}=0\)。它不会凭空推动整个 agent 群的质心。

joint correction 按固定 global agent order flatten 为

\[
g=[g_{A1,x},g_{A1,y},g_{A2,x},g_{A2,y},g_{B1,x},g_{B1,y},g_{B2,x},g_{B2,y}]\in\mathbb R^8.
\]

同一 eta 在 rollout 内保持不变，但三个 basis 及 `g` 每个物理 timestep 从当前 4-agent state 与当前第一次全局投影结果重新计算。corrector 位于第一次全局硬投影之后、第二次全局硬投影之前。

### 4.2 为什么首版不选其他 aggregation

| 方案 | 优点 | 首版问题 | 结论 |
|---|---|---|---|
| 固定一个 partner 的 `B_rel` | 最像每个旧 2-agent row | 忽略六个 pair 中四个，注入任意 partner identity；relation correction 与全局 CBF 不一致 | 不用于 `B_rel` |
| 最近邻 | 关注最紧急 pair，unique-nearest 时 permutation-equivariant | tie/switch 不连续；同时活跃多个 pair 时只看一个 | 暂不采用 |
| 未归一化全 pair sum | 简单、equivariant | 4-agent scale 最高变 3 倍，eta 数值不再可比 | 不采用 |
| `bound(sum raw)` | 范数受限 | “sum/cancellation 后才 bound”不再是旧 pair basis 的平均；远近与多 pair 信息不可辨 | 不采用 |
| CBF-margin weighted sum | 可强调危险邻居 | 新增 weighting law、尺度和超参数，把新方法混入容量测试 | 可作为以后独立 ablation，不用于首版 |
| pairwise-bound arithmetic mean | `N=2` 精确退化、无新超参、equivariant、尺度受控 | 对所有 pair 等权；对称时可 cancellation | **推荐首版** |

推荐项并不宣称是充分表示。它故意保留 3D eta 的表达瓶颈，以便实验回答“当前低维 family 是否仍有 basin”。

## 5. Flow dyads 与 global `B_rel` 必须区分

冻结 Flow checkpoint 只能接收一个 two-agent joint observation。首版最小 wrapper 可预声明两个固定 **opposing dyads**，例如 global order `[A1,A2,B1,B2]` 下：

\[
\mathcal D=((A1,B1),(A2,B2)).
\]

每个 dyad 构造原格式 `[2,10]` observation，调用同一冻结 checkpoint 一次，再把两组 `[2,2]` 输出 scatter 回 global `[4,2]`。这保持 checkpoint、Euler steps、normalization、speed-bound 语义不变，但 overall nominal law 是两个冻结 dyadic laws 的乘积，不再是一个 learned 4-agent joint distribution。

**Flow dyad pairing 只解决 checkpoint tensor interface；它不能用于定义 eta 的 relation basis。** 推荐的 `B_rel` 始终聚合每个 agent 与其余三个 agent 的全部关系，第一次和第二次 hard projection 也始终是 global four-agent projection。这样 six-pair collision geometry 不会被固定 Flow pairing 从 corrector 中隐藏。

固定 dyads 的已知假设是：

- 它不对任意 4-agent permutation equivariant，只对保持 dyad/row-role 的置换有一致解释；
- dyad 外交互只经 global safety projection 和 global `B_rel` 反馈给 nominal action；
- checkpoint 没有 wall geometry/environment input，只有 absolute positions、velocities、goals 和一个 paired agent。Double-Bottleneck 是明显的 geometry/interaction distribution shift。因此无 basin 可能来自 frozen Flow representation、3D eta、或两者共同不足，不能单独归因于 basin search。

相比之下，调用所有六个 pair 再平均 Flow output 会把 same-direction pairs 喂给一个 opposing-pair 训练分布，并破坏原 joint correlation；动态 nearest/matching 会引入离散 policy switching。首个保守实现不推荐这些变化。

### 5.1 CRN / pair-key 语义

为了让同一 state 下不同 eta 的比较仍是 matched common-random-number experiment，推荐显式固定：

\[
\begin{aligned}
K_{\rm ep}&=\operatorname{fold\_in}(\operatorname{PRNGKey}(seed), rollout\_id),\\
K_{k}&=\operatorname{fold\_in}(K_{\rm ep},k_{\rm absolute}),\\
K_{k,d}&=\operatorname{fold\_in}(K_k,dyad\_id),\quad d\in\{0,1\}.
\end{aligned}
\]

`dyad_id` 与 global dyad order 固定；eta、outcome、episode length 和 solver branch 均不得进入 key。每个 `K_{k,d}` 驱动该 dyad 的原四维 Flow Gaussian，所以每步 nominal joint noise 总维数为 8，并且两个 dyad 独立。相同 `(state, rollout_id, seed)` 的所有 eta 必须复用相同 keys。若实现选择一次 batched `[2,2,10]` Flow call，也必须证明 batch row 与上述固定 dyad mapping 一致且不同 eta 下不重排。

## 6. 4-agent hard projection 的自然扩展

现有 pair barrier 在 `single_integrator/cbf.py:73-93`。对四个 agent，保持完全相同数学形式，对每个 unordered pair `i<j` 构造

\[
h_{ij}=\lVert p_i-p_j\rVert^2-d_{\rm safe}^2,
\qquad
2(p_i-p_j)^Tu_i-2(p_i-p_j)^Tu_j\ge-\gamma h_{ij}.
\]

对应 `A` row 只在 global action 的 agent `i,j` 两个 2-D blocks 非零。四 agent 有六个 pair rows。若场景有 `M` 个 finite wall segments，另有 `4M` 个 wall rows，以及四个二维 speed SOCs。两次投影都应解同一问题

\[
\min_{u\in\mathbb R^8}\tfrac12\lVert u-u_{\rm target}\rVert^2
\quad\text{s.t.}\quad Au\ge b,\quad \lVert u_i\rVert_2\le v_{\max},\;i=0,\ldots,3.
\]

第一次的 target 是拼接后的 `u_Flow`，第二次的 target 是 `u_safe+g_eta`。这只是 agent/pair 数量的自然推广，不引入 slack、priority、pass order 或 recovery rule。现有 CBF 实现中的 `p.shape==(2,2)`、`reshape(4)`、`range(2)`、两个 SOC 都必须在 Double-Bottleneck scenario-specific 实现中参数化；不能直接复用而假称已支持四 agent。

## 7. Symmetry 与表示容量限制

推荐的 eta corrector 本身对 agent permutation equivariant，但完整 closed loop 不是完全 permutation-invariant：agent-goal assignment、fixed Flow dyads、Flow row roles 和非对称 initial condition 都可破坏 symmetry。这不是硬编码“谁 yield”的许可；fixed dyad mapping 必须在 config 中预声明，并在所有 eta/state 比较中保持不变。

3D shared eta 的主要容量限制如下：

1. 所有 agent 和两个 bottleneck 共享同一组三个 gain，不能直接表达“一处增强 goal gain、另一处减弱”或 per-agent gain。
2. arithmetic mean 把三个 pair vectors 压成一个二维向量；不同 pair configurations 可产生相同 mean。在近对称排列中，危险关系可能 cancellation。
3. 每个 pair 在 individually bounded 后等权；距离很远但已饱和的 agent 与较近且已饱和的 agent 权重相同。
4. same-side teammate 与 opposing agent 使用同一种 separation basis；没有 role/type feature。
5. 另一方面，`u_safe` 是 global projection 的输出，`eta2*u_safe` 已间接包含所有 active constraints；第二次 global projection 也会把 correction 映射回 six-pair safe set。因此 3D family 仍是技术上完整、可运行且值得先测的闭环 family。

若不同初态需要不同 eta、basin 移动/收缩或某些 state 的 basin 为空，这些都是本次实验要观测的结果，不应在实现阶段通过增加 eta 维数消除。

## 8. Strict-deadlock 的 partial-stall masking

当前严格 deadlock candidate 在 `single_integrator/environment.py:192-219` 定义为：success 尚未发生，2 s window 已就绪，

\[
\max_i |\Delta e_i|<0.01\ \mathrm m,
\qquad
\max_i\lVert u_i\rVert<0.025\ \mathrm{m/s},
\]

并连续保持 5 s。把 error/progress/speed arrays 从长度 2 自然扩到长度 4、仍取 global max，是保持当前 predicate 语义的最小实现；success 同样应要求四个 agent 全部到达 goal。

但这一语义有明确的 4-agent 局限：若一部分 agent 在某个 bottleneck 完全停滞，而另一部分仍有速度或明显 goal progress，则 global max 条件为假，**partial/group stall 被遮蔽**。只有所有未成功系统成员整体低速且整体无进展时才触发 strict deadlock。首个 smoke implementation 不应静默引入 per-subset deadlock 来“修复”它；应保留并记录此限制，分别报告 timeout 和 strict deadlock。否则会同时改变 coordination complexity 与 outcome definition，破坏本研究的隔离目标。

## 9. 审计结论与集成门槛

结论：**当前 3D eta 在四 agent 下技术上可明确定义**，最小可辩护方案是 global all-pair、pairwise-bound-then-mean 的 `B_rel`。它保留 eta 维数、参数意义、episode-level persistence、两次硬投影插入点和 `N=2` 精确退化。

在开始任何 basin sweep 前，集成 smoke test 至少应验证：

1. `B_rel` 的运算顺序确为 each-pair bound → arithmetic mean；
2. 对随机 2-agent states，新 generic expression 与旧 `DiagnosticCorrector` 数值一致；
3. 4-agent `g`/`u_safe`/`u_exec` 分别为 `[4,2]`，CBF `A` 的列数为 8、pair rows 为 6、speed SOC 为 4；
4. fixed Flow dyads 的输出正确 scatter 回 `[A1,A2,B1,B2]`；
5. 不同 eta 的 matched rollout 使用完全相同的 per-step/per-dyad Flow keys；
6. `eta=(0,0,0)` 给出严格零 raw correction，第二投影输入等于第一投影输出；
7. agent permutation test 对 `B_goal/B_rel` 成立，同时将完整 policy 的 fixed-dyad 非不变性明确列为 expected limitation；
8. monitor 测试覆盖 all-agent stall 可触发以及 partial stall 不触发（记录为已知限制，不能误判为实现 bug）。

首个 Double-Bottleneck 结果应解释为“冻结 dyadic Flow wrapper + global hard projection + 3D aggregated eta”这一完整 family 的 basin 诊断。若 individual states 没有成功 basin，下一步首先应区分 Flow dyad/OOD 限制与 mean-relation 压缩限制，而不是在本阶段直接扩大 eta。
