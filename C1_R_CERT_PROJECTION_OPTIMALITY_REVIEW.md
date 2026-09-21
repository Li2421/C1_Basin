# R_CERT 投影最优性证书：集成前语义审查

> 历史审查稿：后续用户冻结的最终规格把 dynamics-derived goal/progress
> bad sets 也加入八个 VI witness，计数改为2727/378，并固定
> `sigma=kappa=1`。实现与失败结论见 `C1_VI_R_CERT_VALIDATION.md`；本稿中的
> 1111/362和“等待审阅”不再是当前状态。

日期：2026-09-18。状态：**等待审阅；未集成、未做机器人实验、未批准完整风险或训练。**

本文只实例化用户指定的 projection-optimality 机制。它不修改原死锁事件、控制器、R_CERT 聚合或已有逻辑证书，也不补造不受支持的控制原子。

## 1. 冻结的轨迹、控制与事件索引

记 `x_n` 为执行 `n` 个动作后的联合位置，`n=0,...,850`；`u_k` 为从 `x_k` 到 `x_(k+1)` 的四维联合执行速度，`k=0,...,849`。单位分别为 m 和 m/s。动力学代码严格执行

```text
x_(k+1) = x_k + 0.05 u_k,
physical_velocity_k = u_k.
```

没有执行器滞后、跟踪控制或碰撞后的隐式速度修改。因此在**当前仓库的直接单积分器**中，monitor 的低物理速度原子可以等同于第二次投影后的低执行控制原子。该结论不能外推到其他动力学。

控制器保持：

```text
u_safe,k = Pi_U(x_k)(u_FlowBC,k)
w_k      = u_safe,k + G_phi(x_k,u_safe,k)
u_k      = Pi_U(x_k)(w_k).
```

`U(x_k)` 是 `A(x_k)u>=b(x_k)` 与两个逐机器人欧氏速度球 `||u_i||_2<=0.5 m/s` 的交。`project_velocity` 最小化 `0.5||u-w||²_2`，所以这是欧氏投影；没有替换度量。速度球通过外切平面迭代，但只有在原 CBF 可行性、实际速度球、KKT stationarity 和 complementarity 检查通过时才接受。解析证明假设精确欧氏投影；实现验证必须另外量化 `feasibility_tol=1e-9`、`optimality_tol=2e-6`、`speed_tol=1e-10` 的数值影响。

### 1.1 Strict deadlock

令 `d_i,n=||x_i,n-g_i||_2`。进展窗为 `2.0/0.05=40` 个动作。动作 `k>=39` 后的原候选条件为：

```text
G_k: max_i d_i,k+1 > 0.08 m                         # not success
P_k: max_i |d_i,k-39 - d_i,k+1| < 0.01 m
V_k: max_i ||u_i,k||_2 < 0.025 m/s                 # 0.5*0.05
```

第一次候选样本的 stuck timer 为0；维持5秒需101个候选样本、跨100个间隔。为保留“以后恢复不能抹去以前死锁”，把 strict 事件拆为历史 onset 分支

```text
W_t = {t-100,...,t},                  t=139,...,849
D_strict,t = first strict deadlock occurs at t,
D_strict = OR_t D_strict,t.
```

每个 `D_strict,t` 必然推出 `G_k AND P_k AND V_k` 对所有 `k in W_t` 成立。first-event 顺序保持 `agent collision > wall collision > success > strict deadlock > timeout`。完整评分必须保留所有已经发生的 onset 分支；不能只看终点是否恢复。

### 1.2 Stalled timeout

独立六分类器只在 first-event 为 `other_timeout` 时检查 stalled。timeout 轨迹有850个动作；最后40个动作后样本对应 `k=810,...,849`，首末误差样本为 `d_i,811` 和 `d_i,850`，实际跨39个时间间隔：

```text
Vstall_k: max_i ||u_i,k||_2 < 0.05 m/s,       k=810,...,849
Pstall_i: |d_i,850-d_i,811| < 0.02 m,         i=0,1
D_stalled also requires the exclusive first event other_timeout.
```

`D = (OR_t D_strict,t) OR D_stalled`。普通 moving timeout 保持独立类别。

## 2. 可以支持的控制集合原子

对阈值 `rho>0` 定义联合低速集合

```text
B(rho) = {q in R^4 : ||q_0:2||_2 <= rho and ||q_2:4||_2 <= rho}.
```

这是两个闭欧氏球的笛卡尔积。精确距离为

```text
dist²(q,B(rho)) = sum_i max(||q_i||_2-rho,0)².
```

虽然 monitor 使用严格 `<rho`，事件仍推出属于闭集 `B(rho)`；等号轨迹不属于事件但距离为0，造成保守性而不破坏 inclusion。

唯一支持的映射是：

| 事件分支 | 必要原子 | 集合 | 时间索引 |
|---|---|---|---|
| `D_strict,t` | `V_k` | `B(0.025 m/s)` | 每个 `k in W_t`，共101个 |
| `D_stalled` | `Vstall_k` | `B(0.05 m/s)` | `k=810,...,849`，共40个 |

这里低物理速度等于低控制已经由当前动力学代码证明，而不是从变量名称推断。

不支持 projection-optimality 映射的原子：

- `G_k` 是目标集状态条件；
- `P_k` 是相隔40个动作的历史状态条件；
- `Pstall_i` 是终端两个状态样本之间的条件；
- `other_timeout` 资格、无更早 first event、历史 latch 和剩余 horizon 是离散/历史条件；
- 当前 monitor 没有 local-deadlock 子事件。一个机器人停止而另一个“outsider”移动会使系统 `max_speed` 超阈值，从而不满足现有速度原子；把它计为 local deadlock 会重新定义事件；
- jitter、retreat、task abandonment 和 horizon-delay 是否属于事件，只能由上述原 monitor 逐条件判断，不能从名称加入新原子。

特别地，`D_stalled` 的终端进展条件不能由单个 `u_k in B` 推出或等价表示。VI 证书只覆盖它的40个速度必要条件；不能声称覆盖完整 stalled 语义。

## 3. 保留的 exact atom-negation certificates

为比较前后相同的逻辑结构，先列出不依赖 VI 的单原子证书。每个证书只有一个 margin；正值足以排除对应事件分支，零值不作证明。

所有 margin 先无量纲化，建议固定 `sigma_ejℓ=1`，避免单位混合：

```text
c_goal(t,k) = (0.08 - max_i d_i,k+1) / 0.08
c_prog(t,k) = (max_i |d_i,k-39-d_i,k+1| - 0.01) / 0.01
c_speed_exact(t,k) = dist²(u_k,B(0.025)) / s_u²

c_stall_prog(i) = (|d_i,850-d_i,811| - 0.02) / 0.02
c_stall_speed_exact(k) = dist²(u_k,B(0.05)) / s_u²
```

因此：

- 每个 `D_strict,t` 的原逻辑证书数为 `101 goal + 101 progress + 101 exact speed = 303`；
- `D_stalled` 的原逻辑证书数为 `2 terminal progress + 40 exact speed = 42`。

这些证书不包含离散 `other_timeout` 的 detached margin。省略一个必要原子的 negation 只会减少证书、增加保守性，不影响其余证书的 inclusion。

建议的固定物理控制尺度为

```text
s_u = max_speed = 0.5 m/s.
```

这是控制器已有的物理尺度，不是 deadlock 阈值或按结果调出的超参数。`c_speed_exact` 与下述 `c_VI` 都以 `s_u²=0.25 (m/s)²` 归一化，便于公平比较。

`kappa` 在用户给出的机制和仓库中仍没有数值。它显著影响 normalized soft-min 和新增证书数量的作用；本审查不擅自选择。集成前还需审阅并固定 `kappa`。若不接受上述 `sigma=1` 规范，也必须在集成前给出固定值，不能用机器人结果调参。

## 4. VI 证书与包含证明

四维联合控制有8个固定 signed-coordinate witnesses：

```text
a_(r,+) = +s_u e_r,
a_(r,-) = -s_u e_r,                    r=0,1,2,3
v_j(x_k) = Pi_U(x_k)(a_j).
```

`v_j` 只是在**同一状态、同一可行控制集合**上计算的几何点。它们不得执行，不得生成替代轨迹，不得用于选择动作或路径。

对每个受支持的 `(event branch, k, rho, j)` 定义

```text
m_kj = (w_k+v_j(x_k))/2
c_VI(k,rho,j) = [dist²(m_kj,B(rho)) - ||w_k-v_j(x_k)||²/4] / s_u².
```

若 `u_k=Pi_U(x_k)(w_k)` 且 `v_j in U(x_k)`，欧氏投影的一阶最优性给出

```text
(w_k-u_k)^T (v_j-u_k) <= 0.
```

若事件速度原子成立，则 `u_k in B(rho)`，于是

```text
dist²((w_k+v_j)/2,B(rho))
 <= ||(w_k+v_j)/2-u_k||²

dist²((w_k+v_j)/2,B(rho)) - ||w_k-v_j||²/4
 <= [||w_k-u_k+v_j-u_k||²-||(w_k-u_k)-(v_j-u_k)||²]/4
 = (w_k-u_k)^T(v_j-u_k)
 <= 0.
```

所以 `C_VI={c_VI>0}` 与该速度原子互斥，进而 `C_VI subset complement(D_e)`。除以正数 `s_u²` 不改变符号。

该证明不要求 `B(rho) subset U(x)`，也不要求执行 witness；它只要求实际第二投影和 witness 使用同一欧氏 `U(x)`。若 solver 返回失败、非有限值或 KKT/可行性证书未通过，该样本必须报告 numerical failure，不能赋任意风险。

## 5. 增广后的证书数量与保守性

保留所有 exact certificates，并为每个速度原子添加8个 VI certificates：

| 事件分支 | exact-only `M_e` | 新增 VI | augmented `M_e` |
|---|---:|---:|---:|
| 每个 `D_strict,t` | 303 | `101×8=808` | 1111 |
| `D_stalled` | 42 | `40×8=320` | 362 |

每个新增 VI certificate 只有一个 margin，建议同样使用 `sigma=1`。事件发生时，原证书和 VI 证书都满足 `c<=0`，故每个 `S_ej=h(-c)>=1`；不论证书数量如何，原 `R_e>=1` 和 `R_CERT>=1` 证明保持成立。

但 normalized soft-min 的数值不会因“保留旧证书”而单调。令

```text
Abar = mean_old exp(-S/kappa),
Bbar = mean_new exp(-T/kappa).
```

则

```text
R_old = -kappa log Abar,
R_aug = -kappa log[(M Abar+N Bbar)/(M+N)].
```

若 `Bbar>Abar`，增广风险下降；若 `Bbar<Abar`，增广风险上升。证书数从303到1111、42到362本身就改变归一化，soft-min 上界间隙也从 `kappa log M` 变成 `kappa log(M+N)`。因此后续必须同时报告 exact-only 与 augmented，不能只把风险降低解释为证书更好，也不能复制 witness 来人为改变结果。

VI 是保守的充分证书，不是 exact atom negation 的替代品。固定反例

```text
U=[-2,2]×{0}, w=(2,10), B=unit ball, s_u=0.5,
u=Pi_U(w)=(2,0), dist²(u,B)/s_u²=4 > 0.
```

四个二维 signed witnesses 为 `(±0.5,0)` 及两次 `(0,0)`。对应未归一化 `c_VI` 约为

```text
-8.3078, -10.1119, -9.1980, -9.1980,
```

全部不大于0；VI 没有检出 `u outside B`，而 exact certificate 检出。这是预期保守性，不是数值失败。

## 6. 审阅后才允许的实现与验证

本阶段没有新增风险代码、rollout 适配器、Slurm 作业或实验结果。进入实现前需确认：

1. 事件分解采用711个 `D_strict,t` 分支加一个 `D_stalled` 分支；
2. exact certificate 表和单原子组织方式；
3. `s_u=0.5 m/s`、所有上述无量纲 margin 的 `sigma=1`；
4. 固定 `kappa` 数值；
5. 是否接受证书数变化本身会改变 normalized soft-min，且比较必须同时报告 exact-only/augmented。

审阅通过后，验证才依次实现：解析距离和 VI 单元测试；projection plateau、零控制、对称、active-set switch 及上述保守反例；完整历史 rollout；normal yielding、retreat、nonzero-control stall、local-deadlock 不受支持项、jitter、timeout、horizon delay 和 deadlock-recovery 反例；多步长有限差分与同刻/更早动作梯度覆盖；最后才是冻结预算的独立预测和 early-only `-grad/random/+grad` 配对回放。

任何 unsupported semantics、projection 数值证书失败、早期梯度不能到达更早动作，或负梯度不能降低独立真实死锁且保持成功/timeout，均给出 INCONCLUSIVE 或 FAIL。风险下降本身不通过。
