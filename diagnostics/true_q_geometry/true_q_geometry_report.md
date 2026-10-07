# C1 真实全时域死锁风险几何诊断

日期：2026-09-22（Asia/Singapore）

## 结论摘要

真实映射

\[
(z_t,\phi)\mapsto Q_D(z_t;\phi)
\]

在当前三参数、逐步重算的闭环 `G_phi` 家族中具有强烈的策略级变化，因此不是“多数平坦”。但是，其局部形状主要由平台、0.75–1.0 的相邻 probe 跳变、硬投影 active-set 切换以及 deadlock/timeout 终止盆地边界构成；它不是一个得到数据支持的平滑风险曲面。

三个位于失败轨迹上的新鲜方向验证均通过预声明规则：低风险方向的 `Q_D` 分别为 0、0.125、0，高风险方向均为 1，配对精确检验分别为 `p=0.0078125`、`0.015625`、`0.0078125`。不过低风险方向的所有非死锁结果都是 timeout，`Q_success=0`。因此这里发现的是“严格死锁概率的真实控制几何”，而不是成功逃逸几何。

最终分类：

> **CASE B — CONTROL GEOMETRY EXISTS BUT IS NONSMOOTH / MULTIMODAL**

## 冻结语义

完整重建见 [`system_and_policy_semantics.md`](system_and_policy_semantics.md)。本研究没有修改环境、Flow、两次硬投影、严格死锁监测器或事件优先级。

- 两台二维单积分器，`dt=0.05 s`，联合动作维数 4，H=850 步。
- Flow 每个物理步使用新的四维高斯；所有 `Q_D` 估计包含这项部署随机性。
- `G_phi` 是确定性 4D residual，参数维数 3：goal、`u_safe` 与相对位置三个状态相关基函数的共享增益。
- 同一个 `phi` 在完整 continuation 中保持不变，但 `G_phi(z_k,u_safe,k)` 每步重新计算。
- 两次 Euclidean CBF 硬投影均保留。
- outcome 为 collision、success、strict deadlock、timeout；timeout 和 collision 的 deadlock value 都为 0。

没有训练 `G_phi`、神经 critic、certificate 或任何新 `R_risk`。

## 预声明设计与计算量

在查看新 `Q_D` 结果前固定了 [`predeclared_protocol.json`](predeclared_protocol.json)：

- 10 个增广起点：5 个失败轨迹的 8/6/4/2/1 s pre-deadlock 状态，4 个成功轨迹状态，以及 1 个较早成功状态；
- 9 个局部 probe：`phi0`、三个坐标的 `±0.125`，以及 goal 坐标的 `±0.25`；
- 每个 `(state,phi)` 4 个 Flow 随机种子；
- 非平坦后只选中心有限差分范数最大的 3 个状态，用 8 个新鲜种子验证 `phi_minus/phi0/phi_plus`。

主映射执行 350 条新完整 continuation，复用 10 条语义兼容的缓存尾段，共 94,841 个新物理步。方向验证执行 72 条新 continuation、16,430 步。用于推断的总计为 422 条新 continuation、111,271 步，低于 500 条初始目标。另执行 CPU/GPU 各一条相同的 80 步计时轨迹；它们未进入统计。CPU 用时 0.289 s，Slurm GPU 用时 0.563 s，因此正式计算使用 CPU。

单个 state-policy 的 4 样本估计报告 Wilson 95% 区间；方向验证使用 8 个新鲜共同随机数种子与配对精确 McNemar/binomial 检验。这是有限状态与有限随机种子的经验几何，不是连续域定理。

## 1. 真实控制几何是否存在

存在。三个对称坐标比较汇总结果为：

| 坐标 | `Q_D(plus)-Q_D(minus)` | 配对样本 | 两侧精确 p |
|---|---:|---:|---:|
| goal | +0.325 | 40 | 0.0009766 |
| safe-action gain | 0.000 | 40 | 1.0 |
| relative | -0.300 | 40 | 0.0018311 |

由于单状态只有 4 个样本，即使 4/4 配对结果全部相反，两侧精确 p 也只有 0.125；因此没有单状态 pair 在 p<0.10 达到传统显著性。这里以预声明的 `Q_D` range 门控和跨状态汇总配对证据进入方向确认，并用 8 个全新种子做最终方向检验。

| 状态 | 近似 source offset | `Q_D` probe range | `||g_TRUE||` | 分类 |
|---|---:|---:|---:|---|
| D8_pair225 | 8 s | 1.00 | 1.000 | DISCONTINUOUS / SWITCHING |
| D6_pair226 | 6 s | 1.00 | 4.243 | DISCONTINUOUS / SWITCHING |
| D4_pair227 | 4 s | 1.00 | 4.243 | DISCONTINUOUS / SWITCHING |
| D2_pair228 | 2 s | 1.00 | 4.243 | DISCONTINUOUS / SWITCHING |
| D1_pair231 | 1 s | 1.00 | 5.657 | DISCONTINUOUS / SWITCHING |
| S8g_pair229 | 8 s | 0.75 | 0.000 at δ=.125 | DISCONTINUOUS / SWITCHING at δ=.25 |
| S4g_pair230 | 4 s | 0.75 | 0.000 at δ=.125 | DISCONTINUOUS / SWITCHING at δ=.25 |
| S8z_pair225 | 8 s | 0.00 | 0.000 | SATURATED |
| S2z_pair231 | 2 s | 0.00 | 0.000 | SATURATED |
| Searly_pair232 | 21.5 s | 0.25 | 1.414 | PIECEWISE / weak |

![Q_D goal-coordinate slices](figures/q_d_vs_phi_coordinates.png)

## 2. 局部真实方向

中心有限差分使用 `δ=0.125`。按预声明规则选择 `D1_pair231`、`D2_pair228` 和 `D4_pair227`，再以 `eta=0.125` 构造 `phi_minus/phi_plus`。

| 状态 | `Q_D(phi_minus)` | `Q_D(phi0)` | `Q_D(phi_plus)` | minus vs plus p | `Q_success(phi_minus)` |
|---|---:|---:|---:|---:|---:|
| D1_pair231 | 0.000 | 1.000 | 1.000 | 0.0078125 | 0 |
| D2_pair228 | 0.125 | 1.000 | 1.000 | 0.015625 | 0 |
| D4_pair227 | 0.000 | 1.000 | 1.000 | 0.0078125 | 0 |

三个状态均满足方向验证规则，说明真实 full-horizon risk 本身具有可重复的局部方向。不过 `phi_minus` 的其余结果全为 timeout，不能叫作 true escape。

![Fresh direction validation](figures/q_d_directional_slices.png)

## 3. 哪些物理量解释 Q_D

在 90 个 state-policy 聚合点中：

- 100 步持久状态位移与 `Q_D` 的 Spearman `rho=-0.833`；其最低位移四分位全部 `Q_D=1`，最高位移四分位全部 `Q_D=0`。
- 中间两个位移分箱仍分别包含 `Q_D` range 1.0 和 0.75，所以单一阈值不充分。
- 100 步 task progress 的 `rho=-0.323`，且在失败起点内方向会反转：低 `Q_D` 的 timeout 策略常有较差甚至负的任务进展。
- projection residual 与 `Q_D` 的 `rho=0.130`，单独解释力弱。
- 初始 timer、速度、距离等相关性主要混合了不同起点，不能当作同状态策略因果解释。

因此“100 步持久状态位移”值得作为未来方法的排序成分继续检验，但它不是充分统计量，也不能区分 timeout 与真正任务成功。预声明的 progress、progress+order、progress+projection 低维分箱中仍有很大的条件 `Q_D` 重叠。本研究没有拟合可选诊断模型。

![Features versus Q_D](figures/feature_vs_q_d.png)

## 4. escape、延迟死锁与 timeout

相对同状态/同种子的 `p0`，160 个 baseline-deadlock policy comparisons 中：

| TRUE_ESCAPE | DELAYED_DEADLOCK | TIMEOUT | NO_EFFECT |
|---:|---:|---:|---:|
| 0 | 22 | 47 | 91 |

这重现并扩展了旧 timer 风险的核心警告：22 个策略只推迟了死锁，47 个策略则把 deadlock 替换成 timeout。详细个案见 [`escape_vs_delayed_deadlock.md`](escape_vs_delayed_deadlock.md)。

![Escape versus delayed deadlock](figures/escape_vs_delayed_deadlock.png)

## 5. 硬投影的作用

对所有 `|Delta Q_D|>=0.5` 的对称 probe，以及全部新鲜方向 pair，共审计 84 条同状态/同种子成对轨迹：

- executable collapse：0/84；
- 18 个策略 pair 聚合点的平均 residual RMS 差：0.163；
- 平均 executed-action RMS 差：0.069；
- 平均 state-trajectory RMS 差：0.159；
- 平均 active-set disagreement：0.970。

因此投影没有摧毁这些结果相关的策略差异；它们穿过第二硬投影并造成持续状态分离。与此同时，极高的 active-set 分歧支持“piecewise/basin switching”而非平滑几何的解释。这里不能仅由相关性断言投影创造了几何，最保守分类是 `PRESERVED_BY_PROJECTION`。

![Projection role](figures/projection_active_set_vs_q_d_change.png)

## 6. 时间控制窗口

五个失败 source state 在 8、6、4、2、1 s offset 上的 probe range 都是 1.0。中心差分范数从 8 s 的 1.0 增至 1 s 的 5.657，说明接近 latch 时局部跳变更陡，而不是控制影响消失。

但每个 offset 的最低 `Q_D` probe 都是 100% timeout、0% success。因此本数据只确定“deadlock-versus-timeout 的策略敏感窗口”至少延伸到 source deadlock 前 1 s；它没有找到 true-escape window。

![Temporal geometry](figures/q_d_geometry_vs_time_to_deadlock.png)

## 7. 平滑性结论

10 个状态中，7 个分类为 `DISCONTINUOUS/BASIN_BOUNDARY`，2 个为 `FLAT/SATURATED`，1 个为 `PIECEWISE_SMOOTH`。goal 坐标从 `δ=.125` 到 `.25` 的有限差分方向在失败状态上符号一致，但幅值明显改变；成功状态在 `.125` 内完全平坦，却在 `.25` 发生 `Q_D=0.75` 的跳变。

因此，未来若继续从真实几何构造方法，数据更支持分段、排序或分布式处理，而不支持假设一个全局平滑、单一 scalar-gradient 的风险面。本报告只给出诊断，不定义新的 `R_risk`。

![Sparse local basins](figures/success_deadlock_basin.png)

## 局限

- 状态数为 10，局部 probe 维数为 3，不代表全状态/参数域。
- 主映射每格只有 4 个样本，单格 Wilson 区间较宽；方向结论依赖单独的 8-seed 新鲜确认。
- 本样本未观察到 collision，也未观察到 baseline-deadlock 到 success 的 true escape，因此不能学习或证明 true-escape 特征。
- 结果是经验 Monte Carlo 几何，不是形式化连续域证明。
- 没有 broad search、参数优化、神经拟合、controller training 或 `R_risk` 设计。

## 最终决策

**CASE B — CONTROL GEOMETRY EXISTS BUT IS NONSMOOTH / MULTIMODAL**

