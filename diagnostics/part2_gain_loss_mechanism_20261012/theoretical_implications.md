# Part 2 的最小理论增补：正规链与离散事件链

## 1. 控制计算图与有限修正

固定物理状态 `x`、Flow key `k`、修正 `δ=B(x,k)η`。四条真实链仍是 `TT=Π(a_safe+δ)`、`TF=Π(H0(δ))`、`FT=Π(HF(0)+δ)`、`FF=HF(δ)`；不能将 TF/FF 写成终端注入的单个 Jacobian。对于一条固定正规活跃集的 η 路径，有限动作差可写作 `u_p(η)−u_p(0)=∫_0^1 G_p(sη)η ds`（绝对连续的分段光滑路径几乎处处成立）。因此零点 `G_p(0)` 对 Q16 有限 η 不自动足够。首动作控制反事实在同一 state/key 下操控 placement，能识别总动作差；不能单独把其中“Flow 反馈”“投影”“方向”和“幅度”识别为独立因果中介。

本轮主要 pair 的 η=0 初态动作偏移：Toy FT/FF 在 24 个预定状态均为 0；Ring TT/TF 中位 `7.96×10^-8`、最大 `1.26×10^-7`（缓存的 seed0 固定 key）。这使有限 η 对比主要体现修正 placement，但不自动消除沿闭环状态路径的基准差异。FT 有非零 η 的 terminal closure，FF 无末端投影；Toy 的 FT→FF 不能化简成“仅改 H 的注入位置”。

四个预定初态 action-only Flow trace 显示 `||h_{10}(Bη)-h_{10}(0)||/||Bη||` 为 Toy `0.274/0.078`、Ring `0.299/0.403`；Toy 的投影分支还有子步活跃，Ring `H0` 则由 Flow 反馈改变路径。它们是有限 η 传播的**示例级数值事实**，不构成场景总体的衰减定理，也不识别某一子步的独立因果作用。若论文需陈述子步反馈，使用实际离散递推与固定分支的乘积 Jacobian，而非把修正写成每步简单累加。

## 2. 物理解释的可推导部分

对每个机器人，`e_i=(g_i-p_i)/||g_i-p_i||`。在固定状态、非零目标距离且小时间步时，`||p_i+dt u_i-g_i||−||p_i-g_i|| = −dt e_iᵀu_i + O(dt²||u_i||²/||p_i-g_i||)`。因此**实际目标向执行动作**下降会降低这一小步的目标距离改进。Condition B0 测量的是执行动作沿 `P_goal a_safe` 的有符号分量；只有该方向与 `e_i` 对齐且分母正常时，才可直接使用上述符号推论。B1 是实际 H20 最慢机器人进展差，规避了 B0 的方向替代问题；仍不足以证明末端 success 差。

Ring 的切向单位 `t_i=(-p_{iy},p_{ix})/||p_i||`。在远离圆心且小步时 `Δθ_i = dt·t_iᵀu_i/||p_i|| + O(dt²||u_i||²/||p_i||²)`。A0 测量 `||P_tan u||`，只有**绝对大小**，没有各机器人正确通行方向或同步约束；它不能成为成功的充分条件。A1 的短期目标退让也不能推出后期共同完成。Ring B15 Gain 的 seed-level 救回中，398/586 属于 TT 有机器人从未进入其目标区，188/586 属于各自曾进入但未同时完成；后一个分支尤其不能由初始环向速度单独解释。

## 3. Toy 离散 monitor 的精确分支

真实源码为 [`environment.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/environment.py:195)。令 `d_i(t)=||p_i(t)-g_i||`，窗口 `W=2.0/0.05=40` 步；代码在非整数 W 时线性插值历史距离。候选 guard 等价于：

```
window_ready
and not all_i(d_i(t) <= goal_tol)
and max_i |d_i(t-W)-d_i(t)| < progress_epsilon (=0.01)
and max_i ||u_i(t-1)|| < max_speed * speed_epsilon_fraction (=0.025)
```

若 candidate 连续为真，`candidate_since` 保持首次进入步，`stuck_timer=(t-candidate_since)dt`；否则立即清零。达到 `deadlock_hold_seconds=5.0` 触发 deadlock；完整连续运行从首次 candidate 到触发相差 100 步。仿真终止优先级为 collision、success、deadlock、timeout（[同文件](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/environment.py:223)）。本轮从 768 条 Toy seed0 缓存轨迹重放 `max_stuck_timer` 与正式 summary 全部一致。进入 candidate 本身并非不可逆：Toy 的 both-success 轨迹也可能短暂进入后退出。

**有限时域分支稳定命题。** 假设固定 seed、同一正规投影/Flow 分支上，至时域 H 的状态和动作对 η 局部 Lipschitz，且所有实际访问步的目标、进展、速度、碰撞等 guard 与阈值均有严格非零余量。在足够小 η 邻域内，所有 guard 布尔值及其先后顺序不变，candidate 计时器和终止分支也不变；在该分支内 Part 2 的 `S_{t+1}=(F_x+F_uK)S_t+F_uG` 可继续使用。这是有限步连续性与严格不等式的直接推论。若 guard 余量接近 0 或投影/clip/最近墙分支切换，则没有统一双侧 Jacobian；可报告方向差分、分段灵敏度或有限扰动界。**离散采样 monitor 不自动具有标准连续时间 hybrid saltation 所需的横截性。**

Toy 缓存的 1,669 条 FT-success/FF-fail 逐 seed 配对中，FF 第一个 candidate 中位约第 229 步；FT 进入 bay 的中位约第 160 步，且 1,479 条 FF 由 deadlock、190 条由 timeout 结束。它支持事件分支机制的时序与几何关联；尚未通过独立介入“仅恢复 bay entry/仅改变 guard margin”来证明中介因果性。Toy 旧 UID 820013 邻域的成功/死锁事件在约 0.003 η 变化时切换，也不允许把完整轨迹的终止函数视作固定光滑分支。

## 4. 三类陈述的边界

**数学成立：** 以上固定正规分支的局部闭环灵敏度、离散 guard 严格余量下的有限步分支稳定性；离散一步目标距离/环角的一阶展开。它们需要相应前提，不能从 B15 成功率倒推前提。

**实验支持的假说：** Toy field correction 使初始目标向动作与 H20 进展下降，通常错失让行 bay，进而沿低速/低窗口进展分支被 monitor 停止；Ring field correction 经 Flow 与最终投影后更常保留环向动作，可救回一部分遍历失败，也存在后期同步救回。placement 是操控变量，二者的完整效果在配对仿真中可归因于 placement 的总变化；各中间量的单独中介作用目前仍是相关性证据。

**尚不能主张：** A0/A1/B0/B1 为跨 η 或跨场景充分/必要条件；局部 M/G/K 可预测有限 Q16 的 B15 翻转；所有 Ring Gain 都是早期绕行；Toy 全部 Loss 都是独立的 monitor artifact；普通 smooth Jacobian 可穿越 deadlock/success 事件。当前同 η 分层 held-out 检验未显示 A/B 指标提供超出 η 身份的稳定状态信息，因此不添加新的“保证 gain/loss”定理。
