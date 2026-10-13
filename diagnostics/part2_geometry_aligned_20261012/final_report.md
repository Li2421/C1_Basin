# C1 Part 2：几何对齐的干预机制审计

日期：2026-10-12。主比较：Ring TT→TF、Toy FT→FF。**总判定：局部动作映射可读出几何方向，但“唯一有利的环向/让行方向导致 B15 Gain/Loss”的强版本不成立。** Ring 得到投影后绝对环向运动的行为证据；Toy 得到 Flow 反馈、侧向协调与事件路径的分段证据。二者均没有独立中介因果证明，也没有超过 η 身份基线的 B15 预测条件。

## 1. 复用、冻结定义与计算边界

先核对 [`reuse_inventory.md`](reuse_inventory.md) 中的 49,152 条确认评估、旧 [`机制报告`](../part2_gain_loss_mechanism_20261012/final_report.md)、H1–H4、`transfer_mechanism_v2` 的 16-key 首动作/Jacobian NPZ，以及 [`Part 2 局部报告`](../part2_closed_loop_sensitivity_validation_20261012/final_report.md)。旧完整 rollout 的控制器、初态、η 和 noise key 均未改。B15 定义始终为固定 16 seed 中至少 15 次成功；Ring TT→TF Gain/Loss `48/1`，Toy FT→FF `10/178`，每场景分母 384。两场景另有 Both-success/Both-failure：Ring `29/306`、Toy `25/171`，无 unknown。

[`protocol.json`](protocol.json) 在本轮数值汇总前固定：`v=η/||η||`、Ring 短圆弧/CW/CCW/安全 nominal 方向、Toy 目标推进/双机器人净空导数/向 bay 上移机会/侧向动作不对称，以及 UID 01–08/09–16/17–24 的初态分组。**历史研究已经看过全部 Q16 标签；这里的 held-out 仅是本轮计算规则的分组核对，不是前瞻未见状态。** 精确数据路径、checkpoint 哈希与命令在本目录 [`reproduce_from_saved.sh`](reproduce_from_saved.sh) 和 [`verification.json`](verification.json)。

按用户后续限制，不再开展任何新实验。该限制到达前已完成按协议固定 UID 01/09/17×η[0,4,8,12]×两场景的 24 个 **action-only** Flow 子步例（每例零/修正两次积分，共 48 次）；没有推进 simulator、完整 continuation、短 continuation、训练或 GPU 占用。后续只从已保存的 NPZ/CSV 重算与写报告。这 24 例是补充的有限样本，不能估计总体机制频率。

## 2. 真实映射和四个证据层级

执行式为 `TT=Π_T(a_safe+Bη)`、`TF=Π_T(H0(Bη))`、`FT=Π_T(HF(0)+Bη)`、`FF=HF(Bη)`；TT/FF 调用与 TF/FT terminal closure 见 [`factorial_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py:23)。`B` 由当前安全 nominal 重算；Flow 从相同 `h0` 起跑，每个物理步按 state 更新，场内投影的目标及反馈见 [`field_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:215)、[子步递推](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:222)。本轮固定 state/key 下比较：

1. **局部微分** `ΔE=wᵀ(G_q−G_p)v`，`G=M B`；旧 probe 的 `J_eta` 与 `M B` 做数值交叉检查（[`probe.py`](/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/probe.py:110)）。
2. **有限首动作** `wᵀ[u_q(η)−u_p(η)]`，同时拆为零点偏移和两个 correction secant。Ring TT/TF 旧 seed0 零点偏移中位约 `8×10⁻⁸`；Toy FT/FF 为零，但公式仍保留偏移。
3. **闭环运动**：同 seed0 的配对 H10/H20/H80 真实缓存轨迹，另复用全部逐 seed Gain/Loss 事件审计。seed0 不代替 B15 标签。
4. **B15 成员**：固定 16 seed 成功数，独立于上述单次动作或早期轨迹。

精确有限恒等式含 `u_q(0)−u_p(0)` 与沿 `sη` 的 Jacobian 积分，不能直接写 `u_q−u_p=(G_q−G_p)η`。若以终点 Jacobian 代积分，Ring 短圆弧读出的中位绝对误差 `0.00273 rad/s`，Toy 目标向均速 `0.1956 m/s`；Toy 178 个 Loss 中 **119** 个局部差为正，实际有限配对差却为负。见 [`theory_implications.md`](theory_implications.md)。

## 3. Ring：投影后环向行程有证据；“有利方向”强解释失败

定义 `r_i=p_i/||p_i||`、`t_i=(-r_{iy},r_{ix})`、`θ̇_i=t_iᵀu_i/||p_i||`。初始目标接近对跖：384 个候选中 **320** 个至少一台机器人距 `π` 不到 `0.05 rad`；48 个 Gain 中有 **40** 个。短圆弧顺/逆时针符号因此不稳定，不能事后根据成功路径选方向。CSV 同时报告固定 CW/CCW、短圆弧以及安全 nominal 所选方向。径向动作也单列；更大的 `|θ̇|` 不等于安全且同步完成的交换。

[`ring_directional_mechanism.csv`](ring_directional_mechanism.csv) 从全部 384 候选×16 同 state/η/key 的旧首动作 NPZ 重算，结论为：

|Ring 初态 16-key 指标|Gain 48|Both-failure 306|解释|
|---|---:|---:|---|
|TF−TT 最短圆弧有符号动作差 >0|10|84|**不是** Gain 必要条件|
|TF−TT 绝对角速度差 >0|34|96|部分运动对应，不保证完成|
|末端投影对绝对角速度的相对保留 >0|48|286|投影塑形普遍发生，区分力弱|
|TF−TT 最短圆弧局部 `ΔE>0`|20|109|局部符号不能推出 B15 翻转|

Gain 的投影前绝对角速度差中位 `−0.0171 rad/s`，投影相对变化 `+0.0258 rad/s`，最终差 `+0.0115 rad/s`：TF 的 Flow 输入相对 TT **更易经末端速度/CBF 投影保留环向量**，与旧动作分解一致。但 Both-failure 的投影相对变化中位还达 `+0.0317 rad/s`。唯一 Ring Loss（UID 920006、η[10]，`16/16→11/16`）的局部、有限短圆弧差、绝对角速度与投影保留均为正，直接否定“这些量为正足以防 Loss”。

四类候选在投影前/后差值上大面积重叠，见 [`首动作几何散点图`](figures/geometry_action_overlap.svg)。

闭环上，48 个 B15 Gain 的 seed0 H80 绝对环向行程差中位 `+1.2205 rad`、**47/48** 为正；Both-failure 也有约 **290/306** 为正。先前严格逐 seed 救回中，398/586 的 TT 从未让所有机器人各自进入目标区，另 188/586 为各自进入但未同时完成。故“更多可执行环向遍历”可解释一部分行为路径，尤其前者；不能解释全部同步分支，也不能识别唯一有利方向。12 个固定子步例中 `H0` 场内投影贡献为零，Flow feedback 改写修正路径，末端投影再作用于端点；有符号短圆弧贡献的符号混合，见 [`flow_substep_decomposition.csv`](flow_substep_decomposition.csv)。

## 4. Toy：Flow 反馈抑制首步通行推进；净空减少的说法被否定

目标向瞬时速度 `a_i=e_iᵀu_i`；双机器人净空函数及导数 `h=||p_1−p_2||²−D_safe²`、`ḣ=2(p_1−p_2)ᵀ(u_1−u_2)` 与真实 CBF 一致（[`cbf.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/cbf.py:73)）。`ḣ>0` 仅表示此刻分离；让行也可能需要减速、侧移或暂时退让，所以本轮还报告向上 bay 机会和两机器人横向速度不对称，不强造单一 `w_T`。Toy 通道宽 `0.4 m`、bay 从 `y>0.2 m` 起，见 [`environment.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/environment.py:17)。

[`toy_yielding_mechanism.csv`](toy_yielding_mechanism.csv) 的 16-key 首动作中，**178/178 B15 Loss** 的 FF−FT 目标向均速为负，中位 `−0.1689 m/s`；但 **10/10 Gain、25/25 Both-success** 也为负。更关键的是 **178/178 Loss 的 FF−FT `ḣ` 为正**，中位 `+0.8512 m²/s`，Gain/Both-success 也全部为正。因此“FF 因减小即时净空而失败”被直接否定。两机器人的横向速度不对称在 155/178 Loss 下降，符合通行协调弱化的一种读法；相同符号也出现在全部 Gain 和 Both-success，不能单独决定结果。

目标向缺口主要在投影**前**出现：Loss 的 FF−FT pre-action 中位 `−0.2364 m/s`，末端/最终子步投影相对变化中位 `+0.0574 m/s`，实际执行后为 `−0.1689 m/s`。这里的 `pre_FF` 是第 10 个 Euler proposal、`pre_FT=HF(0)+Bη`，两者的投影前后差是精确动作分解；不能把 FT 的末端 closure 漏掉。12 个固定初态 action-only 例的 Toy 目标向直注贡献中位 `+0.1537`、Flow 反馈 `−0.1191`、场内投影中位 `0`；仅 **12/120** 个 η 子步有可测投影形变。6 个示例级 B15 Loss 中直接贡献中位 `+0.2343`、反馈 `−0.1819`、场内投影 `−0.0032`。原始缓存首动作逐数重建误差为 0，安全证书最大违例 `3.17×10⁻¹²`。**这些例子支持 Flow feedback 而非场内投影主导首步方向变化；范围只限 12 个预定例。**

固定 UID01、η[4] 的两场景 10 子步方向分解见 [`子步示意图`](figures/substep_directional_example.svg)；它只显示已固定的示例，不是按成功标签选出的总体估计。

完整轨迹中，1,669 个逐 seed FT-success/FF-fail 配对的 FT 全部进入 bay，FF 仅 46；FF 1479 次因 deadlock、190 次 timeout 终止。首步动作差、H20 最慢机器人进展差和实际 bay/monitor 时序构成一致的行为链；但 H20/H80 的相对横向分离或 bay 高度在各 B15 类别大量重叠。真实 monitor 以 40 步窗口进展 `<0.01`、最大速度 `<0.025`、连续 5 秒 candidate 触发 deadlock，终止顺序由 [`environment.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/environment.py:195) 定义。此前 768/768 条 Toy seed0 监视器重放与正式 `max_stuck_timer` 相符。这个离散事件链支持**事后解释**，不能用光滑 `ρ`/S 一步外推 B15 Loss。

## 5. 按初态分组的预测核对与反例

[`heldout_validation.csv`](heldout_validation.csv) 保留每候选的 B15 标签、全部预定义连续分数和分组；[`heldout_metric_summary.csv`](heldout_metric_summary.csv) 给 AP、按 8 初态 bootstrap 区间与同 η 标签置换。没有拟合阈值。held-out Ring Gain 基率 `12/128=0.094`；绝对环向首动作差 AP `0.199`，最短方向有限差 `0.099`、局部差 `0.151`、原始动作差幅度 `0.088`，均低于只用 discovery B15 标签形成的 η 身份 AP `0.434`。Ring 只有 1 个全局 B15 Loss，不能评价双向分类性能。

held-out Toy Loss 基率 `61/128=0.477`；首动作目标向缺口 AP `0.567`、原始动作差幅度 `0.571`、H20 最慢机器人进展差 `0.529`。横向不对称削弱 AP `0.702`、同 η 置换未校正 p≈`0.016`，但 η 身份 AP `0.888` 更强，8 个 held-out 初态 bootstrap AP 区间约 `[0.455,0.873]`；本轮并列考察了多项几何指标且历史标签已被检查，p 不能当独立确认。该特征在 Both-success/Gain 中也常见。**不判定 held-out B15 预测条件通过。** 具体反例与数值见 [`counterexamples.md`](counterexamples.md)。Flow key 的 16-key 平均与 seed0 早期两链模拟均需离线噪声协议；真实部署不预知未来噪声。

## 6. M/G/K、理论提升与结论等级

本轮从旧 `J_eta` 与 `M B` 算 `wᵀGv`，固定 state/key 的局部动作方向有数值依据：16-key 候选平均的链式相对误差中位 Ring TT/TF `0.00119/0.00424`，Toy FT/FF `5.8×10⁻¹¹/2.4×10⁻⁷`；旧 probe 中不稳定活跃点与较大尾部仍保留。**K 与闭环 S 未在全部 384 候选重新验证**；旧 Part 2 只在登记的正规 Ring 小样本显示 H80 局部预测较好，Toy 切换附近为 NONSMOOTH。不能由 M/G 的局部符号跳到有限首动作，更不能跳到 bay 事件或 B15。Toy 的有限目标向缺口甚至常与当前局部导数相反。闭环传播的正确对象仍是逐步重算 `B` 与 CBF/Flow 状态导数的 `S_{t+1}=(F_x+F_uK_t)S_t+F_uG_t`；跨监视器 guard 分支时普通 Jacobian 无统一适用性。

**七问回答：** (1) Ring 的**绝对环向投影保留**获直接动作级支持，唯一任务有利有符号方向未获支持；(2) Toy 的首步反馈抵消与后续通行/事件路径获分层支持，“净空下降”及“场内投影主导”未获支持；(3) M/G 可解释固定状态的局部响应，K/S 只限既有正规样本，不解释有限 B15；(4) 几何量未超过 η 身份基线，不能提前可靠识别翻转；(5) 有直接反例，详见附录；(6) 没有非循环且已核前提的新严格 Gain/Loss Proposition；(7) 保留 Part 2 正规局部机制和 Toy 离散事件限制，收缩场景几何表述。

**独立判定：** `GEOMETRIC_MECHANISM_SUPPORTED=PARTIAL`（Ring 绝对运动、Toy 行为链；强方向版本被否定）；`LOCAL_THEORY_VALIDATED=PARTIAL`；`HELDOUT_PREDICTION_VALIDATED=FAIL`；`CAUSAL_MEDIATION_SUPPORTED=INSUFFICIENT_EVIDENCE`；`INSUFFICIENT_EVIDENCE=true`。RSS 正文可写真实四链、正规局部 M/G/K/S、上述动作/轨迹/事件分层证据与明确反例；不得写几何指标保证 B15 Gain/Loss，或把代理裕度、单个 Jacobian、事后 bay 事件称作完整盆边界或独立因果中介。
