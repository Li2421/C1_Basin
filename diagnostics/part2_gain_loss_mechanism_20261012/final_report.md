# C1 Part 2：placement 依赖的 Gain/Loss 机制发现

日期：2026-10-12。范围：Toy Give-Way **FT→FF** 与 Ring Exchange **TT→TF**；冻结控制器、η、初态和配对 Flow/未来噪声。**结论：可识别物理行为链与 Toy 离散事件链；尚未验证超出固定 η 候选身份的可迁移 B15 预测条件。** 这不是 FF 普遍优于 TT 的结论。

## 1. 证据、设计与成本

实验前复核 [`reuse_inventory.md`](reuse_inventory.md)：`Basin_C1_flow_field_poc_20261004` 的四链控制器、`Basin_C1_hard_field_20261006` 的 49,152 条 Q16 确认评估、`transfer_mechanism_v2` 的 16-key 首动作测量、H1–H4 与上一轮 [`Part 2 报告`](../part2_closed_loop_sensitivity_validation_20261012/final_report.md)。上轮的 `gain_loss_audit.csv`、完整轨迹及 boundary 数组均复用；本轮**没有重新训练、M/G/K 大扫描、GPU 计算或新增 continuation（完整 0/320，短时 0）**。正式 B15 始终是同一初态/η 的固定 16 seeds 中至少 15 次成功；seed0 轨迹只用于机制特征，不代替 B15。

[`protocol.json`](protocol.json) 在本轮条件评分前固定每场景 24 个初态的分组：UID 01–08 discovery、09–16 validation、17–24 held-out；同一初态全部 η/seed 同组。每组 8×16=128 个候选。阈值定义及可证伪预言先写入 [`mechanism_hypotheses.md`](mechanism_hypotheses.md)，评分见 [`heldout_prediction.csv`](heldout_prediction.csv)。需要坦承：旧 `transfer_mechanism_v2` 已对全 Q16 做过 outcome 分析，本轮早期探索也看过跨全部候选的 η 分布；所以这里是**本轮阈值选择的分组留出检查**，不是真正从未观察过数据的前瞻实验。最严格的泛化问题因此仍未决。

## 2. 主 pair 的控制差别与零点

真实控制式为 `TT=Π(a_safe+Bη)`、`TF=Π(H0(Bη))`、`FT=Π(HF(0)+Bη)`、`FF=HF(Bη)`（[`factorial_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py:23)、[`field_controller.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py:222)）。Toy FT/FF 均有 field safety；FT 另有 terminal closure，FF 从头把修正放入每个 Flow 子步。Ring TT/TF 均有 terminal safety；差别主要是 terminal 注入与无 field 投影的 Flow 内注入。B 每物理步重新计算，η 全程固定。初态 η=0 的 24 状态 seed0 动作偏移：Toy FT/FF 全为 0；Ring TT/TF 中位 `7.96×10^-8`、最大 `1.26×10^-7`（[`baseline_offset_summary.json`](baseline_offset_summary.json)）。所以本轮主要 pair 的首步差不是大的零修正基线偏移。

## 3. Toy FT→FF：先失去让行通路，再进入离散监视器分支

Toy B15 候选 **Loss 178、Gain 10、Both-success 25、Both-failure 171，分母 384**；没有 unknown。逐 seed 的 178×16=2,848 条属于 B15-Loss 候选，其中 FT-success/FF-fail 为 1,669 条、双方成功 1,155、双方失败 21、反向 gain 3。1,669 条实际翻转中，FF 以 **deadlock 1,479、timeout 190** 结束。死亡不是“不安全”：正式投影证书和仿真碰撞判据未被改动；`deadlock` 与 `timeout` 严格区分（[`event_pair_audit.csv`](event_pair_audit.csv)）。

同 state/η/key 的首动作已经不同，翻转配对首动作 L2 差中位 `0.24375`；联合位置分离超过 5 cm 的首步中位为第 5 步，最差机器人 H20 进展 `FF−FT` 中位 `−0.17876 m`。旧 16-key 首动作表的目标向安全-nominal 有符号分量（精确定义见 [`mechanism_hypotheses.md`](mechanism_hypotheses.md)）在 B15-Loss 候选中：FT 投影前/执行后中位 `0.432/0.341`，FF 为 `0.179/0.179`。两链投影**前**就已有目标向差，FT terminal closure 将其自身分量从 `0.432` 减至 `0.341`，不能把 FF Loss 单独归因于 FT 末端投影。场内修正及 Flow 反馈改变可执行动作；CBF 又改写每链动作。B15-Loss 实际逐 seed 翻转的前 80 步，FT 线性 CBF 近活跃比例中位 `1.0`，FF 为 `0.3875`；这说明活跃状态不同，**不**证明活跃集切换是独立原因。速度球活动未由这个线性残差统计代表。

让行 bay 的几何行为非常清楚：在 1,669 条逐 seed 翻转中，FT 轨迹 **全部**进入 bay（机器人中心满足 `|x|≤0.2,y>0.2`），FF 只有 46 条进入（`2.76%`）；1,623 条为 FT 进入而 FF 未进入。FT 首次入 bay 的中位是第 160 步；FF 首次进入 Toy monitor 的 candidate 状态中位第 229 步，其中 1,294 条是 FT 已先入 bay。所有逐 seed 翻转的 FF 都曾进入 candidate；仅凭“进入过”并不足以断言死锁，因为 candidate 可重置，B15-Loss 候选在 seed0 双方成功时也可能短暂进入。这个时序支持**早期推进/让行路径不同 → FF 后续低速且窗口进展不足 → deadlock 或 timeout** 的行为解释，不能仅凭时间先后证明 bay 或目标向分量是唯一因果中介。完整行见 [`critical_event_rows.csv`](critical_event_rows.csv)，历史固定 η[4]、UID 820001、seed0 的轨迹图见 [`Toy 图`](figures/toy_give_way_eta4_seed0.svg)。

Toy monitor 的候选 guard 是 `window_ready ∧ ¬success ∧ max_i|d_i(t−40)−d_i(t)|<0.01 ∧ max_i||u_i(t−1)||<0.025`；连续 5 秒（首次进入到触发相差 100 个 0.05 秒步）才触发 deadlock，任一非候选步会清零计时。终止优先级为 collision、success、deadlock、timeout（[`environment.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/source/single_integrator/environment.py:195)）。从已存位置/动作重放 768 条 Toy seed0 轨迹，其 `max_stuck_timer` 与正式 summary **768/768 相符**。这给出了真实事件分支证据；跨事件的普通双侧轨迹 Jacobian 不受保证。

## 4. Ring TT→TF：可执行环向运动与两种救回路径

Ring B15 候选 **Gain 48、Loss 1、Both-success 29、Both-failure 306，分母 384**；没有 unknown。48×16=768 条种子评估中，逐 seed TT-fail/TF-success 为 586、双方成功 172、双方失败 7、反向 loss 2、数值未知 1。586 条救回的 TT 失败轨迹分成两类：**398** 条至少一台机器人从未进入自身目标区，属于遍历/通行失败；**188** 条所有机器人曾各自进入目标区，但从未同时完成，属于后期同步失败。不能用“初始环向速度较大”统一解释后者（[`event_summary.json`](event_summary.json)）。

逐 seed 救回时，同 state/key 首动作差中位 `0.07429`，5 cm 联合位置分离的首步中位为第 8 步。H20 最慢机器人进展 `TF−TT` 中位 `−0.05287 m`，到 H80 转为 `+0.05388 m`；H80 总绝对环向行程优势中位 `+1.3534 rad`，586 条中约 `96.76%` 为正。这支持一条**先接受局部目标距离代价、再完成更多环向遍历**的分段行为路径。旧 16-key 初态动作表在 B15-Gain 候选中：TT 的安全-nominal 切向有符号分量，投影前/后中位 `0.094/0.058`；TF 为 `0.099/0.091`。两者投影前接近，TT 被 terminal 投影削减更多；terminal 目标到执行动作的形变中位 TT `0.531`、TF `0.117`。因此 Ring 的优势并非简单的“Flow 在投影前已放大环向速度”：**Flow 改变投影输入与速度球的相容性，末端投影后的可执行环向动作才更常保留。** 旧初态 Ring 投影活跃主要为速度球，前 80 步线性 CBF 残差近零比例的组中位为 0，不表示安全投影未起作用。历史固定 η[4]、UID 920001、seed0 图见 [`Ring 图`](figures/ring_exchange_eta4_seed0.svg)。

上述是 placement 总干预下的受控同 state/key 首动作差和配对轨迹差；**环向运动/末端投影形变作为中介仍是关联与动作级反事实证据**。旧 H1 的整段动作族 donor 交换覆盖 TT/FF，不能冒充本轮 TT/TF 的中介实验；尤其 188 条后期同步救回仍缺单独机制归因。投影前/后与四类 outcome 的数值在 [`initial_action_decomposition.csv`](initial_action_decomposition.csv)。

### Flow 子步的直接检查（补充，示例级）

旧 `transfer_mechanism_v2` 的缓存有每子步**投影形变**，没有两条修正路径的完整 `h_k`；不能把形变数组说成 correction 传播。为补这个窄缺口，按历史固定 η[4]、seed0、每场景事先固定 UID 01/02，在初态用冻结 checkpoint 做了**四个动作级、10 子步配对探针**，没有调用 `env.step`。对每例从完全相同的真实 `h0` 起跑，Toy 比较 `HF(Bη)` 与 `HF(0)`，Ring 比较 `H0(Bη)` 与 `H0(0)`；逐子步记录 `||h_k(Bη)−h_k(0)||/||Bη||`。Toy 两例终点比率为 `0.274/0.078`，Ring 为 `0.299/0.403`；线性无反馈参考终点为 `1`。Toy 的路径差先增长后衰减，η 路径分别有 `2/6` 个 Flow 子步出现可测投影形变；Ring 的 `H0` 无子步安全投影，衰减来自 Flow 状态反馈与其离散积分。执行动作按真实末端逻辑重建：Toy `FF=HF(Bη)`，Ring `TF=Π_T(H0(Bη))`，四例重建误差均为 0（数值记录见 [`field_substep_summary.json`](field_substep_summary.json)、[`子步图`](figures/field_substep_eta4_seed0.svg)）。四例恰好都是主 contrast 的 B15 翻转：Toy `16→1/16`，Ring `0→16/16`，**只作示例，不用于总体频率或新预测条件**。`Bη` 的子步效应因反馈/投影明显不是简单相加；这还不足以单独归因于某个子步、投影或中介。

四例重建的旧链、新链首动作还分别与正式缓存中的同 state/η/key rollout 首动作逐数一致，L2 误差均为 `0`；这验证了探针的 key 和 checkpoint 对齐。

## 5. 冻结条件的分组预测检验

两个机制条件均不以最终标签定义：**B0** 为 Toy 首步 16-key 目标向安全-nominal 动作差 `FF−FT<−0.14`；**B1** 为 H20 最慢机器人进展差 `FF−FT<−0.12 m`。**A0** 为 Ring 首步 16-key *执行动作总量*的环向投影范数差 `TF−TT>0`；**A1** 还要求 H20 最慢机器人进展差 `TF−TT<0`。B0/A0 是初态控制器计算量；B1/A1 要运行两条配对早期轨迹，属于 early simulation predictor。16 个 Flow key 和未来噪声在真实部署时不保证事先知道，故离线 16-key 特征不等于可直接在线使用的单步控制规则。

下表的混淆矩阵为 `TP/FP/FN/TN`，positive 分别是 Toy B15 Loss 与 Ring B15 Gain；其他已知三类均为 negative。AP 为连续规则分数排序得到的 average precision，不是二值规则的 precision。

|Scene / rule|Validation (128)|Held-out (128)|Held-out P / R / BA / AP|
|---|---:|---:|---:|
|Toy B0|43/28/17/40|50/33/11/34|0.602 / 0.820 / 0.664 / 0.688|
|Toy B1|53/41/7/27|57/42/4/25|0.576 / 0.934 / 0.654 / 0.529|
|Ring A0|12/43/5/68|9/48/3/68|0.158 / 0.750 / 0.668 / 0.215|
|Ring A1|11/30/6/81|9/39/3/76|0.188 / 0.750 / 0.705 / 0.324|

Held-out Toy Loss 基率 `61/128=0.477`，Ring Gain `12/128=0.094`。固定预测负类的 BA 为 0.5、AP 等于基率；discovery 中位数阈值的首动作差幅度基线在 Toy/Ring held-out 的 BA 为 `0.615/0.478`、AP `0.539/0.102`；初始最远目标距离基线 BA `0.430/0.592`。按 8 个独立初态重采样的 95% 描述性区间：Toy B0 BA `[0.575,0.748]`；Ring A1 `[0.587,0.831]`。PR 图见 [`heldout_pr.svg`](figures/heldout_pr.svg)。Ring A1 有 **39 个 false positives 对 9 个 true positives**，因此不能把它写作可靠 Gain 保证；Ring 仅 1 个 B15 Loss，不存在可信的 Ring Loss 分类器。

**关键反证：η 身份解释了相当多的表面预测力。** 只以 discovery 标签给每个固定 η 编号估计正例频率的非物理对照，在 held-out 的 Toy/Ring BA 为 `0.824/0.879`、AP `0.888/0.434`，均强于 B0/A1；它只作诊断对照，**不是可接受的机制条件**。固定 η 编号内置换 held-out 初态标签，B0/B1 的单侧 p=`0.565/0.992`，A0/A1 为 `0.681/0.662`（5,000 次；[`stratified_eta_summary.json`](stratified_eta_summary.json)）。这些 p 值不证明条件无作用，但现有证据**不能证明它们提供超出 η 候选身份的可迁移状态信息**。Toy B0 相比首动作幅度 BA 的按初态 bootstrap 差约 `+0.049`，区间 `[-0.034,+0.118]`，也未稳健胜出。最典型反例：Toy UID 820017 η1 满足 B0/B1，却是 FT 7/16、FF 0/16 的 **Both-failure**；Toy UID 820017 η8 是 FT 16/16→FF 3/16 的 Loss，却未满足 B0。Ring UID 920017 η0 满足 A0 却双方 0/16；UID 920017 η1 为 TT 1/16→TF 16/16 的 Gain，却不满足 A0/A1。全部反例在 [`heldout_prediction.csv`](heldout_prediction.csv)。

**H10/H20/H80 统一复核（补充、事后探索）：** 不调阈值，直接用缓存的最慢机器人进展差做 Toy Loss 排序，held-out AP 分别为 `0.543/0.529/0.486`，基率 `0.477`；Ring Gain 用绝对环向行程优势排序，AP 为 `0.166/0.119/0.157`，基率 `0.094`。Ring 使用早期退让分数（负进展差）得到 `0.187/0.214/0.165`。全部三时域、三 split 和按初态 bootstrap 区间在 [`horizon_ablation.csv`](horizon_ablation.csv)。H80 在本项目通常仍早于完整完成，但并非事前可观测：需要双 placement 配对短轨迹及未来噪声。这里是**在已见 held-out 结果后的描述性时域敏感度检查**，不能作为第二套独立验证，也没有解决 η 身份混杂。Toy H80 排序接近基率，Ring 的环向优势未形成可靠候选级 Gain 排序；因此完整轨迹中清晰的行为差异不能自动转成早期 B15 预测。

## 6. 四链交互复核

[`four_chain_interaction.csv`](four_chain_interaction.csv) 对原 Q16 的同一 384 个候选重算五个 contrast，**不是** Sobol 有界盒体积分数。Toy：correction placement 的 TT→TF 为 gain/loss `6/111`，FT→FF 为 `10/178`；safety placement 的 TT→FT 为 `92/9`，TF→FF 为 `20/0`。所以 Toy 的 FT 优势同时体现 field safety 与 terminal correction 的交互；候选级二值交互 `y_FF−y_FT−y_TF+y_TT` 平均 `−0.1641`。Ring：TT→TF `48/1`，FT→FF `43/5` 加 10 unknown；TT→FT `8/3` 加 10 unknown，TF→FF `2/4`；已知四链完整的 374 个候选交互平均 `−0.01337`。Ring correction placement 的净 gain 在两种 safety placement 下均可见；Toy 的两种 placement 强烈不同。未知不改判为失败，也不与各 contrast 的分母混算。先前 H3 Sobol64+128 的 TT/FF 有界盒平均分数差 Ring `+0.18142`、Toy `−0.12847`、Combined `+0.02648`，测度与本段 Q16/B15 不同。

## 7. 理论与 RSS 使用范围：逐问结论

**Q1 Toy Loss：** 最早的可测变化是首步目标向动作显著减弱，前 20 步最慢机器人推进落后；多数实际翻转的 FF 未形成 bay 让行，随后进入低速/低窗口进展分支，1,479/1,669 因 deadlock 终止。placement 的**总效果**由同 state/η/noise、冻结代码的配对仿真支持；“目标向动作/入 bay 是独立且充分的因果中介”尚未由只改变该中介的整段安全反事实证明。

**Q2 Ring Gain：** Field correction 经 Flow 形成较易被末端速度球投影保留的环向执行动作；部分轨迹先有局部目标进展代价、后有环向遍历收益。398/586 逐 seed 救回是 TT 遍历失败，188/586 是后期同步失败；后者机制仍未解决。结论限于本 Ring checkpoint/η 候选。

**Q3/Q4 可预测条件与反例：** B0/A0 可从初态和冻结控制器离线计算；B1/A1 需 H20 两链模拟。分组 held-out 有高于多数类的表面 BA，但 η-only 标签对照更强，同 η 分层无稳定增量，且 false positives 明显。**不判定 `PREDICTIVE_CONDITION_VALIDATED`。** 每场景仅 8 个 held-out 初态、旧审计又曾观察全 Q16，不能声称新状态泛化。

**Q5 从 M/G/K 推导：** 可推导正规分支内的有限修正积分、目标距离和环角一步响应、`S_t` 传播；不能从零点 M/G/K 推出大 η 下 bay entry、同步完成或 B15 标签。首动作投影前后分解支持 Flow/CBF 共同塑形，但不提供单独中介因果识别。

**Q6 非光滑理论：** Part 2 应最小增加**有限步离散 guard 分支框架**：定义目标、窗口进展、速度、碰撞 guard 余量；严格余量且活跃集正规时事件序列局部稳定并可用原 `S_t`；余量消失/guard 顺序变化时只给方向差分或分段/有限扰动界。Toy 的 monitor 不是自动满足连续时间 saltation 横截性的光滑混合系统。数学命题与证据分级见 [`theoretical_implications.md`](theoretical_implications.md)。

**Q7 RSS：** 可以陈述四链的真实控制图、正规局部灵敏度、Toy 的让行/monitor 路径、Ring 的投影后环向/遍历路径，以及同样本 B15 gain/loss 的**机制一致性证据**。A/B 指标只作为探索性早期物理特征，不能写成已验证的场景无关充分条件、可提前确定 B15 的理论或完整因果中介。

**独立判定：** `MECHANISM_IDENTIFIED`＝行为链支持、独立中介未证；`PREDICTIVE_CONDITION_VALIDATED`＝否；`EVENT_MECHANISM_SUPPORTED`＝Toy 是；`INSUFFICIENT_EVIDENCE`＝可迁移预测与 Ring 后期同步机制是。机器可读 [`verification.json`](verification.json)；复现命令见 [`reproduce.sh`](reproduce.sh)。
