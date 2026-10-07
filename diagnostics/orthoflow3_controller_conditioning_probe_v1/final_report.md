# 下一步判别实验：controller 条件与 state–eta 反转

## 判定

本轮取得两项直接证据。

1. **允许未来 base controller 改变时，当前 h 不能唯一决定 Q。** 在完全相同的当前物理状态、当前 Flow reference、eta 和 matched seeds 下，仅更换后续 frozen Flow，8 个 state–eta pair 从 16/16 成功变成 0/16。该结论来自 controller 干预，不是 critic 预测。
2. **Ring 确实存在可复现的 state-dependent eta preference。** 两个不同 source family 的开发状态对同一对 eta 给出相反排序，且在独立新 seeds 上复现。但现有大面板中的一枚固定 eta 仍覆盖全部 24 states，因此这种交互尚不等于“为了成功必须 state-adaptive”。

最合理的研究对象进一步明确为：

\[
Q(h,\eta,c_{\rm controller}),\qquad
\mathcal B(h,c_{\rm controller}).
\]

这里 c 指未来闭环 controller 的相关行为条件，不能仅用“当前 Flow 输出相同”代替。实验尚未证明哪种低维 response representation 已经足够，也没有证明这一项就是原四折 LOSO 失败的全部原因。

## 1. 同一 h，不同未来 controller

从原 Toy 200-state cohort 按 source-group 哈希预先选 16 个不同 families。每个 state 测 Safety、fixed common eta、冻结 generator mean、冻结 sample-0，共 64 pairs。原 controller 的 1,024 个 matched seed 结果全部复用。

对照 controller 与原 controller 共享 step0 的 Flow0 reference、hard-safety projection、OrthoFlow3 correction 和执行动作；从 step1 起使用另一个已存在的 frozen Flow checkpoint（同架构、同训练环境，seed1）。未来 RNG keys、horizon、success 和 safety 定义相同。该 controller schedule 有独立数据库 fingerprint。这是明确规定的 counterfactual continuation controller，不是原生 Flow1 从第一步部署。

当前 214-D h 数值核对最大差异为 5.97e-16。Unified physical input 的所有组成量也相同：相同 raw state、当前 raw Flow0、geometry、margins、remaining time 和 committed-reference flag；未来 checkpoint 没有进入输入。

| 方法 | 原 Flow0 后续 B15 | Flow1 后续 B15 | 原 mean Q16 | 新 mean Q16 |
|---|---:|---:|---:|---:|
| Safety | 10/16 | 12/16 | 0.789 | 0.887 |
| Fixed common eta | 15/16 | 16/16 | 0.992 | 1.000 |
| Frozen generator mean | 9/16 | 5/16 | 0.699 | 0.328 |
| Frozen sample-0 | 8/16 | 5/16 | 0.535 | 0.324 |

64 pairs 中：17 个 |ΔQ16|≥0.25，12 个 ≥0.5；9 个 pair 在对 64 个 paired seed tests 做 Holm 校正后 p<0.05，涉及 7 个 source families；其中 8 个是 16/16→0/16。8 个 pair 失去 B15，4 个获得 B15。不是替代 Flow 整体崩溃：Safety 和 fixed common eta 的表现保持或提高。

例如 episode168、116、22、149、163 的 generator mean 均由 16/16 变成 0/16。这直接证明了：在此 controller family 上，将 c 隐藏后，同一个 (h,eta) 可以对应显著不同的 empirical Q。对这张成对实验表，任何不观察 c 的单值预测器，即使知道表中所有数据，也有至少 0.1025 的 pair-equal empirical MAE；这只是该平衡实验表上的代数下界，不是对总体误差的估计。

### 当前局部 response 能否解决？

我们记录了相同 step0 后物理状态、相同 seed 下，两 controller 的 step1 raw Flow、安全动作及执行动作差异。平均执行动作差异范数仅 0.0121。发生 16/16→0/16 的 case 中，该差异可低至 0.00065。跨 64 pairs 的 |ΔQ| 与 raw-response 距离 Spearman 约 0.25，与 executed-response 距离约 −0.29；这些受 eta 类型混杂的描述统计不支持“一个局部差异大小标量就足够”。小的即时动作差异可以在后续闭环中累积为完全不同的结果。完整 controller conditioning 的可学习低维表达仍未验证。

### 因果解释的范围

这个干预确认未来 controller 是 Q 的决定变量，并确认仅当前 Flow reference 不能在任意兼容 controller 间替代它。它没有证明：原四场景的不同 h 已发生同样的输入碰撞；原 LOSO 一定主要由该缺项导致；或加入任意 controller embedding 就能 zero-shot。四固定场景的 geometry 可能在已知支持内间接标识 controller，但这种关联无法自动保证对未见 controller/scene 外推有效。

## 2. 共享 eta 面板与独立 seed 复核

直接复用另一任务完成的 `orthoflow3_state_eta_interaction_panel_v1`。Ring、Four-Way 各 24 个不同 parent families 的 correction-needed TRAIN/validation **中途状态**，每个 state 测同一组 24 eta。它们不是新的 true-t0 TEST cohort。选择事先限定 Safety correction-needed，因此不能称为对所有 outcomes 完全盲选；eta panel 在本批 eta outcomes 前冻结。

Ring 发现 709 个幅度双侧至少 0.25 的 state-pair×eta-pair 反转，涉及 191/276 state pairs。重复组合相互相关，不能当成 709 次独立试验。Four-Way 为 28 个，涉及 24/276 state pairs。数值未决使用 Q 区间，未插补为失败。

与此同时，Ring 面板的 fixed eta [0.8806116178166121,0.02854111511260271,0.024503270164132118] 在 24/24 states 达 B15；Four-Way 的 fair fixed eta 也为 24/24。Ring 全面板 state-specific oracle 对最佳固定 eta 的 mean-Q 优势只有约 0.005–0.008。因此：state–eta interaction 存在，但当前面板的 robust-success 目标仍可由一个场景内 common eta 满足。

### Fresh-seed confirmation

按冻结规则从完整四格、非零 eta 的候选反转中，选择最小双侧 gap 最大者，ties 用 state UID/eta index。发现数据 seeds0..15 为 (16/16,0/16) 与 (0/16,16/16)。只对这 4 个格子运行未参与发现的 future indices16..31：

| 开发状态 | eta index4 | eta index17 | η4−η17 的 Q 区间 |
|---|---:|---:|---:|
| parent000690, t196 | 12 successes /15 valid +1 numerical unresolved | 0/16 | +0.750…+0.813 |
| parent000014, t350 | 1/16 | 16/16 | −0.9375 |

两状态的 matched-seed ordering tests 分别 p=0.000488 和 0.000061；对两项检验校正仍显著。即使未决 seed 取任一科学结果，反转方向和≥0.25 幅度都保持。这确认 Ring state dependence 不是仅由 discovery Q16 的偶然排序造成。

但第一状态原先 16/16 的 eta 在新 seeds 只有 12 successes/15 valid。**排序反转复现了，原 B15 certification 没有在这组新 seeds 上复现。**本轮没有将 seed16..31 冒充标准 B15 seeds，也没有将原 Q16 当作精确真实概率。

冻结 supervised unified critic 对这个 witness 的两 eta 分数为 state A: .874/.340，state B: .591/.914，方向正确。该 witness 是根据真实 discovery outcomes 选择而非模型获胜选择；但它仍只是已研究开发状态上的两候选诊断，不能构成 unseen-state 或 LOSO 成功证明。

## 3. 对 formulation 的更新

应区分三件事：

- **完整 feasibility field 的可辨识性**：对不同 future controllers，当前 h 缺少条件量，已经有直接反例。
- **state-conditioned preference 是否真实存在**：Toy 已有强证据；Ring 现在也有 fresh-seed 反转证据。
- **部署是否需要复杂 state adaptation**：当前 common eta 在这些面板仍很强，交互存在不等于复杂 generator/critic 必然带来成功率收益。

下一轮最有判别力的模型检验应是：用明确 controller-response 条件构造一个简单的 context-level eta prior + state residual，并与仅 prior 比较；模型选择限于 source families/scene。需要新的独立确认集才能声称修复跨场景泛化。此轮没有训练模型，也没有改变 generator 或 safety。

## 4. 数据与执行完整性

- Toy controller probe：requested2,048；exact reuse1,024；partial/aggregate0；new1,024。33 个 append-only journals 由单一 merger 原子入库；postflight2,048全部 exact，零 missing/conflict/collision/numerical failure。
- Ring fresh-seed witness：requested64；reuse0；new64，4 个 journals 入库。63 个有效结果，1 个数值未决。Postflight48 exact-complete slots +15 partial-reusable slots；1 个未决槽位保持不可复用。没有为完成表格修改数值失败定义或重试规则。
- 本线程新增总数：1,088 physical continuations，1,087 valid，1 numerical unresolved，0 collision。另一线程共享矩阵的12,350次执行仅复用结果，未在此重复采集。
- 首次提交1804在 import authority guard 处退出，执行continuation=0；记录了 import 顺序修复并重新冻结 runtime hash 后，1807完成Toy对照。1825完成Ring复核。
- 所有新记录保留明确 controller UID；使用显式 journal merger，避免历史 heuristic ingestor 把不同 Toy Flow checkpoint 合并为同一 controller。

复现入口：在 Basin_C1 根目录，使用 `.venv-c1/bin/python -m diagnostics.orthoflow3_controller_conditioning_probe_v1.experiment analyze`；Ring使用同目录模块 `ring_confirm analyze`。冻结协议、pair manifest、cache pre/postflight、逐pair结果、local response audit、raw journals引用与 `final_decision.json` 均保留。
