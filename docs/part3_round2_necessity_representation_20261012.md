# C1 Part 3 Round 2 — 独立 necessity 确认与表示修复诊断（2026-10-12）

## 结论

**固定的两个真实 Ring Flow controller、FF、共同 η₁/η₂ 和五个全新 true-t0 state 上，controller-specific Q 与有限双候选选择冲突得到独立的经验确认。** 预注册的 seed 半分交叉拟合 selection value 平均为 **+0.175**，但五个 state 中有一个反例，四个正 state 的精确单侧符号检验为 `p=0.1875`。这支持一个**限定于该 controller pair、Ring 几何及双候选的经验 necessity 论断**；尚不足以宣称总体 population necessity，也未证明可部署的学得 selector 会获益。

R5 表示确实保留了旧 R3 压缩掉的按 pose/time 索引的 near-goal moving response。它在现有 P1 标签的留一真实 checkpoint 最近源诊断中**没有**改善 Q 误差或 η 选择；表示充分性、跨 checkpoint 泛化和有用的理论上界仍未得到验证。**Part 3 比上一轮更接近 RSS 论文的限定性 necessity 证据，但还没有达到广义 controller-conditioned selector/generalization claim 的 paper-ready 状态。**

## 1. 冻结设计、旧证据和缓存

执行前重查了 [Part 3 主报告](part3_theory_validation_20261012.md)、[P1 真实 Flow 面板](../diagnostics/controller_q_transfer_round3_v1/REPORT.md)、P1 的 checkpoint、状态/η 协议、Q4 原始标签及 [共享缓存规则](../shared_rollout_db/README.md)。本轮未修改 Flow、OrthoFlow3、投影、物理、事件或评价器；R5 只用于静态诊断，不参与 Q16 outcome 生成。

[预注册分析方案](../diagnostics/part3_necessity_representation_round2_20261012/PREREGISTRATION.md)、[冻结协议](../diagnostics/part3_necessity_representation_round2_20261012/protocol.json)、[320 个精确 case](../diagnostics/part3_necessity_representation_round2_20261012/cases.json)和 [SHA256 收据](../diagnostics/part3_necessity_representation_round2_20261012/freeze_hashes.sha256)均在读取新结果前保存。协议 SHA256 为 `c7a09486d9f88c82bde0b37368819cecc5759e3199d211127669494924282873`。checkpoint、adapter、manifest 指纹与旧 P1 冻结 profile 逐项重放核对。

- **controller**：`base_u_v2_macflow`、`base_u_v10_local_macflow`。二者是真实不同的 Flow checkpoint；其观测 chart 也分别遵循自身训练配置，因此本实验识别的是完整 controller program 差异，不能单独归因于网络权重。
- **共同候选**：旧 P1 seeds 0–1 上，逐对最大化相反的跨 state 平均偏好分数，按旧两候选 V 和索引破平局，冻结原 K16 的 `η₁=[1.0193532314151525,0.2190246358513832,0.0731143222656101]`、`η₂=[1.142077801283449,−0.11184126418083906,0.5354755260050297]`。没有使用 P1 seeds 2–3 或新标签选候选。发现分数 `0.35417`，旧 discovery V `0.14583`。
- **新 state**：沿用既有自然 Ring `sample_instance` 生成器，取 `part3_round2_confirm` 的连续 draw seed `2026101200–2026101204`，全部保留。执行前数据库中五个 state UID 和物理 content hash 均不存在；没有依据 outcome、困难程度或静态 controller 响应更换 state。
- **随机流**：既有 root `2026100403`，同一 state UID、seed `0–15`、物理 step 经相同 fold-in 生成；两个 controller/η 使用 common random numbers。相同归一化 Gaussian key 在不同 chart 下不意味着相同世界坐标的初始伪动作。
- **预算**：`2×5×2×16=320` 个精确请求。正式 [cache preflight](../diagnostics/part3_necessity_representation_round2_20261012/cache_preflight.json) 为 exact reuse 0、ambiguous 0、genuinely missing 320。四个 CPU worker 各执行 80 个，新增唯一 continuation **恰为 320**，合计仿真 CPU 时间约 1,684 秒；GPU rollout 用量 0。纯 Q 确认按仓库规则不要求视频。

单进程 journal [merge](../diagnostics/part3_necessity_representation_round2_20261012/merge_result.json) 插入 320、重复 0。[数据库 postflight](../diagnostics/part3_necessity_representation_round2_20261012/postflight.json) 验证 valid **318**、numerical-unknown **2**、missing **0**、quarantined **0**。重跑官方 [cache postflight](../diagnostics/part3_necessity_representation_round2_20261012/cache_postflight.json) 显示 318 个有效 seed 可精确复用（288 exact cells + 30 partial-cell seeds）；2 个 numerical-unknown 继续按规则显示为 missing，**没有**被悄悄重跑或计作失败。真实终止分解：success 99、timeout 219、numerical failure 2；本面板无 collision/deadlock 终止。原始逐 seed JSON、工作日志、静态特征和复现入口均保存在 [实验目录](../diagnostics/part3_necessity_representation_round2_20261012/)。

## 2. Task A：新 state Q16 与双候选冲突

以下 `s/16` 为 16 个固定 seeds 的成功数。带 `+?` 的 cell 有一个 numerical-unknown，其 Q16 区间是 `[s/16,(s+1)/16]`；其余 cell 是完整经验 Q16。每个 cell 的 observed-valid Wilson 95% 区间、终止原因与每 state 偏好区间见 [完整数值结果](../diagnostics/part3_necessity_representation_round2_20261012/necessity_results.json)。

| 新 state draw seed | v2：η₁ / η₂ | v10 local：η₁ / η₂ | η₂−η₁ 的 controller 交互 `I` | Q16 两候选 V | seed 交叉拟合 V |
|---:|---:|---:|---:|---:|---:|
| 2026101200 | `0/16` / `3/16 + ?` | `2/16` / `3/16` | `[0.125,0.1875]` | `0` | `−0.03125` |
| 2026101201 | `0/16` / `2/16` | `13/16` / `0/16` | `0.9375` | `0.0625` | `0.0625` |
| 2026101202 | `0/16` / `16/16` | `15/16` / `0/16` | `1.9375` | `0.46875` | `0.5` |
| 2026101203 | `0/16` / `6/16` | `15/16` / `0/16` | `1.3125` | `0.1875` | `0.1875` |
| 2026101204 | `0/16` / `5/16 + ?` | `14/16` / `5/16` | `[0.875,0.9375]` | `0.15625` | `0.15625` |

预注册的 `I_s=[Q(v2,η₂)−Q(v2,η₁)]−[Q(v10,η₂)−Q(v10,η₁)]` 在五个 state 都为正；未知值任意赋值后的均值区间为 **`[1.0375,1.0625]`**。按 state 重抽样的未知值保守下界 95% 区间为 **`[0.5125,1.5625]`**；额外的 state＋配对 seed 分层 bootstrap 为 **`[0.5,1.5625]`**。后一项是敏感性分析，没有更改预注册判定。Seed 在四个 controller/η cell 内配对；各 state 的 seed bootstrap 区间保存在结果 JSON。候选 η₁ 在 v2 下五个 state 全为 `0/16`，而 η₂ 在 v2 下差异很大，说明冲突并非单纯 Q 整体平移。

**critic necessity：有限实验条件下确认。** 相同物理 state、η、seed 配对的两个真实 controller 产生大幅不同的经验 Q16，且预注册交互门槛在新 state 上通过。一个只依赖 `(state,η)` 而不区分这两个 controller 的精确 Q 函数不可能同时表达这些观测条件值。该判断不保证所有 Ring state、全部 K16、其他 checkpoint 或连续 η 域的总体差异。

四个 state 的 Q16 区间给出严格相反的 η 偏好：v2 选 η₂，v10 local 选 η₁。**state `2026101200` 是必须保留的反例**：两个 controller 的经验 Q 都偏向 η₂ 或近似持平，Q16 插件 V 为 0，交叉拟合 V 为负；其配对 seed 交互 bootstrap 区间 `[-0.1875,0.4375]` 跨 0。这个失败来自状态异质性和低成功率/有限 seeds，并非 numerical-unknown 可以解释掉；它也展示双候选的覆盖限制。

Q16 插件 V 的五 state 均值为 **0.175**，未知值保守区间 `[0.175,0.1875]`。由于在同一 16 seeds 上取最大值会向上偏，**主选择终点**先用 seeds 0–7 为各 controller 和 controller-blind 混合体选择 η，再在 seeds 8–15 配对评价，并交换两半求平均。其均值为 **`+0.175`**，numerical-unknown 保守区间 **`[+0.175,+0.18125]`**；五 state grouped bootstrap 的未知值保守下界 95% 区间为 **`[+0.03125,+0.35]`**。四个 state 为正，一个为负；若只对五个符号作对称零假设的精确单侧检验，`p=0.1875`。小样本 bootstrap 正下界与符号检验不显著同时报告，不能选择其中有利的一项冒充总体证明。

**selector necessity：预注册的独立经验门槛通过，范围限于这个 pair、两候选和五个新 state。** 这证明一个知道 controller 身份、且有这些 state 的 Q 信息的选择规则在这个有限面板上具有经验信息价值。它不证明部署时可从 state/controller 表示准确预测最优 η，也不证明对新 checkpoint 的 learned selector 优于固定 η。五个 state 不构成 population necessity 认证；双候选甚至未覆盖各 controller 的全 K16 最优，更不能用于连续 basin 形状论断。

## 3. Task B：表示修复与真实行为距离

[R5 静态脚本](../diagnostics/part3_necessity_representation_round2_20261012/repaired_features.py)在 P1 原来的 14 个真实 checkpoint × 24 个旧 state 上复用同一套无 outcome probes：四个 nominal draws、七个物理/pseudo-state pose、三个 pseudo-time、五个 pseudo-action。R3 的旧 64 维/agent 被逐值重建，最大回放差 `2.38e−7`；R4_action 保持旧文件。R5 为每 agent 的 `R3(64)+按 7×3 索引的 2×2 response Jacobian(84)+近目标 moving pose 4–6 的原始五响应(90)`，共 238 维/agent。指定 near-goal response 从 R5 回放的最大误差 **0**。[静态收据](../diagnostics/part3_necessity_representation_round2_20261012/feature_receipt.json)证明 0 次 `env.step`、0 次 outcome 读取、14 个 checkpoint 全完成；仅使用一个 GPU shard，约 40 秒。

[代数反例](../diagnostics/part3_necessity_representation_round2_20261012/alias_counterexample.json)以一个旧真实 probe 张量的数值为底，交换远场与近目标 moving 的条件响应块，保持每处 intercept 不变。构造后的 R3 **完全相同**，near-goal response 差 `1.197`、R5 差 `1.296`。这证明旧压缩器本身不能保存响应来自哪个 pose/time；**构造张量不是另一个真实 Flow checkpoint**。在真实 P1 的 2,184 个 checkpoint-pair×state 比较中，尚未发现“R3 精确相同而 raw probe 行为不同”的实例。

[表示诊断](../diagnostics/part3_necessity_representation_round2_20261012/representation_results.json)将距离与相同 state 下的**采样** raw field/nominal response、R4 已执行动作 probe、旧 Q4 差异比较。这里的最大值和相关性只属于有限 probe 集，不是全域行为上界。

| 距离或留一 checkpoint 指标 | R3 | R4_action | R5 |
|---|---:|---:|---:|
| 与 near-goal moving 原始响应距离的 Spearman | 0.665 | 0.319 | **0.891** |
| 与全部采样 raw behavior 距离的 Spearman | **0.980** | 0.696 | 0.870 |
| 与已执行动作 probe 距离的 Spearman | 0.697 | 自身 | 0.446 |
| 与旧 Q4 `|ΔQ|` 的 Spearman | 0.140 | 0.231 | 0.201 |
| 留一 checkpoint 最近源 Q MSE（旧 P1 Q4） | **0.0826** | 0.0984 | 0.0843 |
| 留一最近源 η 选择 − 源固定 η 的 Q | **−0.1257** | −0.1466 | −0.1436 |

R5 对指定近目标响应的距离更敏感，且最接近的 5% 表示 pair-state 中旧 Q4 差异最大值为 `0.109`，低于 R3 的 `0.328`；但 2,184 个 pair-state 高度相关，阈值与欧氏坐标尺度均未认证。对完整 raw behavior，旧 R3 的整体距离排序反而更好。R5 在 14 个留一 checkpoint 中只有 3 个 Q MSE 优于 R3、3 个 η 选择差优于 R3。留一实验只在旧 P1 Q4、已打开的 state 上进行；每折特征归一化仅用源 checkpoint 静态特征、近邻选择只用源 Q，held 标签仅用于评分，因此无 held-label 泄漏。Numerical-unknown 的区间保存在结果文件，未默认为成功。

**representation 判定：信息遗漏已被修复，但任务充分性未验证。** 对发现 pair 的旧 24 state，R3/R5/R4 平均距离分别为 `14.90/26.32/1.276`（各自特征单位），near-goal moving 差平均 `14.68`；这对真实 controller 没有接近零的 R3 别名。新表示没有将行为距离可靠地转为 Q 排序或 selector 收益，更不能支撑非空的 transfer guarantee。

## 4. Task C：逐项 verdict 与下一道门

| 问题 | Verdict | 可主张范围与剩余缺口 |
|---|---|---|
| 新 state Q16 是否支持 controller-specific critic necessity？ | **`VALIDATED`（固定 pair/场景/双候选的独立经验条件值）** | 五个未见 state、318 有效+2 unknown、预注册交互均值未知值保守下界 1.0375；总体 controller family 与 population Q 尚无证明。 |
| 新 state 共同 η 是否支持 selector necessity？ | **`PARTIAL`（预注册有限面板经验门通过）** | 交叉拟合 V `+0.175`，4/5 state 正；一例负，符号 `p=0.1875`，无部署型选择器或全 K16/连续域结论。 |
| R5 是否保留旧表示遗漏的任务相关响应？ | **`VALIDATED`（probe 信息保真）** | indexed near-goal moving response 精确保留，构造别名被区分；真实 harmful alias、Q/选择上的充分性未证实。 |
| 是否已具备 held-out real Flow checkpoint selector transfer 条件？ | **`REJECTED`（当前条件）** | R5 的旧 Q4 留一最近源选择差于 R3 且两者均差于源固定 η；缺独立 held checkpoint Q16 headroom、校准与非空界。 |

这轮确认使论文可以谨慎写成：“**在预先固定的一对真实 Ring FF Flow controller 与两个共同 η 上，五个独立新 state 的 Q16 证实了 controller×η 交互，有限 seed 交叉拟合显示 controller information 有正的经验选择价值。**”不能写成“controller-conditioned critic/selector 对一般场景或 population 必要”“新表示使跨 controller 泛化成立”“理论上界有效”或“RSS 所需的 learned held-checkpoint selector 已验证”。原 [Part 3 主报告](part3_theory_validation_20261012.md)中的局部机制结论和 `VACUOUS_BOUND` 判断仍然成立；此处取代其 Step 2 仅探索性判定。

**最小下一门**：先在既有 P1 标签上预注册一个只用源 checkpoint 选择特征尺度/模型的 R5 ablation，要求留一真实 checkpoint 的 Q 排序及选择相对 R3、R0 和源固定 η 同时改善；若静态门通过，再冻结一个此前未用真实 checkpoint，做新 state、共同候选、独立 seeds 的 Q16 选择确认。单个 held checkpoint ×16 新 state×2 η×16 seed 就需最多 **512** 个新 continuation，超过本轮 320 上限，且单一 held checkpoint 仍不足以证明跨 controller 普遍泛化；更可靠的多 checkpoint/K4 设计需要另定预算。当前不启动 critic 重训练或额外 rollout，也不改 pair/候选以追逐正结果。

## 复现与审计

完整命令、Slurm job ID（CPU `19417`、单进程合并 `19421`、静态单 GPU shard `19422`）、状态与哈希保存在 [RUN_LOG.md](../diagnostics/part3_necessity_representation_round2_20261012/RUN_LOG.md)。主要入口：`freeze.py`、`run.py`、`analyze.py`、`repaired_features.py`、`representation_analysis.py`、`alias_counterexample.py`；配置与所有 raw/feature/result/log 文件保存在同一实验目录。最终报告之前已检查脚本语法、协议重放、R3 静态回放、320/320 数据库合并以及报告引用路径。没有删除或改写历史研究证据。
