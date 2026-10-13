# 下一步实验选择：先确认 state×η 选择 headroom 是否存在（2026-10-11）

> **状态更新（同日）：E0 已由 `diagnostics/controller_eta_selection_db_audit_v1/` 完成。**
> - Ring FF（H700，K16，候选中包含最强通用 η13）存在可靠的 state 选择收益：+5.8pp [3.4, 8.8]。
> - 历史上已有一次前瞻确认：256 个新 state 上，Q512 比最强固定 η 高 +5.2pp [2.7, 7.9]。
> - 因此，§0 的机制猜想（“η 是全局增益，单一 η 普遍够用”）和 E1 的悲观分支**已被证伪**；饱和只出现在 Toy FF / DB / Gap1 等部分面板。
> - 下一轮选择见 `docs/next_experiments_ring_state_selection_20261011.md`。

本文件只做实验选择，供实现 agent 执行。选择者没有新增任何 rollout 或训练。

上游证据：`diagnostics/controller_state_geometry_transfer_v1/`（HANDOFF/REPORT/PREREGISTRATION），以及 `docs/controller_generalization_audit_20261011.md` §11。

## 0. 选择依据：只读检查的结论

定义：
- U：交叉拟合固定 η 的失败率，即 1 − (用一半 seed 选出的 per-(G,C) 固定 η 在另一半 seed 上的成功率)。它是任何“按 state 选 η”所能获得收益的上界。
- I：在全部 η 与全部 seed 上都没有成功的 state 所占比例。这些 state 的失败不可能靠换 η 修复。

H850 下的 Toy pilot 结果：

| 面板 | (宽度, C) 格子数 | U = 0 的格子数 | U > 0 的格子 |
|---|---|---|---|
| 自然初态 | 8 | 7 | 只有 w0.41 isotropic |
| 加深 bay | 8 | 7 | 只有 w0.41 isotropic |
| 广初态 | 4 | 3 | 只有 w0.41 isotropic |
| 合计 | 20 | 17 | 3 个，U = 0.06–0.125 |

结论：H850 下 pilot 失败的主因是**候选饱和**，即存在一个对几乎所有 state 都成功的固定 η。这不是标签精度不足造成的。在 U = 0 的格子上，再多 seed 也测不出 headroom。

缩短 horizon 的结果：
- H400：w0.59 仍然饱和（U = 0）。w0.41 isotropic 的 U = 0.156，其中 I = 0.06；anisotropic 的 U = 0.406，其中 I = 0.38。
- H200：几乎所有 state 都 I = 1。
- 缩短 horizon 把“饱和”变成了“不可行”，并没有产生按 state 而异的 η 偏好。

对机制的猜想（待检验）：OrthoFlow3 的修正场每一步都按 state 重新计算，η 只是一组全局反馈增益。因此按 state 自适应的部分已经内置在闭环里，单一 η 很容易对整个场景通用。

## 1. 四个问题的当前判断

| 问题 | 当前状态 | 依据 |
|---|---|---|
| Q 预测 | 在单一共享坐标的几何族内能泛化，几何覆盖能改善 Q（Toy MANY−FEW 的 MSE 为 −0.032，CI 小于 0）。跨拓扑或跨 N 不可靠（FAMILY_REVERSE、COUNT）。 | DB/Toy 宽度族；factor 面板 |
| 给定 C 选 η | 在几乎所有面板上被饱和掩盖，成功率约等于 1，无法区分不同方法。不饱和的 factor 面板中，state oracle 收益约 ±0.5pp，成败由配置级的 η regime 决定。 | pilot；factor 面板；§0 |
| 目标 η prior 校准 | **成立。** 1–4 个目标 state 即可修复 DB/Ring 的 regime。条件结构没有可靠的额外价值（Ring k4 为 −1.8pp）。 | calibration_results |
| 连续 success basin 泛化 | **没有证据。** K4/K8 加 Q2/Q4 标签不能推断 basin 的形状或体积。 | — |

## 2. 选定的实验（按顺序，带决策门）

### E0 已有高精度数据的 headroom 普查（不新增 rollout，第一优先）

**目的。** 回答“现有任何环境里是否存在可靠的 state×η 选择 headroom”。这里的 headroom 必须是环境本身的性质，而不是由候选面板人为造成的。

**数据。** 使用 `shared_rollout_db/rollout.sqlite` 中的 cell，筛选条件为：每个 (controller, state, η) 至少 16 个有效 seed，且不在冲突隔离中。选择者已查到 57,223 个这样的 cell。在每个 controller 下，取“共享同一组 η 的 state 集合”作为一个面板。下表列出候选；实现 agent 必须先追溯每个 UID 的来源实验、controller 类型、placement 和 state 来源（true-t0 还是中途状态）。

| controller_uid 前缀 | 场景 | 完整的 state × η |
|---|---|---|
| ctl_6ef16b8eb8c94f40 | toy | 32 × 16 |
| ctl_d1de2ed5b6ecafdc | toy | 32 × 16 |
| ctl_0e56fc46cc1c6772 | toy | 24 × 16 |
| field_external_toy_v | toy | 23 × 16 |
| ctl_ceef00016f37f243 | DB_4A | 48 × 13 |
| ctl_790d13021b72de22 | DB_4A | 32 × 16 |
| ctl_f262448d380b7c42 | ring | 15 × 16 |
| ctl_68504985c54feb85 | ring | 12 × 16 |
| ctl_dfbc896b68ce8a04 | ring | 10 × 16 |
| ctl_f1d8d989cbbab340 | ring | 10 × 16 |
| ctl_ff4013f13f82acee | ring | 10 × 16 |
| ctl_eae6875700502bfe | ring | 9 × 16 |
| ctl_8eef4063bda464ba | Gap1_N2 | 8 × 16 |

这些可能包括 C6/C7 的 FINAL 面板，需要核实。

**指标。** 对每个面板，用偶数编号 seed 和奇数编号 seed 交叉拟合，正反两个方向取平均，并对 state 做 bootstrap（2000 次）：
- H_xfit：mean_s B[s, argmax A[s,:]] − mean_s B[s, argmax mean_s A]。
- U 与 I，定义同 §0。
- 鲁棒版本：用 B15 代理（B 半上 ≥ 7/8 成功）重算 H。
- 单一 η 的最大覆盖率，以及 B15 的覆盖数（用贪心法计算最少需要几个 η 才能覆盖所有可行 state）。

**合格门（先冻结再计算）：**
- state 数 ≥ 12；
- H_xfit ≥ 0.05 且 95% CI 下界 > 0（均值版或鲁棒版任一满足即可）；
- U − I ≥ 0.05；
- 覆盖数 ≥ 2。

**判读：**
- 有面板合格：该环境进入 E2。
- 全部不合格：在这些控制器与场景配置下，“候选饱和”被确认为普遍现象，进入 E1。

**能排除的解释：** “pilot 选的环境太偶然”；“只是 Q2/Q4 精度不够”（普查用的是 Q16 级精度）。

### E1 对唯一有非零上界的格子做精度确认（约 4,096 次 rollout；仅在 E0 全部不合格时运行）

**目的。** 把 pilot 中仅剩的“可能有 headroom”的窄窗口做最终封口。

**设计。**
- 只用 Toy 广初态 pilot 的 16 个 state，宽度 w = 0.41，两个 controller，K8 η。
- 新的 future seed 2–17：2–9 用于选择，10–17 用于评估。
- 在 H850 下执行；H400 的结果按已审计的前缀规则从 H850 记录推导。
- 不改变 η、state、几何或成功判定标准。

**门槛。** 与 E0 相同。另外在 H400 下单独报告 I。

**判读：**
- 不通过：说明在 Toy 宽度 / 初态 / horizon 这个设计空间里，不存在可确认的 state×η 选择收益。结合 E0，结论是：在当前 η 参数化下，选 η 本质上是场景级问题。此后的研究重心应转向 prior 校准（见 E4），不再寻找“强 state 选择”面板。
- 通过：进入 E2。

### E2 合格环境的独立确认（预计 2,000–4,000 次 rollout；只在 E0 或 E1 有合格者时运行）

- 在合格环境中使用全新的物理 state（新的随机序列）和全新的 future seed，并预注册与 E0 相同的门槛。
- 确认用的面板必须是在看到结果之前就固定的 η 集合。其中必须包含该环境中已知的“最佳通用 η”，防止通过排除通用 η 来人为制造 headroom。
- 只有确认通过，才能进入 E3。

### E3 强 state 条件下的跨几何迁移正式检验（只在 E2 通过后运行）

- 沿用 `controller_state_geometry_transfer_v1/PREREGISTRATION.md` 的 Phase 1 设计（MANY/FEW/FEW_x2/ALL/EXTRAP，ENT/MLP/state-only），但必须建立在 E2 确认的环境之上，围绕一个共享坐标的几何族展开。
- 主要终点：在新 state × 新几何上，给定 C 选 η 的成功率，同时要求相对“每个 (G,C) 的固定 η”的收益为正，以及 matched − mismatched 的 CI > 0。
- Q 的相关性只作为次要指标。

### E4 并行支线：场景 prior 校准的可预测性（不新增 rollout，或少量 rollout）

- 已经确认目标标签能修复 η regime。要回答的问题是：在不使用目标标签的情况下，能否从可计算的机制量预测场景级 η regime？
- 先复用 factor 面板和四场景已有数据：用 leave-config-out 方式，从机制坐标预测“最佳固定 η / 最佳 η 区域”，与 k=1 校准比较成本和收益。
- 这条线在 E0/E1 得出“选 η 是场景级问题”时，就成为主线。

### E5 暂缓：连续 success basin 的泛化

- 需要对每个 state 做稠密的 η 采样（如 Sobol 64 点以上）并达到 B15/Q16 精度，成本高。
- 只有在需要回答 basin 形状或体积是否迁移时才启动。
- 在此之前，不对 basin 的形状、体积或拓扑做任何泛化表述。

## 3. 明确不做的事

- 不在 U = 0 的格子上增加 seed。
- 不靠扩大 η 面板来寻找 headroom：面板越大，越容易出现通用 η。
- 不靠排除通用 η 来人为制造 headroom。
- 不运行 broad-IC main。
- 不把 Q 相关性或接近 100% 的选择成功率当作“强 state 迁移”的证据。
- 所有新 rollout 必须遵守 shared_rollout_db 的 preflight/merge 流程。
- 本文件中的实验都是纯 Q 实验，不需要视频；如有 batch launcher 例外，以 AGENTS.md 为准。
