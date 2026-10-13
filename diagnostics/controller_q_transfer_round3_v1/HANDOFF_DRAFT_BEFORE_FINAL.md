# Controller–Q Round 3：独立研究交接

> 本文件在最终结果审计后定稿；P0/P1 已完成，P2′ 尚在执行。请以最终 `results.json` 的 stage status 判断哪些结果真正完成。

## 核心问题

在固定 Ring geometry、初始 state、K16 episode 级 \(\eta\)、H700 物理 dynamics 和成功判据时，真实 Controller（Flow checkpoint、FF/OrthoFlow3 执行算子、硬安全投影）如何改变闭环 \(Q(s,C,\eta)\) 和 \(\eta\) 排序？项目 Critic 是否能借助 Controller 表示，在没有训练标签的 Controller 上选择比全源最佳固定 \(\eta\) 更好的参数？主要终点是未见 Controller 的闭环选择成功率及配对 CI，Q 误差仅为辅助。

## 前序证据与本轮实际工作

上一轮原生 Ring FF Q512 在先前使用的 256 个开发 state 上的已复核成功率约 93.54%，全 K16 最佳固定 \(\eta_{13}\) 约 88.33%，差 +5.21pp；这只支持相同 Controller、相同几何下的 state selection，不能推断 Controller 迁移。上一轮还发现 FF/TT 会改变 state 层的 \(\eta\) 排序，投影后的 Jacobian 与未投影基底不同，且 \(\eta=0\) 时 Controller 动作已不同。详见 `../ring_state_selection_transfer_v1/REPORT.md` 与 `HANDOFF.md`。

本轮先审计历史标签、checkpoint、execution UID、JAX x64 与 DB preflight。P0 对三种同为 2,048 个 state–\(\eta\) 标签的 16-head 训练布局各用 3 个训练子集和 3 个 fit seed 进行开发校准；复用历史 106,496 条标签，不新增 rollout。选定 256state×8η/Q4 和验证 NLL checkpoint，冻结文件为 `P0_DECISION_SUPPLEMENT.md` 及 SHA256 收据。P1 对 14 个历史真实 Ring Flow checkpoint 做观测 chart 与静态首步兼容审计，并在先前打开的 24 个分层 state 上请求 24×16×Q4×14=21,504 个成对标签；其中精确缓存 1,536 个（9 个为 unknown），实际新增 19,968。单进程入库和 postflight 后有效 21,232、numerical-unknown 272、缺失/冲突均为 0。P1 的 24 个 state 包含按旧 native 结果选的 12 个 adaptive-demand，因此 P1 仍是 DEVELOPMENT pilot，不能作为独立确认。P1 未通过预注册资格门，因此原 P2 不执行；P2′ 在固定原生 v10 Flow 的结构不同的 placement/κ Controller 上进行留出测试，尚在执行。

## 已得到的定量结果

P0 在旧开发 256 state 上、相对全 K16 最佳固定 \(\eta_{13}\)（88.33%）的结果：128×16 为 90.82%，增益 +2.49pp、同时 95% CI [+0.62,+4.49]pp；256×8 为 91.04%，+2.71pp、[+1.05,+4.55]pp；512×4 为 90.34%，+2.01pp、[0,+4.16]pp。验证 selected-success checkpoint 对 256×8 的改进未达预注册切换标准，保留 NLL。原始数据见 `P0_RESULTS.json`、`p0_fits/`。这是样本/标签预算设计证据，不能作为未见 Controller 的 selection gain。

P1 的 14 个真实 Flow 中仅 **7 个**最佳固定 η 成功率位于预注册 40%–95% 区间，未达到至少 8 个的要求，故 gate **FAIL**。在这 7 个上，跨 Controller 使用源最佳固定 η 的 global 遗憾中位数 **17.19pp**，保守 unknown、两终点同时 95% CI **[3.13,28.39]pp**；state-specific cross-controller oracle 遗憾中位数 **27.60pp**，区间 **[17.71,33.33]pp**。这表明 Controller 明显改变最佳离散 η，在开发面板上不是纯 Q level 平移；但不同 Flow 的难度分布过宽，7 个合格 Controller 不满足主试验预设资格。探索性 Q 分解：Controller 平均 Q 的 SD 0.119，两两 η 排序 Spearman 中位数 0.704，60.4% 的 Controller 对有不同 hindsight 最佳 η。完整每 Controller、每有向配对、unknown 与交互能量见 `P1_RESULTS.json`、`P1_Q_DECOMPOSITION.json`。所有这些仍是 DEVELOPMENT，未检验可部署 selector 对未见 Flow 的收益。

## 假设状态与结论边界

- **已支持（开发范围）**：在原生 Ring v10 固定 Controller 上，同等标签预算的 16-head MLP 可用分布于较多 state 的部分 η 标签产生高于 K16 最佳固定 η 的选择结果；该结论仍需独立未见 Controller 数据。
- **已支持（开发范围）**：在固定 Ring state/geometry/η/seed/evaluator 下，真实 Flow checkpoint 能改变 Q 水平及离散 η 排序；有向跨 Controller 遗憾在预注册统计意义上为正。P1 因合格 Controller 数不足而不能升级为正式真实 Flow 迁移 benchmark。
- **尚未确定**：R3、R4_action 或其组合能否让 Critic 对未见真实 Flow 取得可靠闭环选择收益；Controller 数量增加是否改善迁移；Q 水平与排序失败的归因。P2′ 只能回答固定 Flow 下执行算子改变时的部分迁移问题。
- **不能声称**：P1 的 cross-controller oracle regret 是可部署 selector 的收益；静态局部 Jacobian 是完整闭环的精确等价；离散 K16 支持连续 success basin 的形状结论；固定 Flow、留出 placement 的 P2′ 等于留出真实 Flow。

## 执行分支与已有硬边界

P1 资格门为至少 8 个 Controller 的交叉拟合最佳固定 η 成功率处于 40%–95%，并且 global 或 state 层的有向 cross-controller regret 中位数至少 +5pp、保守 unknown 的同时 CI 下界大于 0。P1 的门 B 通过、门 A 失败，故按规则进入 P2′。原版 P2 独立还有最低新标签 73,728（超 70,000 上限）和严格完整 R4 凸包无法容纳两个留出 Controller 的问题；计算见 `P2_MINIMUM_BUDGET_AUDIT.json`、`P2_HULL_FEASIBILITY_AUDIT.json`。P2′ 使用固定真实 Flow v10、源 FF/TT/κ 与留出 TF/FT，官方 preflight 确认 67,584 请求全部 genuinely missing、ambiguous 0；其闭环标签生成正在运行。P2′ 不能代替留出真实 Flow 的 P2，不能把未完成阶段写成结果。

## 直接可复用资产

- `REPORT.md`、`results.json`：最终证据和机器可读阶段状态。
- `PREREGISTRATION_P0.md`、`PREREGISTRATION_P1.md`、`PREREGISTRATION_P2PRIME.md`、`PREREGISTRATION_P3.md` 及各收据：先于相应 outcome 的冻结规则。
- `P1_PROTOCOL.json`、`P1_EXECUTION_PROTOCOL.json`、`P1_CASES.json`、`P1_CACHE_PREFLIGHT.json`：真实 Flow pilot 的精确身份、请求及复用账本。
- `p1_raw/`、`P1_LABELS.npz`、`P1_RESULTS.json`：P1 原始记录、配对标签和统计（后两项仅在 postflight 后生成）。
- `p0_fits/`、`P0_RESULTS.json`：逐拟合 checkpoint/预测与 P0 决策。
- `flow_adapter.py`、`p1_features.py`：真实 Flow 观测适配与不调用 `env.step` 的静态表示。
- `RUN_COMMANDS.md`、`ENVIRONMENT.json`：复现入口、代码版本和计算环境。

完整实验结束后补入 P1/P2′/P3 的数值、原始证据路径、独立确认状态、根因判断及下一步判别性研究问题。
