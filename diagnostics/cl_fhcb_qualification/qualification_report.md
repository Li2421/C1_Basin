# CL-FHCB 独立 qualification 报告

## 结论

**CASE F — CONTINUATION CERTIFICATE INVALID OR VACUOUS**

具体是 **invalid，而不是 vacuous**。冻结 certificate 在独立 test split 的
`damp035/unresolved` cell 上违反预注册经验 Bellman gate：平均
slack `-0.007109`，cluster-bootstrap 95% CI
`[-0.011294, -0.003159]`；
上界仍低于预注册阈值 `-0.001`。因此按 compute gate 停止，没有运行 Stage 2 或 G，
也没有训练、重拟合或修改冻结方法。

## 最终属性表

| 属性 | 含义 | 状态 |
|---|---|---|
| P | Predictive / Early Prediction | **PASS** |
| Q | Closed-loop Intervention Ranking / Control Relevance | **NOT_TESTABLE** |
| G | Closed-loop Directional Fidelity | **NOT_TESTABLE** |
| PROJ | Multi-step Projection Consistency | **PARTIAL** |
| LHC | Long-horizon Closed-loop Causality | **FAIL** |
| OAS | Outcome-aligned Sensitivity | **FAIL** |

## Stage 1：P 与 continuation validity

共运行 32 条独立 test-split full-horizon trajectory，实际 20973
个物理步，CPU。四个 `phi` 在 8 个 test 初态上使用 paired common random numbers，
`G_phi` 在 K 后继续运行至 success、strict deadlock 或 timeout。

P 的预注册规则在临近终点的非终止 prefix 上通过：
- K=20，terminal_minus_40：AUROC=0.712，95% CI [0.615,0.808]，prefix 后仍有 1.0s lead。
- K=100，terminal_minus_150：AUROC=0.712，95% CI [0.615,0.808]，prefix 后仍有 2.5s lead。

但初态 P 明显反向：K=20 的 AUROC 仅 `0.038`，deadlock
减 nondeadlock 的平均 R separation 为 `-0.516`。
因此 P_PASS 只表示 certificate 在 strict-deadlock 临近时含信息，不表示可用于政策
排序。

certificate 并非恒等于 1：独立访问状态中 value=1 的比例为
`0.384`。失败来自 transition expectation，
不是 vacuity。`goal025` 在初态的冻结 B 为 0.373952，而独立 8-state mixture 中
strict-deadlock 为 6/8；其 Wilson 95% 下界仍高于 B，提供第二项上界反证。

## 长时域与 outcome alignment

Stage-1 full continuations 已足以发现强反例：初态 apparent low-R `goal025` 在 8 个
test 初态中产生 6 次 strict deadlock，而 `zero`、`damp035`、`relative025` 均为 0。
在 low-R/high-R 个体配对中共有 `18` 个 adverse
deadlock change；没有 TRUE_ESCAPE、DELAYED_DEADLOCK 或 TIMEOUT_SUBSTITUTION 被
误计为改善。至少一个 paired exact test 达到 p<0.05，所以 LHC_FAIL。

K=20 与 K=100 的政策排序结论一致：均有更多 `R_down/Q_up` 而没有
`R_down/Q_down`，policy-level Spearman 为负，因此 OAS_FAIL。full remaining 只作
事件 bookkeeping，未作为新的 K 或 intervention duration。

## 投影与 G

Stage-1 的双硬投影 trajectory 显示 correction 差异能传到 executed action 与 state，
K=20/100 的 collapse fraction 都低于 0.10；但预注册 Stage-2 stochastic distribution
复验因 gate 停止，故只给 PROJ_PARTIAL，不继承旧 NO_MASS_ALIASING 结论。

G_NOT_TESTABLE 有两个独立原因：Q gate 未通过；且冻结 `B` 仅在四个离散 `phi` ID
定义，同时规范禁止插值或外推，所以合法的 central FD `grad_phi R_K` 不存在。没有
用 soft projection、straight-through 或任意替代梯度。

## K 消融与计算预算

主 K 严格保持 20 与 100。较长 K 提高了临近 deadlock 的可见窗口，但没有修复初态
政策排序；不存在使 Q/G 变得有意义的冻结 `K*`。总计仅新增 32 条 rollout，远低于
800/1600/2200 预算。Stage 2 的 256 条与所有 G rollout 均未启动。

## 可复现文件

- `property_p.json`：全部 AUROC、rank-biserial、overlap/tie 与 bootstrap CI；
- `continuation_certificate_audit.json`：逐 phi/cell 独立 Bellman slack；
- `property_q.json`、`property_g.json`：gate 与 NOT_TESTABLE 原因；
- `property_proj.json`：双投影后的 K/full trajectory separation；
- `long_horizon_causality.json`、`outcome_alignment.json`：完整终局因果分类；
- `stochastic_semantics_audit.json`：deterministic `G_phi` 与共享 Flow law；
- `historical_failure_regression.md`：九项历史问题；
- `figures/`：请求的轻量图。
