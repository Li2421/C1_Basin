# UTC 2026-10-13T08:09:07.101723+00:00

# Round10 universal eta schedule preregistration

UTC is recorded verbatim at the top of this file and in the SHA256 receipt. This registration precedes fresh state sampling, new rollouts and any new endpoint analysis. The schedule was selected using opened round9 evidence; the new-state endpoint below is prospective. All interpretations and follow-up suggestions remain DEVELOPMENT.

## Historical evidence and scope

Read AGENTS.md, shared_rollout_db/README.md, docs/batch_videos.md and the complete round10 task first. Historical inventory: round9 diagnostics/controller_eta_switch_time_v1/{PREREGISTRATION,protocol,core,run,launch,gate,results,REPORT}; round4 diagnostics/controller_loco_real_flow_v1/{protocol,CASES,design}; round3 controller_q_transfer_round3_v1/{P1_EXECUTION_PROTOCOL,P1_CHECKPOINT_COMPATIBILITY,state_generator,flow_adapter}; native real_state_expansion/protocol. Existing opened states cannot supply prospective endpoints. Check all DB state identities/content and recoverable physical coordinates before registering fresh states. No checkpoint training, eta tuning or parameter selection is performed. Fixed-arm old checkpoint/runtime identities and round9 schedule identities are reused where exact; new state/seed requests alone make the main evidence missing.

## Frozen cohort and arms

All14 controller_order entries of round3 P1_EXECUTION_PROTOCOL, exact checkpoint/adapter/native-v10 profiles. Schedule-dev is exactly base_u_macflow, base_u_v10_local_macflow, base_u_v9_local_macflow, base_u_plus_w_v4_wide_macflow; held-out is the other10. No data-dependent exclusion.

64 fresh states: original round4 generator generate(split='round10_universal_fresh',count=64,seed_start=2026160000,native_protocol=real_state_expansion/protocol), whose implementation calls sample_instance(seed,'development',original LocalFrameConfig). Seed interval2026160000..2026160063 is fixed now. Compare state_uid/content_hash against the entire historical DB, and canonical numerical positions/velocities/goals/timestep against every recoverable physical state (normalize numerical zeros and float64 representation); scan historical alias/provenance seed information and record unresolved physical provenance explicitly. Any exact duplicate stops the main batch; do not redraw until unique after observing results. UID/content deduplication must cover all DB rows; the audit must not pretend coordinate recovery is complete for legacy rows lacking coordinates. Fresh physical coordinates are not outcomes.

Seed0..3 use original future_root2026100403 and unchanged key_for(state,seed,physical_step), including fresh alias SHA/rng_namespace. Ring, FF, H700, dt.05, pseudo_steps10, physical safety projection/collision/success events and JAX x64false all unchanged.

Arms F10,F13,S300,S400,S500 for every controller/state/seed. Freg only base_u_plus_w_v3_macflow and base_u_v3_macflow with eta2, base_u_v8_local_macflow with eta4. 18688 main requests, plus32 runtime validation requests, total18720 maximum completed seed requests. Unknown is preserved and never repeated or counted as failure.

Schedules are eta13 before action index t and eta10 at index>=t, t300/400/500, main S500. Earlier native termination is retained. Fixed arms use the identical eta for every action and old fixed-controller/eta UID. No timing, eta, controller or success-standard adjustment based on new evidence.

## Frozen implementation, identity, consistency gate

Import round9 run.py as immutable source and reuse its execute/make_runtime/SafetyTracker/key_for loop. Round9 core helper implementation is copied with only experiment paths/identity/inventory constants changed for isolated IO; the numerical rollout loop and field/safety/controller modules are not edited. Record both original source hashes and the new IO binding hash. As in round9 launch.py, normalize numpy snapshot arrays only at journal transport. Schedule payload format retains eta_a/eta_b float64 hex, action-index rule, parent controller and frozen round9 runner/core hashes. For four existing controllers, use exact existing round9 schedule controller UIDs; remaining ten use the same identity construction. The IO binding has its own registered provenance but does not redefine the physical execution semantics.

Gate reproduces all32 round9 gate cases, including both pure effective etas, branches t0/t700, old physical states0/1 and seeds0/1. New validation-only controller identity includes the frozen round10 IO binding fingerprint, enabling authorized fresh validation without falsely presenting a cached old result as execution. Compare to original round9 raw, and to round4 counterpart, using exact outcome, episode_length and first_action float64 bytes (signed zero included); no tolerance. Any disagreement stops main and produces a report. Gate old states are never confused with the fresh cohort. Before main submission verify the gate PASS and official preflight ambiguous=0.

## DB and blindness

Design -> register new experiment/new states/new schedule profiles -> official preflight -> only genuinely missing requests -> worker append-only experiment journals -> one official single merger -> postflight. Exact old IDs/configs remain unchanged. No DB schema changes; local exact journal ingestor preserves full modern seed_key and physical timestep, as in round9. SQLite writes are single-process registration/merging/postflight only. The requested shared DB/journals lifecycle is the explicit exception to experiment-directory-only artifacts. No git commit, cleanup, historical asset modification or new training.

Workers write complete per-step trajectories and every10-step position/velocity/projection snapshots, original augmented-initial-state hash and simulator swept clearance. Partial numerical failures are explicitly partial/unknown. No analyst reads or summarizes fresh raw labels, predictions or outcomes before final endpoint opening. Monitoring reads only queue, filename counts and worker completion/performance logs. After every planned main raw file exists and workers are done, write ENDPOINT_OPENING.receipt with UTC, preregistration/protocol/case hashes and raw inventory; only then merge/read endpoint values. Gate labels are historical and exempt from fresh-cohort blindness.

## Endpoint, oracle, bootstrap and unknown

Primary per-controller oracle=max(F10,F13,Freg where present) on this same new64/Q4 cohort, never a hindsight eta outside these arms. Primary schedule S500. S300/S400 secondary only, never replace a failed primary. Known-success/256 is a success lower bound; add unknown/256 for upper bound. The registered point gate uses lower-bound point rates, with all unknown counts and conservative feasible gate bounds reported. This does not call unknown a failure.

State-cluster bootstrap10000 draws, rng202610130100. Seed outcomes first average within each state. One state draw of64 is shared across controller/arm comparisons; each draw also independently resamples controllers with replacement within the group (all14, dev4, held10). Recompute each controller's max fixed-arm oracle within each bootstrap draw. Report paired per-controller95% percentile CIs and the hierarchical group's95%CI. Robust lower/upper envelopes use schedule_lo-max(fixed_hi) and schedule_hi-max(fixed_lo); no guessed unknown imputation. Group means weight controllers equally. Primary overall criterion is lower95% CI>0; exactly0 fails the strict positive gate. Per-controller noninferiority is point difference>=-.05 (report95%CIs); E3 is count of S500-F10<-.10, exactly-.10 is not catastrophic. Report raw point gates and conservative unknown-feasibility gates separately; if unknown can flip the mapped conclusion, H_schedule is INCONCLUSIVE.

G_universal requires all14 positive overall lower95%, no point controller deficit worse than5pp versus oracle, and E3=0. Held-out separately requires its own positive overall lower95% and no point deficit worse than5pp; additionally report its E3. Dev uses the same three-component gate for the conclusion mapping. All gate margins/failing controllers are explicit. Primary passes all14 and held -> H_schedule SUPPORTED (prospective new-state endpoint, Ring-only); dev pass/held fail -> partial-overfitting result, INCONCLUSIVE for universal support unless a robust deficit is a direct counterexample (then REJECTED); dev and held both fail -> REJECTED per task mapping. Other unmapped mixed case -> INCONCLUSIVE. Failure to clear a wide CI alone is not proof of a mechanistic controller effect. Unknown feasible bounds override an otherwise definitive mapped conclusion.

Secondary same metrics for S300/S400; correlation between S500-oracle gain and F10-F13 across14 controllers, Pearson and Spearman point plus crossed state/controller bootstrap10000, rng202610130101 (resampled repeated controllers retained). Constant samples yield NA, not a fabricated coefficient. Also report paired source/oracle outcome transitions, outcomes and occurrence steps; timeout remains timeout. Mechanistic correlations are DEVELOPMENT/descriptive, not proof that the regime is caused by a particular event.

## Resources, video gate, delivery

zhihan tasks in this workstation together CPU<=18, GPU shards<=3; count owner correctly (ming jobs are not zhihan quota). CPU FF workers max17 plus1 orchestration/analysis, BLAS/OMP/MKL1, no GPU needed. Deduct other zhihan jobs before submitting; keep all arms. Initial estimate2.5-4h and calibrate gate costs.

Unlike round9, this is a Ring large evaluation batch: video requirement applies. Do not invoke the pure-Q exception. Use the existing batch_videos load_trace/render_video/CLI entry; its four-scene hard-coded scope may be narrowed in a new-directory wrapper to ring_exchange only, explicitly as requested, with no repository tool modification. Required roles remain deadlock_show,deadlock_resolution,safety_success. A genuine deadlock baseline needs already documented actual detector/criterion and full identical-initial/seed traces; native Ring has no deadlock termination branch and no new detector or relabeling is allowed. Search compatible historical video evidence; if no qualified deadlock cases exist, retain the two missing roles in an incomplete manifest and report the gap. The gate must return nonzero and no complete batch marker may be created. Deliver a real independent full safety_success MP4, plus up to two clearly named full timeout-to-success comparison MP4s when such pairs exist; comparison panel labels must say TIMEOUT, never DEADLOCK. These diagnostic videos cannot fill missing deadlock roles. Every MP4 uses every actual step, simulator swept-clearance safety checks and trace/video hashes. No supplemental rollout solely to manufacture videos.

Finalization: verify complete raw inventory -> endpoint-opening receipt -> inherited-asset SHA audit -> single merger -> postflight -> fixed endpoint statistics/plots -> videos/strict video gate -> REPORT/HANDOFF/results even if video gap -> numerical-completion receipt. Only if numeric and strict video delivery both pass may a complete batch marker be written. Video gap means NUMERICAL_COMPLETE_VIDEO_INCOMPLETE, not completed batch. Report first paragraph answers H_schedule overall and held-out, labels SUPPORTED/REJECTED/INCONCLUSIVE, distinguishes facts and interpretation, and limits claims to Ring/FF/H700, fixed two-eta schedules and14 real Flow (plus three prespecified Freg comparators). Q4 is not B15 certification.

The exact user prompt is appended below before the receipt is hashed.


# Frozen user task

你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`，并严格遵守。

# 一、要检验的命题（只检验这一个）

**H_schedule：在 Ring 场景中，“先用 η13、到第 t 步换成 η10”这个与 controller 无关的 η 时间表，在所有真实 Flow controller 上都不劣于该 controller 自己的最佳固定 η。** 如果成立，之前各轮看到的“controller 依赖的 η regime”主要是“整个 episode 只能用一个固定 η”这一动作空间限制造成的。

## 依据

依据来自 round9（`diagnostics/controller_eta_switch_time_v1/`，DEVELOPMENT）。这些数据所用的 state 与 seed 0–3 都已被打开过；t = 500 也是看过数据后才挑出的：

| controller | 纯 η10 | 纯 η13 | η13 → η10，t = 500 | 相对最佳固定 η |
|---|---|---|---|---|
| base_u | 23.4% | 53.1% | 84.0% | +30.9pp |
| v10_local | 57.8% | 92.2% | 94.5% | +2.3pp |
| v9_local | 96.1% | 79.3% | 93.8% | −2.3pp |
| plus_w_v4_wide | 63.3% | 26.2% | 96.9% | +33.6pp |

一个可能的机制解释：前期推进需要 η13，后期收敛需要 η10。固定 η 偏好哪一个，取决于该 controller 的失败主要集中在哪个阶段。

本轮的任务，是在新 state 和全部 14 个真实 Flow 上做前瞻检验，其中 10 个 Flow 从未参与时间表的选择。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

## controller

全部 14 个真实 Ring Flow。分组如下：
- **schedule-dev 组：** base_u、v10_local、v9_local、plus_w_v4_wide。它们是 round9 中用于选择时间表的 4 个。
- **held-out 组：** 其余 10 个。

## state

- 按 round4 loco_test 的生成分布与流程，用新的、预先登记的随机种子抽取 **64 个全新 state**。
- 抽取后，确认它们与历史 DB 中所有 state 都不重叠。
- 在本轮终点评价之前，不读取这些 state 的任何结果。

## 臂（每个 controller × state，seed 0–3，即 Q4）

| 臂 | 定义 |
|---|---|
| F10 | 全程 η10 |
| F13 | 全程 η13 |
| Freg | 全程使用该 controller 的 Stage 0 regime η，仅对 regime 不是 η10 / η13 的 controller 执行：plus_w_v3 与 v3 用 η2，v8_local 用 η4 |
| S500（主） | 前 500 步 η13，之后 η10 |
| S300、S400（次） | 同上，切换步分别为 300、400 |

- **切换的实现：** 复用 round9 冻结的切换实现（`core.py`）。唯一的干预是从第 t 步起替换 η；OrthoFlow3 修正仍然每步重算。
- **一致性门：** 先在 32 个 case 上复现 round9 的一致性门，结果须逐位一致。
- **规模：** 14 × 64 × 5 × 4 = 17,920，加上 Freg 的 3 × 64 × 4 = 768，**共 18,688 条完整 rollout**。
- **共享 DB：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。切换臂沿用 round9 已登记的身份格式。

## 不变项

Ring、FF、H700、安全投影、成功判定、JAX x64 = false。不修改控制器、物理或事件判定。

## 终点与判定（全部预注册）

对每个 controller，定义 **最佳固定 η = max(F10, F13, Freg)**。它是在同一批数据上事后选出的 oracle，对时间表不利，因此是保守的比较对象。

**主判定 G_universal 以 S500 为准，须同时满足：**
1. **整体收益：** 14 个 controller 上，S500 − 最佳固定 η 的平均值，其 95% 下界 > 0。bootstrap 时按 state 成簇，并在 controller 层重抽。
2. **逐 controller 不劣：** 没有任何 controller 的 S500 比其最佳固定 η 低 5pp 以上（按点估计；同时报告逐 controller 的 CI）。
3. **相对源先验不出灾难：** 没有任何 controller 的 S500 比 F10 低 10pp 以上（E3 = 0）。

**泛化检查：** 对 held-out 组（10 个）单独重复第 1 条和第 2 条。这是对“时间表不是在 4 个 controller 上过拟合”的直接检验。

**次要分析：**
- S300、S400 的同样指标。
- 在 14 个 controller 上，比较时间表的增益与 F10 − F13 的关系，用来检验“固定 η 的 regime 由失败阶段决定”。
- 失败类型与发生步数。

# 三、预注册的结论映射

| 观察结果 | 结论 |
|---|---|
| G_universal 通过，且 held-out 组也通过 | **H_schedule SUPPORTED。** 零样本 controller 泛化失败的根因，是动作空间设定错误：用单个固定 η 压缩了随阶段变化的最优 η，于是“哪个 η 好”变成了一个看不见的 controller 属性。在 Ring 上，一个与 controller 无关的时间表即可实现泛化。下一步是让 critic 学习 Q(s, t, η) 或在中途重新选择 η，并检验其他场景 |
| schedule-dev 组通过，held-out 组不通过 | 时间表部分过拟合，controller 依赖部分是真实的。报告哪些 controller 失败，以及它们的失败阶段 |
| 两组都不通过 | round9 的现象不具普遍性，H_schedule REJECTED |

# 四、硬性规则

- 只在新目录 `diagnostics/controller_universal_eta_schedule_v1/` 中写文件。不修改、不删除历史资产，不做 git commit。
- 时间表的参数（η13→η10，t ∈ {300, 400, 500}，主 t = 500）已写死，不得根据本轮结果调整。
- unknown 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **证据等级：** 新 state 首次打开前的预注册部分是前瞻的；其余全部标为 DEVELOPMENT。
- **资源：** 本 workstation 上用户所有任务合计不超过 18 个 CPU shard 和 3 个 GPU shard。
- **视频：** 这是 ring_exchange 的大型评价 batch，按 AGENTS.md 和 `docs/batch_videos.md` 交付 ring_exchange 的完整仿真视频。Ring 的原 runtime 没有 deadlock 判据；如果找不到合格的死锁解决案例，就如实报告缺口，不得用 timeout 冒充。

# 五、交付

全部交付物放在 `diagnostics/controller_universal_eta_schedule_v1/`：
- PREREGISTRATION.md 及收据；
- 新 state 的抽取与去重审计；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 图表：每个 controller 的 F10 / F13 / Freg / S300 / S400 / S500 柱状图，以及 held-out 组单独一张图；
- 视频及其 manifest；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接回答：H_schedule 是否成立，以及在 held-out 组上是否成立。每个假设标注 SUPPORTED / REJECTED / INCONCLUSIVE；事实与解释分开写，不夸大。结论边界：Ring、FF、H700、2 个 η、14 个真实 Flow。
