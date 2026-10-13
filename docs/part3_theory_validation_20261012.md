# C1 / OrthoFlow3 Part 3 — Theory validation and closure (2026-10-12)

> **Round 3 transfer update (same date):** [Real Flow checkpoint transfer feasibility](part3_round3_transfer_feasibility_20261012.md) reuses the opened P1 14-checkpoint × 24-state × K16/Q4 panel with zero new rollouts. Strict source-only nested leave-one-checkpoint-out soft matching lets R5 improve held Q MSE over R3 (0.0675 vs 0.0714), but R5 top-1 Q (0.5856) remains below R3 (0.5945) and source-fixed η₁₀ (0.6019). The preregistered decision is **`TRANSFER_GATE_FAILED`**. This supersedes the Round 2 nearest-source diagnostic as evidence about this particular transfer reader; it does not change the local mechanism, fixed-pair necessity or `VACUOUS_BOUND` conclusions below.

> **Round 2 evidence update (same date):** [Independent necessity confirmation and representation repair diagnostics](part3_round2_necessity_representation_20261012.md) supersede this report's Step 2 evidence status for the fixed Ring FF pair `base_u_v2_macflow` / `base_u_v10_local_macflow` and η₁/η₂. Five previously unseen true-t0 states received a complete 320-continuation Q16 panel (318 valid, 2 numerical-unknown). The preregistered finite-panel critic interaction and seed-crossfit selection-value gates passed, with one state showing no optimal-choice conflict and negative crossfit value. R5 preserves indexed near-goal moving responses but did not improve exploratory held-checkpoint Q/selector transfer. The local Step 1 mechanism findings, evaluator-margin limits and `VACUOUS_BOUND` conclusion below remain unchanged. No population or general cross-controller transfer claim follows from the update.

## Executive answer

**Part 3 is not yet sufficient for an RSS claim that a controller-conditioned critic or selector is generally necessary, or that a representation-based cross-controller generalization guarantee is useful.** A conditional local mechanism theorem is correct, and one smooth Toy case passes an independent controller-parameter derivative check for all four placements. Real Flow checkpoints on a shared Ring FF panel show empirical controller-dependent Q and a positive finite-bank information value. Those states were previously opened, confirmation uses only two fresh seed indices per cell, and the result does not establish population Q or a deployable held-controller selector. The proposed transfer bound has no certified behavioral or representation constants and becomes `VACUOUS_BOUND` under the available conservative numerical range.

This report is a **read-only evidence reconciliation plus CPU static probes**. New rollouts: **0**; new fits: **0**; GPU shards: **0**. Pure Q analysis requires no video batch. It does not change frozen controller, projection, plant, evaluator, or success criterion.

### Evidence audit before computation

| Evidence | Compatibility and reuse | Remaining gap |
|---|---|---|
| [Part 1 audit](part1_implementation_theory_audit_20261012.md), [four-chain audit](../diagnostics/placement_action_family_audit_v1/final_report.txt), [field source](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py), [factorial dispatch](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py) | Frozen execution order, source hashes, basis and projector already audited. No new rollout needed for implementation identity. | Controller-weight derivative H was not checked. |
| [Part 2 derivative records](../diagnostics/part2_closed_loop_sensitivity_validation_20261012/jacobian_validation.csv), [reuse summary](../diagnostics/part2_closed_loop_sensitivity_validation_20261012/reuse_summary.json), [boundary records](../diagnostics/part2_closed_loop_sensitivity_validation_20261012/boundary_validation.csv) | Same Toy/Ring checkpoint, state, η and noise keys; 49,152 old result records reused. G/K and conditional completion margin evidence retained. | H and full evaluator equivalence were not established. |
| [Ring real-Flow P1](../diagnostics/controller_q_transfer_round3_v1/REPORT.md), [labels](../diagnostics/controller_q_transfer_round3_v1/P1_LABELS.npz), [protocol](../diagnostics/controller_q_transfer_round3_v1/P1_PROTOCOL.json) | Same Ring geometry, FF operator, true-t0 state, K16 and four common seeds; 14 genuinely different checkpoint programs. 21,504 old requested labels, 272 numerical unknown, no new labels. | Previously opened development states; Q4 and only Q2 per split. |
| [P1 static descriptors](../diagnostics/controller_q_transfer_round3_v1/p1_features), [P2′ alias audit](../diagnostics/controller_q_transfer_round3_v1/P2PRIME_R3_ALIASING_DIAGNOSTIC.json), [generalization audit](controller_generalization_audit_20261011.md) | Existing no-step probes and held-placement result can test sampled representation behavior. | No certified supremum or held-out real-checkpoint selection test. |

The official [shared cache policy](../shared_rollout_db/README.md) requires exact semantic preflight before any continuation. There was no evidence gap requiring a new outcome here within the 320-continuation first-round budget. Changing only critic input coding would not invalidate these frozen outcome labels.

The requested historical [field `runner.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/field_pipeline_v1/runner.py) and [`assess.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/field_pipeline_v1/assess.py) were also checked. The runner calls frozen FF action, then actual `env.step`, and requires both `collision_free_success` and `termination=='success'`; exceptions become `numerical_failure`. Its DB midstate restoration deliberately starts with empty hidden monitors and has a different RNG schedule, so those midstate labels cannot silently confirm the present true-t0 Ring P1 panel. The assessor leaves unresolved Q16/B15 cells unresolved instead of assigning successes or failures. For the four-placement H probe the relevant continuation entry is [stage-2 `run_rollouts.py`](/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/run_rollouts.py), which preserves the same first-event rule.

## Step 1 — Mechanism, controller derivative, and success margin

### Actual program and formula correction

At fixed physical state `x` and **the same normalized Gaussian draw** `ξ`, native Flow starts at `z₀=ξ`, uses ten Euler steps at `s=k/10`, decodes with checkpoint mean/scale, componentwise clips to `[-1,1]`, then applies a per-agent speed ball. Ring applies its observation/action local-to-world chart and a world-frame speed bound. The native action `a_ref` is projected onto the frozen state-dependent CBF/speed set `U_x` to obtain `a_safe`. OrthoFlow3 computes `δ=B(x,ξ,c)η`; its second column is `κ[a_safe−(a_safe·g)g/(||g||²+ε²)]` with `κ=3.303687238760696`, `ε=10⁻⁶v_max`, and goal column `g`. Consequently **controller variation changes B even at fixed `x,η,ξ`**.

Write `P=P_{U_x}`, `F⁰_c(δ)` for the ten unconstrained field steps and `Fᴾ_c(δ)` for the ten admissible-field steps. In a field step the projected object is `q=h+(1−s)[f_c(h,s)+δ]`, followed by `h⁺=(1−ρ)h+ρP(q)`, `ρ=min(1,0.1/(1−s))`. This projects the **remaining-time predicted action**, not the vector field. The final FF step has `ρ=1`; there is only a safety assertion, no extra terminal projection. The audited executable map is:

| Placement | Executed action |
|---|---|
| TT | `P(a_safe+δ)` |
| TF | `P(F⁰_c(δ))` |
| FT | `P(Fᴾ_c(0)+δ)` |
| FF | `Fᴾ_c(δ)` |

The Part 1 normal-cone elimination and nonexpansive projection arguments apply directly to TT and, with `Fᴾ_c(0)` as its own base, FT. They are **not** formulas for the whole TF or FF map. The previously proposed `B(x)` notation must be `B(x,ξ,c)` or condition explicitly on controller/noise. An active speed sphere contributes its tangent normal to lineality; `rank(B)−rank(A_I B)` counts only lineality for complete active linear rows, not all effective directions of the SOCP. These corrections were already identified in [Part 1](part1_implementation_theory_audit_20261012.md) and are retained here.

For a smooth controller parameter direction `d`, set `r=∂_c a_ref[d]`, `a'=D P(a_ref)r`, and `δ'=D_c B[d]η`. The nonzero second-column term is

`δ'₂ = κ η₂ [I − ggᵀ/(||g||²+ε²)] a'` (per agent, with fixed physical goal vector).

Let `L⁰_δ,L⁰_c` and `Lᴾ_δ,Lᴾ_c` be the field endpoint derivatives with respect to correction and controller, respectively. Then the **correct** controller derivatives are

| Placement | `H=∂u/∂c[d]` in a smooth branch |
|---|---|
| TT | `DP(a_safe+δ)(a'+δ')` |
| TF | `DP(F⁰_c(δ))(L⁰_c+L⁰_δ δ')` |
| FT | `DP(Fᴾ_c(0)+δ)(Lᴾ_c(0)+δ')` |
| FF | `Lᴾ_c(δ)+Lᴾ_δ δ'` |

Inside FF, `dq=dh+(1−s)(D_h f_c dh+∂_c f_c[d]+δ')` and `dh⁺=(1−ρ)dh+ρ DP(q)dq`; for TF omit `P` in the recurrence. `DP` is the local KKT derivative, including active speed-ball curvature. These formulas require stable active sets, a regular KKT system, fixed chart and noise, and no clip or retry branch switch. A different checkpoint can change normalization/chart, so a discrete checkpoint pair is **not** a smooth `d` without a specified parameter path.

### Independent local checks and scope

[New H probe](../diagnostics/part3_theory_validation_20261012/h_bias_probe.py) applies a scalar direction to the final Flow vector-field output bias in normalized action coordinates. It uses a frozen Toy checkpoint, state `hard_confirm_toy_1_820013`, η index 4, and one fixed seed-0 key. JAX JVP differentiates all ten native/field responses; local KKT differentiation handles projection; a separate centered difference calls the full four-chain action with perturbed controller bias. The native reconstructed action agrees to `2.8e−17`. `||δ'||=0.01336` and `||a'||=0.01828`, so the anchor-to-basis channel is material even though terminal projection attenuates TT/FT. All observed KKT multipliers are nonweak and residuals are at floating-point scale.

| Placement | `||H||` | Relative error vs independent FD, ε=`10⁻³` / `3×10⁻⁴` |
|---|---:|---:|
| TT | 0.000315 | `4.22e−7 / 3.80e−8` |
| TF | 0.085558 | `1.88e−7 / 1.69e−8` |
| FT | 0.000315 | `4.22e−7 / 3.80e−8` |
| FF | 0.048950 | `1.94e−7 / 1.74e−8` |

The exact output is [h_bias_results.json](../diagnostics/part3_theory_validation_20261012/h_bias_results.json). This establishes a **single smooth local direction**, not an all-state or whole-network Jacobian. It does not verify weight paths between different checkpoints, clip boundaries, active-set changes, solver retries, singular KKT systems, or float32-dominated differences. Those must remain a separate non-smooth/error category.

For `x=(p,v)` and `p⁺=p+0.05u, v⁺=u`, the tangent recursion is `S[t+1]=A_cl[t]S[t]+[0.05I;I]H[t]`, with `A_cl=[[I+0.05K_p,0.05K_v],[K_p,K_v]]`. [Two-step check](../diagnostics/part3_theory_validation_20261012/closed_loop_bias_probe.py) uses the same keys under state perturbations, independently differences K/H at step 1, and compares with a separate end-to-end two-step controller perturbation. It manually propagates the audited plant map; it produces no outcome label and calls no evaluator.

| Placement | Two-step relative error |
|---|---:|
| TT | `9.91e−5` |
| TF | `4.61e−4` |
| FT | `9.91e−5` |
| FF | `1.60e−4` |

[Part 2 G/K table](../diagnostics/part2_closed_loop_sensitivity_validation_20261012/jacobian_validation.csv) already compares correction and state derivatives on Toy/Ring. Median relative G errors are TT/FT `0.02386` and TF/FF `0.00010/0.00027`; TT/FT maxima reach `0.258`, consistent with active/weak branch sensitivity and cannot be pooled into a clean smooth statistic. K median errors are `0.00009–0.00050`, maxima `0.00177–0.00214` across placements. Its projection P and geometry Q probes include up to `0.277` and `1.0` relative errors respectively near weak/singular cases; those are **exceptions**, not validation. H+G+K is therefore locally connected on the tested Toy branch, while a full regular-strata error map for all four scenarios is still missing.

### Evaluator semantics and margin theorem

The [Toy implementation](../single_integrator/environment.py) (re-exported by `toy_giveway`), [DB](../double_bottleneck/environment.py), [Four](../four_way_intersection/environment.py), and [Ring](../ring_exchange/environment.py) evaluators check **simultaneous** goal entry after each step. Their collision geometry is scene-specific: Toy/DB and Ring record swept wall/obstacle and pair checks; Four checks wall clearance at the endpoint and swept agent-pair clearance. Collision uses `<=` thresholds; goal uses `<=` tolerance. Toy/DB compute windowed progress, strict `<` candidate thresholds, a held candidate timer, reset the timer when the candidate condition fails, and trigger deadlock at `timer >= hold−1e−10`. Their first-event termination priority is collision, success, deadlock, timeout. Ring/Four have collision, success, timeout. At the last allowed step, success precedes timeout. Numerical controller/solver failure is recorded separately; it is not a negative continuous margin.

The old completion margin `ρ_goal=max_{1≤t≤T}(tol−max_i goal_error_i(t))` agrees with the saved binary label in the [Part 2 finite Toy/Ring set](../diagnostics/part2_closed_loop_sensitivity_validation_20261012/reuse_summary.json) for non-numerical seed-0 records (Toy 384 per placement; Ring 383–384 depending on numerical failures). **It is not globally equivalent to success**: a trajectory can hit a goal after a prior collision/deadlock, or hit goal and collision simultaneously (collision wins). Endpoint distances cannot replace swept geometry. At equality, `ρ_goal=0` is evaluator-successful, so even the shortcut `ρ_goal>0` is not exact on the boundary. A deadlock candidate exactly at its strict progress/speed threshold is false, while the timer equality triggers; the event map is piecewise and history-dependent. A trajectory reaching goal only after `max_steps` fails; reaching on step `max_steps` succeeds if collision-free. These are constructive counterexamples to a globally faithful smooth scalar margin.

A **strict local sufficient condition** is: there is a step `τ≤T` with all goal errors *strictly* below tolerance; every swept collision clearance through `τ` is strictly above its threshold; no earlier deadlock timer reaches hold; all actions/solves are finite; and all these event gaps remain positive under the perturbation. On such a fixed first-event and active-margin stratum, the minimum event gap is locally Lipschitz (and differentiable if its active component is unique). A local root displacement is `−(∂_c ρ·d)/(∂_ηρ·e)` along an η direction `e` only when the denominator is nonzero and the same event branch remains active. The old [boundary response](../diagnostics/part2_closed_loop_sensitivity_validation_20261012/boundary_response_summary.json) had 4 regular TT/FF Ring cases, same displacement sign, mean absolute root-fraction error `0.0353`; this is local evidence, not a global basin shape theorem. We did **not** independently validate a controller-weight-induced full-horizon margin displacement.

## Step 2 — Real-controller conditioning necessity

[Read-only script](../diagnostics/part3_theory_validation_20261012/reuse_analysis.py) reanalyzes P1 without changing candidate coverage. For each of the 91 real Flow checkpoint pairs, the same physical state, FF placement, Ring task, K16 bank, and seed keys are paired. `Y` is success and `U` is numerical unknown; unknown remains an interval. Seeds 0–1 rank pairs and η choices; seeds 2–3 evaluate them. The 24 states had been opened before P1, so this is **seed-disjoint development**, not untouched new-state confirmation. State bootstrap intervals describe these 24 opened states; they are not simultaneous population guarantees after screening 91 pairs.

The discovery-leading pair, `base_u_v2_macflow` versus `base_u_v10_local_macflow`, has confirmation mean `|ΔQ₂|=0.2253` over 24×16 cells; arbitrary unknown assignments leave a lower bound `0.2240`. The finite-bank uniform-prior information value `V=½ max Q_c+½ max Q_d−max ½(Q_c+Q_d)` is `0.1458` on the seed-2/3 empirical Q₂ estimates; conservative unknown interval arithmetic still gives `0.1458` for this realized panel. A 24-state bootstrap yields `[0.0833,0.2083]`, conditional on pair screening and the opened state panel. Discovery-optimal η differs for all 24 states; using the other controller's discovery choice loses `0.6042` on the confirmation Q₂ estimates (unknown-safe lower `0.5938`). [Pair table](../diagnostics/part3_theory_validation_20261012/pairs.json) retains all 91 pairs. These are larger than a pure common Q-level offset and show genuine **empirical finite-candidate choice conflict** for this pair. The source P1 [Q decomposition](../diagnostics/controller_q_transfer_round3_v1/P1_Q_DECOMPOSITION.json) independently reports controller×η and state×controller×η energy, while its preregistered eligibility gate failed (7/14 qualified, minimum 8).

**Critic necessity: `PARTIAL`.** A model forced to output one Q for the same `(s,η)` across these checkpoints cannot exactly fit the differing empirical Q₂ values. General population Q conflict has not been certified by Q16/Q64 independent seeds and new states. **Selector necessity: `PARTIAL`.** This particular finite K16 panel has positive empirical V, so a controller-blind oracle over this panel sacrifices value; a deployable controller-conditioned learned selector has **not** been shown to beat the strongest fixed or blind choice on held-out real checkpoints. The [P2′ test](../diagnostics/controller_q_transfer_round3_v1/REPORT.md) held out placement with the *same* Flow checkpoint, failed its joint selector gate, and cannot substitute for real-checkpoint transfer. A ranking reversal alone would not have sufficed; here an optimal-choice conflict is observed on the finite panel, with the above precision limits.

## Step 3 — Behavior, representation, and transfer

### Conditional behavior-to-Q theorem and measured terms

On a common physical state/action/task chart and a coupled noise path, define at bounded `||ξ||≤R`: `d₀` for decoded Flow initial physical action, `d_f` for vector-field response at the **same** observation, physical pseudo-action and pseudo-time, `d_a` for native safe anchor, `d_B` for basis, and `d_u` for executed action. Local recurrences through ten field steps and regular projection strata propagate these discrepancies to a path bound `E_T`; if all event gaps of one path exceed the corresponding Lipschitz image of `E_T`, its first-event label is unchanged. Consequently a probability Q bound has the form

`|Q_c−Q_d| ≤ Pr(||ξ||>R) + Pr(nonregular/solver exception) + Pr(event gap ≤ L_event E_T)`.

This is a **conditional coupling statement**, not an evaluated global guarantee. Gaussian noise is unbounded; the projection set changes with state; local KKT derivatives do not certify a global Lipschitz constant; swept collision and deadlock event switches need the explicit exceptional probability. Distances measured on finite probes are sampled distances, never certified suprema.

P1 static descriptors offer sampled executed-action and correction-response probes (`R4_action`) and a compressed nominal/field fingerprint (`R3`). For the discovery pair, their mean Euclidean distances over the 24 states are `14.90` and `1.276` in their own feature units. Across 91 checkpoint pairs, pair-mean R3/R4 distances correlate only `0.143/0.029` with confirmation mean `|ΔQ₂|`; correlations with empirical V are `0.527/0.501`. These descriptive correlations have no calibrated scale or causal guarantee. The [P1 geometry audit](../diagnostics/controller_q_transfer_round3_v1/P1_STATIC_GEOMETRY_AUDIT.json) measured real action/Jacobian differences; old [closed-loop mechanism evidence](../diagnostics/placement_boundary_mechanism_audit_v1/paired_trajectory_sensitivity.json) is for TT/FF placement, not two different Flow checkpoints. P1 does not store full paired trajectories or event margins for every real-checkpoint pair, so the last links of the proposed chain remain **unmeasured for this pair**.

### Representation sufficiency

The actual compact fingerprint [compressor](../diagnostics/controller_q_diversity_v1/train.py) keeps a field intercept for each of 7 pseudo-poses × 3 times, but **averages** its finite-difference Jacobian over all pose/time probes and keeps only overall variance. The raw [probe design](../diagnostics/controller_q_diversity_v1/features.py) includes near-goal moving, opposite-moving, and lateral states, yet their **specific response identity is discarded** by this compression. Swapping two pose/time Jacobians preserves mean and variance while changing the near-goal moving response: a constructive representation alias. Thus the previously identified omission persists as a structural possibility. It is not proof that an exact alias of this kind occurs among the 14 P1 real checkpoints. In another held-placement panel R3 *does* alias FF/TT and TF/FT while Q differs, per [P2′ alias diagnostic](../diagnostics/controller_q_transfer_round3_v1/P2PRIME_R3_ALIASING_DIAGNOSTIC.json). R4_action distinguishes those placements but did not produce reliable held-placement selection benefit.

**Minimal testable repair direction:** retain the near-goal moving response and its local finite-difference Jacobian **indexed by probe pose and pseudo-time** rather than only global mean/variance; first test representation-distance versus raw behavior distance and Q ranking on existing labels. This is a proposed probe/representation test, **not** a critic retraining plan or a claimed solution.

### Transfer inequality and numerical usefulness

A valid triangle bound for a common aligned domain is

`||Q_t−Q̂_t||∞ ≤ ||Q_s−Q̂_s||∞ + ||Q_t−Q_s||∞ + ||Q̂_t−Q̂_s||∞`.

To replace the last two terms by `representation discrepancy + behavioral transfer error`, one must certify a predictor Lipschitz constant in the representation and a behavior-to-Q bound under the coupled first-event semantics. Neither is certified. For the discovery pair, even an **oracle source critic** has source error zero while the observed confirmation target-source cell difference reaches `1.0`; the only conservative binary-Q range bound is `1`, and the conventional regret bound `proposal regret+2×critic error` reaches at least `2` before capping at the effective Q/regret range `1`. This is **`VACUOUS_BOUND`**, not validation. The sampled `0.2253` mean `|ΔQ₂|` is an empirical transfer error estimate, not a uniform bound. A population Q confidence interval and calibrated descriptor Lipschitz constant are absent. The source [P2′ generalization result](../diagnostics/controller_q_transfer_round3_v1/REPORT.md) likewise found lower Q MSE for R3+R4 than R0 without reliable selector gain; prediction fit and regret must remain separate.

Across **scenes**, robot count, observation/action chart, normalization, OrthoFlow3 basis, geometry, task/horizon, candidate η identity and success event must all be mapped into a common measurable domain before this inequality has a physical interpretation. Current four-scene LOSO mixes different geometry and often different failure mechanisms; no verified alignment map exists. No cross-scene theorem has been validated.

## Verdicts and minimum paper path

| Claim | Verdict | Reason |
|---|---|---|
| Four-chain executable formula and anchor-mediated H formula | `VALIDATED` as source identity and local chain rule | Source audit plus one independent JVP/KKT/FD Toy branch; global H numerics remain limited. |
| Smooth local H and two-step `S` prediction | `PARTIAL` | All placements pass one Toy state/direction; non-smooth strata and DB/Four/Ring H are untested. |
| Globally exact continuous success margin | `REJECTED` | First-event precedence, swept collision, deadlock history, equality and numerical failure give counterexamples. |
| Local regular-margin displacement | `PARTIAL` | Old 4 Ring placement roots agree locally; controller-weight root displacement is untested. |
| Real Flow controller-dependent empirical Q | `PARTIAL` | Same-state/η/seed Ring Q₂ conflict; population/new-state confirmation missing. |
| Real Flow controller-dependent empirical optimal η selection | `PARTIAL` | Finite K16 V positive on seed split; no held-real-checkpoint deployable selector proof. |
| Current compact representation sufficiency | `REJECTED` as a general sufficient-statistic claim | Compression structurally loses indexed near-goal moving Jacobian; exact real-checkpoint harmful alias remains unverified. |
| Useful representation/behavior transfer bound | `REJECTED` for current evidence (`VACUOUS_BOUND`) | Missing certified constants/tails; conservative numeric bound exceeds Q range. |
| Cross-scene transfer theorem | `BLOCKED` by alignment evidence | No checked common state/action/basis/task map. |

For a paper-grade necessity claim, freeze the discovery pair and a **new-state** generator before observing outcomes, choose a shared η subbank from the old discovery data, run cache preflight, and confirm on independent Q16 seeds with simultaneous confidence intervals for Q difference and selection value. A 2-controller × 5-new-state × 2-η × 16-seed **320-continuation pilot** fits the first-round budget but is unlikely to establish a population statement; a 24-state × K4 × Q16 confirmation would require up to 3,072 requests before exact cache reuse and therefore requires a separate budget decision. For a generalization claim, first retain the indexed near-goal response, measure same-noise behavior and event gaps on a held real Flow checkpoint, calibrate a nonvacuous bound on an independently selected domain, then test held-checkpoint Q and selection against the blind/fixed baselines. Keep failure/unknown outcomes and nonregular strata explicit. No large critic retraining follows from this audit.

## Reproduction and limitations

Use `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python` with `JAX_PLATFORMS=cpu`:

```bash
python diagnostics/part3_theory_validation_20261012/reuse_analysis.py
python diagnostics/part3_theory_validation_20261012/h_bias_probe.py
python diagnostics/part3_theory_validation_20261012/closed_loop_bias_probe.py
```

Scripts write [results.json](../diagnostics/part3_theory_validation_20261012/results.json), [pairs.json](../diagnostics/part3_theory_validation_20261012/pairs.json), [h_bias_results.json](../diagnostics/part3_theory_validation_20261012/h_bias_results.json), and [closed_loop_results.json](../diagnostics/part3_theory_validation_20261012/closed_loop_results.json). They require archived external field and hard-field directories named in the scripts. The existing P1 files and checkpoints remain authoritative; this report does not reinterpret placement difference as a difference between real Flow controllers. No source data were deleted or rewritten.
