# Round19 Four controller axis preregistration

Registered before any Round19 gate execution, unlabeled probe, or new success label. All writes for this study stay in `diagnostics/four_controller_axis_v1/`, except the prescribed shared rollout DB lifecycle. No historical asset or PoC code is modified.

## Historical evidence and gap

- `diagnostics/four_controller_census_v1/`: Round14B metadata audit, v13 best E8 eight case byte gate, 32 Four states and 384 Q2 labels for v13 best/final. These labels remain valid for their exact execution identities, but their old states are not the 32 new states specified here.
- `diagnostics/controller_ff_factorial_repair_v1/` and `diagnostics/controller_flow_endpoint_probe_v1/`: E8 Flow reference, FF `flow_isotropic`, four scene state/eta cases and raw gate records. Reuse their source code and historical records read only.
- `diagnostics/cross_scene_eta_schedule_v1/`: eta switching wrapper and t=471/786 design. Reuse implementation semantics read only.
- `diagnostics/orthoflow3_controller_intervention_generalization_v1/` and `second_variant/`: historical v12 and v10-wide intervention under canonical Flow at t0, alternate Flow for future steps. Their controller identity and conditioning differ from this study's pure per-checkpoint Flow and cannot be reused as census labels. Their raw journal records are candidates for G1 only if all required comparison fields exist.
- Shared DB: preflight decides exact seed reuse; never infer compatibility from checkpoint name or changed critic encoding. Existing Round14B labels on old states do not answer the new-state census.

## Fixed controller cohort

The 13 best checkpoints with environment fingerprint prefix `73910125d2` are listed below. Include the same run's final checkpoint if its network parameter tree differs from best; the independent parameter hash audit currently indicates 12 such finals, all except `base_u_plus_w_v5_outer_buffer_dense_macflow_seed3111_superseded`. Actual cohort is 25 checkpoints, subject only to G2 verifying this read-only audit. The nine different-environment best runs and their finals are excluded.

1. `base_u_plus_w_v5_outer_buffer_dense_macflow`
2. `base_u_plus_w_v5_outer_buffer_dense_macflow_early`
3. `base_u_plus_w_v5_outer_buffer_dense_macflow_seed3111_superseded`
4. `base_u_plus_w_v5_outer_buffer_macflow`
5. `base_u_plus_w_v6_policy_uniform_macflow`
6. `base_u_v10_world_timeout_goal_recovery_source_balanced_macflow`
7. `base_u_v10_world_timeout_goal_recovery_source_balanced_wide_macflow`
8. `base_u_v11_world_timeout_agent_recovery_source_balanced_macflow`
9. `base_u_v12_world_timeout_goal_recovery_canonical_source_balanced_macflow`
10. `base_u_v13_broad_global_source_balanced_macflow`
11. `base_u_v5_outer_buffer_macflow`
12. `base_u_v9_world_goal_recovery_macflow`
13. `base_u_v9_world_goal_recovery_source_balanced_macflow`

## Stage 0: strict equivalence gates

- Adapter uses E8 native FF Flow-reference `flow_isotropic` runtime, each checkpoint's own obs/act normalization arrays and actor width, and the Round13 eta switch wrapper. No changes to plant, safety, OrthoFlow3, physics or success rule. Force JAX FP32 after checkpoint loading.
- G0: replay the fixed eight v13 best E8 cases selected as Four, Flow-isotropic, train states 0/1, eta10/15, seed0/1. For every case compare terminal outcome, episode length and float64 encoding of the first action against the historical raw record. Any mismatch fails.
- G1: replay eight historical v12 best and eight historical v10-wide best intervention records, deterministically the lexicographically first eight recoverable matching journal records for each checkpoint under each original intervention's canonical-t0-then-alternate-future conditioning. For every record, compare terminal outcome, episode length and the exact bytes of its recorded first action. If fewer than eight records or historical first-action bytes are unavailable, G1 fails as unverifiable. Recomputing a historical first action from the same code is not independent historical evidence and does not pass this requirement.
- G2: for all 25 candidates compare environment fingerprint and config fields against v13; only the four normalization arrays and v10-wide actor width may differ. Same-key repeated forward outputs must match bytewise and be finite, and FP32 must be explicitly restored after import. Any failure stops the study.
- Any failed or unverifiable G0/G1/G2 stops before Stage 1 and before any new success label. Report the exact mismatch or missing historical evidence as `INCONCLUSIVE`; no threshold relaxation.

## Stage 1: unlabeled probe, only if all gates pass

Draw 16 fresh Four t0 states, deduplicate against all recoverable DB physical states. On each eligible controller run 200 steps with each fixed eta10, eta13, eta15; record the R5 five process summaries for every state/eta. No terminal success label may be read or generated. Freeze raw features and PCA-2 transform with SHA256 UTC receipt before Stage 2. Fit PCA-2 on all eligible controllers' unlabeled features only, using deterministic sorted controller order, missing-value imputation by training-column mean and sign orientation by largest loading positive. This is transductive without target labels.

## Stage 2: fixed-arm census

Draw a disjoint set of 32 fresh Four t0 states, DB deduplicated. For each controller, state and seeds 0/1, run F10, F13, F15 at H1100. Numeric failure is unknown, never rerun or counted as failure. For reported rates use successes / known seeds and show unknown counts. A controller qualifies for regime difference if its best arm differs from the most frequent best arm (tie by F10,F13,F15 order), with best-minus-majority-arm difference at least 10pp. Gate passes if at least three controllers qualify, or at least three controllers have best fixed-arm success rate at most 85%. Otherwise stop after census and report absence of demonstrated Four controller-axis headroom in this cohort.

## Stage 3: schedules and decomposition, only if headroom passes

On the identical Stage 2 states/seeds run S15→10 and S10→15 at t=471 and 786, plus S13→10 at t=786. The full arm order is F10,F13,F15,S15to10_471,S15to10_786,S10to15_471,S10to15_786,S13to10_786. For each schedule, E=S−F_late, L=S−F_early, N=F_early−F_late, all in percentage points. For each left-out run, both best/final leave together. Choose the source-controller average best action from fixed or expanded menu, tie by listed arm order; apply it to held-out controllers and report mean and max oracle regret. `H_compress_Four` is SUPPORTED iff expanded mean regret ≤ fixed mean regret /3 and at least half of controllers have at least one eta pair with both E and L ≥10pp; else REJECTED or INCONCLUSIVE if unknowns prevent the comparison.

## Stage 4: E learnability, only with Stage 3

Across grouped leave-one-run-out folds, find the most frequently selected exact time schedule arm from the expanded source-average choices; tie by listed schedule order. That arm defines E=S−F_late. If no schedule is ever selected, E learning is INCONCLUSIVE. Using only the frozen probe PCA-2, fit ridge alpha=1 on source runs and predict held-out run's best/final E. Standardize PCA scores using source-fold mean/scale and compare with source-mean E baseline. SUPPORTED iff held-out Spearman ≥0.6 and RMSE ≤0.7 times baseline; otherwise REJECTED, or INCONCLUSIVE when known E labels are insufficient. Do not split siblings across train/test.

## Execution, resources, video and reporting

For every new rollout: serialize exact state/eta/controller/seed requests, official preflight, execute only genuinely missing seeds, journal from workers, single-process merge, official postflight. Register adapter controllers as new execution identities. Before every submission check all user jobs and stay at or below 18 CPU / 2 GPU total. Four video delivery requires full simulator trace and the uniform `new_benchmark_common.batch_videos` entry. Four has no documented deadlock criterion; timeout never fills the two deadlock roles. If those roles lack valid evidence, report video delivery incomplete without fabrication. Report G0/G1/G2, Four regime question, H_compress and E learning in the first paragraph, marking each SUPPORTED/REJECTED/INCONCLUSIVE and separating facts from interpretation.
