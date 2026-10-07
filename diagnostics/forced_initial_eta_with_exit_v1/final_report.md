# Forced initial structured eta with learned exit — development audit

## Frozen protocol

This is a **development-only**, matched 200-episode WIDE evaluation. The manifest was frozen at `2026-09-26T11:22:09.868611+00:00` before any rollout (file SHA256 `e76e0411ef8409d0003eab2478c74e570dd166ecdae4519e11480f950ba0bc84`). Exact prior IC overlap was zero. No model was trained or calibrated. The tested controller has no learned entry, no H/L schedule, no periodic trigger, no Direct-g action, and no online oracle/eta search. Every episode predicts eta exactly once at step 0, executes at least one structured transition, then uses the frozen learned exit and Safety forever after exit.

G_eta: `/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz` (SHA256 `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`). Exit head: `/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/entry_exit_training_history/exit_policy_iteration/exit_eta_lambda1_seed17.npz` (SHA256 `7b3bd0d9caff3bfa4baed95d227a6484b5f6f176d7c5b21a33715acb151db694`), architecture 217→64→64→1 SiLU with frozen probability threshold `0.25` (logit `-1.0986122886681098`).

## Primary results

| Controller | Success | Strict deadlock | Timeout | Collision | Q (Wilson 95% CI) | Mean J_def | Median successful completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safety | 133/200 | 5 | 62 | 0 | 0.665 [0.597, 0.727] | 0.00000 | 28.55 |
| Forced-Eta+Exit | 123/200 | 54 | 23 | 0 | 0.615 [0.546, 0.680] | 0.05191 | 27.15 |
| Learned-Entry+Exit reference | 174/200 | 5 | 21 | 0 | 0.870 [0.816, 0.910] | 0.02177 | 28.35 |
| Direct-g H8 reference | 190/200 | 0 | 10 | 0 | 0.950 [0.910, 0.973] | 0.02473 | 28.00 |


Forced-Eta+Exit versus Safety: rescue **30**, break **40**, net rescue **-10**, paired Delta Q **-0.050** (episode-bootstrap 95% CI [-0.130, +0.030]). Safety had 133 successes, 5 strict deadlocks, and 62 timeouts.

Failure-type rescue: Safety timeout 26/62; Safety strict deadlock 4/5.

## Exit and startup behavior

The forced controller exited in **194/200** episodes. Exit time was mean **2.023 s**, median **0.525 s**, P10/P25/P75/P90 **0.050/0.050/2.700/7.400 s**, range **0.050–23.200 s**. Safety-success episodes exited earlier (mean/median **1.668/0.500 s**) than strict-deadlock episodes (**10.790/9.550 s**); timeout episodes were **2.044/0.525 s**. This is descriptive, not causal.

The matched learned-entry reference still entered in **200/200** episodes, but not at startup: entry mean/median were **12.162/10.825 s** (range **4.800–29.350 s**) and it exited in **200/200**. Its all-episode entry behavior therefore implements a state-dependent onset time, not forced step-0 activation.

Startup integrity passed: feature dimension 214, exact step-0 left padding with 41 copies of initial goal error, one eta query, frozen eta, authoritative de-normalization/clipping, and second projection. Eta clipping fraction was **0.295**; eta norm mean/median/P95 were **0.637/0.589/1.033**.

Forced-Eta+Exit J_def mean/median/P95/max was **0.05191/0.03719/0.17111/0.51029**. Successful completion time mean/median was **27.73/27.15 s**. On the 93 episodes successful under both Safety and Forced-Eta+Exit, paired completion-time delta mean/median was **-1.61/+0.00 s**.

## Interpretation

**EXIT_NOT_SUFFICIENT_TO_PROTECT_NOMINAL**

The frozen exit fired, but not before substantial Safety-success break.

All 40 breaks terminated as strict deadlock. Of those breaks, 34 eventually exited with median exit time **2.950 s**, while 6 never exited; preserved Safety successes exited at median **0.200 s**. Delayed/unsupported exit behavior on startup states is therefore the leading proximate limitation. This does not prove the first structured action is harmless, so startup eta mismatch cannot be fully excluded.

The suitable semantic description is an **initial structured coordination phase**, not recovery before a failure has developed. The prior frozen learned-entry system and old Direct-g H8 are development references only; neither is a component of the forced controller.

Hard safety: agent collisions 0, wall collisions 0, invalid actions 0, NaN/Inf 0, solver failures 0.

Runtime/resources: 432.3 s rollout wall time; 800 controller-episode rollouts; 2 GPU shards; 6 requested CPU threads; 36 GiB host-memory allocation (observed batch MaxRSS about 1.10 GiB during execution).

## Smallest justified next experiment

Freeze this result. The smallest justified next experiment is a development-only **forced-start exit-latency counterfactual** using the same frozen eta and saved matched states: compare the current exit against immediate Safety handoff after exactly one structured transition. This separates damage from the first startup eta action from damage accumulated because exit is late. Do not retrain from this cohort.
