# Physical-only state conditioning: joint-four test and causal diagnosis

**Verdict for source-only four-scene LOSO: `CROSS_SCENE_STATE_CONDITIONING_NOT_ESTABLISHED`.**
Within-scene state use is established in Toy and previously in Ring under its separately validated 512-state execution profile. The current four-scene experiment has not established transferable within-target-scene adaptation. The joint-four control is generalization to new states of seen scenes, not LOSO.

## Frozen protocol and cache reuse

One joint physical-entity + eta critic and a fair eta-only critic were fitted to exactly the existing TRAIN labels: Toy 256, DB 48, Four-Way 48, Ring 112 independent states (464 total), plus 32/16/16/16 independent source validation states (80 total). All source labels are Q4 on K16. Equal-scene sampling and optimization budgets were shared. Concat and prior-plus-ranking-residual were selected using source validation only; concat was selected. No scene ID, Flow, response, controller fingerprint or rollout preview enters the model.
No new TRAIN rollout was needed. Existing full K16 and common-bank outcomes on 32 states per scene were reused only for DEVELOPMENT diagnosis. A different 32 states per scene (128 total) were frozen from the original unfiltered TEST sampler order before outcomes; all four scenes are represented in TRAIN, none of these physical states is. Model, predictions and 100 within-scene derangements were frozen before fresh Q16 generation.
Fresh confirmation requested 8,832 selected-policy seed outcomes; exact preflight reused 0, leaving 8,832 genuinely missing. Simulate the union of joint, fair eta-only, frozen LOSO, frozen within, 100 mismatches and Safety selections, once per exact state/eta/controller/seed. This avoids regenerating a 34,816-seed full pool. Unmeasured entries are explicitly -1; **there is no fresh full-pool oracle or best-fixed certificate**. Complete all-fixed falsification is reported separately on the existing common-bank development panel.
B15 means at least 15 successes in 16 predeclared seeds; numerical failures remain unresolved upper/lower bounds. All paired intervals use 20,000 physical-state bootstrap resamples. No video is required for this pure Q audit. CPU rollout concurrency is capped at 12 (at least 5 of 24 CPUs remain outside this task).

## New test: train on four scenes, test on fresh states of all four

Main values are B15 states /32. Mismatch is the mean count over 100 derangements, not another independent sample.

| Scene | Safety | Joint Q(eta) | Joint Q(s,eta) | Mismatched state | Frozen LOSO state | Frozen within state |
|---|---:|---:|---:|---:|---:|---:|
| Toy | 16/32 | 25/32 | 27/32 | 22.86/32 | 12/32 | 28/32 |
| DB | 7/32 | 32/32 | 32/32 | 32.00/32 | 32/32 | 32/32 |
| Four-Way | 0/32 | 30/32 | 29/32 | 29.73/32 | 31/32 | 30/32 |
| Ring | 22/32 | 26/32 | 27/32 | 27.00/32 | 7/32 | 30/32 |

Four-Way has numerical unresolved bounds: joint state is 29 confirmed /32, up to 32 possible; joint eta-only is 30 confirmed, up to 32 possible. All non-numerical selected Four-Way trials succeed. Its small negative differences below are differences in certified-success lower estimates, not established physical-outcome degradation. JSON also reports paired worst/best-case numerical sensitivity; do not claim state hurts from this row.

Mean selected rollout success (%), followed by paired differences in percentage points and 95% CI:

| Scene | Safety | Joint eta | Joint state | Mismatch | State − eta | State − mismatch | State − Safety |
|---|---:|---:|---:|---:|---|---|---|
| Toy | 62.89 | 86.52 | 90.04 | 81.41 | +3.52 [-2.73, +11.13] | +8.63 [+3.18, +14.23] | +27.15 [+8.39, +45.51] |
| DB | 71.09 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +28.91 [+20.70, +38.09] |
| Four-Way | 60.55 | 97.07 | 95.51 | 96.00 | -1.56 [-3.52, +0.39] | -0.49 [-1.09, -0.02] | +34.96 [+29.49, +40.43] |
| Ring | 78.52 | 93.55 | 96.68 | 96.68 | +3.12 [-0.78, +8.98] | +0.00 [+0.00, +0.00] | +18.16 [+6.25, +31.05] |

| Scene | B15 state − eta (pp, CI) | B15 state − mismatch (pp, CI) | B15 state − Safety (pp, CI) |
|---|---|---|---|
| Toy | +6.25 [-6.25, +18.75] | +12.94 [+6.50, +20.03] | +34.38 [+12.50, +56.25] |
| DB | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +78.12 [+62.50, +90.62] |
| Four-Way | -3.12 [-12.50, +6.25] | -2.28 [-6.41, +0.00] | +90.62 [+78.12, +100.00] |
| Ring | +3.12 [-9.38, +15.62] | +0.00 [+0.00, +0.00] | +15.62 [-3.12, +34.38] |

## Original completed confirmation: within-scene versus LOSO

These are the earlier, separate 32 states per scene; do not pool them with the new test after diagnostic use. Oracle and all K16 candidates were actually measured here.

| Regime | Scene | Safety | Eta-only | State | Same-scene mismatch mean | Oracle |
|---|---|---:|---:|---:|---:|---:|
| within | Toy | 20/32 | 27/32 | 30/32 | 25.03/32 | 31/32 |
| within | DB | 4/32 | 32/32 | 32/32 | 32.00/32 | 32/32 |
| within | Four-Way | 0/32 | 30/32 | 30/32 | 30.00/32 | 32/32 |
| within | Ring | 22/32 | 29/32 | 30/32 | 30.30/32 | 32/32 |
| loso | Toy | 20/32 | 22/32 | 17/32 | 17.00/32 | 31/32 |
| loso | DB | 4/32 | 29/32 | 31/32 | 30.81/32 | 32/32 |
| loso | Four-Way | 0/32 | 31/32 | 28/32 | 28.00/32 | 32/32 |
| loso | Ring | 22/32 | 5/32 | 6/32 | 5.97/32 | 32/32 |

## State mechanism and all-fixed falsification

Common fixed K16 bank, all eligible pairs; discovery threshold 3/8, independent evaluation half. `Both-side switch` requires correct positive ranking in Group A AND correct negative ranking in Group B. Unlike unbalanced pair-state accuracy, a fixed ranking necessarily scores zero. Equal-pair score is bootstrapped by resampling physical states, never pair cases.

| Regime | Scene | State effect corr. | True reversal pairs | Both-side switch | Selection entropy (bits) | State eval success | Best fixed eval success | State − bootstrap-reselected fixed (pp, CI) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| within | Toy | 0.707 | 47 | 86.1% | 0.974 | 96.09% | 90.23% | +5.86 [-6.25, +18.36] |
| within | DB | -0.142 | 0 | — | -0.000 | 100.00% | 100.00% | +0.00 [+0.00, +0.00] |
| within | Four-Way | 0.051 | 0 | — | -0.000 | 100.00% | 100.00% | +0.00 [+0.00, +0.00] |
| within | Ring | -0.072 | 8 | 1.0% | -0.000 | 82.42% | 82.42% | +0.00 [+0.00, +0.00] |
| loso | Toy | -0.066 | 47 | 0.0% | -0.000 | 44.53% | 90.23% | -45.70 [-63.67, -29.30] |
| loso | DB | 0.000 | 0 | — | -0.000 | 100.00% | 100.00% | +0.00 [+0.00, +0.00] |
| loso | Four-Way | 0.024 | 0 | — | -0.000 | 92.97% | 100.00% | -7.03 [-12.11, -3.12] |
| loso | Ring | 0.023 | 8 | 3.9% | -0.000 | 0.39% | 82.42% | -82.03 [-91.41, -71.48] |
| joint | Toy | 0.525 | 47 | 57.8% | 0.896 | 86.72% | 90.23% | -3.52 [-19.53, +12.50] |
| joint | DB | -0.043 | 0 | — | 0.201 | 96.88% | 100.00% | -3.12 [-9.38, +0.00] |
| joint | Four-Way | 0.060 | 0 | — | -0.000 | 92.97% | 100.00% | -7.03 [-12.11, -3.12] |
| joint | Ring | 0.038 | 8 | 0.0% | -0.000 | 82.42% | 82.42% | +0.00 [+0.00, +0.00] |

All LOSO models choose a single fixed eta on the common bank. Their decision value cannot exceed the best of those 16 fixed policies and their correct-state value is exactly zero there. In joint training, Ring changes to the appropriate scene-level fixed eta, but still does not switch across Ring states. Toy joint training does use the within-scene state signal, while its common-bank success does not reliably exceed the posthoc best fixed policy with this sample.

## Direct diagnosis: two separable failures

1. **Wrong eta prior on an unseen execution regime.** A diagnostic intercept adjustment uses target discovery-half mean Q for each eta, without fitting any state dependence. On independent evaluation seeds it changes LOSO selected success Toy 44.53→90.23%, DB 100→100%, Four-Way 92.97→100%, Ring 0.39→82.42%. All four then equal their best fixed eta and have zero matched-versus-mismatched gain. This is a privileged causal diagnostic, **not a source-only LOSO repair**: it uses target labels. It shows the catastrophic errors can be removed by correcting the eta main effect alone, without recovering state adaptation.
2. **Missing transferred state×eta interaction.** Correcting the eta prior still leaves no decision-level use of target state. Training on all four scenes restores Ring success largely by learning the Ring eta prior; same-scene mismatch and common-bank fixed-policy checks expose this. Toy provides a positive control showing the same code and physical input can learn a real within-scene interaction.
The supported training limitation is coverage of mechanisms and interactions, not merely the number of rollout rows. DB/Four source proposal labels are close to ceiling (mean Q approximately .936/.974), so many states provide little selection contrast. Ring has only 112 current-profile source states, and only 16 source states use the broad fixed bank. Source validation may reward a good scene prior even when no within-target-state interaction transfers. A ranking residual and source-state expansion were already tested; neither established LOSO matched-state value. Q4 seed-half preference signs persist at roughly 98% in informative Toy/Ring cases, so label noise alone is not a supported explanation.
The source-zero Ring curvature channels really are untrained, but setting those channels to source-zero changes **none** of the current Ring selections. Therefore their arbitrary learned weights are not the direct explanation for these failed decisions. Ring target eta support is close to source support; candidate coverage alone also cannot explain Ring. Toy has a separate eta-support shift (60.4% of candidate vectors have some coordinate outside source bounds); this is a contributing confound, not a universal cause.

## Is the same Q(s,eta) input theoretically valid within-scene and in LOSO?

**The same input signature is valid; it does not guarantee the same target function.** With execution contract c fixed, the target is Q^c(s,eta)=P(success | s,eta,c). A within-scene critic can learn that fixed controller, dynamics and evaluator implicitly in its weights. To identify a controller-independent target from physical state alone across changing contracts would require Q^{c1}(s,eta)=Q^{c2}(s,eta) on overlapping physical inputs, or another justified invariant mechanism. Otherwise Q(s,eta) is at best a conditional average over the training distribution of c; changing that distribution at deployment changes the target.
The current encoding already includes relative agent/goal/obstacle geometry, radii and masks. It does not omit the map. It deliberately removes Flow, controller response, policy-frame information and global execution constants. The four pipelines have different frozen Flow hashes, safety fingerprints and horizons (Toy/DB 850, Four 1100, Ring 700). Thus current LOSO changes both geometry and execution mechanism. Joint-four training can infer seen regimes from physical geometry without an explicit scene ID; that is compatible with strong scene-prior shortcuts and does not establish state-level transfer.
An exact archived controller intervention was re-audited in the shared database, reusing 2,048 seed records and no new simulation: the same physical state, eta, current Flow reference, step-0 action and matched seeds are kept; only the future frozen Flow from step 1 changes. 8 of 64 state–eta pairs change from 16/16 to 0/16; 9 pairs pass Holm correction. Any deterministic physical-only encoding is identical in each such pair. This proves information insufficiency **across that allowed controller family**. It does not prove equal-input contradictory labels within the present four fixed-scene supports, nor quantify what fraction of this LOSO failure is due to hidden controller conditions.
Archived short-response context ablations already attempted to expose controller behavior and still failed LOSO (selected Ring 1/60 versus eta-only 27/60 under that historical protocol). These are diagnosis-only controls from a different execution/data cohort, not current baselines. Adding arbitrary context is therefore neither a demonstrated repair nor part of the final physical-only method.

## What the evidence supports, and the corresponding method

- For new states of known frozen scene/controller regimes, keep Q^c(s,eta), reuse compatible labels, and demand matched-state versus within-scene mismatch evidence. Joint-four training addresses this deployment scope; it must not be described as LOSO.
- For unseen scene AND unseen frozen controller, additional physical states of the same few source mechanisms do not establish identifiability. A physical-only transfer claim needs a justified shared execution mechanism or source diversity that demonstrates invariant state×eta effects, followed by untouched target confirmation. With the pipeline fixed as requested, the present source-only model fails that empirical test.
- The current failure is not established as generic memorization or a broken state input. It is observed loss of state×eta transfer plus a wrong domain eta prior. Source training and source validation can improve while this domain relation fails. No test here proves that all future physical-only architectures must fail, and no controller-conditioned deployment is being proposed as an already validated solution.

## Safety, failures and reproducibility

All fresh per-state methods, exact seed records, numerical bounds, failure states and source manifests are retained. Collision counts are checked independently of success. Numerical failures are not silently recoded as scientific failures; B15 is reported conservatively as confirmed success, with possible counts retained in JSON.

Shared working-tree MACFlow source had changed for an unrelated experiment since protocol freeze. Exact controller-identity guards rejected 68 first-seed attempts (zero accepted labels). The original source with matching SHA was restored in an isolated module snapshot; all original controller payload fields and UIDs were verified before continuing. Shared edits and all completed outcomes were preserved. These 68 rejected attempts are additional execution overhead, not part of the 8,832 accepted seed labels. See `frozen_execution_restore.json` and `rejected_identity_attempts.json`.

| Scene | All executed collisions | All executed numerical failures | Joint selected collisions | Joint selected numerical failures |
|---|---:|---:|---:|---:|
| Toy | 0 | 0 | 0 | 0 |
| DB | 0 | 0 | 0 | 0 |
| Four-Way | 0 | 40 | 0 | 23 |
| Ring | 0 | 5 | 0 | 1 |

Artifacts:

- `protocol.json`, `source_data/joint_four_scenes/manifest.json`, `final_models_frozen.json`: source-only model and split provenance.
- `fresh_confirmation/final_protocol.json`, `frozen_splits.json`, `predictions/`: prospectively frozen 128 fresh states, candidate choices and 100 mismatch controls.
- `fresh_confirmation/final/proposal/*/{cases.json,cache_preflight.json,selected_outcomes.npz,paired_bootstrap.npz,safety_issues.json}`: exact state/eta/seed cases, raw labels and bootstrap draws/outputs.
- `fresh_confirmation/results.json`, `per_state_methods.csv`, `failure_cases.json`: final paired results and every state-level outcome.
- `development/results.json`, `mechanism_audit.json`, `mechanism_bootstrap/`: all-16-fixed and both-direction switching audits, with independent discovery/evaluation halves and bootstrap re-selection of the fixed comparator.
- `../posthoc_cause_audit/{results.json,prior_intervention.json,controller_information.json}`: direct interventions and evidence boundaries.
- Earlier complete within/LOSO results: `../stage_v3_coverage/final_phase1_32/final/INTERPRETATION.md` and `consolidated_results.json`.

Paper claim: physical state has demonstrated within-scene selection value, but the tested physical-only critics have not established transferable within-held-out-scene eta adaptation. Joint training on all four scenes is a different, easier generalization question and must be reported separately.
