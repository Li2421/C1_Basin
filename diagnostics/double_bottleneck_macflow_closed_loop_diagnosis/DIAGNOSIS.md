# Double-Bottleneck MACFlow Stage-I closed-loop diagnosis

Date: 2026-09-23  
Scope: diagnostic only; no permanent policy change, no eta/basin experiment, and no `G_phi` training.

## Executive conclusion

The current 4-agent MACFlow does **not** fail because of a demonstrated mismatch with the Toy MACFlow primitives, nor because its first action is generally unusable. On held-out expert states, one-step errors are small and comparable to Toy. The strongest observed discriminator is support under feedback: the Double-Bottleneck policy leaves the expert-state support almost immediately, after which errors compound near long wall segments and eventually produce wall collisions. This diagnosis is associational until a coverage intervention is tested.

The evidence does not support explicit coordination labels as the remedy. Reusing one base-noise vector for an episode makes actions smoother but gives 0/12 success. A temporary LTR/RTL-conditioned oracle also gives 0/12 success. Mode switching is less frequent than in successful Toy rollouts under the same approximate offline classifier. Hard safety removes the collisions, but it is active on 82.31% of steps and all protected rollouts time out, so it is correcting a substantially wrong off-support action field rather than a small safety defect.

The minimal supported next change is therefore **Toy-style local state augmentation with expert re-query**, concentrated first around mode transitions, approaches/final approaches, and wall-near states, while keeping the deployable policy as unconditioned `p(u|x)` with the same joint MACFlow architecture. Before generating a larger dataset, the centralized expert must be validated as a local re-query/recovery oracle from perturbed intermediate states. This recommendation is not implemented in this phase.

## Protocol and artifacts

- Double-Bottleneck checkpoint: `diagnostics/double_bottleneck_macflow_8mode_pilot/best.pkl`, SHA-256 `4a6d9a49d7b8d76c8db9aa637742ea1b981e261df678333271768d429827af65`.
- Double-Bottleneck data: `diagnostics/double_bottleneck_expert_dataset_8mode/`, manifest SHA-256 `034570ffab5313a23f0a4a7bda7240c093a351929ea0b17f68b084b180fb8ce3`.
- Toy checkpoint: `/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl`, SHA-256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`.
- All new diagnostic rollouts ran on CPU. No large-scale data generation or hyperparameter search was performed.
- Exact numerical outputs are in the JSON/CSV files beside this report. Plots are in [`plots/`](plots/) and representative trajectories are in [`trajectories/`](trajectories/).
- “Off support” is a calibrated nearest-neighbor diagnostic, not a theorem: normalize observations using each model's own training statistics, compute held-out-expert-to-training-expert nearest-neighbor distances, and use that model's 99th percentile as its threshold.

## Toy reference

### Training data

The working Toy Stage-I baseline starts from 250 paired initial conditions and 500 source expert trajectories: two coordination modes, A-first and B-first, for each identical initial condition. The split is by initial-condition pair:

| Split | Initial-condition pairs | Trajectories |
|---|---:|---:|
| Train | 200 | 400 |
| Validation | 25 | 50 |
| Test | 25 | 50 |

Paired modes from one initial condition cannot cross splits. Toy therefore also contains locally ambiguous supervision: exact/rounded-state grouping finds 250 ambiguous groups and 633 rows, with maximum joint-action spread 0.4032 m/s. Its success cannot be attributed to eliminating multimodality.

The important data difference is that the successful Toy checkpoint is trained on a uniform-state expert-requery dataset, not merely the nominal trajectory states. It contains 180,877 train and 22,595 validation transitions. Each source transition contributes a local perturbation/requery sample using independent position perturbations up to approximately 0.02 m and last-action/velocity perturbations up to approximately 0.2 m/s, with the corresponding expert mode/stage held fixed for the re-query. Training minibatches sample transitions uniformly. This gives Toy direct local coverage around the nominal manifold; no equivalent augmentation exists in the current Double-Bottleneck dataset.

Mean nominal source-episode length is 452.19 steps for Toy train and 451.90 for validation. The configuration permits longer training, but the frozen reference evaluated here is the 25,000-update checkpoint.

### Observation and action

Toy's observation has shape `[2,10]`, flattened to 20 dimensions. For controlled agent `i`, a row is

`[p_i(2), last_applied_velocity_i(2), goal_i-p_i(2), p_j-p_i(2), v_j-v_i(2)]`.

The joint action has shape `[2,2]`, flattened to 4 dimensions, and represents the two agents' commanded planar velocities. Agent order is fixed. Training-only mean and standard deviation normalize both condition and action; sampled actions are denormalized and radially clipped to the speed limit.

There is no explicit wall map, bottleneck indicator, coordination-mode label, future trajectory, or recurrent state. Geometry is visible only indirectly through absolute position and goal-relative quantities. The last applied velocity supplies one-step actuator context but not a hidden policy history.

### Flow training and sampling

Toy and Double-Bottleneck use the same official MACFlow primitives. The Toy vector field is an MLP with three 256-unit hidden layers. Conditional flow matching samples `x0 ~ N(0,I)` and `t ~ Uniform[0,1]`, forms the linear interpolation `x_t = (1-t)x0 + t x1`, and regresses the vector field to `x1-x0`. Optimization is Adam at `3e-4`.

Inference integrates from the Gaussian base sample with 10 Euler steps. It generates one complete joint action per physical environment step. In the ordinary Toy rollout, a fresh base sample is drawn at each environment timestep by folding the timestep into the PRNG key. No flow sample, action latent, smoothing filter, recurrent state, or coordination label persists across the episode.

### Closed-loop execution

The raw Stage-I Toy comparison has no hard-safety projection. On the authoritative 200-start wide evaluation, seed 0 reaches 75.0% success, with 18.0% wall collisions and 2.5% agent collisions; successful completion averages 25.34 s, approximately 507 control steps. Seed 1 reaches 79.5% success, with 20.5% wall collisions and no agent collisions. Thus Toy Stage-I is not perfectly safe, but it demonstrably sustains a useful closed-loop policy.

On the 12-rollout matched diagnostic subset used here, Toy obtains 12/12 success, zero collisions, and mean length 502.33 steps. Only 0.144% of visited states exceed Toy's calibrated expert-support threshold, and departures, when present, occur late (6/12 rollouts; mean first departure 396.8 steps).

The operational answer to “what makes Toy work” is therefore: the same per-step stochastic MACFlow mechanism as Double-Bottleneck, but on a shorter, simpler task with two agents and, critically, dense local expert-requery coverage around every nominal transition.

## Direct Toy versus Double-Bottleneck comparison

The losses below use the same fixed 8-draw × 256-state evaluation protocol, so they are directly comparable within this diagnosis.

| Property | Toy Give-Way | Double-Bottleneck |
|---|---:|---:|
| Agents | 2 | 4 |
| Observation dimension | 20 | 72 (`[4,18]`) |
| Joint-action dimension | 4 | 8 (`[4,2]`) |
| Train trajectories | 400 | 72 |
| Validation trajectories | 50 | 24 |
| Train transitions | 180,877 | 52,873 |
| Validation transitions | 22,595 | 17,629 |
| Unique train initial families | 200 | 9 |
| Mean train episode length | 452.19 | 734.35 |
| Coordination modes | 2 | 8 |
| Local state perturb/requery samples | 180,877 | 0 |
| Geometry | one bottleneck plus yielding bay | two gates plus central chamber |
| Long wall exposure | lower | substantially higher |
| Bottleneck-duration proxy on held-out expert data | 113.6 steps/trajectory in priority-passage stage 3 | 80.0 first-gate + 81.0 second-gate geometrically tagged steps/trajectory |
| Exact/rounded ambiguous groups | 250 | 84 |
| Max action spread in ambiguous group | 0.4032 m/s | 0.7801 m/s |
| Fixed train CFM loss | 0.21551 | 0.19664 |
| Fixed validation CFM loss | 0.21164 | 0.20147 |
| Raw closed-loop success | 75.0% on 200 starts, seed 0 | 0/12 |

Double-Bottleneck has fewer initial families and no local state augmentation despite a 3.6× larger observation, twice the joint-action dimension, about 1.6× longer episodes, two bottlenecks, and prolonged wall proximity. Average CFM loss is slightly *better* than Toy, demonstrating that this scalar transition-level metric does not measure closed-loop support or critical-state accuracy.

The bottleneck-duration row is only a within-dataset exposure proxy: Toy uses a discrete expert stage, while Double-Bottleneck uses geometric phase masks that may overlap. It should not be read as an exact cross-environment duration equality.

Machine-readable version: [`tables/toy_vs_double.csv`](tables/toy_vs_double.csv).

## Diagnostic A — K-step expert-state rollout divergence

### Method

For each of 24 held-out trajectories, starts were selected from six semantic phases: approach to the first bottleneck, waiting/yielding, first-bottleneck traversal, chamber traversal, second-bottleneck traversal, and final approach. Two MACFlow sampling seeds were evaluated per start, giving 288 continuations per horizon. The rollout feeds predicted states back to the policy for `K in {1,5,10,25,50,100}` and compares with the recorded expert continuation.

“Clear position divergence” is joint position RMSE above 0.08 m. “Off support” uses the calibrated 99th-percentile normalized-observation nearest-neighbor threshold, 0.03957 for Double-Bottleneck.

### Results

| K | N | Position RMSE (m) | Action RMSE (m/s) | First-action RMSE (m/s) | Min wall clearance (m) | Min pair clearance (m) | Mean goal-progress deviation (m) | Collision rate | Off-support rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 288 | 0.00068 | 0.01363 | 0.01363 | 0.05870 | 0.31743 | -0.00055 | 0.0% | 75.69% |
| 5 | 288 | 0.00572 | 0.02741 | 0.01363 | 0.05583 | 0.31383 | -0.00459 | 0.0% | 100.0% |
| 10 | 288 | 0.01650 | 0.04070 | 0.01363 | 0.05142 | 0.30935 | -0.01072 | 0.69% | 100.0% |
| 25 | 288 | 0.05744 | 0.06544 | 0.01363 | 0.01773 | 0.27025 | -0.03580 | 23.26% | 100.0% |
| 50 | 288 | 0.12969 | 0.08842 | 0.01363 | -0.06297 | 0.18951 | -0.10355 | 81.25% | 100.0% |
| 100 | 288 | 0.30021 | 0.12265 | 0.01363 | -0.12778 | 0.03078 | -0.62127 | 98.61% | 100.0% |

Across the continuations that eventually diverge/collide:

- first off-support step: mean 1.27, median 1, range 1–4;
- first position divergence above 0.08 m: mean 36.27, median 34, range 13–74;
- first collision: 284/288, mean step 38.08, median 36, range 10–96.

K=1 action RMSE by phase is 0.01821 (approach), 0.00941 (waiting), 0.01066 (first bottleneck), 0.01041 (chamber), 0.01144 (second bottleneck), and 0.02163 m/s (final approach). There are no K=1 collisions.

Clearances in the table are means of each continuation's minimum clearance; negative values denote penetration in the counterfactual continuation. Goal-progress deviation is predicted improvement in summed goal distance minus expert improvement, so negative values mean that the policy falls behind the expert.

Conclusion: the model's first action is normally locally plausible; the decisive failure is rapid support departure followed by compounding error. This matches Interpretation Case 2, not Case 1. The calibrated support test is stricter than the 0.08 m physical-divergence criterion, explaining why support departure is detected earlier than obvious geometric divergence.

Detailed table and plots: [`tables/diagnostic_a_divergence.csv`](tables/diagnostic_a_divergence.csv), [`plots/diagnostic_a_position_divergence.svg`](plots/diagnostic_a_position_divergence.svg), and [`plots/diagnostic_a_action_divergence.svg`](plots/diagnostic_a_action_divergence.svg).

## Diagnostic B — Teacher-forced local action quality

### Method

At held-out expert states, eight independent actions were sampled without feeding model states back. Metrics include joint RMSE, best-of-eight RMSE, direction cosine, speed error, wall-normal error, and goal-directed error. Phase subsets are geometric and may overlap. A mode-transition subset is defined by expert joint-action variation above the greater of that episode's 90th percentile and `1e-4`.

### Overall comparison

| Metric | Toy | Double-Bottleneck |
|---|---:|---:|
| Action MSE | 0.001028 | 0.000758 |
| Joint RMSE (m/s) | 0.01815 | 0.01928 |
| Best-of-8 RMSE (m/s) | 0.00635 | 0.01175 |
| Mean vector error (m/s) | 0.02159 | 0.02301 |
| Direction cosine | 0.93225 | 0.97039 |
| Speed error (m/s) | 0.01574 | 0.02013 |
| Expert wall-directed component (m/s) | 0.00282 | 0.03268 |
| Predicted wall-directed component (m/s) | 0.00311 | 0.03549 |
| Wall-normal absolute error (m/s) | 0.00091 | 0.00495 |
| Expert goal-directed component (m/s) | 0.08539 | 0.17796 |
| Predicted goal-directed component (m/s) | 0.08958 | 0.17687 |
| Goal-directed absolute error (m/s) | 0.01727 | 0.01421 |

Average Double-Bottleneck local quality is not grossly worse than Toy, but wall-normal error is 5.44× Toy's. This matters more in the two-gate geometry.

The goal-directed component is the velocity projection onto the current agent-to-goal unit direction. The wall-directed component uses the nearest collidable wall normal with positive direction toward the wall; the error rows report the absolute predicted-versus-expert component difference before aggregation.

### Double-Bottleneck critical subsets

| Subset | States | Joint RMSE | Best-of-8 RMSE | Direction cosine | Wall-normal error |
|---|---:|---:|---:|---:|---:|
| Approach | 5,173 | 0.02468 | 0.01513 | 0.9764 | 0.00657 |
| Waiting/yielding | 10,622 | 0.02025 | 0.01226 | 0.9879 | 0.00517 |
| First bottleneck | 1,920 | 0.00995 | 0.00580 | 0.9993 | — |
| Chamber | 4,053 | 0.01014 | 0.00587 | 0.9992 | — |
| Second bottleneck | 1,944 | 0.01476 | 0.00914 | 0.9965 | — |
| Final approach | 4,177 | 0.02435 | 0.01501 | 0.9906 | 0.00601 |
| Near-wall quartile | — | 0.02059 | 0.01260 | 0.9616 | 0.00523 |
| Mode-transition region | 1,042 | **0.06975** | **0.05275** | **0.7933** | **0.01718** |

All eight actual expert modes have similar overall RMSE, 0.0190–0.0196 m/s; no single mode is uniquely broken. The local weakness is concentrated at behavioral transitions and, to a lesser degree, approaches/final approaches and wall-normal control. Low average validation loss hides these rare but closed-loop-critical states.

Full statistics and plot: [`diagnostic_b_teacher_forced.json`](diagnostic_b_teacher_forced.json) and [`plots/diagnostic_b_phase_error.svg`](plots/diagnostic_b_phase_error.svg).

## Diagnostic C — offline coordination-mode consistency

### Classifier

This is analysis only. For each query state, the classifier finds five nearest normalized expert states separately in the LTR-first and RTL-first reference sets. For each direction it computes the minimum joint-action RMSE to the corresponding expert actions and labels the lower score. When the relative score margin is below 0.10, it carries forward the previous label. The resulting label is an approximate local behavioral similarity, not a ground-truth latent variable; stage changes can change the label even in a coherent rollout.

### Results

| Rollout family | Rollouts | Mean switches/episode | Switches/100 steps | Mean consistent-segment length |
|---|---:|---:|---:|---:|
| Double-Bottleneck, fresh noise | 12 | 0.750 | 1.882 | 25.58 |
| Double-Bottleneck, fixed noise | 12 | 0.667 | 1.643 | 33.62 |
| Toy, fresh noise | 12 | 27.33 | 5.38 | 19.13 |

All 12 Toy rollouts in this comparison succeed. Double-Bottleneck therefore does not exhibit an unusually high rate of classifier mode switches. Because every unprotected Double-Bottleneck rollout collides, collision/no-collision switch correlation cannot be estimated. The Toy result also shows why this offline label must not be interpreted as literal episode-level commitment.

Conclusion: the analysis provides no evidence that frequent LTR/RTL switching is the main failure mechanism.

Time series and summary: [`tables/diagnostic_c_mode_timeseries.csv`](tables/diagnostic_c_mode_timeseries.csv), [`diagnostic_c_mode_consistency.json`](diagnostic_c_mode_consistency.json), and [`plots/diagnostic_c_mode_switches.svg`](plots/diagnostic_c_mode_switches.svg).

## Diagnostic D — fresh-noise versus episode-fixed-noise

The canonical policy draws fresh Gaussian base noise each environment timestep. The fixed-noise behavior was implemented only in the temporary diagnostic runner and reused one base vector throughout an episode.

| Variant | Success | Wall collision | Agent collision | Mean steps | Action total variation | Goal progress |
|---|---:|---:|---:|---:|---:|---:|
| Fresh noise | 0/12 | 12/12 | 0/12 | 39.42 | 0.02233 | 2.177 |
| Episode-fixed noise | 0/12 | 11/12 | 1/12 | 43.00 | 0.01256 | 2.041 |

Fixed noise reduces action total variation by 43.7%, but does not produce a single success, does not materially delay failure, and introduces one agent collision. This supports the narrow claim that independent samples contribute to visible action jitter. It does **not** support temporal sampling inconsistency as the primary cause, nor does it justify fixed noise as a permanent method.

Summary and plot: [`diagnostic_d_noise_ablation.json`](diagnostic_d_noise_ablation.json) and [`plots/diagnostic_d_noise_outcomes.svg`](plots/diagnostic_d_noise_outcomes.svg).

## Diagnostic E — temporary explicit-mode oracle

### Method

A temporary in-memory diagnostic vector field appended a fixed one-hot first-direction label, LTR-first or RTL-first, to the 72D observation condition. It retained the official MACFlow flow-matching objective, optimizer, integration, dataset, and 8D joint action. It had 155,144 parameters versus 154,632 for the unconditioned diagnostic-sized model. The direction label does not resolve the four within-direction convoy-order modes, so the ablation tests only the stated LTR/RTL ambiguity.

The oracle was trained for 2,000 updates with batch size 256, without tuning or checkpoint persistence. Validation loss fell from 2.035 to 0.22465.

### Result

On 12 rollouts (3 initial states × 2 direction labels × 2 sampling seeds), it obtains 0/12 success and 12/12 wall collisions. Mean survival is 51.17 steps and action total variation is 0.02367.

Conclusion: removing the two-way first-direction ambiguity is not sufficient to restore closed-loop control. This result does not prove that every possible learned persistent latent is useless, but it removes the main empirical motivation for permanently adding a handcrafted direction label. In accordance with the mandatory cleanup rule, no oracle checkpoint or source code was retained. The aggregate JSON, training log, representative trajectory, method description, parameter counts, rollout design, and deleted-script hash make the result auditable, but the temporary implementation is intentionally not directly rerunnable from retained source. Exact temporary-run PRNG integers were not retained, which is a reproducibility limitation.

Full result: [`diagnostic_e_mode_oracle.json`](diagnostic_e_mode_oracle.json).

## Diagnostic F — limited hard-safety probe

Six matched rollouts were evaluated with and without the existing global hard projection. This was a diagnostic probe, not a safety-baseline or basin experiment.

| Variant | Success | Collision | Timeout | Mean steps |
|---|---:|---:|---:|---:|
| Raw MACFlow | 0/6 | 6/6 | 0/6 | 39.17 |
| Hard projection, no eta | 0/6 | 0/6 | 6/6 | 850.00 |

For projected rollouts:

- projection is active on 82.31% of timesteps;
- mean correction magnitude is 0.0572 m/s;
- mean correction/nominal-action norm ratio is 35.95% (up to 47.8% across rollouts);
- minimum wall clearance is 0.00513 m;
- mean per-rollout summed goal-distance improvement is 13.58 m, but no rollout completes;
- no legacy strict/global deadlock event is emitted; all terminate by timeout.

Hard safety can eliminate immediate collisions, but it is rewriting the policy on most steps and cannot recover liveness. The evidence favors case B: the off-manifold Flow field is fundamentally inadequate for the task, rather than broadly correct with small unsafe deviations.

Full result: [`diagnostic_f_safety_probe.json`](diagnostic_f_safety_probe.json).

## Distribution coverage

Nearest-neighbor distances use normalized observations and model-specific thresholds; raw Toy and Double-Bottleneck distances should not be compared directly because dimensions and normalization differ. Threshold-exceedance fractions are the meaningful comparison.

| System/variant | Expert held-out→train mean NN | 99% support threshold | Visited-state mean NN | Fraction outside support |
|---|---:|---:|---:|---:|
| Toy, fresh | 0.09020 | 0.13965 | 0.08465 | 0.144% |
| Double, fresh | 0.00269 | 0.03957 | 0.5144 | 98.305% |
| Double, fixed | 0.00269 | 0.03957 | 0.6510 | 98.19% |
| Double, mode oracle | 0.00269 | 0.03957 | 0.5302 | 98.58% |
| Double, hard safety | 0.00269 | 0.03957 | 0.6328 | 99.92% |

In ordinary Double-Bottleneck rollouts, first support departure averages 0.667 steps: 4/12 held-out initial states are already beyond the threshold relative to the small training split, and the rest leave by approximately the first feedback step. Toy remains within calibrated support for essentially its entire successful rollout.

The hard projection does not return the policy to expert support; it creates safe states on which the policy was not trained. That explains the collision-to-timeout conversion without success.

The centralized expert was not queried from these off-manifold states in this phase. Its current validation is as a fresh-start scheduled planner, not as an arbitrary mid-trajectory recovery oracle. Treating it as one without validation would confound the diagnosis. Establishing that local re-query contract is part of the recommendation, not an accomplished result.

Full data: [`distribution_coverage.json`](distribution_coverage.json) and [`plots/representative_rollouts.svg`](plots/representative_rollouts.svg).

## Attribution

Ranked from strongest to weakest current evidence:

| Rank | Candidate | Assessment | Evidence |
|---:|---|---|---|
| 1 | Insufficient dataset/recovery coverage | **Strongly supported** | 9 train initial families, no perturb/requery augmentation, 98.3% of fresh rollout states outside calibrated support versus 0.144% for successful Toy. |
| 2 | Covariate shift / compounding error | **Strongly supported** | K=1 is locally reasonable and collision-free; position/action error increases monotonically, with median physical divergence at step 34 and median collision at 36. |
| 3 | Insufficient local learning in critical states | **Supported** | Mode-transition RMSE 0.06975 versus 0.01928 overall; wall-normal error is 5.44× Toy; approach/final errors are elevated. Average validation loss hides this imbalance. |
| 4 | Geometry/horizon sensitivity | **Supported contributor** | Four agents, 72D condition, two gates, long wall exposure, and 734-step episodes amplify small wall-normal errors relative to Toy. This is a task property, not by itself a model defect. |
| 5 | Stochastic resampling inconsistency | **Weak contributor, not primary** | Fixed noise halves action variation but remains 0/12 and collision-prone. |
| 6 | LTR/RTL mode inconsistency | **Not supported as primary** | Few inferred switches; fixed noise barely changes them; explicit two-way mode oracle remains 0/12. |
| 7 | Need for a persistent latent | **Not established** | Neither the fixed-noise proxy nor the explicit-direction oracle gives a large causal improvement. A learned latent remains a future hypothesis only after coverage is repaired. |
| 8 | Need for horizon/trajectory-level modeling | **Premature** | Local coverage and critical-state imbalance provide simpler, directly evidenced explanations. The interpretation hierarchy says to repair/test these first. |
| 9 | MACFlow implementation bug | **No positive evidence** | Primitive-class parity, checkpoint reload/fingerprint tests, finite sampling, speed-bound tests, and 42 regressions pass. K=1 is reasonable. A hidden bug can never be logically excluded, but none was demonstrated. |

The headline attribution is: **nominal-only data plus sparse initial-family coverage is strongly implicated in the immediate feedback distribution shift; critical transition and wall-normal errors then compound in a geometry that is much less forgiving than Toy.** This mechanism is the best-supported current hypothesis, not yet a causal result: no augmentation/requery retraining intervention was run, and 4/12 held-out starts are already beyond the small training split's calibrated support threshold.

## Recommendation: minimal next permanent change

Keep the canonical policy and execution semantics unchanged: joint 8D `p(u|x)`, no explicit mode input, no handcrafted LTR/RTL indicator, no oracle future, fresh per-step base noise, and no horizon model.

The next phase should:

1. Validate that the centralized expert can accept perturbed intermediate Double-Bottleneck states and return dynamically and geometrically valid local recovery actions/continuations without silently imposing one universal passage order.
2. Build a **small** Toy-style perturb-and-requery augmentation, grouped by original initial family and mode to prevent split leakage. Start with mode-transition, approach/final-approach, and wall-near states; include modest position and last-applied-action perturbations whose magnitudes are justified by the measured early rollout errors.
3. Audit the augmented pilot for collisions, duplicates, mode balance, and train/validation grouping.
4. Retrain the same unconditioned architecture at pilot scale, with enough updates to assess critical-subset teacher-forced metrics rather than only mean CFM loss.
5. Repeat Diagnostics A, B, coverage, and the limited raw/safety rollouts. Consider persistent-latent or horizon models only if K=1/critical-state quality and support coverage improve while long closed-loop behavior still fails.

This is the smallest change that is both Toy-aligned and directly supported by the evidence. It has not been implemented here.

## Cleanup proof

### Temporary implementation created and deleted

- `/tmp/double_bottleneck_macflow_diag.5blKPh/run_diagnostics.py`
- its generated `__pycache__` bytecode
- the temporary directory itself
- the in-memory explicit-mode vector-field class and episode-fixed-noise rollout hook contained only in that script

No explicit-mode checkpoint was written. A search of canonical `double_bottleneck`, `flowbc`, `toy_giveway`, and `shared_control` sources finds no `OracleModeAgent`, `episode_fixed_noise`, `mode_input`, explicit-mode input, or persistent-noise hook.

### Diagnostic results retained

- this Markdown report;
- seven numerical JSON summaries plus metadata and Toy/direct-comparison JSON;
- three CSV tables;
- six SVG plots;
- four representative `.npz` trajectory artifacts.

These are analysis artifacts only and are not imported by the canonical policy or rollout code.

### Canonical implementation proof

No canonical source file was edited during this diagnostic phase. Current source hashes are:

| File | SHA-256 |
|---|---|
| `double_bottleneck/flowbc_4a_agent.py` | `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8` |
| `double_bottleneck/flowbc_4a_dataset.py` | `e503da22beb4b7308ecfb6c45207bbe1836424a16a156b1d5c54d255b3deb2f3` |
| `double_bottleneck/train_flowbc_4a.py` | `c1c2b6d1138999a516cc0eb0092db787f91be849c53c59e6364c379840859367` |
| `double_bottleneck/evaluate_flowbc_4a.py` | `97e707dd13803ffa7ad749cdcc51b3ae19d2e9ffd94112d62f4fbfe3bd6794a8` |
| authoritative Toy `flowbc/giveway_flowbc_agent.py` | `8b94fb64b7448cdcfb3c7bc882d10266427339a22a155defa6d71aee449f5adc` |
| authoritative Toy `flowbc/giveway_dataset.py` | `a81b887ba41cdb1e10bfb9fb635429f7fe0552fab409bb58249ce88d35e277af` |
| authoritative Toy `single_integrator/evaluate.py` | `294d233ff6f7a3137730100fbfbaf0a5e2902d31311642cb98a84515821de4ae` |

The Double-Bottleneck agent/dataset/train hashes match the checkpoint's recorded source provenance. The Toy Stage-I source used for the parity audit is byte-identical to the working reference inspected before diagnostics.

The repository was already dirty and the scenario trees were already untracked at phase start. Initial and final `git status --short` show the same top-level pre-existing entries (`README.md`, `scripts/test.sh`, `diagnostics/`, `double_bottleneck/`, `toy_giveway/`, and unrelated existing files); therefore ordinary `git diff` cannot provide line-level history for those untracked trees. The hashes, source search, temporary-directory deletion, and regression suite provide the cleanup evidence without overwriting or staging user work.

### Regression result

Command:

```text
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m unittest -v \
  double_bottleneck.tests.test_flowbc_4a \
  double_bottleneck.tests.test_expert \
  double_bottleneck.tests.test_expert_dataset \
  tests.test_double_bottleneck \
  toy_giveway.tests.test_smoke
```

Result: **42/42 tests passed** in 6.52 s. This covers official MACFlow primitive parity, exact checkpoint reload and fingerprinting, real dataset validation, all eight expert hypotheses, Double-Bottleneck dynamics/safety/termination, and Toy namespace/rollout regressions. JAX printed a CUDA-plugin initialization warning despite the CPU-only setting, then selected CPU and completed successfully; no GPU resource was used.

## Final status

**BLOCKED for formal safety-baseline or eta/basin evaluation.** The proper joint 4-agent MACFlow exists and has good average transition loss, but it is not a valid closed-loop baseline. The next scientific gate is the small Toy-style recovery-coverage pilot described above—not explicit mode labels, fixed-noise deployment, a horizon redesign, basin search, or `G_phi`.
