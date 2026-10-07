# Toy Give-Way → Double-Bottleneck MACFlow transfer audit

Date: 2026-09-23  
Scope: forensic reconstruction plus small causal transfer tests only. No safety-baseline, eta/basin, or `G_phi` experiment was run.

## Executive conclusion

Toy Give-Way did encounter the relevant failure pattern. Its early Stage-I joint Flow-BC had low one-step error and exact dynamics consistency, yet raw closed-loop evaluation was `0/25`, with wall contact in `25/25`. Shortening the scene alone did not repair it. The retained Toy experiment with the cleanest causal contrast used the same 500 nominal expert demonstrations and the same Stage-I objective, but replaced nominal transition training rows with independently perturbed, expert-requeried rows. At every retained checkpoint in that experiment, nominal-only training remained `0/25`, while the requery data achieved `25/25`.

The current successful Toy recipe therefore is not merely “the same MACFlow architecture.” It combines:

- a reactive expert that can be queried at perturbed states while retaining the source stage and coordination mode;
- one perturbed/requeried row for every train/validation source transition;
- transition-uniform training on those rows, train-only standardization, and 25,000 updates;
- checkpoint acceptance by closed-loop behavior rather than by flow-matching loss alone.

A matched Double-Bottleneck pilot supports a partial transfer of that explanation. Merely training nominal data for 25,000 updates improved nominal action RMSE from `0.01926` to `0.00806` but stayed at `0/12` success. A Toy-style perturb/requery union, trained for the same 25,000 updates with the same official Stage-I model, reached `4/12`, reduced fixed-coordinate expert-support OOD from `98.39%` to `80.53%`, and reduced K=100 continuation collision rate from `98.61%` (canonical 2k) / `68.75%` (nominal 25k) to `6.25%`.

This is a credible transferable mechanism, not a solved baseline. Seven wall collisions and one agent collision remain in 12 rollouts. The canonical implementation was therefore not changed.

## 1. Toy historical reconstruction

### Evidence limits

The Toy repository has only three reachable commits:

| Commit | Timestamp | Meaning |
|---|---:|---|
| `cc8579e` | 2026-09-21 15:02 | frozen shared baseline |
| `6848ab9` | 2026-09-21 15:05 | later Direction-A work |
| `aba7d6d` | 2026-09-21 15:07 | later risk work |

The local reflog exposes one additional unreachable post-freeze commit, `d578cb3` at 15:09. It does not expose the September 5–9 development sequence. Consequently, the pre-freeze chronology below is reconstructed from retained scripts, checkpoint metadata, JSON results, reports, file timestamps, and content hashes. It is artifact-backed but not commit-authenticated. Where a raw result referenced by a report is missing, that is marked explicitly.

### Chronology

| Toy version / retained date | Dataset and setup | Closed-loop behavior | Change | Result after change | Evidence strength |
|---|---|---|---|---|---|
| Sep 5–6 original VMAS Stage-I | Original long Give-Way scene; early joint Stage-I policy | Archived report states 20 randomized starts: `2/20` success, `18/20` deadlock, `9/20` agent collision, all 20 with a wall flag | Early baseline only | Demonstrated rollout instability | The contemporaneous report remains; the raw summary it cites was not retained |
| Sep 7 direct single-integrator audit | 100k checkpoint `single_integrator/runs/flowbc_seed0/ckpt_0100000.pkl`; validation flow loss fell from `1.02271` to `0.0075377` | `0/25` success, `25/25` wall, `21/25` agent under the old non-first-event accounting; later first-event replay places first wall contact at steps 17–58 | Replaced VMAS ambiguity with exact SI replay and verified action/state integration | Dynamics residual `8.85e-16`, tracking error `0`; failure remained | Raw `identity_comparison_verified.json` plus checkpoint metadata |
| Sep 7 wall-scale and recentering audits | Same checkpoint and expert set | Wall scales 1.0/1.2/1.5/2.0: success always 0%; wall rate 100/84/72/68%. One-step RMSE `0.004756`, model mean `|v_y|=0.002853` versus expert `0.000445` | Enlarged wall scale; perturbed expert states by 1–2.5 cm and tested return | Expert recovered `60/60`, zero walls; Flow recovered `9/180`, with `154/180` walls | Raw audit outputs retained; strong evidence for feedback sensitivity and lateral bias |
| Sep 7–8 early recovery / ablation audits | Original geometry; several data/preprocessing probes | A preliminary recovery-data comparison remained `0/25`, although median failure was delayed to about 10.2 s | Previous-velocity intervention, masking, weighting, normalization checks | Strong velocity carry-over (`v0y→u0y` gain `0.992`); later audit retracted a supposed loader bug: original sampling was uniform and raw+normalization still failed | Retained intervention and `unweighted_v5` audit artifacts |
| Sep 8 clean short-scene experiment | Corridor half-length reduced from 2.5 m to 1.3 m; regenerated 250 initial pairs × two modes = 500 clean demonstrations | Nominal transitions with normalization: `0/25` at 25k, 50k, 75k and 100k | For each train/val source row, perturb position/last velocity, preserve source mode/stage, requery reactive expert; no stage weighting | Requery data: `25/25` at every retained checkpoint | Strongest controlled historical evidence; same scene, demonstrations, architecture and checkpoints |
| Sep 8–21 frozen successful Toy baseline | Short geometry; 500 original trajectories; independent perturbed/requery training rows; 25k checkpoints, two seeds | Same-support reproduction in this audit: each seed `25/25`; broader frozen evaluation: raw Flow `309/400` | Freeze successful Stage-I baseline | Stable enough for subsequent Toy research | Frozen checkpoints, planning results, and current reproduction |

The historical claim supported by the evidence is **teacher-forced/local-good but feedback-closed-loop-bad**, not successful long-horizon open-loop prediction. No retained experiment establishes the latter.

The precise historical evidence locations are below. `A` denotes `/home/zhihan/research/_archive/02_C1_Toy_GiveWay_legacy_20260909`; `R` denotes `/home/zhihan/research/02_C1_Toy_GiveWay`.

| Stage | Primary evidence | Verification boundary |
|---|---|---|
| Original VMAS | `A/flowbc/README.md`, retained 20k weight `A/flowbc/checkpoints/ckpt_0020000.pkl` | The README's referenced 100k summary and weight are absent; its 20-rollout numbers cannot be recomputed |
| Direct SI identity | `A/single_integrator/runs/flowbc_seed0/{metrics.jsonl,benchmark.json,ckpt_0100000.pkl}`, `A/single_integrator/results/identity_comparison_verified.json`, `A/single_integrator/RESULTS_ZH.md` | Training and failure are raw-verifiable; the cleaned original dataset is represented only by digest `992e5d…` and metadata |
| First-event stop | `A/single_integrator/results/collision_stop_window_v2/{summary.json,regression_check.json,config.json}` | Raw JSON; explains why agent-collision count becomes zero when wall is the first terminal event |
| Wall/recentering | `A/single_integrator/results/wall_scale_audit/{comparison.json,policy_diagnostics.json,data_geometry_check.json,REPORT_ZH.md}`, `A/single_integrator/results/recentering_audit/{summary.json,closed_loop_episodes.json,REPORT_ZH.md}` | Raw JSON supports the numerical comparisons; “covariate shift” is the evidence-backed interpretation |
| Early recovery/shortcut | `A/single_integrator/results/recovery_comparison/`, `A/single_integrator/results/collision_diagnosis_v4/`, `A/single_integrator/results/unweighted_v5/` | Raw comparisons retained; `unweighted_v5` explicitly retracts the earlier loader-bug interpretation |
| Clean short scene | `A/single_integrator/results/short_baseline_v1/{REPORT_ZH.md,frozen_selection/all_validation_results.json,augmentation_audit.json}`, plus its `validation_raw/`, `validation_augmented/`, `paired_nominal_test/`, `paired_wide_*`, and `replication_seed1/` summaries | Raw JSON supports the same-scene nominal-versus-requery contrast |
| Frozen broad result | `A/single_integrator/results/short_baseline_v1/wide_scale_200/`, and current `R/baseline_309_314/planning/` | Per-rollout summaries and current/archive content hashes agree; `RESULT_ZH.md` SHA is `d169aced…` |
| Git freeze | `R/docs/assets_manifest.md` and commits listed above | Git-native evidence begins at the freeze, not at the earlier experimental stages |

### What changed before Toy succeeded

The causal short-scene comparison rules out two common stories:

1. Shorter geometry alone was insufficient: nominal training was still `0/25` through 100k updates.
2. Lower validation loss alone was insufficient: later nominal checkpoints had lower flow loss but still failed.

The decisive retained change was local state coverage through perturb-and-requery. It was not DAgger, not a trajectory model, and not explicit mode conditioning. Toy did not store recovery trajectories. It stored independent supervised state/action rows generated by querying its reactive finite-state expert at perturbed source states.

## 2. Exact successful Toy recipe

### Expert and data

- Expert: deterministic reactive six-stage finite-state controller with `A_FIRST` and `B_FIRST` hypotheses. The yielding agent enters a bay, waits, and proceeds after the opposing agent clears.
- Original data: 250 initial-condition pairs, two modes per pair, 500 successful collision-free episodes, 226,087 transitions.
- Initial state: nominal positions `A=(-0.85,0)`, `B=(0.85,0)`; each agent's x-coordinate is randomized by `U[-0.03,0.03]`, y by `U[-0.002,0.002]`.
- Split is by initial-condition pair, so the two modes of a pair never cross splits:
  - train pairs 0–199: 400 episodes, 180,877 transitions;
  - validation pairs 200–224: 50 episodes, 22,595 transitions;
  - test pairs 225–249: 50 episodes, 22,615 transitions.
- Train episode length: mean `452.1925`, min 448, max 457 transitions. Validation: mean `451.9`, min 448, max 456. Test: mean `452.3`, min 449, max 456.
- Original dataset digest: `da2bdefdca9085d0bf83d036f5ab23a0279e935a64daff18e9daadff24880cd6`.

The successful training corpus is the one-to-one perturbed/requery corpus:

- one independent row per original train or validation transition; test is excluded;
- position perturbation independently `U[-0.02,0.02]` m per coordinate;
- stored last velocity perturbation independently `U[-0.2,0.2]` m/s per coordinate, followed by the expert's normal bound;
- source mode and source finite-state stage are retained, then the reactive expert is queried once at the perturbed state;
- predecessor/current/next swept-safety checks reject invalid samples; 1,472 attempts were rejected and resampled;
- 450 files, 203,472 independent rows total;
- digest `5f5bdee6297cc7cad6d81bf3fb2d7724280b088529cf139394e7810a99bd396f`.

There is no evidence of trajectory-level recovery collection, iterative DAgger, repeated replanning along learned rollouts, critical-stage weighting, or a mixture of nominal and requery rows in the frozen recipe. Each initial-condition pair is demonstrated twice from the same initial observation, once per global mode, so identical initial states can have different actions; nearby later states can as well. Mode is metadata for expert generation, not policy input.

### Sampling and preprocessing

- Samples are drawn uniformly with replacement over transitions.
- Batch size 256.
- No critical-state or bottleneck oversampling.
- Observation per agent is 10D:
  `p_i(2), v_i^last(2), goal_i-p_i(2), p_j-p_i(2), v_j-v_i(2)`.
- Joint observation is 20D in agent order `(A,B)`; joint action is the two agents' 2D executed velocities, flattened to 4D.
- Fixed wall geometry is not explicitly observed. There is no hidden recurrence or history beyond last applied velocity.
- Observation and action coordinates are standardized using train-corpus mean and standard deviation only. Standard-deviation floor is `0.01`.
- Generated action components are denormalized, component-clipped to `[-1,1]`, then each agent is radially projected to the `0.5 m/s` speed limit.

### MACFlow model and training

- Official `ActorVectorField`, `ModuleDict`, and `TrainState` primitives.
- Conditional flow matching: `x0 ~ N(0,I)`, `t ~ U[0,1]`, `x_t=(1-t)x0+t x1`, target vector `x1-x0`, coordinate-mean squared error.
- Actor input is 20 observation + 4 noisy action + 1 time = 25 dimensions.
- MLP: `25 → 256 → 256 → 256 → 4`, GELU, no layer normalization; 139,268 trainable parameters.
- Adam, learning rate `3e-4`, batch 256.
- No critic, Q-guidance, or distillation.
- Seed-0 run continued to 100k updates; seed-1 retained run used 25k. Frozen checkpoints are both step 25,000:
  - seed 0 SHA-256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`;
  - seed 1 SHA-256 `3f58b12266d19358a8ef5673e3e4018c1248c526de7d71877218ea176bd53383`.
- Recomputed validation flow losses are `0.22286583` and `0.21643511`. Seed 0 later reached lower logged flow loss (down to `0.12252` at 100k), so 25k was not selected as the global minimum validation-loss checkpoint. Retained notes describe the accepted point as nominal-validation/closed-loop selected; the exact selection sweep is absent, so a stronger checkpoint-selection claim is not verifiable.

### Closed-loop execution

- Ten Euler flow-integration steps generate one joint action per environment timestep.
- Fresh independent Gaussian base noise is sampled at every environment timestep.
- No persistent noise, action smoothing, temporal filter, explicit mode input, passage-order state, or future oracle information.
- Control/environment timestep is 0.05 s; horizon is 850 steps.
- Raw Stage-I success is measured before safety projection. The frozen broad-distribution result is raw Flow `309/400`; Flow+CBF is separately reported as `314/400` and is not the source of raw-policy success.

## 3. Toy vs Double-Bottleneck delta audit

| Class | Difference | Plausible for current failure? | Toy historical evidence | Transfer without changing `p(u|x)`? |
|---|---|---:|---|---:|
| A structural | 2 agents / 4D action versus 4 agents / 8D action | Yes: more coupled errors and collision pairs | No causal Toy ablation; structural fact only | No task-preserving removal; same joint formulation retained |
| A structural | One conflict/bay versus two bottlenecks plus a chamber and long sequential coordination | Yes: longer time to compound error and more wall exposure | Shortening Toy geometry alone did not solve nominal policy, but made a covered policy feasible | Geometry stays fixed; coverage can be transferred |
| A structural | Toy has 2 passage modes; Double has 8 valid global orders | Yes, especially at waiting/switch states | Explicit-mode oracle previously stayed `0/12`, so mode ambiguity is not the primary current blocker | Keep all modes in unconditioned data |
| B coverage | Toy original: 500 trajectories / 250 starts; Double: 96 trajectories / 12 families (train 72/9, val 24/3) | Yes | Broad initial support matters to frozen Toy performance; exact causal contribution not isolated | Yes, through more expert families if needed |
| B coverage | Toy training rows are all locally perturbed/requeried; Double canonical rows are nominal trajectory transitions only | Very high | Direct Toy 0/25 versus 25/25 causal contrast | Yes; only supervised data change |
| B coverage | Toy reactive expert can label off-nominal states; Double scheduled expert has no arbitrary-state recovery API | High, and it blocks faithful data generation | Toy's stage-preserving reactive query is the successful mechanism | Yes, but requires validating an expert-side recovery query, not changing policy |
| B coverage | Current Double failures include a held-out initial-velocity family and sparse critical regions | High | Toy randomizes initial positions but does not isolate velocity-family transfer | Yes |
| C training | Toy frozen point 25k updates; canonical Double only 2k | Moderate | Toy uses 25k, but Toy nominal 100k still failed | Yes; experiment shows duration is necessary for augmented data but insufficient alone |
| C training | Toy 139,268 parameters for 20D→4D; Double 154,632 for 72D→8D | Possible capacity pressure | No retained causal capacity sweep | Technically yes, but evidence does not justify it yet |
| C training | Toy accepts checkpoints using closed-loop behavior in addition to loss; canonical Double selected best validation loss | High as a selection issue, not a standalone remedy | Toy 25k is not its minimum-loss point | Yes |
| D observation | Per-agent construction has the same relative-state philosophy, but Double is 18D and neither policy sees explicit walls | Yes: Double has much more varied wall-relative geometry | Toy wall errors were central historically; no evidence that adding wall features made Toy succeed | Existing fixed-geometry `p(u|x)` can first be tested with coverage; no observation change yet |
| D preprocessing | Both use train-only coordinate standardization and a 0.01 scale floor | Unlikely to explain delta | Toy normalization alone still gave 0/25 nominal | Already matched |
| E rollout | Both use ten Euler steps, fresh per-step noise, component clip then radial 0.5 m/s limit, dt 0.05 | Unlikely | Exact current match; prior fixed-noise diagnostic stayed `0/12` | Already matched |
| E rollout | Double requires substantially longer, multi-phase closed-loop coordination | Yes | Toy same-support mean 502.56 steps; Double successful expert/policy trajectories approach the 850-step horizon | Cannot remove without changing task; improve state coverage |

The largest experimentally grounded delta is not model provenance. It is that Toy's successful model was trained on perturbed/requeried local states while Double's canonical model was trained only on nominal trajectory transitions.

## 4. Ranked transferable mechanisms

1. **Local perturb-and-requery coverage — strong evidence.** It is the only retained Toy change with a same-scene `0/25 → 25/25` contrast, and the Double pilot gives matched improvement in survival, K-step collision, OOD, and success.
2. **Enough optimization after expanding the corpus — strong interaction evidence.** Double recovery-union at 2k greatly improves K-step stability but remains `0/12`; at 25k it reaches `4/12`. Nominal 25k remains `0/12`, so duration is not sufficient by itself.
3. **Broader initial-condition/family coverage — medium evidence.** Toy has 250 starts versus Double's 12 families. All four union-policy trials in the held-out weakly-asymmetric velocity-probe family failed, including two failures at steps 48–49. The exact contribution has not yet been causally isolated.
4. **Rollout-based checkpoint acceptance — medium evidence.** Toy's retained checkpoint is not its lowest-loss checkpoint, while nominal Double 25k demonstrates better local losses without success. This supports evaluating checkpoints in closed loop, not a claim that checkpoint selection alone solves coverage.
5. **Normalization — required practice, weak explanatory evidence.** It is used in successful Toy and all controlled Double arms, but Toy history shows normalization alone did not fix nominal data.
6. **Critical-state balancing — not Toy-supported.** Toy sampled transitions uniformly and did not oversample stages. It may be a future targeted test, but cannot be described as recovered Toy practice.
7. **Larger model / architecture change — currently unsupported.** No historical or present controlled capacity result justifies changing the Stage-I model.

Toy did not succeed merely by collecting more nominal transitions: the clean causal comparison retained the same 500 nominal demonstrations and changed the supervised state support. Nor did Toy use recovery trajectories, DAgger, persistent noise, or explicit coordination labels.

## 5. Controlled transfer experiments

### Intervention construction

The Double centralized expert is an absolute-time scheduled planner and cannot safely be restarted from arbitrary intermediate states. A direct fake reset at exact, unperturbed intermediate states succeeded only `4/32`; 28 runs collided. It therefore could not be used as a recovery oracle.

For this audit only, an isolated reference-tube requery controller was used on real expert intermediate states:

`u_query = radial_bound(u_ref[t] + 3 (p_ref[t] - p_perturbed), 0.5)`.

It tracks the source expert continuation and then applies proportional goal convergence. The complete 8-way expert hypothesis remains inside the expert/data generator; it is never passed to MACFlow. Position and last-velocity perturbations exactly match Toy (`±0.02 m`, `±0.2 m/s`). One independent requery row was generated per nominal row.

Oracle validation covered `3 regimes × 8 modes × 6 phases × 2 seeds = 288` actual-environment recoveries: `288/288` success, zero collisions/deadlocks/timeouts, all 288 preserved their source mode signature, maximum 648 steps, minimum wall clearance `0.03337 m`, minimum pair clearance `0.08648 m`. Full-horizon vector validation also succeeded for all 52,873 train and 17,629 validation queries. This validates the labels for the pilot, but the tube tracker remains an experimental expert-side mechanism and was removed after generating the retained data.

Data sizes:

- nominal train: 72 episodes, 9 families, 52,873 transitions;
- nominal validation: 24 episodes, 3 families, 17,629 transitions;
- recovery rows: 52,873 train and 17,629 validation;
- union: 105,746 train and 35,258 validation transitions.

All arms use seed 0, batch 256, Adam `3e-4`, official Stage-I `ActorVectorField` with three 256-unit hidden layers, 10 Euler steps, identical unconditioned `p(u|x)` inference, and CPU only. No safety projection was used.

The five controlled arms are:

- canonical nominal 2k: existing checkpoint;
- nominal 25k: duration-only control;
- recovery union 2k: coverage-only at the original update budget;
- recovery union 25k: coverage plus Toy-scale update budget;
- recovery-only 25k: exact analogue of Toy's replacement corpus, included as an ablation.

Evaluation uses 12 raw rollouts (`3` held-out families × seeds `0..3`), all 17,629 nominal validation states with four samples each, and 144 K-step starts per horizon (`24` held-out expert trajectories × six semantic phase anchors). Support is nearest-neighbor RMS distance in canonical nominal-2k normalized 72D coordinates, with threshold `0.03957`, the q99 held-out-nominal-to-train-nominal distance.

### Results

| Variant | Updates | Train rows | Flow val loss: nominal / recovery | Success | Wall / agent / timeout | Median steps | Fixed-support OOD | Teacher RMSE | Transition RMSE | Wall-directed error | K25 pos / collision | K100 pos / collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical nominal | 2k | 52,873 | `0.1961 / —` | 0/12 | 12 / 0 / 0 | 38.5 | 98.39% | .01926 | .07007 | .00495 | .03103 / 27.78% | .05533 / 98.61% |
| Nominal | 25k | 52,873 | `0.0530 / 7.3692` | 0/12 | 12 / 0 / 0 | 70.5 | 98.96% | .00806 | .05351 | .00170 | .02861 / 11.81% | .09372 / 68.75% |
| Recovery union | 2k | 105,746 | `0.2997 / 0.5952` | 0/12 | 11 / 1 / 0 | 349.0 | 99.79% | .03776 | .07750 | .01066 | .01929 / 1.39% | .05119 / 16.67% |
| **Recovery union** | **25k** | **105,746** | **`0.1292 / 0.2633`** | **4/12** | **7 / 1 / 0** | **734.5** | **80.53%** | **.01838** | **.05779** | **.00421** | **.01355 / 0%** | **.03526 / 6.25%** |
| Recovery only | 25k | 52,873 | `0.1450 / 0.2301` | 4/12 | 6 / 2 / 0 | 732.0 | 91.39% | .02371 | .06297 | .00516 | .01302 / 0% | .03350 / 5.56% |

“Transition RMSE” is the previously defined mode-transition subset. Minimum wall clearance was negative for every arm because every arm still had at least one collision: canonical `-0.01658`, nominal-25k `-0.01672`, union-2k `-0.01413`, union-25k `-0.01078`, and recovery-only `-0.01317` m.

Median step-aligned 0.08 m divergence was 11 (canonical), 10.5 (nominal-25k), 9 (union-2k), 8 (union-25k), and 8 (recovery-only). Earlier divergence does not contradict improved survival: the covered policies can choose harmless timing/action deviations from a single aligned demonstration and still remain safe. The canonical median of 11 is not the earlier diagnostic's median 34, which came from 288 continuations started at six expert phases with two sampling seeds. The metric was held fixed across all five transfer arms here; it is suitable for reporting the intervention comparison but is a poor selector, and the two sampling populations must not be merged.

The recovery-only model's own-calibrated OOD is `1.29%`, but that number is not comparable: its recovery-only q99 radius expands to `0.293`. In the fixed canonical coordinate system, it is still `91.39%` OOD and its recovery states are 100% outside the nominal threshold. The fixed-coordinate figures in the table prevent this calibration artifact.

The 25k union model's four successes were one clearly asymmetric seed and three near-symmetric seeds. All four trials of the held-out weakly-asymmetric initial-velocity family failed. Of its eight failures, two occurred at steps 48–49 and six at steps 317–778. At K=100 it had zero collisions from first-approach, first-bottleneck, chamber, waiting, and second-bottleneck anchors, but 37.5% from final-goal anchors. Waiting/yielding remained the largest deviation (`0.14467 m` positional and `0.12922` action RMSE), even when it did not collide inside 100 steps.

Figures:

- [Closed-loop outcomes](plots/closed_loop_outcomes.svg)
- [Survival and fixed-support OOD](plots/survival_and_support.svg)
- [K-step collision rate](plots/k_step_collision.svg)

Machine-readable results are in [controlled_comparison.csv](tables/controlled_comparison.csv), [phase_k100.csv](tables/phase_k100.csv), [evaluation_comparison_v2.json](evaluation_comparison_v2.json), and [common_support_comparison.json](common_support_comparison.json).

### Negative results

- More nominal training is not enough: nominal 25k remains `0/12` despite the best nominal teacher RMSE and wall-directed error.
- Coverage without adequate optimization is not enough: union 2k survives much longer and has far fewer K-step collisions but remains `0/12`.
- Replacement is not better than union: recovery-only and union both reach `4/12`; replacement has more agent collisions, worse nominal teacher metrics, and a misleadingly broad self-calibrated support threshold.
- Local requery coverage does not fully solve the task: union 25k still has eight failures.
- Earlier diagnostics already showed episode-fixed noise and explicit LTR/RTL oracle conditioning both at `0/12`; nothing here reverses that result.

## 6. Closed-loop comparison

### Canonical Double versus accepted pilot

| Endpoint | Canonical nominal 2k | Recovery-union 25k | Change |
|---|---:|---:|---:|
| Success | 0/12 | 4/12 | +33.3 percentage points |
| Wall collision | 12/12 | 7/12 | −41.7 points |
| Agent collision | 0/12 | 1/12 | +8.3 points |
| Median survival | 38.5 steps | 734.5 steps | +696 steps |
| Fixed-support OOD | 98.39% | 80.53% | −17.86 points |
| K=25 collision | 27.78% | 0% | −27.78 points |
| K=100 collision | 98.61% | 6.25% | −92.36 points |
| Teacher action RMSE | .01926 | .01838 | nearly unchanged |

### Duration-matched causal contrast

Against nominal 25k, union 25k has **worse** nominal flow validation loss (`.1292` versus `.0530`), teacher RMSE (`.01838` versus `.00806`), transition RMSE, and wall-directed error, yet changes success from `0/12` to `4/12`, median survival from 70.5 to 734.5 steps, and K=100 collision from 68.75% to 6.25%. This is direct evidence that average nominal-state local metrics were not measuring the feedback robustness added by the requery corpus.

## 7. Toy reproduction and causal interpretation

### Lightweight frozen-Toy reproduction

Using seed-0 checkpoint SHA `8c17b5…f32`, the 25 untouched test pairs (225–249), rollout seed 42, and raw Flow only:

- success `25/25`; zero wall/agent collision, deadlock, or timeout;
- mean episode length `502.56` steps;
- same support definition adapted to Toy's normalized 20D observation: q99 threshold `0.13965`, mean rollout-to-training distance `0.08367`, outside fraction `0.127%`;
- median step-aligned deviation beyond 0.08 m from either expert mode: step 188. This measure can label harmless timing differences as divergence and did not predict failure;
- teacher-forced four-sample overall action RMSE `0.01820`, best-of-four RMSE `0.00851`, 83 stage-transition states RMSE `0.03869`, wall-directed absolute error `0.000919`.

Thus Toy does not remain successful because each sampled action is almost deterministic or because it never deviates in time from the demonstration. It remains inside a dense local training tube and corrects deviations sufficiently to finish.

### What the present experiments establish

They support the following causal statement:

> Under the same official unconditioned Stage-I MACFlow class and inference semantics, replacing/augmenting nominal transition support with validated local expert queries produces a large improvement in Double-Bottleneck feedback robustness; adequate training on that expanded corpus yields nonzero closed-loop success.

The experiments do **not** establish that:

- perturb/requery alone is sufficient for a reliable 4-agent baseline;
- the temporary tube tracker is the final centralized expert interface;
- 8 modes require an explicit latent or label;
- average validation loss predicts closed-loop success;
- architecture capacity is already proven insufficient;
- the policy is ready for safety, eta/basin, or `G_phi` work.

The intervention changes both local state support and the expert action labels/normalization induced by that support; those are a single “recovery-coverage corpus” treatment in this audit. A finer causal decomposition was not attempted.

## 8. Minimal recommended next step

Do not change MACFlow architecture and do not promote the pilot checkpoint as canonical. The minimum permanent next step, after review, is:

1. implement a Double-Bottleneck-specific, isolated local recovery-query interface on the expert side, with explicit tests that its hypothesis/schedule context never enters policy observations;
2. validate it at the same oracle gate used here;
3. generate a **small union corpus**, retaining nominal rows, but add targeted coverage for the held-out initial-velocity family, waiting/mode-transition states, and final-goal wall-near states;
4. train the unchanged Stage-I model for a Toy-comparable 25k budget and select only after matched closed-loop evaluation.

Use union rather than recovery-only: it preserves a meaningful nominal support calibration, has better nominal local metrics, and caused fewer agent-collision failures in this pilot. Predeclare a closed-loop gate; do not keep increasing data indefinitely if a targeted second shell fails to reduce fixed-coordinate OOD and the residual critical-state failures. Only that negative result would justify revisiting representation or capacity.

## 9. Cleanup, provenance, and repository status

### Retained scientific artifacts

- `toy_reproduction.json`: frozen Toy reproduction and source/checkpoint hashes.
- `dataset_summary.json`, `oracle_gate.json`: exact intervention and validation metadata.
- `recovery_train.npz`, `recovery_val.npz`: diagnostic data with SHA-256 recorded in the summary.
- Four diagnostic training directories, each with configuration, metrics, summary, and checkpoint.
- `evaluation_v2/`, `evaluation_comparison_v2.json`, common-support and residual-failure JSON.
- Tables and SVG figures linked above.

### Temporary implementation removed

The following diagnostic-only scripts were created under `/tmp/double_bottleneck_toy_transfer.aFYTvG` and permanently deleted after results were recorded:

- `build_aug.py`
- `train_transfer.py`
- `evaluate_transfer.py`
- `oracle_gate.py`
- `common_support.py`
- `toy_reproduce.py`
- `analyze_results.py`

An earlier evaluator output (`evaluation/` and `evaluation_comparison.json`) was also permanently deleted because its K-step clearance reduction incorrectly included the full 100-step continuation for every K. The corrected `evaluation_v2` is retained.

### Canonical-code proof

- No Toy source was modified.
- No Double-Bottleneck canonical policy, dataset adapter, trainer, evaluator, environment, or expert source was modified.
- No explicit mode input, persistent-noise hook, future oracle input, action horizon, or trajectory model remains.
- Cleanup hashes match the hashes recorded before diagnostics:
  - Double agent `02dda46a…f8`, dataset `e503da22…2f3`, trainer `c1c2b6d…9367`, evaluator `97e707dd…94a8`;
  - Toy agent `8b94fb64…f5adc`, dataset `a81b887b…77af`, evaluator `294d233f…e4ae`.
- Cleanup regression command ran 42 tests in 6.612 s: **all passed**. It covered MACFlow source parity/checkpoint sampling, real expert dataset validation, all eight hypotheses, Double dynamics/projection/termination/visualization, and Toy namespace/rollout regression. JAX emitted its known unavailable-CUDA plugin warning and then executed on CPU; it was not a test failure.
- The repository was already dirty/untracked before this audit. Initial status was:

```text
 M README.md
 M scripts/test.sh
?? diagnostics/
?? docs/double_bottleneck_eta_audit.md
?? double_bottleneck/
?? gitpush.sh
?? shared_control/
?? tests/test_cl_fhcb.py
?? tests/test_double_bottleneck.py
?? toy_giveway/
```

The audit adds only this diagnostic result directory inside the already-untracked `diagnostics/` tree. Source hashes before/after cleanup are recorded alongside the regression result below. Because the scenario tree was already untracked, `git diff` alone cannot prove its integrity; the hashes and tests provide the relevant additional check.

### Final status

**NOT READY for safety-baseline or basin evaluation.** The Toy-derived coverage mechanism is causally credible and materially improves the 4-agent policy, but `4/12` success with eight collisions is below a reliable Stage-I gate.
