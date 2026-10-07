# C1 critic/generator generalization audit and independent Toy confirmation

## Decision

The frozen Toy W1 continuous-Q critic has real within-Toy closed-loop value on
independent, same-distribution initial conditions, and it performs well on
continuous proposals not exactly present in training. **A transferable
state–eta feasibility law is not yet established.** The strongest eta-only
control remains competitive on the new confirmation set, and the corrected
Four-Way/Ring evidence is either saturated for top-1 selection or favors
eta-only. Keep the frozen Toy critic as an in-domain candidate, not as a
cross-scenario claim; do not change the generator yet.

This experiment did not train a multimodal generator, modify the frozen Toy
generator, alter control/safety/success semantics, or optimize eta using the
critic. The historical Toy 200-state figures remain 194/200 oracle and
181/200 frozen critic; the new cohort does not retroactively change them.

## Independent Toy confirmation

Forty-eight independent wide-IC initial states were drawn outcome-blind from
the same physical distribution as the historical 200-state test, using a new
RNG namespace. Each is its own source group. The full state list, frozen
generator mean and K=16 samples, frozen 3-seed W1 critic scores, eta-only
scores and all choices were saved before any new rollout. No exact initial
state overlaps the old 200. All 768 sampled eta differ exactly from the
549 distinct TRAIN eta; nearest TRAIN eta distance in the frozen normalized
coordinates has median 0.1518 (IQR 0.1053–0.2063). This supports
*in-distribution independent-state and nonidentical continuous-proposal*
generalization, not arbitrary eta-space extrapolation or cross-scene zero-shot.

All methods used the same 16 matched continuation seeds per state. B15 is
at least 15/16 successes. Q16 is a finite-seed empirical estimate.

| Method | B15 / 48 | Mean Q16 | Success / 768 | Deadlock | Timeout | Collision | Mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|
| K16 proposal oracle | 48 | 1.0000 | — | — | — | — | — |
| Frozen W1 critic, same K16 proposals | 46 | 0.9792 | 752 | 6 | 10 | 0 | 0.5572 |
| Continuous eta-only kernel, same proposals | 43 | 0.9128 | 701 | 49 | 18 | 0 | 0.5379 |
| Eta-only MLP, same proposals | 31 | 0.7552 | 580 | 115 | 73 | 0 | 0.4458 |
| Fixed common eta | 42 | 0.9479 | 728 | 27 | 13 | 0 | 0.5056 |
| Frozen generator mean | 38 | 0.8698 | 668 | 49 | 51 | 0 | 0.5145 |
| Hard Safety | 31 | 0.7240 | 556 | 41 | 171 | 0 | 0 |
| MAC-only | 29 | 0.7318 | 562 | 0 | 69 | 137 | — |

The corrected controllers have zero collisions and zero numerical failures
on this cohort. Existing synchronized batched K16 latency is 0.127 ms p50,
0.182 ms p95 on the benchmark GPU, *after* the feature vector is available.
Eta is chosen once at true t0 and latched, so this is one decision delay,
not a per-step safety-loop cost. The critic's higher J_def than fixed common
is a real tradeoff and should not be hidden.

Paired B15 rescues/breaks for critic versus: eta-only kernel 4/1 (net +3,
exact two-sided p=0.375; bootstrap rate-difference 95% CI −2.1 to +16.7
points); fixed common 5/1 (net +4, p=0.219); generator mean 9/1 (net +8,
p=0.0215); Safety 16/1 (net +15, p=0.000275). Thus it clearly helps versus
Safety and its own generator mean, but the **new** 48-state confirmation alone
does not establish a reliable advantage over a strong global eta preference
or fixed common eta. On the already examined 200-state cohort, the same
critic was 181/200 versus the eta-only kernel's 163/200, with 22 rescues and
4 breaks; that supports a Toy state signal but is diagnostic rather than a
second untouched confirmation. The successful K16 proposal oracle at 48/48
also means there is no reason to alter the generator for this cohort.

## Corrected Four-Way, Ring, and DB assets

The original joint generator/critic used DB, Four-Way and Ring labels. The
old Ring training predates the repaired safety definition, and the v1 critic
used `success_count/seed_count` as a Q target even on early-stopped labels;
that ratio is not an unbiased exact Q16. We kept old predictions as a frozen
historical comparator but trained all new Ring diagnostics only on audited
`ring_current_safety_v2` logical B15 labels. Those logical labels are not
silently treated as calibrated full-Q16 counts.

Audited dense coverage is 69 TRAIN / 16 VAL DB states with 22 repeated eta,
and 64 TRAIN / 16 VAL source-family states with 269 repeated eta for each of
Four-Way and Ring. There is no source-group overlap in those audited splits.
For a strict source+spatial-eta holdout, we further froze 48 TRAIN, 16 DEV
and 16 TEST source families, with 161 TRAIN, 37 DEV and 71 TEST eta groups
for Four-Way and Ring. TEST eta are spatially separated from TRAIN (minimum
normalized distance 0.086, median 0.320).

| Diagnostic | Frozen result | Interpretation |
|---|---|---|
| Four-Way source+eta holdout | Oracle 16/16; VAL-selected Four-only and eta-only MLP both 16/16 | Top-1 is saturated; no adaptive-selection evidence. |
| Ring source+eta holdout | Oracle 15/16; VAL-selected Ring-only and eta-only MLP both 15/15 coverable | No top-1 gain; eta-only TEST NLL 0.3015 vs Ring-only 0.4381. |
| Ring existing true-t0 K16 panel, corrected safety | Oracle 60/60; old v1 critic 53/60, corrected Ring-only 54/60, corrected joint 53/60, continuous eta-only kernel 58/60 | Previously examined diagnostic, **not** a new independent confirmation. |
| Four→Ring 25% adapter-only | VAL-selected frozen shared-trunk adapter: 0/15 spatial coverable, 18/60 existing K16 | No few-shot transfer benefit under this representation. |
| DB repeated-eta audit | 22 repeated eta over 69 TRAIN/16 VAL; no TRAIN-selected repeated-eta preference reversal | Current dense panel cannot identify DB state-conditioned advantage. |

Ring does contain state-dependent feasibility structure: ten eta pairs chosen
using TRAIN only reverse their B15 preference across DEV source families.
Among 74 decisive DEV comparisons, the corrected Ring-only model's
VAL-selected seed orders 54 correctly and the eta-only kernel 52. Because the
same DEV split selected checkpoints and the difference is small, this is
diagnostic evidence that state information exists, **not** a reliable
deployment gain. Joint training and Four→Ring adaptation did not consistently
improve the independent source+eta holdout or the existing true-t0 K16 test.

The cross-scene negative result has an identifiable coverage confound.
Four-Way TRAIN states are mid-trajectory (median timestep 410.5) and Ring
TRAIN states are mid-trajectory (median 167.5), whereas their fresh K16
decision states are true t0. Nearest physical-16 distance from fresh states
to TRAIN is 4.9× the VAL-to-TRAIN median for Four-Way and 2.4× for Ring.
DB TRAIN/VAL are both true t0 and do not show this particular phase shift.
This is evidence of a train–deployment representation/support mismatch,
not proof that h is intrinsically insufficient or that sharing is impossible.

Raw-numeric Four→Ring eta-only zero-shot scored just 12–14/60 on the existing
Ring panel, but numeric eta need not transfer directly across scenarios;
therefore this is only a lower-bound control, **not** a falsification of
shared feasibility structure. A proper target-scenario zero-shot critic or
generator was not available: the original joint checkpoints already used
Ring/Four-Way training labels, and state feature dimensions differ (Toy 214,
DB/Four-Way 80, Ring 100). Do not label their held-out-state results
“scenario zero-shot.”

## What is learned, and what remains unresolved

- **Toy:** The frozen W1 critic is more than a pure eta lookup on the
  historical 200-state diagnostic and has good absolute performance on new
  source families/proposals. The new 48-state paired advantage over the
  strongest continuous eta-only model is positive but inconclusive. Claim
  *partial in-domain state–eta signal*, not established universal Q law.
- **Four-Way:** The available source+eta holdout cannot distinguish adaptive
  selection because an eta-only model already reaches the oracle state count.
- **Ring:** Feasibility preference can depend on state, but the corrected
  critics do not reliably turn it into better K16 closed-loop selection than
  a strong eta-only prior. Adapter-only transfer is poor. The true-t0 phase
  shift and old safety-label mismatch prevent a clean structural negative.
- **DB:** Existing repeated-eta evidence lacks a discriminative fixed-eta
  transition panel; state-conditioned learning is underidentified.
- **Generator:** The frozen Toy mode-free generator covers 48/48 new states at
  K16, versus mean 38/48; historical Toy K16 coverage is 194/200. Its
  stochastic proposals are useful when selected, but no generator change is
  justified by this task. Original DB/Four/Ring generator training used
  those scenarios, so its performance is not cross-scene zero-shot.

The most important pending evidence is a **fresh true-t0 Four-Way/Ring/DB
K16 confirmation** with full compatible Q16, independent source families,
frozen choices and a strong eta-only comparator. The separate
`orthoflow3_k16_critic_canonicalization_v1` task is currently collecting
those labels and has not yet frozen its new critic or confirmation cohort.
We did not duplicate its rollouts or count its planned outputs as results.
The read-only `evaluate_independent_confirmation.py` is ready to evaluate
its finished immutable candidate pools. Until then, cross-scenario
zero-shot and target-phase few-shot claims are **UNDERRESOLVED**. The
existing spatial holdout and phase-shift audit already argue against
promoting any newly trained shared critic as the current main model.

## Reproduction, cache, and budget

The audit and split artifacts are `lineage_inventory.json`,
`spatial_source_split.json`, `ring_reversal_corrected_model_summary.json`,
`ring_k16_interaction_summary.json`, and `decision_phase_support.json`.
The independent Toy cohort and all frozen scores are
`toy_replication/frozen_proposals.json`; exact outcomes and paired statistics
are `toy_replication/replication_analysis.json`. Recompute the latter with
`python diagnostics/orthoflow3_c1_generalization_v1/analyze_toy_replication.py`.

Four cache-preflighted batches requested 4,096 + 4,096 + 4,096 + 3,072 =
15,360 continuations. All were genuinely missing before execution, below
the 16,384 total and 4,096 per-batch limits. Append-only journals were
merged into `shared_rollout_db/rollout.sqlite`; each batch postflight now
reports all requests EXACT_REUSE, zero missing, zero ambiguous and zero
incompatible. There were zero duplicated local records. The one shared
rollout executor adjustment was **metadata-only**: new state IDs use the
frozen source-group alias to avoid colliding with old experiment-local IDs;
the physics/control loop was untouched. No other new rollout was launched
by this task.
