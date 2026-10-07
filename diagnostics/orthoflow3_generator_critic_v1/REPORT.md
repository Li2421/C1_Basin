# OrthoFlow3 joint generator + finite-proposal critic v1

## Decision

`GENERATOR_READY`

`CRITIC_READY`

Final pipeline status: **`GENERATOR_AND_CRITIC_READY`**.

The result is a single scenario-aware, mode-free pipeline jointly trained on
Double-Bottleneck, Four-Way, and Ring.  No crossing-order, priority, cyclic
yielding, CW, CCW, mixed-circulation, or equivalent routing label is an input,
target, latent supervision signal, or model-selection variable.  The critic
only ranks a frozen finite proposal set and is never optimized over eta.

## 1. Architecture audit and frozen representation

The detailed audit is in `ARCHITECTURE_AUDIT.md`.  The implementation reuses
the canonical mode-free predecessor's scenario adapters, shared trunk,
diagonal tanh-squashed Gaussian, transformed mean, deterministic stochastic
proposal rule, and `K=4`.  It reuses the continuous-Q critic's binomial
likelihood and finite ranking protocol.  Historical selector mode embeddings
and mode-relative anchors are excluded.

`h` is the lossless frozen `conditioning.flat`: 80 dimensions for
Double-Bottleneck, 80 for Four-Way, and 100 for Ring.  `c` is a 33-dimensional
mechanical encoding of the dataset's physical environment descriptor.  A
native-dimension adapter is selected by scenario, after which both networks
share their computation.  The frozen eta box is
`[0.5,1.25] x [-0.5,0.5] x [0,0.75]`.

## 2. Data and split integrity

The unchanged dataset-v1 split contains:

| Scenario | Train states | Validation states | Eta labels |
|---|---:|---:|---:|
| Double-Bottleneck | 69 | 16 | 2,129 |
| Four-Way | 64 | 16 | 21,520 |
| Ring | 64 | 16 | 21,520 |

Sampling is hierarchical and balanced: scenario, state, then eta evidence.
The generator samples among all verified robust eta points for a state; it
does not collapse supervision to one center.  The critic samples robust and
non-robust evidence within uniformly sampled states and uses observed seed
counts in the binomial loss.  Parent-level train/validation separation is
unchanged.  Frozen benchmark test states read or used: **0**.

The original 245-state / 45,169-label dataset was sufficient: learning curves,
validation proposal coverage, and true closed-loop outcomes did not indicate a
data deficit.  Therefore additional training states and eta labels generated
in this task are both **0**.  New database rollouts below are validation-only,
not appended to the training dataset.

## 3. Generator training

The generator uses native adapters of width 96, a physical-context encoder of
width 32, shared layers 128/64, and six outputs (`mu`, `log_sigma`).  Sigma is
`0.025 + 0.275 sigmoid(raw)` in unconstrained coordinates.  Training minimizes
scenario/state-balanced robust-point negative log likelihood plus a 0.25
within-state positive-over-negative softplus ranking term.  AdamW uses
`lr=5e-4`, weight decay `1e-4`, gradient norm 5, batch 64 states/scenario.

Three seeds were trained.  Validation selected seed 41 at step 1,900
(checkpoint SHA256
`a7bb0476cf25b7e0046a5d977d73f91aa478191c3f59ca73a5c487d4bf5d9965`).

Generator-only label-space validation:

| Scenario | Mean nearest-label Q | Mean nearest robust | Oracle mean+K nearest Q | Oracle robust coverage proxy | Boundary saturation |
|---|---:|---:|---:|---:|---:|
| Double-Bottleneck | 0.949 | 87.5% | 1.000 | 100% | 2.08% |
| Four-Way | 1.000 | 100% | 1.000 | 100% | 0% |
| Ring | 0.982 | 93.75% | 1.000 | 100% | 0% |

These are nearest verified label diagnostics, not substitutes for the true
closed-loop results below.  Mean sigma per dimension is approximately 0.300
for all three scenarios; in raw eta coordinates this remains localized by the
box radii.  It is non-degenerate, while saturation of the generated mean at a
box boundary is negligible.

Among the five frozen proposals (mean plus four samples), the mean number whose
nearest observed label is robust is 4.125 for Double and 4.688 for each new
scenario.  The per-state count distributions are Double `{2:2, 3:3, 4:2,
5:9}`, Four-Way `{4:5, 5:11}`, and Ring `{2:1, 4:2, 5:13}`.  These diagnostics
were frozen before critic ranking.

## 4. Zero-sufficient states

Validation contains 1/16, 1/16, and 14/16 zero-sufficient states for Double,
Four-Way, and Ring.  The generator is not a gate: normalized mean eta norm on
zero-sufficient states is 1.581, 0.850, and 0.824 respectively.  Eta=0 is kept
as the separate B0 safety-only baseline, outside the generator's frozen box.
This is deliberate for the present generator-first test; minimum-deformation
gating remains a future learning objective.

Despite nonzero corrections, the closed-loop cohort shows **zero B2/B5 break
states and zero paired-seed breaks** in Four-Way and Ring; Double generator
mean has one paired-seed break but no robust-state break, while critic selection
removes it.  Thus the current finite selector does not show systematic
unnecessary destruction of baseline success.

## 5. Critic training and validation

The critic uses native adapters of width 96, context and eta encoders of width
32, shared layers 128/64, and one continuous-Q logit.  It is trained only after
the generator checkpoint and proposals are frozen.  The loss is binomial BCE
against empirical seed success Q, weighted by `min(actual seeds,16)`.  AdamW
uses `lr=1e-3`, weight decay `1e-4`, gradient norm 5, and 96 states/scenario.

Three seeds were trained.  Validation selected seed 23 at step 2,700
(checkpoint SHA256
`f0a2dd358a477590cf434c7803d6ddcc307509157a9698c7dbfaafcf302e02ed`).

| Scenario | Q MAE | Brier | Spearman | Within-state Spearman | Pair ranking accuracy (gap >=.25) | Robust AUROC | Near-boundary MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Double-Bottleneck | .051 | .024 | .873 | .869 | 99.07% | .991 | .231 |
| Four-Way | .019 | .008 | .749 | .749 | 99.87% | .999 | .277 |
| Ring | .110 | .058 | .805 | .818 | 96.87% | .969 | .297 |

Calibration means (predicted Q versus empirical Q) are .648/.636 for Double,
.248/.235 for Four-Way, and .389/.351 for Ring.  Near-boundary probability
estimation is weaker than coarse ranking, but finite proposal ranking is the
deployment-critical property.

On all 16 validation states/scenario in label space, oracle mean+K Q is 1.0.
Critic-selected Q is .973 for Double and 1.0 for Four-Way/Ring.  The critic gap
is .027, 0, and 0; robust selection among proxy-coverable states is 87.5%,
100%, and 100%.  There is one Double proxy misranking with regret >=.25 and
none in the two new scenarios.

## 6. Frozen finite proposal protocol

For each state the generator produces its transformed mean and four stochastic
draws using a SHA256-derived deterministic seed.  B3 is `sample_0`.  B4 may
choose the true best of mean+K only as an oracle upper bound.  B5 scores exactly
those five frozen proposals and takes a finite `argmax`.  Eta=0 and the
training-only shared anchor are reported as separate B0/B1 controls.  No oracle
center is inserted, and no continuous critic refinement is performed.

## 7. True closed-loop train/dev validation

The cohort is six validation states/scenario, selected before outcomes by
SHA256 order.  Each tuple requests the canonical 16 continuation seeds.  Cache
preflight requested 2,016 seed rollouts, reused 163 exactly, and identified
1,853 missing.  Execution used 1,867 physical attempts including numerical
retries, persisted incrementally to the shared database.

Two Ring seeds remained numerically invalid after the frozen retries.  They
are not task failures: one tuple has 15/15 observed successes and is therefore
exactly robust; the other has 15 observed failures and is exactly non-robust.
Their Q intervals are retained.  All B2-B5 Q values below are exact.

| Scenario / method | Mean Q16 | Robust states | Robust rescue states | Robust break states | Paired rescue | Paired break |
|---|---:|---:|---:|---:|---:|---:|
| Double B0 safety/eta0 | .417 | 0/6 | 0 | 0 | 0 | 0 |
| Double B1 shared anchor | 1.000 | 6/6 | 6 | 0 | 56 | 0 |
| Double B2 generator mean | .979 | 5/6 | 5 | 0 | 55 | 1 |
| Double B3 random sample | 1.000 | 6/6 | 6 | 0 | 56 | 0 |
| Double B4 oracle mean+K | 1.000 | 6/6 | 6 | 0 | 56 | 0 |
| Double B5 critic selected | 1.000 | 6/6 | 6 | 0 | 56 | 0 |
| Four-Way B0 safety/eta0 | .563 | 0/6 | 0 | 0 | 0 | 0 |
| Four-Way B1 shared anchor | 1.000 | 6/6 | 6 | 0 | 42 | 0 |
| Four-Way B2 generator mean | 1.000 | 6/6 | 6 | 0 | 42 | 0 |
| Four-Way B3 random sample | 1.000 | 6/6 | 6 | 0 | 42 | 0 |
| Four-Way B4 oracle mean+K | 1.000 | 6/6 | 6 | 0 | 42 | 0 |
| Four-Way B5 critic selected | 1.000 | 6/6 | 6 | 0 | 42 | 0 |
| Ring B0 safety/eta0 | .917 | 5/6 | 0 | 0 | 0 | 0 |
| Ring B1 shared anchor | [0,.010] | 0/6 | 0 | 5 | 0 | 87 |
| Ring B2 generator mean | 1.000 | 6/6 | 1 | 0 | 8 | 0 |
| Ring B3 random sample | 1.000 | 6/6 | 1 | 0 | 8 | 0 |
| Ring B4 oracle mean+K | 1.000 | 6/6 | 1 | 0 | 8 | 0 |
| Ring B5 critic selected | 1.000 | 6/6 | 1 | 0 | 8 | 0 |

The true generator gaps (B4 minus B2) are .021, 0, and 0.  The true critic gaps
(B4 minus B5) are 0 for all three scenarios.  Critic exploitation/misranking
cases with true Q regret >=.25: **0**.  The Ring shared-anchor collapse also
demonstrates why scenario/state-conditioned generation is necessary and why a
single global eta is not an acceptable deployed substitute.

## 8. Rescue, break, and deployment interpretation

The cohort was selected uniformly, not restricted to baseline failures.
Double and Four-Way B0 have nonzero mean success probability but no state
meeting the strict 15/16 robust threshold; B5 converts all six states to robust
success.  Ring starts with five robust zero-sufficient states and one failure;
B5 rescues the failure while preserving all five successes.  Across all 18
states, B5 has 18/18 robust states, no robust breaks, no paired-seed breaks,
and no critic exploitation event.

This is a small train/development-distribution validation, not a frozen
benchmark-test generalization claim.  Its role is to verify that label-space
quality survives the actual Flow + OrthoFlow3 + second safety-projection loop.

## 9. Audits and reproducibility

- shared rollout database `PRAGMA quick_check`: `ok`;
- foreign-key check: PASS;
- database rows after validation: 1,323,826;
- frozen test states used: 0;
- parent split changed: no;
- explicit mode token found in executable model/data path: no;
- critic-gradient eta optimization: no;
- new training data generated: none;
- regression: 58 core tests + 2 subtests + 19 scenario/data tests passed;
- G_phi trained: no;
- OrthoFlow3 basis/domain changed: no.

Machine-readable configs, training histories, selected checkpoints, frozen
proposals, per-state closed-loop results, numerical audit, cache pre/postflight,
decision, and hashes are colocated in this directory.

## 10. Final conclusion

The generator itself places its mean and finite stochastic mass in robust eta
regions across all three coordination mechanisms.  The continuous-Q critic
ranks those independently generated proposals closely enough to match the
true finite-set oracle on the closed-loop cohort, without exploiting the
critic and without an explicit coordination mode.  The next experiment may
scale held-out closed-loop evaluation; it does not need a data-expansion or
architecture-revision response from this v1 result.
