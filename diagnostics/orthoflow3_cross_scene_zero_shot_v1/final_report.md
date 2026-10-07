# Toy + DB + Four-Way → Ring: first held-out-scene critic fold

## Decision

**CROSS_SCENE_STATE_CONDITIONING_NOT_YET_SUPPORTED.** The source-selected
state-aware critic achieves **0/60 B15**, versus **17/60** for the continuous
eta-only MLP selected on the same source validation scenes, and **60/60** for
the frozen candidate oracle. This is a strong negative result for the tested
source-only critic, not evidence that state information is intrinsically useless.
Generator training and the other LOSO folds were not opened: the predeclared
critic gate failed. **NEW ROLLOUT = 0.**

## What was frozen and what “zero-shot” means here

The shared physical entity architecture was inherited and every model weight
was initialized anew. Toy, Double-Bottleneck and Four-Way alone provide training
and validation labels. Their source-family splits are disjoint. Eta scaling
uses only source TRAIN extrema. Physical units are the inherited fixed units;
no Ring statistics, parameters, thresholds, early stopping or checkpoint
selection were used. The three scene objectives receive equal weight; pairs
within each scene are equally weighted. Seeds are 17, 23 and 41. Source mean
validation BCE chooses seed 17 for both primary neural methods.

The inherited physical schema was developed previously with Ring structural
and contract tests. Reinitializing weights cannot erase this history. This
is **target-label-free fitting and selection under a fixed inherited schema**,
not a target-naive history of representation design. Ring's base Flow remains
target-trained as explicitly allowed by the task. The existing Ring generator
also had Ring training; it is used only as the frozen candidate provider for
the isolated critic test. This is not an end-to-end zero-shot generator result.

Source proposal-aligned datasets produced by a Ring-trained joint generator
were excluded to avoid indirect target-label influence on source acquisition.
No new source data or revised model was introduced after Ring evaluation.

## Phase 0: Toy integration and controller semantics

Toy raw NPZ snapshots, physical goals, velocities, walls and finite-horizon
contract are mapped into the existing entity schema. Every Toy decision is
true t0, so its monitor history is empty and remaining fraction is one.
Its actual frozen Flow reference was recovered from the original policy/RNG
without advancing the environment. Across 60 states, speed-bounded reference
error against archived features is at most 4.72e-16. Original positions,
velocities and goals are checked against archived features.

CPU/GPU model consistency, passive coordinate transformations, entity
reordering and mixed masked batches passed. CPU/GPU comparisons use highest
matmul precision; the mixed-padding error is 7.45e-9. The historical and shared
OrthoFlow3 implementations produce identical three fields on the checked
two-agent/four-agent inputs. Eta still means goal, scaled goal-orthogonal
safety-Flow, and relative-agent fields; every field is recomputed each timestep.
No physical labels were relabeled because of coordinate changes.

## Source evidence

| Scene | TRAIN states / families | TRAIN pairs | VAL states | VAL pairs | TRAIN unique eta | TRAIN Q≤0.5 pairs |
|---|---:|---:|---:|---:|---:|---:|
| Toy | 48 / 48 | 2,876 | 12 | 144 | 549 | 1,280 |
| Double-Bottleneck | 69 / 69 | 812 | 16 | 365 | 13 | 88 |
| Four-Way | 58 / 58 | 153 | 15 | 39 | 16 | 0 |

Toy combines the prior structured cross-matrix and historical wide-eta pool,
deduplicated by canonical state/eta/controller. All labels are reread from
compatible, non-quarantined, nonnumerical exact seed records in the global DB.
Source TRAIN/VAL family overlap is zero. All canonical keys, seed identities
and rollout UIDs are saved in `source_pairs.parquet`.

The Four-Way source evidence has a serious limitation: most historical failure
searches were early-stopped, so their k/n ratios cannot be treated as unbiased
fixed-budget Q. Only complete evidence was admitted. Of its 153 TRAIN pairs,
141 have empirical Q≥15/16 and 12 are intermediate, with no clear failure pair.
These snapshots are mid-trajectory, whereas Ring evaluation is true t0.
Toy and DB source states are true t0. The source dataset is therefore clean
with respect to target labels but limited as a feasibility-training distribution.
Short-budget Toy labels are empirical Q targets, not certified B15 positives.

## Frozen Ring K16 evaluation

The cohort is the existing independently drawn 60-state Phase-A true-t0
confirmation. Each pool originally held a mean plus 16 random samples. This
test uses exactly **the original 16 stochastic samples**, indices 1–16, for
every method. It does not regenerate or modify any eta. Predictions and all
choices were written to `target_predictions.json` before outcome access.

| Method | B15 / 60 | Oracle gap | Mean selected empirical-Q bounds |
|---|---:|---:|---:|
| Source-only shared critic, VAL-selected | 0 | 60 | 0.0135–0.0292 |
| Source-only eta-only MLP, VAL-selected | 17 | 43 | 0.4438–0.4510 |
| Source-only continuous eta kernel | 15 | 45 | See model_comparison.csv |
| Source-only global linear eta preference | 0 | 60 | See model_comparison.csv |
| Ring-only supervised B15 reference | 53 | 7 | 0.9344 |
| Ring-supervised Phase-A joint reference | 56 | 4 | 0.9646–0.9688 |
| Frozen K16 oracle | 60 | 0 | At least one certified B15 in every state |

The Ring-only reference was selected by its archived Ring DEV metric; it is a
B15 classifier and its output is not interpreted as per-continuation Q.
The Phase-A joint reference also used Ring labels and is not zero-shot.

Shared versus source-selected eta-only: **0 rescues, 17 breaks**, difference
−28.3 percentage points; paired state-bootstrap 95% CI **[−40.0, −16.7]**,
exact two-sided discordant-pair p=1.53e-5. Its successful selection rate given
an available B15 proposal is 0%; eta-only is 28.3%. The shared critic makes
59 severe selected false positives (predicted p≥0.95 and empirical Q upper
bound≤0.5). On complete target pairs its Q NLL is 3.963, MAE 0.763 and Spearman
−0.759; the selected eta-only MLP gives 0.807, 0.397 and +0.481, respectively.

All seeds are retained, not selected by Ring results:

| Seed | Shared B15 | Eta-only MLP B15 |
|---|---:|---:|
| 17 | 0 | 17 |
| 23 | 1 | 23 |
| 41 | 14 (+1 unresolved) | 33 |

The predeclared secondary probability ensembles give shared 0/60 and eta-only
39/60. They do not change the primary source-VAL-selected comparison. In every
matched seed, the state-aware model is worse than eta-only.

## Implementation and failure audit

The Ring raw-state/Flow reconstruction reproduces the archived Phase-A critic
scores with maximum error **7.45e-7**. All 960 candidate identities match the
original stochastic indices. Cached Q bounds and robust statuses agree exactly
with the original frozen evidence. There is no source/target state UID overlap.
This substantially weakens preprocessing, eta-ordering and stale-label errors
as explanations for the observed collapse.

Two direct observations locate the failure:

1. Across the evaluated Ring proposals, eta2 correlates negatively with
   empirical Q (Spearman **−0.846**) but positively with the new shared critic
   score (**+0.865**). It ranks an adverse controller direction highly.
2. Source obstacle primitives have curvature channel identically zero.
   Ring supplies inner/outer circular boundaries with nonzero signed curvature.
   The source data cannot teach the network how Q changes with this physical
   dimension. Four-Way additionally lacks complete failure labels and source
   decision-stage coverage. Equal scene weighting does not fix those gaps.

These observations support **unsupported target geometry plus harmful
state-conditioned extrapolation**. They do not isolate model inductive bias
from source data insufficiency, and do not prove the physical input lacks the
necessary information. That distinction remains UNDERRESOLVED. The evaluation
does not justify choosing a different source model based on this target cohort.

## State–eta reversals

The predeclared subset requires the exact same two eta values in at least two
independent states with reliably opposing Q order. All **960 target eta values
are distinct across states**. Therefore the K16 cache cannot identify this
particular reversal test: `DATA_COVERAGE_INSUFFICIENT`. Nearby floats are not
silently treated as the same controller. This does not imply Ring lacks
state dependence. The aggregate test is nonetheless discriminative: oracle
is 60/60 while source-only methods are far below it.

## Cache, limits and next decision

Cache preflight: 15,360 requested seed slots; 14,096 exact-reusable slots and
1,179 reusable slots in partial pairs; 85 missing valid slots correspond to
numerical-uncertified evidence. No missing seed was executed. Unknown Q is
reported as bounds; enough valid failures can still certify non-B15. There
are zero collisions among valid selected records. Q16 remains finite-seed
evidence, not an exact population probability. No new record required merging.

The next scientifically relevant gap is **source-only training coverage of
failure/intermediate outcomes and decision states**, together with evidence
for physical geometry outside the source support. That gap should be addressed
under a new preregistered protocol and independent confirmation, not by tuning
this fold until these 60 states improve. Full generator zero-shot and the other
LOSO folds remain untested.

## Reproduction

From the repository root, with the project venv:

1. `python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.build_source`
2. `sbatch diagnostics/orthoflow3_cross_scene_zero_shot_v1/train.sbatch`
3. `python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.baselines`
4. After source freeze: `python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.evaluate`
5. `python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.audit_result`
6. `python -m diagnostics.orthoflow3_cross_scene_zero_shot_v1.finalize`

Training scripts refuse to run after the target prediction freeze exists.
Use an isolated output copy for a fresh reproduction; preserve these artifacts.
Model/data/code hashes and source splits are in `manifest.json`,
`source_data_manifest.json`, `source_states.json` and `models_frozen.json`.
