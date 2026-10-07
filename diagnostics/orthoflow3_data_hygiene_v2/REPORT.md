# OrthoFlow3 data hygiene v2 audit

## Executive result

- Dataset status: **DATASET_V2_RECOMMENDED**.
- Model-data status: **CURRENT_MODELS_USE_PARTIALLY_STALE_DATA**.
- v1 is structurally complete and fully traceable, but is not scientifically current for Ring because all 21,520 Ring eta rows used the old adapter without the outer-boundary constraint.
- v2 preserves all 245 state identities and 45,169 eta rows, replacing every Ring eta summary with current-safety evidence. No model was trained.

## 1. Canonical freeze

Canonical hashes are in `canonical_hashes.json`; final verification is **PASS** in `canonical_hash_verification.json`. The current Ring safety hash is `18419e7a15d3dbeb9b49bacefbaf6e376a7e3551833a89bdfec52c3bab1861be` and the old hash is `37b0f6ad63e7ebcc2467045de6f40a336171b1cdbec55a4fdb954de90b9236e3`. v1, environments, Stage-I checkpoints, OrthoFlow3, normalization, generator, and critic hashes remained unchanged.

## 2. v1 provenance classification

All 45,169 rows were traced. Initial classes were 23,620 `CURRENT_COMPATIBLE`, 21,520 `STALE_SAFETY`, and 29 `NUMERICAL_UNCERTIFIED`; `STALE_ENVIRONMENT`, `STALE_MACFLOW`, `STALE_ORTHOFLOW3`, incomplete provenance, and missing DB evidence were all zero. All 256,956 rollout UIDs referenced by v1 were present and their summaries matched DB records.

## 3. Ring safety audit and recomputation

All 80 Ring states and all 21,520 Ring labels came from old safety provenance. Old rows comprised 5,502 robust, 16,012 non-robust, and 6 uncertified labels, including 80 eta=0 probes. Current-safety recomputation produced 121,394 unique seed records from 122,783 physical attempts; full 16-seed execution would have required 344,320 seed records, so exact early stopping saved 64.74% of unique seed executions.

## 4. Ring label drift

Certified robust/non-robust drift was **0.3674%**: robust→robust 5462, robust→non-robust 33, non-robust→robust 46, non-robust→non-robust 15963. Ten current rows remain numerically uncertified. Eta=0 changed on **0/80** states. Mean empirical-success-fraction change was 0.001242, median 0.000000; these fractions use actually executed canonical seeds and are not misreported as universal full Q16. Old collision presence disappeared on 827 labels (827 `1→0`); current v2 contains no collision seed.

## 5. Evidence strength and label validity

No robust label is under-evidenced. Exact robust seed-count evidence is `{'15': 9995, '16': 1246, '32': 71, '33': 9, '64': 39}`; 4/4 alone is never accepted. v2 contains 33770 certified negatives, 1 weak negative, and 38 uncertified labels. Numerical retries affected 702 label tuples, but zero were incorrectly certified from a numerical failure. There are zero successful-collision inconsistencies and zero current collision-containing robust labels.

## 6. State density and boundary evidence

| Scenario | Train/val states | Eta rows | Robust | Certified negative | Weak | Uncertified | Zero-sufficient |
|---|---:|---:|---:|---:|---:|---:|---:|
| Double-Bottleneck | 69/16 | 2129 | 1054 | 1074 | 1 | 0 | 15 |
| Four-Way | 64/16 | 21520 | 4794 | 16698 | 0 | 28 | 3 |
| Ring | 64/16 | 21520 | 5512 | 15998 | 0 | 10 | 56 |


No state has fewer than four robust eta values or lacks a certified negative. Median robust counts per state are 10 / 60 / 65 for Double/Four/Ring. Boundary evidence is retained as point clouds and nearby certified negatives; no unvalidated ellipsoid or inner ball was fabricated.

## 7. Scenario balance, splits, conditioning, and normalization

Training used equal per-step scenario exposure: generator 64 states/scenario and critic 96 states/scenario. Independent leakage audit: **PASS**; no frozen test state, crossed parent/family, corrected validation descendant in train, or exact/near conditioning duplicate was found. Conditioning schema audit passed for all 245 states. `NORMALIZATION_LEAKAGE = NO`; train-only recomputation matched the frozen statistics exactly (maximum absolute difference 0.0).

## 8. Model-training provenance

The frozen joint models used v1. Ring critic training used 17,216 old-safety rows and validation used 4,304; Ring generator training used 4,165 old-safety robust in-domain rows and validation used 1,281. Because one full scenario has stale provenance but certified label drift is limited to 0.367% and eta=0 did not change, the assessment is **MODEL_TRAINING_DATA_PARTIALLY_STALE**, not current and not evidence of model failure. No retraining was performed.

## 9. Reproducibility, DB integrity, and regression

The pre-registered replay selected 25 labels/scenario. All **75/75** label Q values and **695/695** seed outcomes, terminal reasons, and collision flags reproduced exactly. DB `quick_check` is `ok`; foreign-key violations, duplicate exact tuple keys, orphan states/etas, malformed seed IDs, missing rollout provenance, and missing source paths are zero. Historical conflicts remain quarantined/resolved as recorded; 63 historical LIVE sources belong to earlier experiments, while this hygiene experiment has zero LIVE sources. Regression: **137/137 PASS**.

## 10. Learning-readiness matrix

For audited v2, direct-center regression, generator robust-point likelihood, Q(h,eta), critic ranking, and margin/set learning are **READY** for all three scenarios. Minimum-deformation learning is **PARTIAL** for all three because eta=0 and robust alternatives exist but executed-deformation metadata is not dense enough. In v1, Ring objectives are `NOT_READY` under current safety semantics; Double and Four retain the v2 readiness shown in `learning_readiness_matrix.json`.

## 11. Required answers

1. v1 structurally complete: **yes**.
2. v1 scientifically current under fixed Ring safety: **no**.
3. Stale Ring eta labels: **21,520**.
4. Certified robust/non-robust drift after recomputation: **0.3674%** (79 changed certified labels).
5. Eta=0 sufficiency changed: **no, 0/80**.
6. Under-evidenced robust labels: **0**.
7. Certified negatives / weak negatives: **33770 / 1**.
8. Numerical failures incorrectly treated as negatives: **0**.
9. Current collision-containing robust labels: **0** (v1 had 3 stale Ring robust rows with a collision seed; current evidence has zero collision seeds).
10. Train/validation leakage: **none**.
11. Normalization leakage: **no**.
12. Current model checkpoints use partially stale data: **yes**.
13. New rollout work: **121,394 unique seed records; 122,783 physical attempts including exact numerical retries**.
14. v2 materially differs from v1: **yes in safety provenance and certification; empirical class drift is modest (0.367%)**.
15. READY on v2: **direct center, generator robust-point learning, Q(h,eta), critic ranking, margin/set**; minimum-deformation is **PARTIAL**.

## Final status

`DATASET_V2_RECOMMENDED`

`CURRENT_MODELS_USE_PARTIALLY_STALE_DATA`

No generator, critic, G_phi, deformation head, proposal rule, OrthoFlow3 representation, eta domain, benchmark distribution, or frozen model was changed or trained.
