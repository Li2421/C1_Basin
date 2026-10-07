# Held-controller robust-selection benchmark

**Verdict: `HELD_CONTROLLER_SELECTION_NOT_SUPPORTED`.** On an independently trained, previously unseen Ring Flow, the frozen H20 controller-conditioned critic selected B15 eta on **11/12** eligible states for each of seeds 17, 23 and 41. Source-only eta-only and matched wrong-base-context controls also selected **11/12**. Correct context changed top-1 on 6, 5 and 3 states relative to wrong context, but delivered **zero paired B15 rescue**. This is a within-scene controller test, not strict cross-scene LOSO.

## Frozen protocol and data integrity

- New Flow: independent initialization/minibatch seed 88123, trained on the existing compatible Ring expert-transition dataset with the same Stage-I procedure as the canonical Flow. The Flow checkpoint was selected by its own dev CFM loss, never by eta rollout outcomes. It was absent from all critic training, normalization and checkpoint selection.
- Before its task outcomes were opened, the benchmark froze 12 Ring source-VAL families absent from the previous controller probe, and one identical, score-blind panel of 16 exact source-TRAIN eta per state. Four eta were chosen by source-TRAIN frequency; twelve by deterministic farthest-first geometry. These are independent source families, but **mid-trajectory states, not true-t0**.
- Eta-only, correct-H20 and wrong-base-H20 probabilities were frozen before any new controller rollout was journaled; see `frozen_predictions.json` (SHA256 `3ffc43d2d1e5bbbf50f9d140550c27aa37c32f84120b32bcf5b804ca3aaaa6bc`). No generator was modified or used to form candidates.
- Cache preflight: requested **3,072** standard-seed continuations; exact 0, partial 0, aggregate 0, truly missing **3,072**. All 3,072 were appended to journals and atomically merged; conflicts 0. Nine real numerical failures remain separately recorded. Postflight: 2,928 exact seed reuses, 135 partial seed reuses, 9 unavailable numerical seed keys; no incompatible or ambiguous records. Numerical failures were **not** imputed into Q16.
- All 12 states had at least one B15 eta and at least one confirmed non-B15 eta. Oracle coverage **12/12**; 134/192 pairs B15 and 58/192 confirmed non-B15. Of 192 pairs, 183 have complete valid Q16. In particular, the two hardest families have only 3/16 and 2/16 B15 candidates. Many other states have 10–15 B15 candidates, so the panel still has a high global-prior ceiling.

## Same-candidate comparison

| Model / context | B15 selected, seeds 17/23/41 | observed-count NLL, seeds 17/23/41 | Paired B15 rescue/break vs eta-only |
|---|---:|---:|---:|
| Source-only eta-only | 11/12, 11/12, 11/12 | 0.544, 0.533, 0.556 | reference |
| H20, correct new controller | 11/12, 11/12, 11/12 | 0.693, 0.740, 0.709 | 0/0, 0/0, 0/0 |
| H20, wrong base controller | 11/12, 11/12, 11/12 | 0.584, 0.615, 0.628 | same B15 states |
| Oracle | 12/12 | — | — |

The failure is the same `dev_nominal_000773` source family for all methods and seeds. It has two actual B15 eta (both 16/16), but each model selects a 1/16 or 13/16 eta depending on seed. For seed 17, the wrong 1/16 eta is assigned 0.982 by eta-only and 0.999 by correct H20. Correct context therefore **does not suppress the severe high-confidence false positive**. Full top-2/top-3 and Q16 interval metrics are in `held_selection_metrics.csv`; selected true-Q lower/upper intervals are reported when a numerical seed prevents exact Q16.

The fixed cross-matrix contains **107 strong true state–eta ranking reversals** across these new states. Eta-only cannot represent such reversals; correct H20 predicts both sides on 30/41/11 cases across seeds. But wrong-base H20 scores 29/42/18. Thus state-aware predictions do vary with state, while the *correct controller context* has no stable advantage in the interaction test. On the frozen first-four frequent-eta diagnostic subpanel, B15 selection is 8/9 for eta-only and 9/9, 8/9, 8/9 for both correct **and wrong** H20. This post-hoc K4/K8 analysis is diagnostic only, not independent confirmation.

The local H20 response is not simply constant: correct-vs-base median normalized feature distance is 0.439, and top-1 changes on 3–6 states. Only 86/4,608 normalized target features exceed absolute z=3, so a wholesale scale explosion is not evident. Yet correct H20 has **worse** Q likelihood than wrong-base H20 in every seed. Earlier fresh Flow v8 showed a positive probability-context signal but had oracle B15=0/4; this new oracle-eligible controller shows that probability evidence does not transfer reliably to robust selection.

## Interpretation and next boundary

This test refutes promotion of the current H20 model as a held-controller robust selector. It does **not** prove controller-conditioned feasibility is impossible. The strongest directly supported explanation is that the learned context-to-Q/ranking relation is not stable on a new controller; source support is sparse for some broad eta (the false 1/16 top choice appeared at only one source-TRAIN state), and 9/12 states have many robust choices, leaving little room to outperform eta-only. Because correct and wrong H20 tie on B15 and wrong context gives better NLL, missing proposal coverage is **not** the main failure here.

Do not tune on this opened panel. A further independent test, if pursued, must pre-register new true-t0 hard families and a source-only candidate rule that reduces global-prior saturation without consulting model scores or target outcomes. Strict cross-scene LOSO remains unproven.

Reproduction: `held_robust_benchmark.py`, `held_robust_predict.py`, `held_robust_evaluate.py`, `held_robust_diagnose.py`; frozen design, predictions, pre/postflight, merge audit and full per-pair outcomes are in this directory.
