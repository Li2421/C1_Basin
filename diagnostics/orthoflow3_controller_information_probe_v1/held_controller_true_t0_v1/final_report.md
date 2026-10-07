# Independent true-t0 held-controller selection dataset

**Dataset built and frozen.** Existing Ring K16 true-t0 labels had already been opened in several diagnostics and used the canonical Flow. The earlier independent Flow-v11 test had 12 mid-trajectory states and 134/192 B15 candidates, so was too easy for a meaningful selection comparison. No existing asset met all four requirements: previously unseen controller, independent true-t0 families, frozen score-blind candidate pool, and unused task outcomes. See `dataset_design.md`.

The new prospective panel sampled 48 independent Ring test-split initial conditions (seeds 6,000,000–6,000,047) and froze the first 24 as Stage 1. Each state has the same 16 exact source-TRAIN eta. The eta panel and state rule were fixed without v11 outcomes or any critic/generator score. All 768 prospective pair scores (eta-only, correct H20, wrong-base H20; seeds 17/23/41) were frozen before task rollouts; `frozen_predictions.json` SHA256 is `e3d1e8379856295bce9472c886d82c2342056dc8bb338d43ff6891ce8e81721f`. The new states have zero UID overlap with critic TRAIN or VAL, and the parser replay matches the former Ring encoder tensor. This is within-Ring held-controller confirmation, **not** cross-scene zero-shot.

Stage-1 cache preflight: **6,144 requested, 0 exact, 0 partial, 0 aggregate, 6,144 genuinely missing**. Records under other controller semantics were not reused. All 6,144 requested continuations were executed, written to append-only journals, and atomically merged into the global rollout DB, with **0 conflicts, 0 collisions, 9 numerical-failure seeds**. Postflight reports 6,000 exact reusable seeds, 135 partially reusable seeds and 9 numerical keys not available as valid Q16; those failures are retained in the DB and never imputed. All 384 candidate pairs nevertheless have a logically determined B15/non-B15 status from observed seeds.

| Stage-1 property | Result |
|---|---:|
| true-t0 states / common eta / pairs | 24 / 16 / 384 |
| oracle-eligible mixed states | 23/24 |
| states with only 1–4 B15 candidates | 9/24 |
| candidate B15 / confirmed non-B15 | 116 / 268 |
| strong cross-state ranking reversals (Q16 difference ≥0.25 both sides) | 334 |
| best fixed eta, post-hoc ceiling diagnostic | 20/24; 6/9 on hard states |

The preregistered Stage-2 trigger required fewer than 8 hard states **or** fewer than 12 mixed oracle-eligible states. Both requirements were exceeded, so **Stage 2 was not executed**. Its 24 states remain unopened, and no result-dependent candidate redesign occurred.

On the 23 mixed oracle-eligible states, source-only eta-only selected B15 **0/23 for all three seeds**. Correct H20 selected **0/23, 20/23, 0/23** for seeds 17/23/41. Matched wrong-base H20 selected **exactly the same states: 0/23, 20/23, 0/23**. Every model/seed selected a *single identical eta across all 24 states*. Seed 23 happened to prefer an eta robust on 20 states; the other seeds preferred eta that failed on all 24. Therefore this panel exposes severe seed-dependent global eta preference and high-confidence false positives, but **does not support learned use of state or the correct controller context for eta ranking**. Correct versus wrong context never changed top-1. The 334 real ranking reversals show that the data contain state–eta interaction even though these models did not exploit it.

This panel is a clean, informative **negative-selection test**. It does not by itself provide a high-powered positive test of adaptive ranking: a post-hoc best fixed eta already covers 20/24, leaving only three states of oracle gain. The post-hoc fixed eta is a diagnostic ceiling, not an admissible trained baseline. Any next adaptive-selection claim needs an independently frozen, controller-varied or harder physical-regime confirmation set; these opened 24 states must not be used to choose a new checkpoint or eta panel. The unopened Stage-2 states remain available only under a new prospective, explicitly frozen protocol.

Reproduction: `design.json`, `states.json`, `pairs.json`, `protocol.json`, `frozen_predictions.json`, `cache_preflight_stage1.json`, `cache_postflight_stage1.json`, `merge_audit_stage1.json`, `evaluation_stage1.json`, and the scripts `held_true_t0_benchmark.py`, `held_true_t0_predict.py`, `held_true_t0_evaluate.py` in the parent directory.
