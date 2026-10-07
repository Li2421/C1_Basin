# Prospective held-controller true-t0 selection test

## Existing evidence checked

- `orthoflow3_ring_k16_diagnostic_v1`: 60 independent Ring true-t0 states and K16 labels, but the panel has been opened in repeated critic/generator diagnostics. It uses the canonical Flow, not independent held Flow v11. Useful historical reference, not clean held-controller confirmation.
- `orthoflow3_ring_revision_v1/fresh_test_manifests.json`: 60 Ring true-t0 initial conditions from test seeds 4,000,000–4,000,059. Also opened and canonical-controller only.
- `held_controller_robust_selection_v1`: independent Flow v11 and 12 disjoint source-VAL families; however, states are mid-trajectory and 134/192 candidates are B15. All three source-only methods tie at 11/12 B15. This is evidence of benchmark saturation, not a clean selection comparison.
- `fresh_controller_v8_v1`: held-controller probability diagnostic with oracle B15 0/4; cannot test robust selection.

None is simultaneously **independent held controller, true-t0, unseen source families, outcome-blind candidates, and sufficiently discriminative**. The new panel is therefore prospective, not retrospectively selected from known failures.

## Frozen design

- 48 fresh Ring test-split initial conditions, seeds 6,000,000–6,000,047, one true-t0 source family each. Their physical state, goals, and state hashes were frozen before held-controller outcomes.
- Same 16 exact source-TRAIN eta used in the earlier held-v11 benchmark: four frequency anchors plus twelve deterministic farthest-first continuous points. Selection was independent of critic scores and held-controller outcomes; no artificial mode ID.
- Independent Flow v11 trained from compatible expert transitions, never used for critic training, normalization or checkpoint selection. At every rollout the first Flow call uses the canonical base Flow and subsequent calls use frozen v11; OrthoFlow3 basis, hard safety and success semantics are unchanged.
- Three pre-existing source-only eta-only critics and H20 controller-context critics (seeds 17, 23, 41). For H20, correct v11 and wrong-base response contexts are both scored. All 768×9 scores are frozen in `frozen_predictions.json` **before any new task rollout**.
- Stage 1: first 24 states ×16 eta ×16 standard future seeds =6,144 continuations. Stage 2: remaining 24 states under a preregistered trigger in `design.json`; no model or outcome-based eta changes between stages.
- B15 means at least 15 observed successes among the 16 fixed seeds. Numerical failures remain explicitly unresolved rather than being imputed as success/failure.

This is a within-Ring, unseen-controller test—not cross-scene zero-shot evidence. The common eta panel may still be saturated or lack robust solutions; candidate coverage and selection eligibility are reported separately.
