# Frozen-Q-guided OrthoFlow3 Direct-eta controlled ablation

## Decision

**Q_GUIDANCE_HARMS_CONTROL**

Baseline checkpoint: `bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7`. Frozen Q: `8cf8605d1aa6c053faddb72ffe1bc4e0f77c0f0975ab752f48b05c0b6a4eb7cd`; Q freeze integrity passed. Tau was the pre-existing validation-only threshold `0.771`. Fifteen actors were trained at five lambdas and three seeds.

The selected actor is `L1.00_S23` (lambda 1.0, seed 23), SHA256 `2cd35ec635c4d980cd5946e3a09b68a16505bfb33f6426be4b129e6c4c11f889`. Stage-1 has 15 rows; Stage-2 promoted 3 rows.

## Held-out TEST

Baseline/guided B63: 16/20 vs 16/20. Mean Q64: 0.9180 vs 0.9227. ZERO harmful false activations: 4 vs 4; ACTIVE B63: 4/4 vs 4/4. Critic-exploitation cases: 2.

## Fresh WIDE 400

Safety/baseline/guided successes: 286 / 281 / 278. Baseline/guided breaks: 88 / 91; rescues: 83 / 83. Baseline break recovery: 21; baseline rescue preservation: 82; new guided breaks: 24; new guided rescues: 1.

Paired guided-minus-baseline success delta: -0.0075, 95% CI [-0.040000000000000036, 0.025000000000000022]. Break-rate delta CI: [-0.03521126760563381, 0.05639097744360899]; rescue-rate delta CI: [-0.025862068965517238, 0.025862068965517238].

Hard-safety integrity: PASS; collisions 0, invalid actions 0, projection failures 0.

Q provides material training-time value beyond MSE: **False**. Next direction: Study critic support and conservative basin learning before further actor optimization. No next experiment was started.
