# Startup-complete deterministic G_phi training

- Selected seed: 17 (validation state-grouped MSE only)
- Architecture: 214 -> 128 -> 128 -> 4, SiLU, linear output
- Optimizer: AdamW, lr=0.001, weight decay=1e-05
- Test state-grouped mean L2: 0.054758575
- Startup test state-grouped mean L2: 0.07923225398457122
- Warm-V3 test state-grouped mean L2: 0.04790594474605186
- Previous V3 checkpoint on exact V3 test: 0.015412729
- New checkpoint on exact V3 test: 0.047905945
- Projection replay integrity gate: PASS
- Test was used for final reporting only, never model selection.

The projection rewrite magnitude is descriptive.  It is not a model-selection
criterion and does not alter the supervised target.
