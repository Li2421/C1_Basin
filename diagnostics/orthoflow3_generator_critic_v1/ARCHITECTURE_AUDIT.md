# OrthoFlow3 generator-first architecture audit

## Frozen implementation recovered

The canonical mode-free predecessor is
`diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1`.
Its reusable pieces are:

- one scenario-aware network with scenario input adapters and a shared trunk;
- a diagonal Gaussian in unconstrained coordinates;
- `tanh` squashing into the frozen eta box;
- state-balanced positive log likelihood plus a within-state negative ranking term;
- the transformed generator mean and deterministic stochastic proposals;
- a frozen proposal count of `K=4`;
- finite proposal scoring, followed by a finite `argmax` only.

The canonical continuous-Q implementation is
`diagnostics/orthoflow3_continuous_basin_critic_v1`, with the later
probability/ranking audits in `orthoflow3_structured_continuous_q_data_v1`
and `orthoflow3_ranking_aware_critic_v1`.  The reusable critic is a state
adapter, a 3-D eta encoder, a shared MLP, and a scalar logit trained with a
binomial likelihood weighted by the number of actually observed seeds.
Ranking-aware losses are not made part of the frozen primary loss: the prior
controlled ablation found that objective mismatch was not the main limiting
factor.  Within-state ranking remains a required validation metric.

## Conditioning frozen for v1

`h` is the dataset's canonical lossless `conditioning.flat` field.  Its native
dimensions are 80 (Double-Bottleneck), 80 (Four-Way), and 100 (Ring).  It is
normalized per scenario using training states only.  `c` is a mechanical
encoding of numeric fields already present in `environment_descriptor`, plus
the descriptor `kind/schema`; it is normalized from training data.  Scenario
identity selects only the native-dimension input adapter.  Shared generator
and critic trunks receive the adapter output and physical environment context.

This preserves the existing adapter/shared-trunk formulation while ensuring
that scenario ID is not a substitute for the available physical geometry.

## Frozen eta and proposal conventions

- raw eta domain: `[0.5, 1.25] x [-0.5, 0.5] x [0.0, 0.75]`;
- normalized eta: box center/radius coordinates in `[-1,1]^3`;
- generator sigma: `0.025 + 0.275 sigmoid(raw_sigma)` in unconstrained
  Gaussian coordinates;
- transformed mean: box center plus radius times `tanh(mu)`;
- deployable stochastic proposals: four deterministic draws per state;
- eta=0 is a separate hard-safety baseline and diagnostic critic candidate,
  not a generator-domain target;
- the critic may rank only the mean and four frozen generator draws.  No
  gradient, grid, random, Bayesian, or other critic-only eta optimization is
  permitted.

## Explicitly excluded legacy components

The historical `orthoflow3_toy_db_conditional_generator_v1` contains a frozen
mode selector, mode embeddings, and anchor-relative deltas.  None of those
components, labels, or targets are used here.  Crossing order, priority,
cyclic-yielding class, CW/CCW/mixed circulation, and any equivalent routing
variable are excluded from inputs, supervision, losses, and model selection.
They may only be used in post-hoc behavioral reporting.

## Split and selection rules

The frozen dataset-v1 parent-level train/validation split is used unchanged.
Sampling is hierarchical: scenario, conditioning state, then eta evidence.
Generator checkpoint selection uses validation likelihood/ranking without
critic feedback.  The selected generator is frozen before critic training.
Critic checkpoint selection uses validation continuous-Q likelihood and
within-state ranking.  Frozen benchmark test states are never read.
