# OrthoFlow3 Q learnability and directional fidelity v2

Status: frozen before base outcome inspection. This is a diagnostic Q-only
experiment. No J, G, Direct-eta, Direct-g, gate, mixture, flow, actor, critic,
or production controller is trained or modified.

## Conditional quantity

The target is

`Q(h(z, xi_0), eta) = P(success | exact augmented z, exact query-step Flow key xi_0, fixed eta)`.

The archived deployment feature variant fixes `xi_0` through its `u_flow`,
`u_safe`, and first-projection fields. Every continuation reuses that exact
query-step Flow key. Only Flow draws after the query transition are resampled.
Future streams are matched across all eta candidates for a state and future
index. Eta is held fixed to terminal outcome or the original absolute 850-step
horizon. The two safety projections and OrthoFlow3 basis implementation remain
frozen.

Existing candidate success counts that varied the query-step Flow seed estimate
`Q(z, eta)`, not this conditional quantity, and are not reused as labels.

## Frozen population and split

The source manifest contains 424 states. Eleven historical strict-deadlock
states lack the RNG namespace needed to bind an archived feature variant to an
exact query-step Flow key, leaving 413 conditioning-compatible eligible states.
Leakage groups are deterministically hash-permuted with seed string
`orthoflow3_q_v2_state_groups_20260927`. One outcome-blind, hash-ranked state is
selected from each of 120 distinct groups: 80 TRAIN, 20 VAL, and 20 TEST. A
single archived Flow variant per state is selected with seed string
`orthoflow3_q_v2_h_condition_20260927`. No outcome, eta label, failure type,
geometry, or earlier controller performance enters selection.

## Eta cloud and evidence

The common cloud contains explicit eta=0 plus the first 23 points of the frozen
authoritative 256-point OrthoFlow3 Sobol design. Every TRAIN and VAL candidate
gets 4 future trials; every TEST candidate gets 8. This is 13,440 base
continuations. Candidate rows retain binomial success/trial counts and all
failure categories.

TRAIN informativeness is accepted only if neither the all-fail nor all-success
bin contains 95% or more candidates and at least 10% of TRAIN states have both
successful and unsuccessful eta candidates. The cloud is never outcome-adapted.

## Q model

TRAIN-only h mean/std and authoritative eta-domain center/ranges normalize the
inputs. The only nonlinear model family is 217 -> 256 -> 256 -> 1 with SiLU and
a sigmoid interpretation. Candidate-level binomial NLL is optimized using Adam
at 1e-3. Seeds are exactly 17, 23, and 41; early stopping and checkpoint
selection use VAL NLL only. Required baselines are TRAIN-prevalence constant and
an untuned logistic-linear model.

## Frozen follow-ups

After TEST evaluation and checkpoint freezing, the first eight TEST states are
extended from 8 to 64 eta=0 trials. A feasibility threshold is selected on VAL
eta=0 screening labels only.

For directional validation, TEST states are scanned in frozen order. At each
state the common-cloud point with predicted Q closest to 0.5 is chosen, subject
to nonzero gradient and an unclipped +/- step. The first six eligible states are
used. In normalized eta coordinates, alpha is exactly 0.05. Eta-minus, base,
and eta-plus receive 32 matched futures; exact base trials are reused. Q is not
updated after these true outcomes are observed. The optional random-direction
control is omitted to preserve rollout budget.

## Resource and decision rules

The base timing batch began on two GPU shards. Its measured throughput projected
past the 60-minute wall cap; after confirming the server was idle at night, the
durable globally deduplicated completion batch uses the user-authorized maximum
of six shards, two CPU threads and 14 GB RAM per shard. JAX preallocation is
disabled with a 0.14 per-process memory fraction. The planned scientific total
is 14,416 valid continuations: 13,440 base, 448
zero follow-up, and 528 gradient follow-up. It remains below 15,000
continuations and the 8-million-step cap.

For a reproducible aid to interpretation, held-out prediction is called useful
only when TEST NLL beats the constant baseline and mean statewise Spearman is
above 0.20. Directional fidelity additionally requires six cases, at least 4/6
with Q-plus > Q-minus, positive mean improvement, at least 4/6 predicted/true
sign agreement, and no Q-plus minus Q-minus collapse of 0.25 or worse. Final
reporting also presents all continuous values so this diagnostic rule is not a
substitute for scientific judgment.
