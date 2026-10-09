# Gap1 N=2/10/20 eta-pool audit (2026-10-09)

## Scientific result

The N=20 Flow prerequisite passed before eta labeling: 95/96 fresh
non-opposing complete successes, zero collisions, and 22/24 frozen opposing
TEST states screened as clean bounded gridlock candidates. The N=20
checkpoint was frozen before opposing evaluation or eta data collection.

The **current global, state-independent 16-eta pool does not yield a usable
three-scale Success-Basin dataset**. All 768 state-eta pairs per N now have an
exact B15 logical decision. N=2 has 69 robust pairs covering 33/48 physical
states, including VAL and TEST states. N=10 has four robust pairs covering
three TRAIN states and no VAL/TEST state. N=20 has zero robust pairs covering
zero states. These are finite-pool findings. N=20 had nine successful
individual continuations across five pairs, so the evidence does **not**
prove that the continuous OrthoFlow3 intervention family contains no basin.
It does show that the frozen, predeclared K=16 global candidate protocol has
no robust N=20 basin in these 48 states. No post-outcome eta sampling was
performed.

| N | Physical states TRAIN/VAL/TEST | State-eta pairs | B15 decisions | Robust pairs | States with robust eta (TRAIN/VAL/TEST) | Valid full Q16 pairs | Recorded numerical rollouts | Collisions |
|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|
| 2 | 32/8/8 | 768 | 768 | 69 | 24/4/5 | 758/768 | 10 | 0 |
| 10 | 32/8/8 | 768 | 768 | 4 | 3/0/0 | 0/768 | 19 | 0 |
| 20 | 32/8/8 | 768 | 768 | 0 | 0/0/0 | 0/768 | 65 | 0 |

N=2 executed all 12,288 prescribed futures; ten included deterministic
`CBFSolverError: identical_nonlinear_polish_failed` after four identical
attempts. Those ten pairs retain their exact seed records and robust B15
decision, but do not have a valid empirical Q16 probability. For N=10 and
N=20, the predeclared exact B15 early-stop rule was used: two valid failures
prove non-robustness, while fifteen valid successes prove robustness. N=10
observed 1,596 valid and 19 numerical rollouts; N=20 observed 1,545 valid and
65 numerical rollouts. All 768 N=20 pairs were certified non-robust from at
least two valid failures each, including pairs with numerical outcomes.
Unrun futures are not counted as failures, and partial B15 observations are
not presented as Q16 probabilities. No full-Q16 expansion was run for N=10
or N=20 after the basin-coverage gate failed.

| N | K=1 | K=2 | K=4 | K=8 | K=16 |
|---:|---:|---:|---:|---:|---:|
| 2 | 24/48 | 24/48 | 24/48 | 28/48 | 33/48 |
| 10 | 0/48 | 0/48 | 0/48 | 0/48 | 3/48 |
| 20 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |

These are oracle **robust** candidate coverage rates over all physical
states, using nested prefixes of the same eta pool. At K=16, the eligible
state denominators are 33, 3, and 0 for N=2, N=10, and N=20 respectively.
The zero denominator for N=20 makes eligible-state coverage undefined, not
100%. Robust prevalence across all 768 state-eta pairs is 8.984%, 0.521%,
and 0%. The N=2 full-Q16 distribution is in the machine-readable audit; the
N=10/N=20 B15 observations have no complete-Q16 distribution.

## Protocol and provenance

The frozen checkpoints have SHA256 values:

- N=2: `7ea73d275ba3b928922cd68da24bf8b0c316c098d4e455050828acf8c0ecd6e5`
- N=10: `d062058c45e87c81a3c10f1f0c86523b111b90501a70f3230aa87b4c4918d980`
- N=20: `0ab7321305863c8e3c362bf80119d78ba026296800219bec23bd1b452bd08c3c`

The shared OrthoFlow3 eta has three dimensions with support
`[0.5,1.25] × [-0.5,0.5] × [0,0.75]`. The same first 16 scrambled Sobol
points (seed 20261001) apply to every state and N. A per-step Flow latent
produces a bounded action; the original goal-stop and accepted hard safety
projection run; OrthoFlow3 adds `B(x,u_safe) eta`; the same accepted safety
projection runs again before simulator execution. The robust criterion is
success in at least 15 of the prescribed 16 stochastic continuations. The
future Q input is `(physical state, eta)` only; it has no controller identity.
Physical-state splits are disjoint and determined by state ID, not rollout.
N=2/N=10 state identities and eta points were reused exactly from v2; N=20
states were fixed before any N=20 eta outcome.

The exact design and evidence are under
`datasets/gap1_eta_scaling_v3_n20/`. The key files are `design_manifest.json`,
`eta_pool.json`, `prior_result_reuse.json`, `health_audit.json`, and
`rollouts/n{2,10,20}/pair_XXXX/result.json`. Preflight reported 1,936 exact
and 2,230 partial reusable seed requests with zero ambiguous identities.
Recovery arrays completed the six indices canceled by the six-shard CPU
governor. The experiment's 1,133 shard journals were merged into the shared
rollout database as
`exp_c9f0a25a6605c942f354717b37906091806841db2fbec1ea97b3638c19d96541`.
The governor remains active with a fixed maximum of six Gap1 CPU shards.

N=20 frozen evidence and trace-linked simulator videos are documented in
`docs/gap1_n20_scaling_20261009.md`. The controller, safety projection,
eta formulation, global pool, geometry, success criterion, and B15 threshold
were not changed in response to these results. No eta generator or critic
was trained.

## Dataset status

- **N=2: DATASET_READY**, with ten numerical Q16 pairs explicitly masked
  from empirical-probability training; all 768 robust B15 labels remain
  available. Its 758 valid full-Q16 pairs and TRAIN/VAL/TEST basin coverage
  are adequate for an N=2-only study.
- **N=10: INSUFFICIENT_BASIN_COVERAGE** under the fixed K=16 pool. Four robust
  pairs occur in three TRAIN states; VAL and TEST have no positive state.
- **N=20: INSUFFICIENT_BASIN_COVERAGE** under the fixed K=16 pool. Zero robust
  pairs were found; this is a sampling/protocol result, not proof that the
  entire continuous intervention family is empty.

Overall **SCALING_DATA_NOT_READY**. Training a state-conditioned generator
or Q critic on this three-scale dataset would not test scalable basin
learning at N=10/20. Further candidate-coverage experiments would need a
separately predeclared, global state-independent protocol and untouched new
TEST physical states; the existing TEST outcomes cannot be used to tune it.
