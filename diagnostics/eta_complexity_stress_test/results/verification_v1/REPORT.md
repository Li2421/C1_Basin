# Coupled Dual-Intersection eta stress test

## Scope

This isolated diagnostic uses no MAC Flow/Flow-BC checkpoint, neural nominal
policy, demonstration collection, or training.  The nominal controller follows
fixed geometric waypoints independently for all eight agents.  The tested law
is the repository's unchanged shared 3-D correction:

```text
g = eta_goal * B_goal + eta_safe * u_safe + eta_relative * B_rel
u_exec = hard_project(u_safe + g)
```

`B_goal` is bounded goal displacement; `B_rel` is the all-other-agents,
pairwise-bound-then-mean basis; both hard projections are the existing generic
N-agent projector.  The search domain was `{'lower': [0.0, -1.0, -1.0], 'upper': [1.5, 1.0, 1.0]}`.

## Environment and solvability gate

The fixed corpus has 96 initial conditions and SHA-256
`33c042f73a63af6678889e3e507b7386f1e76ba32377b8d20fcbf9fd22ecc284`.  The explicit centralized scheduler solved
96/96 (1.000)
using its selected phase ordering.  The oracle is not used by nominal, safety,
or eta rollouts.

## Baselines

| controller | successes / ICs |
|---|---:|
| analytic nominal | 0 / 96 |
| hard safety only | 93 / 96 |

## Eta existence and basin estimate

The global Sobol screen tested 0 eta values
per IC.  Empirical nonempty success sets occurred for
0 / 96 ICs.  The
median sampled success fraction was nan.
These are finite-sample estimates, not proofs that an unobserved eta set is
empty.

## Verdict

**ETA CAPACITY: PENDING_SEARCH**

**ETA SOLVING DIFFICULTY: PENDING_SEARCH**

The capacity label separates oracle-verified physical solvability from sampled
eta existence.  The difficulty label comes from independent fixed-budget
uniform, Sobol, and diagonal-CEM searches and is only reported for
representative ICs with empirical eta success.
