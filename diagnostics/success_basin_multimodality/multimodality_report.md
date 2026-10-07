# SBMA — Success-Basin Multimodality Audit

## Final classification

**CASE B — SINGLE CONNECTED SUCCESS BASIN SUPPORTED**

Within the tested three-dimensional diagnostic eta region and the fixed failing states D1/D2/D4, the evidence supports one broad connected success region. It does **not** support a structural need for a multimodal generative correction representation.

This is finite empirical support, not a proof that the full unbounded success set is mathematically connected.

## Frozen semantics

The parameter is denoted `eta=(eta_goal, eta_safe, eta_relative)`; it is not a future neural-network parameter `theta`. The deterministic state-dependent diagnostic corrector is

`G_eta(z) = eta_goal bounded(goal-position) + eta_safe u_safe + eta_relative bounded(self-other)`.

It outputs the joint four-dimensional residual. The same eta remains active through the complete continuation and `G_eta(z_k)` is recomputed at every physical step. Each transition applies the original stochastic Flow policy, first exact hard projection, residual addition, second exact hard projection, and frozen environment. No one-step or short open-loop interpretation was used.

Outcomes remain COLLISION, SUCCESS, STRICT_DEADLOCK and TIMEOUT. Stalled deadlock was not reintroduced. No G, risk function, certificate, Q critic, or neural network was trained.

## Predeclared criterion and design

A SUCCESS_CELL required no numerical UNKNOWN, at least 16 trials, and a one-sided exact 95% lower confidence bound `LCB(Q_S) >= 0.80`. A FAILURE_CELL required the corresponding upper bound `UCB(Q_S) <= 0.20`. All other cells were UNKNOWN_CELL. This rule was frozen before Phase-A outcomes.

Phase A sampled a structured 3-D lattice:

- goal gain: 0.5, 0.75, 1.0, 1.25;
- safe gain: -0.5, -0.25, 0, 0.25, 0.5;
- relative gain: 0, 0.25, 0.5, 0.75;
- 80 eta points per state and 16 fresh seeds per point;
- D1, D2 and D4, for 3,840 continuations.

This contains three structured slice families rather than one arbitrary `safe=0` plane. Axis-neighbor adjacency was fixed for the graph.

After resolving two numerical attempts without altering their `(state,eta,seed)`, Phase A contained 2,953 successes, 530 strict deadlocks, 357 timeouts, zero collisions and zero numerical UNKNOWN outcomes.

## 3-D connected components

| State | SUCCESS cells | FAILURE cells | Statistical UNKNOWN cells | SUCCESS components |
|---|---:|---:|---:|---:|
| D1 | 48 | 18 | 14 | 1 |
| D2 | 52 | 4 | 24 | 1 |
| D4 | 56 | 8 | 16 | 1 |

There are 48 high-confidence success cells common to all three states. The established point `eta=(1,0,0.25)` lies in each component. The remaining UNKNOWN cells are cells that did not cross either confidence rule; they are not silently treated as failures. No numerical UNKNOWN remains in the final map.

Grid connectivity alone is not the main conclusion. It motivated the stronger path test below.

## Decisive path and averaging test

Before Phase-B outcomes, the two points with maximum Euclidean separation in the three-state common SUCCESS set were selected deterministically:

`eta_A=(0.75,-0.5,0.75)` and `eta_B=(1.25,0.5,0)`, distance `1.3463`.

For each state, the straight interpolation at `lambda=0,0.1,...,1` was evaluated with 32 fresh seeds per point. After same-input numerical reruns for two initially rejected projection calls, the result was:

| State | Path trials | Success | Deadlock | Timeout | Collision |
|---|---:|---:|---:|---:|---:|
| D1 | 352 | 352 | 0 | 0 | 0 |
| D2 | 352 | 352 | 0 | 0 | 0 |
| D4 | 352 | 352 | 0 | 0 | 0 |

Every one of the 33 state/path cells is a SUCCESS_CELL; 32/32 gives a one-sided 95% lower bound of 0.9106. Therefore:

**SUCCESS_PATH_FOUND**

The midpoint and all other tested deterministic averages remained successful. This is direct evidence against the claim that averaging these distant successful policies leaves the success set. Because the straight path succeeded, the predeclared gate did not authorize extra graph-search rollout.

## Fresh validation and cross-state consistency

The known eta received 32 new seeds on each state: 96/96 SUCCESS, with no deadlock, timeout, collision or numerical UNKNOWN. Both distant endpoints were also 32/32 SUCCESS on every state. Their mean completion times differed substantially: eta_A approximately 17.3 seconds and eta_B approximately 9.3 seconds, but both were genuine task completion.

The same 48 coarse cells and the entire selected straight path succeeded across D1/D2/D4. This supports recurring local success geometry that a state-conditioned controller could plausibly represent. The three states are closely related, outcome-selected failure states, so this does not establish population-level generalization.

## Local empirical basin margin

Around `(1,0,0.25)`, six coordinate directions were tested at delta 0.125, 0.375, 0.625 and 0.875, with 16 new seeds per point. All three states produced the same confidence classification:

| Direction | Consecutive high-confidence lower margin | First delta not meeting SUCCESS_CELL |
|---|---:|---:|
| +goal | >=0.875 | not reached |
| -goal | 0.125 | 0.375 |
| +safe | 0.375 | 0.625 |
| -safe | >=0.875 | not reached |
| +relative | >=0.875 | not reached |
| -relative | 0.625 | 0.875 |

These are directional sampled margins, not formal robustness radii. “Not meeting” includes mixed outcome probability and is not equivalent to certain failure.

## Behavioral modes

The endpoints show continuous quantitative differences but the same measured coordination identity. At eta_A, median completion was about 17.3 s, agent0 reached lateral y about 0.491, and ordering changed near 3.1 s. At eta_B, completion was about 9.3 s, agent0 reached y about 0.305, and ordering changed near 2.6–2.7 s. In both cases the same agent made the larger lateral excursion and the agents changed longitudinal ordering before success. All second projections were active along these representative rollouts.

Thus the experiment finds different speeds and path amplitudes inside one connected success region, not two independently established behavioral modes.

## Projection audit and validity

All 754 previous SBGA UNKNOWN records were classified NUMERICAL_SOLVER_FAILURE: their exact mathematical SOCPs were feasible. The final SBMA used a Clarabel exact SOCP plus tightly certified equivalent fallbacks only when necessary; constraints and objectives were unchanged. Four accepted fallback actions occurred. Invalid attempts remain preserved with null outcomes and were never re-labelled.

Across 6,149 raw attempts (including five retained invalid attempts), 1,585,111 physical steps were recorded. The final outcome-bearing dataset has 6,144 continuations: 5,058 success, 599 strict deadlock, 487 timeout and zero collision. These aggregate counts include deliberate margin probes outside the central success region and should not be interpreted as a policy performance estimate.

Read-only verification checked all file hashes and all trajectories' integration, corrector and terminal bookkeeping. Maximum integration residual was `1.11e-16`, maximum corrector residual zero, 18 representative episodes were fully replayed, and 108 hard projections were recomputed. Old diagnostic artifacts and frozen source/config hashes were unchanged.

## Scientific conclusion and limits

The strongest supported statement is:

> In the explored eta box, D1/D2/D4 share a broad empirical success region, and two maximally separated sampled representatives are joined by a freshly validated straight success path.

Consequently the current evidence does not justify a multimodal correction architecture on topological grounds. It does not prove global connectedness, rule out distant success components outside the tested box, establish behavior on unrelated initial states, or identify a future R_risk. Those questions remain separate.
