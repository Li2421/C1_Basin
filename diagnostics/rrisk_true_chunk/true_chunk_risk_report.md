# True persistent-correction deadlock-risk geometry

## Scope and frozen invariants

This diagnostic asks only whether a constant joint correction, applied for a
short physical chunk and then removed, changes the frozen true continuation
event probability. It did not train `G_phi` or `R_psi`, construct or tune a new
surrogate, change Flow-BC, change the environment or event/latch semantics, or
soften either hard projection. Arm construction and all gates used only true
`D_H` continuation outcomes.

For state `z`, constant correction `alpha` and chunk length `L`, the measured
object is

```text
Q_D(z, alpha; L) = P(D_H = 1 | alpha is applied for L physical steps,
                                 then the frozen baseline resumes).
```

`D_H` remains `historical_strict_deadlock OR stalled_deadlock`. First-event
termination remains enabled. The timestep is 0.05 s, maximum episode length is
850 steps, and per-agent speed is limited to 0.5 m/s.

## Reconstruction of old TEST Q

The old intervention perturbed `action_offsets[100]`, a joint 4-D residual
added to `u_safe + r` before the second hard projection. Frozen `G_phi` was
zero, so the diagnostic offset was the only residual. The executed velocity was
`u_exec = Pi_2(u_safe + offset)`. The offset had norm 0.2 and lasted one
physical step (0.05 s); the continuation then used the unchanged frozen
baseline. Four frozen random unit directions, their negatives, and baseline
gave nine arms for each of trajectory IDs 0, 2, and 3. Each arm had eight
continuations, for 216 rollouts total.

All 27 arms produced 8/8 `D_H`. All 108 score-different action pairs remained
physically distinguishable after projection, so the old failure was not mass
aliasing. At action 100 the states were alive and not latched. In the existing
reference traces they were 197, 130, and 261 steps (9.85, 6.50, and 13.05 s)
before the eventual reference event. Thus they were not uniformly adjacent to
the event. A single norm-0.2 step can change position by at most about 0.01 m,
so lack of temporal commitment was a hypothesis, not a reinterpretation of the
old result. The full frozen reconstruction is in
[`old_test_q_reconstruction.md`](old_test_q_reconstruction.md).

## Exact chunk definition

For action indices `t,...,t+L-1`, with joint flattening
`alpha=[u_0x,u_0y,u_1x,u_1y]`, the diagnostic sets

```text
r[t+j] = alpha in R^4,       j = 0,...,L-1
w[t+j] = u_safe[t+j] + alpha
u_exec[t+j] = Pi_2(x[t+j])(w[t+j]).
```

After the chunk, `alpha=0` and the frozen policy continues. The first
intervened action latent and the complete prior state/history are replayed from
the old reference trace; fresh continuation randomness starts at `t+1`. At
`t=100,L=1` this exactly matches the old TEST-Q split semantics. Every rollout
checks the replayed state against the archived reference state to `2e-10` in
max norm. The frozen initialized `G_phi` output is also recorded and checked.
The constant chunk was already discriminative, so the optional tapered chunk
was not introduced.

`Pi_2` is the existing joint Euclidean projection in R4. It enforces the joint
agent-pair CBF inequality, all per-agent finite-wall CBF inequalities, and the
two per-agent Euclidean speed balls. The unchanged CBF configuration is
`gamma=gamma_wall=1`, separation buffer `1e-4`, feasibility tolerance `1e-9`,
KKT optimality tolerance `2e-6`, speed tolerance `1e-10`, SLSQP `ftol=1e-12`,
at most 200 iterations and 32 adaptive speed-ball cuts.

At every live chunk step the raw requested correction, pre-`Pi_2` command,
executed command, executed correction relative to `u_safe`, removed vector,
removal fraction, active safety/speed sets, and active-set switching are saved.

## Budget normalization

The primary across-horizon comparison fixes requested chunk energy:

```text
||alpha|| = 0.2 / sqrt(L),     sum_j ||alpha||^2 = 0.04.
```

Two predeclared per-step families test a local dead zone without a broad sweep:
`||alpha||=0.1` and `||alpha||=0.2`. Their requested energies are respectively
`0.01 L` and `0.04 L`. Arms are the eight symmetric coordinate corrections
`+/- delta e_i`, `i=0,...,3`, plus baseline. At `L=1`, the energy-matched and
fixed-0.2 arms are identical and were run once. Results always report both
per-step norm and total requested/executed chunk energy; conclusions do not
hide the different budget conventions. First-event termination can truncate a
planned chunk; raw records therefore also retain live chunk steps and live
requested energy rather than pretending a post-terminal correction executed.

## State selection

Inclusion rules and all nine states were frozen before new outcomes:

- **Early:** old TEST-Q action 100, with latch and candidate false.
- **Emerging:** immediately before the first candidate-deadlock action in the
  archived reference trace.
- **Late:** ten action indices (0.5 s) before the archived event, still alive
  and unlatched.

The pilot used `rid0_early_t100`, `rid2_emerging_t130`, and
`rid3_late_t351`. The gated expansion used the remaining six combinations:
`rid2_early_t100`, `rid3_early_t100`, `rid0_emerging_t197`,
`rid3_emerging_t261`, `rid0_late_t287`, and `rid2_late_t220`. No state was
chosen after observing which correction worked.

## Horizon selection and pilot gates

The predeclared pilot horizons were `L={1,10,20}`, corresponding to
`{0.05,0.50,1.00}` s. Every arm initially received eight common-seed
continuations. Exact two-sided Clopper-Pearson intervals are reported per arm.
Within each state/horizon, planned baseline-versus-arm and matched `-` versus
`+` contrasts use two-sided Fisher tests with Holm family-wise correction.
Pilot discrimination requires both `|Delta Q_D| >= 0.5` and adjusted
`p <= 0.05`.

The pilot was decisive at `L=10`, which triggered the predeclared six-state
expansion, `L=5` refinement, 24-seed independent replication, and—only after
replication succeeded—the true-gradient validation. `L=40` was not triggered.
The optional old-score comparison was not used: neither old score defined an
arm or gate, and establishing the true geometry did not require reopening or
tuning either failed surrogate.

## True Q_D results

The pilot comprised 201 arm scenarios and 1,608 continuations. At `L=1`, all
51 state/arm scenarios (408 continuations) were 8/8 deadlock, including both
norm-0.1 and norm-0.2 coordinate corrections. This agrees with the old
norm-0.2 random-direction result and supplies no evidence for action-local
controllability.

At `rid0_early_t100`, `L=10`, the baseline was 8/8 deadlock while the
predeclared norm-0.1 `+e_1` arm was 0/8. Its baseline contrast had
`Delta Q_D=-1`, raw exact `p=1.554e-4`, and within-cell Holm-adjusted
`p=0.00559`. All eight corrected pilot outcomes were successful terminations,
with no collision or timeout. The same arm was then selected by the frozen
gate—not manually—and repeated on 24 fresh seeds: baseline was 24/24 deadlock
(`Q_D=1`, exact 95% CI `[0.858,1]`), while the correction was 1/24
(`Q_D=0.0417`, CI `[0.00105,0.211]`). The independently replicated effect was
`-0.9583`, Holm-adjusted `p=3.10e-12`; 23/24 corrected runs ended in success.

TODO_FULL_MAP

## First control-discriminative horizon

TODO_FIRST_HORIZON

## Projection and physical-distinctness audit

Projection clipping does not explain the validated contrast. Across nonzero
pilot arms, the median arm-level removal fraction was 0.614, below the
predeclared 0.8 “mostly removed” threshold. In the 24-seed replication, the
successful norm-0.1 `+e_1` arm had mean removal fraction 0.329, mean executed
per-step correction norm 0.0756, and mean 2.96 active-set switches over the
ten-step chunk. The unsuccessful matched-norm `+e_3` arm was also physically
distinct (mean executed norm 0.0296). In the 32-seed directional validation,
every nonbaseline arm had a strictly nonzero cumulative executed correction;
the least-clipped random arm had mean removal 0.168 and still produced 30/32
deadlock. Thus different true outcomes cannot be attributed to corrections
being projected to a common action or to a generic “more correction survived”
effect.

All per-step active sets and switches are in the raw records. Reference-state
replay error was zero for the validation sets, and the largest recorded frozen
`G_phi` numerical residual was approximately `1.4e-17`.

## Temporal controllability analysis

TODO_TEMPORAL

## Local true-risk direction and independent validation

No gradient was estimated while risk was saturated. After pilot
discrimination and independent replication, the four coordinate-wise central
differences at the selected state/horizon defined

```text
g_TRUE[i] = (Q_D(+delta e_i) - Q_D(-delta e_i)) / (2 delta).
```

No differentiation through the environment, projection, event, or historical
latch was used. The baseline, `-g_TRUE`, `+g_TRUE`, and four predeclared random
matched-norm directions were then evaluated with 32 fresh continuation seeds.

At `rid0_early_t100`, `L=10`, and `delta=0.1`, the pilot coordinate estimates
were

```text
g_TRUE = [0, -5, 0, 0],       ||g_TRUE|| = 5,
```

so the matched-norm descent intervention was `-g_TRUE/||g_TRUE|| = +e_1`.
On 32 fresh seeds, baseline was 32/32 deadlock (`Q_D=1`, exact 95% CI
`[0.891,1]`), `-g_TRUE` was 1/32 (`Q_D=0.03125`, CI
`[0.00079,0.162]`), and `+g_TRUE` was 32/32. The Holm-adjusted exact comparison
of `-g_TRUE` with baseline had `p=2.16e-16` and effect `Delta Q_D=-0.96875`.
The four predeclared random norm-0.1 directions yielded 32/32, 32/32, 30/32,
and 32/32 deadlock. Thus the descent direction reduced frozen `D_H` on fresh
rollouts and was not reproduced by generic matched-budget perturbation.

The strict three-term inequality `Q_D(-g)<Q_D(0)<Q_D(+g)` cannot hold because
baseline and `+g` are both saturated at one; the supported ordering is
`Q_D(-g) << Q_D(0) = Q_D(+g)`. This is sufficient evidence for a descent
direction at finite scale `delta=0.1`, not for an infinitesimal derivative,
local smoothness, or a two-sided linear model. The correction
was physically executed: for `-g_TRUE`, the mean executed per-step correction
norm was 0.0744, mean requested fraction removed was 0.341, and all 32 runs had
cumulative executed-correction norm at least 0.583. A minimally clipped random
direction (mean removal 0.168) still gave 30/32 deadlock, so the directional
result is not explained simply by projection throughput.

## Statistical and geometric limitations

- Eight continuations per exploratory arm only identify large effects; the
  selected effect therefore receives 24-seed replication and 32-seed
  directional validation.
- Coordinate arms characterize a local four-dimensional cross, not the full
  action ball or a globally smooth risk surface.
- A useful direction at one state need not transfer to every trajectory or
  severity stratum. The temporal map is descriptive outside the independently
  validated cell.
- Projection removal and active-set switching make requested and executed
  correction geometry different; both are retained in raw records.
- The experiment establishes controllability under a short open-loop constant
  commitment. It does not establish that any old analytic score represents
  that geometry.

## Final decision

**CASE B — SHORT-HORIZON CONTROLLABILITY EXISTS.** One-step true risk is
saturated under the tested coordinate corrections, whereas a constant
norm-0.1 correction held for `L=10` (0.5 s) produced a large, independently
replicated change in frozen `D_H`, and its finite-difference descent direction
again reduced `D_H` on a second fresh validation set. This is evidence that the
give-way failure is a persistent coordination/commitment phenomenon rather
than an action-local one. It does not rehabilitate either old instantaneous
surrogate.

## Implication for R_risk

- **Is deadlock controllable at one-step action scale?** No evidence of it:
  `L=1` is saturated for all tested norm-0.1/norm-0.2 coordinate arms, agreeing
  with the old random-direction TEST Q.
- **Is it controllable at a short persistent-correction scale?** Yes. A
  constant correction held for 0.5 s changes true continuation risk from near
  one to near zero on fresh seeds.
- **What is the smallest useful horizon?** The smallest independently
  validated horizon is `L=10` (0.5 s). `L=5` is reported separately as a
  predeclared exploratory boundary and is not allowed to retroactively replace
  the validation cell.
- **Is there a useful local true-risk gradient?** There is a useful
  coordinate finite-difference descent direction at the validated
  state/horizon and `delta=0.1`; it strongly reduced fresh-rollout `D_H`. An
  infinitesimal gradient at `alpha=0` was not established, and global
  smoothness should not be assumed.
- **What should the next R_risk approximate?** A horizon- and budget-explicit
  chunk object such as `Q_D(z,alpha;L)` (or its controlled smoothing/ranking),
  not an instantaneous action score. It must represent persistent post-`Pi_2`
  commitment and state-dependent directional geometry.
- **Is there enough evidence to justify redesigning R_risk and eventually
  training G_phi?** Yes for a future, separately specified short-horizon risk
  redesign and subsequent training experiment. No training is performed or
  authorized by this diagnostic.

## Reproducibility artifacts

The frozen protocol is [`predeclared_protocol.json`](predeclared_protocol.json),
machine-readable outcomes and analyses are under `data_v1/`, the final manifest
is [`manifest.json`](manifest.json), and requested plots/tables are under
`figures/` and `tables/`. Raw records contain seed-level outcomes and complete
per-step chunk/projection diagnostics.
