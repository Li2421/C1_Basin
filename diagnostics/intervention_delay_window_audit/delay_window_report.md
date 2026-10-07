# Step-aligned intervention delay-window audit

## Conclusion

**TEMPORAL_TARGET_STILL_AMBIGUOUS**

The old long-horizon target is not adequate for a receding online gate, and
the data strongly reject treating every `y_long=1` state as requiring action
now.  However, this audit found no state with statistically supported loss
from a one-step delay, only one fully resolved short-delay window, and eight
states that remain temporally ambiguous after adaptive sampling.  The
urgency/window concept is the best-supported candidate, but the available
state set does not yet identify a sufficiently stable supervised temporal
target.

## Frozen cohort and semantics

- Audited states: **59** oracle-stable states.
- Result-independent generic cohort: **50**, selected as the ten lowest
  fixed SHA-256 values in each existing category-by-`y_long` stratum.
- Named hard-state anchors: **13**; four overlap the generic cohort.
- Long-horizon labels: **25 zero / 34 one**.
- Delays: `d in {0,1,2,4}` physical steps; `dt=0.05 s`.
- Initial sampling: 64 four-way matched continuation seeds per state.
- Adaptive sampling: ten unresolved states reached 128; nine of those reached
  256.  Only states with nonzero paired discordance and unresolved paired CI
  were extended.

The existing validated oracle exposes a fixed-`eta_best` continuation, not an
online eta search for arbitrary reached augmented states.  Therefore all
branches used the closest frozen oracle-consistent implementation: execute
only `u_safe` during the delay prefix, then use the audited state's fixed
`eta_best` DiagnosticCorrector and the same FlowBC and hard projections.  The
post-switch implementation is identical between delays.  This limitation
must be retained when interpreting the result; it is not a receding oracle
re-query.

## Aggregate recoverability

The primary aggregate is the equal-state macro mean.  Its interval is a
bootstrap over states; pooled continuation values are descriptive because
adaptive states have more seeds.

| Delay | State-macro Q | 95% state-bootstrap CI | Mean Delta(Q0-Qd) | 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 0.997285 | [0.995299, 0.998941] | 0 | [0, 0] |
| 1 | 0.996756 | [0.994770, 0.998477] | 0.000530 | [-0.000463, 0.001523] |
| 2 | 0.995697 | [0.993247, 0.997881] | 0.001589 | [-0.000066, 0.003509] |
| 4 | 0.991459 | [0.985567, 0.996292] | 0.005826 | [0.001324, 0.011653] |

Thus no aggregate loss is supported at one or two steps, whereas a four-step
delay has a supported population-level loss.

The pooled rollout outcomes, reported only descriptively, were:

| Delay | Success | Deadlock | Timeout | Collision | Mean duration (steps) |
|---:|---:|---:|---:|---:|---:|
| 0 | 5550/5568 | 15 | 3 | 0 | 239.22 |
| 1 | 5542/5568 | 17 | 9 | 0 | 242.73 |
| 2 | 5526/5568 | 24 | 18 | 0 | 246.13 |
| 4 | 5467/5568 | 42 | 59 | 0 | 251.73 |

Failures increasingly comprise both deadlocks and timeouts; there were no
collisions.

## State-level temporal patterns

The paired classification uses no hand-chosen `Delta_Q` magnitude.  A delay
has supported loss only when its paired bootstrap 95% interval lies above
zero.  A state is called strictly tolerant only for an exactly zero observed
paired success effect or a statistically supported improvement.  Every
nonzero unresolved difference remains ambiguous.

Across all 59 states:

- `IMMEDIATE_INTERVENTION_REQUIRED`: **0**
- `SHORT_DELAY_TOLERATED`: **1**
- `MULTI_STEP_DELAY_TOLERATED`: **50**
- `DELAY_EFFECT_AMBIGUOUS`: **8**

The single resolved short-window state was `Q_pair227_m040`: `Q0=Q1=Q2`
`=126/128=0.984375`, while `Q4=121/128=0.9453125`; paired
`Delta_4=0.0390625`, 95% CI `[0.0078125,0.078125]`.

Seven states have supported loss at `d=4`, three at `d=2`, and none at `d=1`.
This is evidence that delay length matters, but six of the longer-delay-loss
states still have a nonzero unresolved one-step effect and therefore cannot be
declared safely delay-tolerant under the preregistered rule.

## Old long-horizon label comparison

Among all **34** old `y_long=1` states:

- intervention-now required: **0/34**;
- one-step delay strictly tolerated: **28/34**; six remained ambiguous;
- two-step delay strictly tolerated: **27/34**;
- four-step delay strictly tolerated: **25/34**.

In the result-independent generic cohort alone, the corresponding counts are
`0/30`, `25/30`, `24/30`, and `22/30`.

All 25 `y_long=0` states were invariant across delay branches because their
validated `eta_best` is exactly zero.  No reverse case (`y_long=0` with a
supported one-step loss) occurred.

These results directly show that `y_long=1` is not an instantaneous
intervention label.

## Previous hard 13

- **12/13** are `MULTI_STEP_DELAY_TOLERATED`.
- **1/13** is `DELAY_EFFECT_AMBIGUOUS`.
- **0/13** require immediate intervention.
- **0/13** have a fully resolved short-delay window.

The sole ambiguous anchor is
`RBV_Q_pair228_m080_s95401001_p018`: at 256 matched seeds,
`Q0=1`, `Q1=Q2=0.98828125`, and `Q4=0.984375`.  The one- and two-step paired
effects remain ambiguous (`Delta=0.01171875`, CI `[0,0.02734375]`), while the
four-step loss is supported (`Delta=0.015625`, CI
`[0.00390625,0.03125]`).

## Target assessment

- **Target A, old long-horizon binary:** contradicted by the delay audit.
- **Target B, current-step necessity:** semantically aligned with online
  control, but this cohort produced zero statistically supported positive
  states, so it does not yet define a trainable binary problem.
- **Target C, urgency/delay window:** the best candidate because the aggregate
  and several individual effects grow with delay.  Nevertheless, only one
  state has a fully resolved short window and eight remain ambiguous, so the
  evidence is not yet sufficient for `URGENCY_WINDOW_TARGET_SUPPORTED` under
  the stated acceptance rule.

## Smallest justified next step

No gate-training run is justified from these labels yet: the current-step
binary target has zero positive examples, and a resolved urgency target has
only one short-window example.  The smallest next experiment is to apply this
same frozen `d={0,1,2,4}` matched-delay protocol to the remaining existing
oracle-stable pool using the same result-independent hash-stratified rule—no
new states and no learned control.  Only if that produces adequate resolved
immediate/short-window examples should the next training experiment be a
small source-group-held-out offline urgency probe, with ambiguous states
excluded from hard supervision.

