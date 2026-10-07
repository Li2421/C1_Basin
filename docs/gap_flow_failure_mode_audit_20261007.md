# Frozen Gap1 Flow failure-mode audit (2026-10-07)

## Scientific decision

**B — CURRENT FLOW IS NOT YET A CLEAN DEADLOCK BASELINE at N=2, 10, or 50.**
The reported nominal test pipeline has **no safety projection**. All 32 test
episodes terminate at a swept collision, not a deadlock or timeout. N=2 and
N=10 strike walls; N=50 usually collides with another robot before reaching
the gap. Conflict-free controls using the same frozen checkpoints and plant
also fail. Thus 0% success is currently evidence of insufficient safe
navigation, not a certified coordination deadlock. This audit did not retrain
or modify any Flow checkpoint.

## Exact provenance and pipeline

The published table uses `bottleneck_family/evaluate_flow.py` with the test
nominal archives, `--split test --final-evaluation --seed 17
--samples-per-step 16`, **without `--safety`**. The dataset comes from
`bottleneck_family/flow_dataset.py`, with train/dev/test episode seeds drawn
independently from root seed 20261007. Its expert is the sequential grid A*
planner, but that planner is absent at nominal inference. Separate
`flow_guided_sequential_route_v4` test success (12/12, 12/12, 8/8) uses
explicit route coordination and is not a Flow-only result.

| N | Dataset / test trace root below `diagnostics/gap_flow_v1/` | checkpoint SHA256 | selected train step | train/dev/test nominal | timeout limit |
|---:|---|---|---:|---:|---:|
| 2 | `n2_recovery_wide` | `a321b6fad7a607ee379db4daccb4f20707988a5ed5085c797abb2673a507cc91` | 9500 | 48/12/12 | 2000 |
| 10 | `n10_wide_recovery` | `1e078260be745a29806c8661fac0f1acdd0dba64a7439042882cf6a8e6acaeb4` | 10000 | 48/12/12 | 8000 |
| 50 | `n50_wide_recovery` | `ad22605d71bb2732ebf99f05707f9d56c466d751d3e4488d7d0e16a292471480` | 7000 | 24/8/8 | 40000 |

For each N, `train/best.pkl` is a distinct fixed-size flattened **joint**
MACFlow: input `[N,8]`, output `[N,2]`, normalized observations/actions,
three 256-unit hidden layers, 10 Euler flow steps. At each physical step the
evaluator draws 16 joint actions with a step-keyed JAX RNG, radially bounds
each, averages them, and bounds the average at 0.5 m/s. The 8 features per
agent are position, last velocity, goal displacement, and a public static
gap-waypoint displacement (`bottleneck_family/observation.py`). The model
sees all agent rows jointly, but there is no explicit occupancy reservation,
yield rule, or route planner at inference. The gap is 0.62 m wide; the agent
radius is 0.16 m, `dt=0.05 s`, goal tolerance 0.08 m.

`BottleneckEnv.step()` applies single-integrator motion and checks **swept**
wall and pair clearance. The first collision terminates the episode;
otherwise all robots must reach their goals before `max_steps`. The optional
`stalled_groups()` diagnostic is *not called* (`diagnose_stalls=False`) by
this evaluator and would only be a stall candidate even if called. Timeout is
not deadlock. `HardSafetyFilter` is only constructed for `--safety`, absent
in the published test run. Existing *dev* safety-projected runs are a
different chain: N=2 has 4/12 success and 8 collision-free timeouts; N=10
has 0/12 success and 12 collision-free timeouts; N=50 was not evaluated.
Those timeouts have not been adjudicated as genuine deadlocks.

## All published nominal test failures

The case-level audit uses the stored simulator position at **every** step and
its saved continuous-motion swept clearance. It independently recomputes
the final swept wall/pair clearance. Categories are mutually exclusive:
every collision is C, even if poor navigation is its underlying cause.
For a measurable opposing encounter, an opposite-direction pair must be
within 0.8 m of one another while both centers satisfy `|x|<1.5 m` near
the barrier; entering the gate region means `|x|<1.5 m, |y|<1 m`.

| N | runs | A: genuine no-progress deadlock | B: navigation-only terminal | C: swept collision / safety failure | D: other | wall / agent / both | gate-region reached | opposing gate encounter |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 2 | 12 | 0 | 0 | 12 | 0 | 12 / 0 / 0 | 12/12 | 0/12 |
| 10 | 12 | 0 | 0 | 12 | 0 | 12 / 0 / 0 | 8/12 | 0/12 |
| 50 | 8 | 0 | 0 | 8 | 0 | 1 / 6 / 1 | 0/8 | 0/8 |

| N | median terminal step (seconds) | median mean-agent goal progress before termination | median closest agent distance to gate center | median mean-agent speed in final fifth | final-fifth low-speed episode fraction |
|---:|---:|---:|---:|---:|---:|
| 2 | 238 (11.90 s) | 2.867 m | 0.247 m | 0.250 m/s | 0 |
| 10 | 180 (9.00 s) | 0.421 m | 0.603 m | 0.061 m/s | 0 |
| 50 | 17 (0.85 s) | 0.006 m | 3.146 m | 0.092 m/s | 0 |

The last column is the fraction of final-fifth steps with *mean* speed below
0.05 m/s, then median over episodes. It is not a deadlock predicate. N=2
does make goal-directed progress toward the gate, but strikes the wall while
still moving; the two agents do not meet at the gap under the stated
encounter definition. N=10 is mixed in approach quality but also terminates
at walls, not at a persistent standstill. N=50 fails in the starting room.
All 32 case-level rows include initial/final/minimum goal distance, total
travel, gate proximity, pre/post encounter fields, continuous clearance,
speed, and progress diagnostics in `diagnostics/gap_flow_audit_20261007/`.

## Frozen Flow competence controls on the test initial states

All controls retain the **same N-specific checkpoint**, 8-feature
observation, 16-sample action mean, no safety projection, plant, map,
dynamics, collision detector, and all-agent success condition. All planned
tuples were sent through `shared_rollout_db` preflight before execution.
`one_way` reuses the existing start and goal point sets, reversing each odd
agent so everybody traverses the gap left-to-right. `temporal_*` preserves
*exactly* the original starts/goals and externally holds one direction at
its start until the first group reaches all its goals; it adds no route
command to Flow. `park_opposite_*` changes the parked group's goals to its
own starts and holds it still. At N=2, this is the closest fixed-shape
single-moving-agent control possible without changing the checkpoint's
required two-row input; for larger N it is a one-direction group control.
These interventions alter task composition, so they are competence probes,
not matched estimates of the original benchmark's treatment effect.

| Control | N=2 success / runs | N=10 | N=50 | N=2 collision / timeout |
|---|---:|---:|---:|---:|
| One-way | 0/12 | 0/12 | 0/8 | 12 / 0 |
| Temporal, even side first | 0/12 | 0/12 | 0/8 | 9 / 3 |
| Temporal, odd side first | 0/12 | 0/12 | 0/8 | 1 / 11 |
| Park opposite, even traverses | 1/12 | 0/12 | 0/8 | 11 / 0 |
| Park opposite, odd traverses | 0/12 | 0/12 | 0/8 | 12 / 0 |

The temporal controls complete their first moving group only **1/12**
(even first) or **0/12** (odd first) at N=2; no N=10 or N=50 first group
completes. In N=2 parked-group controls, 23/24 trials fail; all 23 terminal
collisions are with walls, so an opposing-moving-agent encounter is not
needed to cause the failure. In the N=2 one-way control, 11/12 collisions
are walls; the remaining one is an agent collision.

The contrast requested in Task 3 cannot be quantified as a *pre- versus
post-interaction* change, because **none** of the original test traces has
an observed opposing gate encounter. Instead, the comparison shows that
N=2 nominal median goal progress is 2.867 m before wall collision, while
the same checkpoint's one-way and parked single-traversal medians are
1.699 m and 2.462/2.709 m before failure. For N=10, median original
progress is 0.421 m and one-way progress is 0.078 m; for N=50, original
progress is only 0.006 m and all failures are essentially pre-interaction.
The N=2 `temporal_odd_first` trials are especially diagnostic: no moving
agent enters the gate region in 12/12 trials; 11/12 time out, with median
final-fifth mean speed 0.009 m/s and median total goal progress 0.185 m.
Low speed can therefore occur **without** adversarial gate occupation. The
reported nominal chain has no projection to measure; no projection-to-stall
causal claim is made.

## Eta eligibility and interpretation

There is **no qualifying deadlocked nominal test state** to query: 0/32
nominal failures are collision-free persistent no-progress deadlocks. The
count of states with a robust successful eta is therefore **not evaluated**;
the fraction among deadlocked states has denominator zero and is undefined,
not 0%. The existing generic three-gain P0 corrector and hard projection
exist in `shared_control/`, and the current B15 robust convention is at
least 15/16 matched seeds. However, the runnable
`new_benchmark_common/safety_eta3.py` scenario registry currently contains
Four-Way and Ring only. Applying P0 or OrthoFlow3 to these Gap1 raw
collision trajectories would introduce the two-projection safety chain and
new scenario plumbing; it would test a **different baseline**. We did not
redesign eta, target failure regions, or label explicit path coordination
as an eta intervention. Its 12/12, 12/12, 8/8 success proves these tested
joint tasks are feasible under a stronger controller, not that an existing
low-dimensional eta resolves Flow deadlock.

## Evidence and next experimental gate

`diagnostics/gap_flow_audit_20261007/videos/manifest.json` records five
full initial-to-terminal MP4s with frame count, SHA256 and source trace:
N=2/N=10 wall collisions, N=50 early agent collision, N=2 one-way collision,
and an N=2 conflict-free temporal timeout. There is **no honest deadlock
video** from this nominal Flow test set. These audit videos are not the
formal four-scene batch video delivery.

Before this can support a paper claim about coordination deadlock, a
frozen evaluation chain must first pass same-gap single-agent and one-way
controls at high rate while remaining collision-free. If standalone Flow is
the desired chain, retraining or changing its navigation/safety controller
is necessary; the checkpoint should remain frozen until a separate design
decision. A safety-projected Flow is a possible *different* chain, but its
timeout traces need full gate-interaction and persistent-stall adjudication,
plus the same competence controls, before being called deadlock.
