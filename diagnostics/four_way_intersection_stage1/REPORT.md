# Four-Way Intersection — Stage-I report

**Decision: `PASS_STAGE1`**. The frozen v13 test is the terminal evaluation;
no data, checkpoint, sampling rule, or geometry was changed after it.

## 1. Scenario and liveness mechanism

Agents use the canonical order `A, B, C, D`: North→South, East→West,
South→North, West→East. They are disks in an open square (`half extent=4.75`,
radius `0.20`) and all cross a shared *open* central area. The central area has
no gate, lane, signal, priority variable, or fixed right-of-way. A collision
ends an episode; it is not used to prescribe an order.

This is cyclic multi-agent crossing coordination, not a narrow-resource
Give-Way variant: agents can occupy the central square concurrently when their
paths permit it, but local yielding can form cyclic waits, simultaneous
attempts, and oscillatory release/crossing decisions. The outer buffer was
expanded only to eliminate incidental reset-near-wall failures; starts, goals,
and the central conflict geometry were unchanged. The pilot's minimum reset
outer clearance was 0.5505 m.

Observation is four rows of 18 world-frame physical values: own position,
velocity, goal displacement, then every other agent's relative position and
velocity. Thus all agents, goals, and fixed square geometry are recoverable;
there is no crossing-order feature.

## 2. Centralized expert and multimodality

The centralized expert enumerates crossing-order hypotheses internally and
realizes each with continuous velocity control. A successor is released after
the predecessor clears the central conflict region, rather than after it
reaches a remote goal. Agents already in goal tolerance are permanently held
in recovery plans. Neither mechanism is an environment rule or deployed-policy
input.

The outer-buffer pilot passed Gate A/B: 30/30 expert successes, zero
collisions, and 16 distinct realized orders. The same representative initial
state admits `A→C→B→D`, `B→D→A→C`, and `A→B→C→D` expert solutions. See
[`expert_pilot_outer_buffer.json`](expert_pilot_outer_buffer.json) and
[`expert_pilot_outer_buffer.svg`](expert_pilot_outer_buffer.svg).

## 3. Initial states and dataset

Each independent draw perturbs all four approach longitudinal distances
(±0.55), lateral offsets (±0.16), a correlated lateral asymmetry (±0.06), and
initial world velocity components (±0.10). Named RNG streams are disjoint for
train, development, and test.

The final v13 root is
[`base_u_v13_broad_global_dataset`](base_u_v13_broad_global_dataset). It
preserves v11 *training* recovery coverage only, adds independent global
nominal draws (120 train, 30 dev, 60 new test) and 8 uniform anchors per new
train/dev nominal trajectory. Old dev/test were not copied or opened. The
manifest records 180 train nominal starts, 30 dev nominal starts, and 60 sealed
test nominal starts; 4,799 uniform-recovery trajectories; 1,010 targeted-wall
trajectories; and 9 targeted-agent trajectories.

The final trainer loaded 94,143 nominal transitions, 887,999 uniform-recovery
transitions, 180,707 targeted wall transitions, and 1,473 targeted-agent
transitions from the train split. These are transition counts; the trajectory
counts above remain the acquisition/provenance counts.

Uniform recovery uses the common local perturbation distribution (position SD
0.075, velocity SD 0.055) and centralized expert re-query. Targeted wall data
came only from train/dev wall precursors; each audit record names its source
rollout/time, wall identity/distance where available, seed, and expert result.
Targeted agent data were introduced only after walls were resolved: 9 valid
pre-contact train/dev precursor continuations with pair, offset, distance,
seed, and expert result. Invalid collided/inside-wall states were never used.

The early Base-U versus Base-U+W check was intentionally retained. Under the
same development rollout seed (113), outer-buffer Base-U v5 had 0/20 success,
80% wall collision, 15% agent collision, and 5% timeout; the matched dense
wall-recovery model had 0/20 success, 60% wall collision, 15% agent collision,
and 25% timeout. Thus wall recovery reduced its named failure by 20 points but
did not solve task completion, so it was not credited as the final mechanism.
The collision locations are shown in
[`targeted_wall_ablation_before_after.svg`](targeted_wall_ablation_before_after.svg).
The later goal-hold, generic policy-rollout recovery, and broad-global coverage
sequence is documented in [`ITERATION_LOG.md`](ITERATION_LOG.md).

## 4. MACFlow baseline

The deployed model is the conventional joint Stage-I density
`p(u_A,u_B,u_C,u_D | x)` over `u∈R^8`: official `ActorVectorField` in a
`ModuleDict`, `TrainState`, conditional flow-matching loss, Adam, and 10-step
Euler sampling. The final run uses a standard 3×256 actor, batch 256, 100k
updates, seed 8123. It has no recurrence, persistent latent, action chunk,
trajectory generator, crossing-order label, CW/CCW label, or coordination-mode
input. A data-only source-balanced transition sampler and 50% early-transition
sampling over the first 15 transitions of *all* sources prevent long recovery
continuations from drowning out release decisions.

Training artifacts are in
[`base_u_v13_broad_global_source_balanced_macflow`](base_u_v13_broad_global_source_balanced_macflow).
Its best fixed development CFM loss was 0.07777 at step 95,000.

## 5. Development support and closed-loop diagnostics

Before v13 training, support was calibrated on train nominal timestep-0 states
only (the appropriate comparison for globally sampled deployment starts): 1/30
new dev starts was OOD (3.3%). The separate frozen dev diagnostic reports
30-rollout success 63.3%, wall 0%, agent collision 3.3%, timeout 33.3%,
timestep-0 OOD 6.7%, rollout OOD 76.7%, and mean episode length 892.97.
Teacher-forced mean RMSE is 0.00924 (best-of-8 0.00494).

K-step mean divergence for K=1/5/10/25/50/100 is
0.0120/0.0605/0.1191/0.2724/0.5154/0.9945; collision rate is zero at every
reported K, including K=100. See
[`final_diagnostics_v13_dev/summary.json`](final_diagnostics_v13_dev/summary.json).

## 6. Frozen untouched test

Master explicitly opened the independently generated v13 test exactly once,
after development was frozen. Its selection audit lists all 60 test archives;
there were **no post-test data acquisitions, retraining runs, geometry edits,
or checkpoint selections**.

| Metric | Frozen v13 test (60) |
|---|---:|
| Success | 29/60 = 48.33% |
| Wall collision | 1/60 = 1.67% |
| Agent collision | 5/60 = 8.33% |
| Timeout | 25/60 = 41.67% |
| Timestep-0 OOD | 5/60 = 8.33% |
| Rollout OOD | 73.33% |
| Mean length | 905.20 |
| Teacher-forced RMSE | 0.00982 |
| K=100 divergence / collision | 1.12805 / 0% |

The test success is slightly below 50% and timeout remains the main residual
failure. Those failures are now liveness/stochastic rollout failures rather
than an obvious initial-support, wall-boundary, missing local-recovery, or
undertraining bug; 48.33% is treated as approximately the intended maturity
target, not a hard quota. Full frozen evidence is
[`final_diagnostics_v13_frozen_test/summary.json`](final_diagnostics_v13_frozen_test/summary.json).

## 7. Mode statistics, failures, and visualizations

Frozen analysis found 16 realized crossing signatures, including 15 among the
29 successful episodes. This is strong analysis-only evidence that the learned
policy does not collapse to one permanent order. The most frequent signatures
were `B→A→C→D` (14) and `A→B→C→D` (11), but successful outcomes also use 13
other signatures. See
[`final_diagnostics_v13_frozen_test/analysis_only/crossing_order_statistics.json`](final_diagnostics_v13_frozen_test/analysis_only/crossing_order_statistics.json).

The geometry/expert trajectories are in
[`expert_pilot_outer_buffer.svg`](expert_pilot_outer_buffer.svg). Frozen
failure endpoints are shown in
[`frozen_failure_locations.svg`](final_diagnostics_v13_frozen_test/analysis_only/frozen_failure_locations.svg);
this visualization was rendered solely from the already completed frozen
summary and did not reopen test archives.

## 8. Hard-safety compatibility

The compatibility-only smoke test exposes all six pairwise constraints
`AB, AC, AD, BC, BD, CD` plus 16 square-boundary rows and reports nominal
feasibility. It is recorded in
[`hard_safety_smoke_v13.json`](hard_safety_smoke_v13.json). No formal
hard-safety comparison, OrthoFlow3, eta/basin run, or `G_phi` training was
performed.

## 9. Maturity decision

`PASS_STAGE1`. The environment has a distinct cyclic-crossing liveness
mechanism, the centralized expert is reliable and demonstrably multimodal, the
final train/dev/test protocol is independent and auditable, and the joint
MACFlow baseline has meaningful frozen performance with safety-compatible
infrastructure. Residual timeout and agent-contact populations remain valuable
for later hard-safety research, but do not justify further test-driven
development.
