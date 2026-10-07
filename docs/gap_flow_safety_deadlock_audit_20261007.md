# Gap1 frozen Flow + accepted safety projection audit

Date: 2026-10-07. **Final verdict: C — frozen Flow competence is insufficient.**
No Flow weights, eta, route planner, MAPF, benchmark geometry, physical thresholds, or
safety algorithm were changed.

## Exact nominal safety chain and Gap1 adapter

The historical TT/Success-Basin chain in
`diagnostics/success_basin_multimodality/run.py` samples the frozen Flow each
physical step, bounds its velocity, builds the hard CBF constraints, and
projects once to obtain `u_safe`; a second projection is used **only after
eta**, which this audit does not use. The newer accepted N-agent realization
is `HardSafetyFilter` in `shared_control/hard_projection.py`, wrapped by
`CertifiedHardSafetyFilter` in
`diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py`.
That wrapper retries the *identical* optimization if the primary solver's
certificate fails. `new_benchmark_common/safety_eta3.py` uses the same
wrapper for its formal `hard_safety` chain. The older TT 2-agent projector
is shape-specific; the generic N-agent implementation is the necessary
existing adapter for Gap1 at N=10 and N=50.
The TT protocol's own frozen checkpoint is
`/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl`
(SHA256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`),
and its locked two-agent Clarabel projector source has SHA256
`841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48`.
Those weights have a different observation/action shape and are **not** used
for Gap1. Gap1 uses its own frozen N-specific checkpoints and the accepted
dimension-generic projection with the same mathematical CBF objective and
constraints.
Its source SHA256 values are
`847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79`
(`shared_control/hard_projection.py`) and
`94a478f7b1dc1897e1e3113a78fe4c4afa134cbb02ce924f4bc19befc6ce2f94`
(the identical-problem retry wrapper). Neither source was modified.

For this audit, the executed law is exactly

```
o_k = [position, last executed velocity, goal - position,
       static gap waypoint - position] for each robot
a_k = bounded_joint_MACFlow(o_k, xi_k)  # 10 Euler steps, [N,2] velocity
u_ref,k = a_k                            # opposing and one-way nominal runs
u_safe,k = argmin_u 0.5 ||u-u_ref,k||^2  subject to existing hard CBF set U(x_k)
p_{k+1} = p_k + 0.05 u_safe,k
```

The Gap1 model was trained and previously evaluated with a mean of 16
independent sampled joint actions per step (MC16). To check the literal
single-sample TT/Success-Basin sampling convention, **MC1 is the primary
audit**, and MC16 is an explicitly separate sensitivity check that only
inserts the same safety projection into the earlier Gap1 protocol. Both use
the *same frozen* Gap1 `best.pkl` for each N, `policy_observation` from
`bottleneck_family/observation.py`, split `test`, evaluator seed 17, and the
existing test initial states. Sampling and episode key construction are in
`bottleneck_family/evaluate_safety_audit.py`; official MACFlow model/sampling
are in `new_benchmark_common/macflow.py`. There is no critic, generator,
eta, priority signal, or inference-time path guidance.

The constraint set is inherited, not new:

- Every unordered robot pair has
  `h_ij=||p_i-p_j||²-(2*r_agent+agent_margin+0.0001)²`, with its existing
  first-order CBF inequality.
- Every robot and **every finite wall segment** has
  `h_iw=distance(p_i,segment)-r_agent-r_wall-wall_margin-0.0001`, with the
  existing wall CBF inequality. This includes the outer boundary and both
  faces/ends of the Gap1 barrier.
- Each robot has its exact Euclidean speed ball `||u_i||<=0.5 m/s`. Clarabel
  solves the 2N-dimensional Euclidean projection; retry uses the same
  objective and feasible set, never a relaxed fallback.

The Gap1 snapshot already supplies `[N,2]` positions, all unordered pairs,
12 finite wall segments and plant parameters. A smoke check produced the
expected 24/120/600 wall rows for N=2/10/50 and 1/45/1225 pair rows.
With the **same projector** and a constant forward reference, a robot
initialized at the door entrance moved from `x=-1` to `x=0.805` in 80 steps
without collision (minimum swept clearance 0.138 m). Thus there is no hard
wall-geometry integration blocker. No new obstacle handling was added.

`BottleneckEnv.step` performs the actual swept wall/pair collision check;
the first collision terminates. Success requires **all** agents within
0.08 m of their goals without prior collision. The published horizons are
2000/8000/40000 steps at N=2/10/50. Gap1's optional `stalled_groups()`
is only a candidate detector and is not invoked in these evaluations.
Timeout remains timeout. For diagnostics only, the offline analysis applies
the existing TT strict candidate rule (2 s goal-error window <0.01 m,
all-agent speed <0.025 m/s held for 5 s) without changing simulator events.

In the single-active-task controls, the fixed N=2 checkpoint still receives
its required two observation rows. The passive robot's goal is set to its
start and its **reference velocity is set to zero before the unchanged
joint projection**. Its measured physical displacement was zero in all
left-to-right runs and 10/12 original right-to-left runs. In two original
right-to-left cases, pairwise CBF coupling moved the passive robot by
0.30–0.36 m. A follow-up `solo_odd_remote` control places it at the
other task's *existing* far goal point, while keeping the active robot's
exact start and goal. This is a control-task adapter, not a new
nominal controller or a claim that an N=1 checkpoint exists. One-way uses
the same start/goal point sets but reverses odd agents' tasks so both
robots traverse left-to-right. Temporal controls keep the original starts
and goals and set the waiting direction's reference velocity to zero until
the first group reaches its goals. The opposing baseline has no masking.
All new rollout tuples passed the shared rollout DB preflight before
execution; traces record every simulator state and every safety result.

## N=2 results and mechanism

Each entry uses the 12 frozen test initial states and the same checkpoint
`a321b6fad7a607ee379db4daccb4f20707988a5ed5085c797abb2673a507cc91`.
`MC1` is the literal one-sample Flow-to-projection chain; `MC16` preserves
the previous Gap1 sampling/averaging rule. All completed episodes are
collision-free. Failure counts below classify a timeout as **B** only after
checking the matched non-conflicting controls and the absence of a certified
opposing gate encounter/strict stall; timeout alone is never A.

| Mode | MC1 safe success / 12 | MC1 B / C / D / E | MC16 safe success / 12 | MC16 B / C / D / E |
|---|---:|---|---:|---|
| Single active, left-to-right | 8 | 4 / 0 / 0 / 0 | 5 | 7 / 0 / 0 / 0 |
| Single active, right-to-left | 0 | 12 / 0 / 0 / 0 | 0 | 12 / 0 / 0 / 0 |
| Single active, right-to-left with remote passive robot | 2 | 10 / 0 / 0 / 0 | not run | not run |
| One-way, both left-to-right | 0 | 12 / 0 / 0 / 0 | 0 | 12 / 0 / 0 / 0 |
| Temporal, even side first | 3 | 9 / 0 / 0 / 0 | 3 | 9 / 0 / 0 / 0 |
| Temporal, odd side first | 0 | 12 / 0 / 0 / 0 | 0 | 12 / 0 / 0 / 0 |
| Original opposing traffic | 4 | 8 / 0 / 0 / 0 | 2 | 10 / 0 / 0 / 0 |

The MC1 Level-A measurements below use the *moving agent's* goal distance
and gate counts; other columns are medians across the 12 episodes. Wall
distance is the minimum **center-to-segment** distance attained in each
episode, including the simulator's swept motion.

| N=2 Level A | safe success | moving agent entered / crossed gate | median minimum center-wall distance | median final goal distance | projection-active step fraction | mean `||u_safe-u_ref||` |
|---|---:|---|---:|---:|---:|---:|
| Left-to-right | 8/12 | 12/12 / 11/12 | 0.187 m | 0.080 m | 0.091 | 0.014 m/s |
| Right-to-left, original passive position | 0/12 | 11/12 / 4/12 | 0.172 m | 5.044 m | 0.861 | 0.305 m/s |
| Right-to-left, remote passive position | 2/12 | 12/12 / 8/12 | 0.172 m | 4.240 m | 0.740 | 0.299 m/s |

The same MC1 metrics for Level B count *all* moving-task agents, so the
gate denominators are 24 at N=2 and 120 at N=10. Temporal masking is
applied **before** projection; its reported correction excludes that
deliberate mask.

| Non-opposing control | safe success | moving-task agents entered / crossed gate | median minimum center-wall distance | median final mean-agent goal distance | projection-active step fraction | mean `||u_safe-u_ref||` |
|---|---:|---|---:|---:|---:|---:|
| N=2 one-way | 0/12 | 12/24 / 10/24 | 0.172 m | 7.406 m | 0.831 | 0.256 m/s |
| N=2 temporal even-first | 3/12 | 21/24 / 19/24 | 0.172 m | 1.560 m | 0.228 | 0.066 m/s |
| N=2 temporal odd-first | 0/12 | 0/24 / 0/24 | 0.749 m | 11.160 m | 0.000 | 0.000 m/s |
| N=10 one-way | 0/12 | 30/120 / 15/120 | 0.172 m | 8.813 m | 0.998 | 0.217 m/s |

The left-to-right/right-to-left asymmetry is not caused by a physically
closed door: the projector's constant-reference smoke traversal works.
In the remote-passive control, 11/12 episodes keep the passive robot
*exactly* stationary; the other moves it only 0.060 m by pairwise CBF
coupling. Of the 11 strictly single-moving episodes, only **1** succeeds.
The original right-to-left proxy has 10 strictly single-moving failures.
In MC1 right-to-left single-active runs, the **median** wall-constraint
active fraction is 0.861 and pair-constraint fraction is 0; the moving
Flow's final-5-s goal-direction request has median 0.327 m/s, while the
executed goal-direction component is -0.001 m/s. The median final-5-s
projection correction is 0.359 m/s; the wall constraint is active in
100% of the final 5 s, the pair constraint in 0%. This is safety
projection suppressing commands at the **static wall**, without opposing
motion. In MC1 one-way runs the
median wall-constraint active fraction is 0.823 and no full-task success
occurs. Temporal odd-first fails before the moving agent enters the gate
region in all 12 episodes; in many such cases the projection does not even
activate, showing another non-interaction navigation/conditioning failure.

For MC1 opposing traffic, 4/12 succeed and 8/12 time out. Both agents
reach the broad gate region in all 12, but **0/12** have an opposing pair
within 0.8 m while both are at `|x|<1.5 m`; **0/12** satisfy the existing
TT strict 2-s-window/5-s-hold stall criterion. Pairwise CBF constraints
activate in only one run, which succeeds; wall constraints dominate the
failures. Across the eight timeouts, median final-5-s Flow goal-direction
request is 0.199 m/s versus executed 0.0001 m/s, with 0.432 m/s median
projection correction. The wall constraint is active in 100% of the
final 5 s; the pair constraint in 0%. The measured suppressor is the wall
constraint, not mutually incompatible opposing-agent motion. Median
high-speed reversal fraction in this window is 0.101, so these traces
also do not show dominant high-frequency velocity reversal.
For clarity, "wall active in 100% of steps" means at least one wall row
is active per step; the median *fraction of all 24 wall rows* active in
the final 5 s is 0.049 in opposing failures (0.042 in single-active
right-to-left failures). Minimum robot-center-to-wall distance is about
0.172 m, the physical radius plus wall radius/margin and the 0.1-mm
safety buffer. The per-row and per-step quantities are separately saved.

All per-step Flow/reference/safe velocities, projection magnitudes, active
pair/wall/speed counts, simulator swept clearances and goals are preserved
in local `diagnostics/gap_flow_safety_audit_mc1_v1/n2/*/traces/` and the
MC16 counterpart `diagnostics/gap_flow_safety_audit_v1/n2/*/traces/`.

## N=10 and scale gate

The frozen N=10 checkpoint is
`1e078260be745a29806c8661fac0f1acdd0dba64a7439042882cf6a8e6acaeb4`.
MC1 used all 12 independent test initial states and the original 8000-step
horizon. The complete N=10 MC16 one-way sensitivity control is also 0/12
safe successes, 12/12 safe timeouts. Because Level A and N=2 Level B had
already failed, N=10 temporal controls were not expanded; the N=10
one-way and opposing cohorts were retained as a scale diagnostic.

| N=10 MC1 mode | runs | safe success | A clean deadlock | B navigation/competence failure | C collision | D numerical/oscillatory | E mixed/unclear |
|---|---:|---:|---:|---:|---:|---:|---:|
| One-way, all left-to-right | 12 | 0 | 0 | 12 | 0 | 0 | 0 |
| Original opposing traffic | 12 | 0 | 0 | 5 | 0 | 0 | 7 |

The N=10 one-way group is collision-free but cannot complete: median
2.5/10 agents enter the broad gate region, 1/10 crosses, and mean-agent
final goal distance remains 8.813 m. At least one wall CBF row is active
in a median 99.7% of steps and at least one pair row in 78.1%; the
median projection correction is 0.217 m/s. These controls reveal safety
and navigation congestion without *opposing* flow.

In opposing traffic, 7/12 episodes show an opposite-direction pair within
0.8 m with both agents at `|x|<1.5 m`. They are **E**, not A: none meets
the unchanged TT strict 2-s-window/5-s-hold persistent-stall criterion,
and the corresponding one-way controls do not establish basic competence.
This global TT predicate can miss a partial group stall while other
robots still move; its non-detection is not proof that no local queue
stalled. Such a local event still could not be isolated as an
*opposing-specific* deadlock with the present failing controls.
The other five opposing timeouts have no measured opposing gate encounter
and are B. All 12 are collision-free. For the seven E episodes, median
pre/post-encounter mean goal-progress rates are **0.033/0.006 m/s**, mean
speeds **0.054/0.036 m/s**, and projection corrections **0.041/0.156 m/s**.
These are temporal associations, not evidence that opposing interaction
caused the loss: the fraction of active wall rows rises from 0.015 to
0.047 (median), while active pair rows remain about 0.001 before and
after. The one-way controls independently stall under the same walls.
Final-5-s opposing median goal-direction request is 0.017 m/s and
executed component approximately zero; this is far weaker Flow demand
than the N=2 wall-stall cases. Full gap-distance, inter-agent swept
clearance, active-row and progress series are in the NPZ traces and
`all_rollout_diagnostics.json`.

N=50 was **not run**: the prerequisite N=2 single-active and one-way
competence controls failed under both sampling protocols, and N=10
one-way also failed. Running the much larger N=50 batch could not
establish the requested clean causal contrast.

## Representative complete videos

`diagnostics/gap_flow_safety_audit_mc1_v1/videos/manifest.json` gives each
full MP4's trace path, hash, simulator seed, checkpoint, frame count and
terminal event. It includes a successful left-to-right single-active
traversal, successful temporally separated traversal, failed right-to-left
single-active traversal, failed one-way traversal, and a representative
opposing timeout, plus N=10 one-way and mixed opposing failures.
**There is no successful one-way video or canonical N=2
deadlock video** in these data; neither was fabricated. These audit videos
are separate from the formal four-scene batch delivery gate.

## Decision gate

The required empirical contrast is broken at Level A/B: the frozen
controller with the accepted projector has only 1/11 strictly
single-moving right-to-left remote-passive successes, and zero
same-direction two-agent successes. It is premature to test eta or call
the opposing timeouts safety/coordination deadlocks. Under the frozen
nominal chain, improving/retraining Flow navigation is necessary before
Gap1 can support the external benchmark claim. No retraining was
performed in this audit.
