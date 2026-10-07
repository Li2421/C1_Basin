# Double-Bottleneck four-agent deadlock audit

Date: 2026-09-23  
Scope: read-only audit and proposed semantics. No monitor, environment, Toy Give-Way,
test, or termination code was changed.

## 1. Finding

The current Double-Bottleneck monitor is the Toy Give-Way global monitor with the
agent axis enlarged from two to four. It is useful as a compatibility diagnostic,
but it is not adequate four-agent ground truth. In particular, one moving agent
prevents three stationary unfinished agents from being candidates, and movement
away from a goal or cyclic movement prevents the event even when it makes no task
progress. The code contains no bottleneck occupancy, passage, subset, or local
blocking atom.

The recommended four-agent event is a separate, versioned event,
`double_bottleneck_resource_frontier_v1`, described in Section 4. It retains a
global quiescence branch, adds independently timed bottleneck-resource cohorts,
and adds a post-passage per-agent branch. This is a proposal for master review,
not an implemented definition. Its numerical constants must be checked against
successful expert trajectories before it becomes ground truth.

## 2. Exact current behavior

Let `x[n]` be the state after `n` actions, `u[n-1]` the last executed velocity,
and `d_i[n]` agent `i`'s Euclidean goal error. At the default `dt=0.05 s`, the
current candidate at state sample `n` is exactly

```text
n >= 40
AND not all(d_i[n] <= 0.08)
AND max_i abs(d_i[n-40] - d_i[n]) < 0.01
AND max_i norm(u_i[n-1]) < 0.025.
```

The candidate must be true for 101 consecutive samples spanning 5.0 s. Thus a
stationary episode first triggers at step 140, time 7.0 s: 2.0 s to fill the
window plus a 5.0 s hold. A false sample resets the current timer. A trigger is
historically latched in `first_deadlock_step`. Event precedence is collision,
success, deadlock, then timeout; timeout occurs at step 850 (42.5 s).

Consequences in four-agent operation:

- **Global deadlock:** all-zero control is detected at 7.0 s, as tested.
- **Subset/partial deadlock:** any sufficiently moving or progressing outsider
  masks every stalled subset. This limitation is explicitly tested by
  `test_partial_stall_is_not_global_strict_deadlock_candidate`.
- **Bottleneck blocking:** neither occupancy nor directional passage is observed.
- **Persistent lack of progress:** only a 2 s endpoint difference is observed.
  Best-so-far progress, route stages, path length, and cycles are absent.
- **Temporary waiting:** a system-wide pause shorter than 7 s is not an event,
  but a valid coordinated pause lasting at least 7 s is a false-positive risk.
- **Latching:** a trigger remains in the summary even if a nonterminating
  diagnostic later moves again. Reset clears it.
- **Timeout:** an incomplete moving, cycling, retreating, or partially stalled
  trajectory is a timeout whenever the strict global trigger never occurs.

These semantics must remain unchanged for frozen Toy Give-Way. A new 4A monitor
must not replace or reinterpret Toy results.

## 3. Evidence from an existing Double-Bottleneck trajectory

The saved engineering rollout
`diagnostics/double_bottleneck_smoke/anchor_eta_1_0_025` is explicitly not a
scientific baseline, but it is a useful monitor counterexample. It is safe for
850 steps and is currently labeled `timeout`; its summary reports no collision,
no success, no strict deadlock, `max_stuck_timer=0.1 s`, and only a transient
strict candidate.

At the end:

| agent | final position | goal error | qualitative state |
|---|---:|---:|---|
| A1 | `(-2.022,-0.298)` | 5.572 m | stopped at left-bottleneck approach |
| A2 | `( 2.895, 0.165)` | 0.389 m | crossed both passages, jittering in right staging |
| B1 | `(-0.778, 0.646)` | 2.785 m | stopped in chamber at the opposite left approach |
| B2 | `(-2.996,-0.224)` | 0.273 m | crossed both passages, jittering in left staging |

No agent ever entered its 0.08 m goal disk. Over the last 10 s, A1 and B1 moved
only about 0.0080 m and 0.00007 m net, while A2 and B2 had instantaneous speeds
as high as 0.135 and 0.098 m/s but made essentially no net goal progress. This
moving tail defeats the current all-agent speed atom.

Using the proposed left-resource calculation below, A1 and B1 form the remaining
left-gate cohort after A2 and B2 clear it. Their directed clearance deficits are
both approximately 1.232 m and cease improving. With the proposed 2 s window,
stable-cohort/engagement guard, and 5 s hold, the left-resource event would latch
at step 499 (24.95 s), instead of becoming a generic timeout at 42.5 s. This
calculation was performed read-only from the stored `trajectory.npz`.

## 4. Proposed `D_H^(4A)`

### 4.1 Signals and route memory

Use `N=850`, `W=40` samples (2.0 s), `K=100` intervals (5.0 s),
`eps=0.01 m`, goal tolerance 0.08 m, and low-speed threshold 0.025 m/s.
All comparisons and sample indexing must be logged exactly.

For each agent, latch directional clearance of each gate. With bottleneck inner
edge `h=0.95`, outer edge `o=1.85`, and agent radius `R=0.16`, clearance planes
for the disk center are:

| route | left gate cleared | right gate cleared |
|---|---:|---:|
| left-to-right (A1,A2) | `x >= -h + R = -0.79` | `x >= o + R = 2.01` |
| right-to-left (B1,B2) | `x <= -o - R = -2.01` | `x <= h - R = 0.79` |

These are historical latches: later retreat does not undo an achieved passage.
For agent `i`, gate `r`, and direction sign `s_i` (`+1` left-to-right, `-1`
right-to-left), let `a_ir` be the applicable clearance plane and define

```text
q_ir[n] = max(0, s_i * (a_ir - x_i[n]))
F_ir[n] = min_{0 <= m <= n} q_ir[m].
```

Thus `F_ir[n-W]-F_ir[n]` is new best directed progress toward clearing that
resource; retreat, oscillation, and re-traversing old ground do not count as new
progress. Let `C_r[n]` be agents that have not yet latched clearance of gate `r`.

For agents that have cleared both gates, define goal frontier

```text
E_i[n] = min_{0 <= m <= n} d_i[m].
```

Using `x` is deliberate for these axis-aligned one-lane gates. If geometry later
changes, replace `q` by a deterministic shortest-path-to-clearance distance in
the disk-center free space; do not silently reuse these planes.

### 4.2 Three independently timed candidate families

Each candidate family has its own consecutive-sample timer. Switching between
candidate types must not accumulate one shared hold time.

**A. Global quiescence `Q[n]`.** `Q[n]` is true when the unfinished set is
nonempty and unchanged over `[n-W,n]`, every currently unfinished agent has
`abs(d_i[n-W]-d_i[n]) < eps`, and every currently unfinished agent's current
speed is below 0.025 m/s. Completed agents cannot mask or create this event.
This is the recognizable strict/global branch.

**B. Gate-resource stall `B_r[n]`.** For either left or right gate, `B_r[n]` is
true when all of the following hold:

```text
C_r[n] is nonempty
AND C_r[n] == C_r[n-W]                 # no membership/clearance handoff
AND min_{i in C_r[n]} q_ir[n] <= 1.32  # 0.90 gate + 2R + 0.10 approach margin
AND max_{i in C_r[n]} (F_ir[n-W] - F_ir[n]) < eps.
```

This cohort includes both travel directions. Therefore an agent legitimately
waiting while an opposing agent advances toward or through the same resource is
not a candidate: that agent improves the cohort frontier or clears the resource,
resetting the timer. An unrelated moving agent, lateral oscillation, or motion
away from clearance cannot mask the cohort. The 1.32 m engagement guard avoids
calling an agent deliberately held far from a not-yet-relevant gate a local
bottleneck deadlock. Log the actual throat/influence-zone occupancy set as
diagnostic metadata, but do not require physical throat occupancy: the observed
A1/B1 case leaves the narrow segment empty while opposing agents refuse to enter.

**C. Post-passage individual stall `P_i[n]`.** `P_i[n]` is true when agent `i`
has cleared both gates throughout `[n-W,n]`, remains outside its goal disk, and
`E_i[n-W]-E_i[n] < eps`. This catches a permanently incomplete agent in a staging
region even while other agents continue to move. It does not use a low-speed
condition, so jitter and cycles do not evade it.

A raw trigger occurs when the same `Q`, same `B_r`, or same `P_i` candidate is
true for 101 consecutive samples (5.0 s span). The first possible trigger remains
7.0 s. Define the historical event

```text
D_H^(4A) = 1 iff a raw 4A trigger occurs by N=850 before any earlier success
           or collision; a same-step collision has priority, then success,
           then 4A deadlock, then timeout.
```

The event latches its first trigger step and type (`global`, `left_resource`,
`right_resource`, or `post_passage_i`). In a nonterminating diagnostic, later
recovery does not erase the event; report `recovered_after_deadlock` separately.
Operational evaluation may terminate on the first trigger after validation.

This is an operational persistent-stall event, not a proof that the physical
state is mathematically irreversible. Irreversibility cannot be inferred from a
finite state/action trace without specifying admissible future controllers.

### 4.3 Timeout distinction

At step 850, label an incomplete collision-free rollout `timeout`, not deadlock,
if no branch completed its full hold. Examples are an agent still setting a new
clearance/goal frontier, a legitimate resource handoff in the final window, or a
stall that began too late to supply 7 s of evidence. Do not reclassify every
failure-at-horizon as deadlock and do not add a short tail-only substitute.

## 5. Required examples and error risks

### True positives

- All four agents remain at rest from reset: global trigger at step 140.
- A1 and B1 remain engaged on opposite sides of the left gate while A2/B2 have
  cleared it: `left_resource` triggers even if A2/B2 keep jittering.
- One agent has cleared both gates but cycles outside its goal while the other
  three finish: `post_passage_i` triggers.

### True negatives

- A1 waits while B1 makes new directed progress toward/through the left gate:
  the shared resource frontier advances and the left timer resets.
- The whole system pauses for less than the 5 s confirmation after the 2 s
  window, then resumes and succeeds.
- At 42.5 s an agent is still making measurable new route progress and no
  branch has completed its hold: the result is timeout, not deadlock.

### False-positive risks

- A valid planner may intentionally hold every member of a resource cohort for
  at least 7 s, for example a long time reservation, and later succeed.
- Progress slower than 0.01 m per 2 s is treated as no meaningful progress even
  if it would eventually finish under a much longer horizon.
- A post-passage agent may validly wait more than 7 s in staging to avoid traffic.

These risks require replay over all successful expert coordination modes before
freezing constants. A recovered-after-trigger expert trajectory is direct
evidence that the proposed hold/engagement semantics need adjustment.

### False-negative risks

- An agent can creep by at least 0.01 m every 2 s yet still fail to finish by H.
- A blockage beginning after 35.5 s cannot provide the full earliest 7 s of
  evidence before H and remains a timeout.
- A stalled subset far outside the 1.32 m engagement zone can be masked while
  other agents move; it is detected only after global quiescence or engagement.
- Monotone `x` progress is not sufficient for success: an agent can continue to
  set a small longitudinal frontier while maintaining a bad lateral plan.

## 6. Review and implementation recommendation

Implementation should later change for Double-Bottleneck, but not until the
master reviews this definition against centralized-expert trajectories. The
safe rollout protocol for that review is:

1. keep the present legacy monitor and Toy semantics untouched;
2. compute the proposed 4A atoms in shadow/nonterminating mode;
3. log per-agent gate-clearance latches, `q`, `F`, `E`, cohort membership,
   engagement, candidate timers, trigger type, and recovery;
4. replay clearly asymmetric, weakly asymmetric, near-symmetric, and alternative
   successful passage orders;
5. revise constants only from declared validation cases, then version and test
   the accepted Double-Bottleneck protocol;
6. only after acceptance, use `D_H^(4A)` as basin ground truth.

No implementation was changed in this audit.
