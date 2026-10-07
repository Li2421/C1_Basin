# Gap1 Flow competence: pre-training diagnosis

This diagnosis was recorded before starting the competence retraining.  The
historical N=2/N=10 Gap1 checkpoints and the accepted safety projection are
unchanged.  The prior full safety audit is in
`docs/gap_flow_safety_deadlock_audit_20261007.md`.

## Data and model behavior

The old data generator (`bottleneck_family/flow_dataset.py`) starts only from
opposing tasks and asks `SequentialExpert` (`bottleneck_family/expert.py`) to
complete each task by moving **one numbered agent at a time**.  In a held-out
sample of the old dev transitions, 23,233/23,245 N=2 steps and
173,063/173,260 N=10 steps had exactly one agent moving; the remaining steps
had zero, and **none had two moving agents**.  The model is a flattened joint
MACFlow with fixed row order (`new_benchmark_common/macflow.py`), so this data
can encode the sequential expert's implicit agent-order policy.  The old
checkpoint has essentially no supervised evidence for two simultaneous
same-direction tasks or for a passive robot whose goal is its current pose.

This is not a simple shortage of right-to-left examples.  The N=2 training
set contains 144 agent trajectories in **each** direction; right-to-left has
57,440 nonzero-action transitions against 34,444 left-to-right.  Near the
gate the counts are 16,126 versus 10,064.  N=10 likewise has 370,314
right-to-left active transitions versus 314,924 left-to-right.  Deliberate
oversampling of the weak direction would target the wrong imbalance.

The model fits its own sequential training distribution.  On sampled old dev
teacher states, N=2 predicted mean speed is 0.496/0.498 m/s for the active
agent, with mean action error 0.008/0.004 m/s; when an agent is waiting, mean
predicted speed is 0.004/0.006 m/s.  These are single-sample predictions from
the frozen checkpoint.  Low held-out CFM loss and accurate teacher-forced
actions therefore do not demonstrate closed-loop competence on the controls.
The old checkpoints were selected by fixed dev CFM loss alone
(`new_benchmark_common/training.py`).

The eight-dimensional observation contains physical position, previous
velocity, relative goal and a relative gate waypoint.  Directional signs and
units are consistent.  However, the original `gate_waypoints` continues to
return the **entry** at the exact entry position, although the sequential
expert's next action points through the gate.  In training, 144 N=2
left-to-right and 288 right-to-left gate-local active steps have a waypoint
norm below 0.03 m with a distant goal.  This is a local target/label
inconsistency, especially harmful when the safety projector keeps the
robot near the entrance.  The new competence observer preserves all eight
features but switches from entry to exit with a 0.04 m look-ahead once
aligned (`gate_waypoints_competence`); the historical observer remains
available to reproduce old results.

The old safe rollouts independently show right-to-left single-active success
of 0/12 (2/12 with the passive robot moved to the remote old goal) and N=2
one-way success of 0/12.  The right-to-left median final-5-s Flow
goal-direction request is +0.327 m/s, executed goal-direction velocity
-0.001 m/s, wall constraints active in all final steps, and pair constraints
inactive.  Thus the safety projector is suppressing a wall-directed request;
there is no opposing-agent explanation for this failure.

## Controlled feasibility check

Before retraining, a deterministic controller using only the *public* gate
waypoint with the corrected entrance switch and the **existing unchanged**
`CertifiedHardSafetyFilter(HardProjectionConfig())` safely completed the
first three old-dev N=2 instances in each of: left-to-right single-active,
right-to-left single-active with remote passive robot, left-to-right one-way,
even-first temporal traversal and odd-first temporal traversal (3/3 in each).
Each physical rollout used the exact Gap1 map, disk dynamics, continuous
swept collision check and 2000-step horizon.  Its mean projection correction
was 0.008–0.022 m/s over those modes.  The probe was preflighted through the
shared rollout cache (`/tmp/gap_reference_pilot_plan.json`).  It establishes
that the geometry and accepted projector can support these ordinary N=2
tasks, without claiming that this deterministic probe is the learned model.

For N=10 the same simple simultaneous waypoint reference failed a small
pilot of one-way cases with heavy pairwise projection.  This is evidence that
high-density one-way traffic needs a separate competence check; it does not
license changing the safety projection or inventing an opposing priority rule.

## Minimal training response

Keep the same joint Stage-I MACFlow architecture, Flow integration, physical
state, eight-feature observation shape, speed bound, simulator and accepted
safety filter.  Train only on successful solo and same-direction traversals
generated from varied independent Gap1 starts/goals.  Reflect each case
across the symmetric gate and swap the two rows, yielding exact left/right
direction balance and both active-agent row positions.  No opposing traffic
or alternating/yielding demonstration enters the training or dev data.
Select checkpoint from held-out **navigation rollouts** and projection
compatibility, not opposing-test outcomes or CFM loss alone.  Evaluate fresh
competence controls before freezing; only a passing frozen checkpoint is
eligible for original opposing Gap1 evaluation.
