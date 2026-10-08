# Gap1 N=50 post-gate Flow competence audit

This report concerns the frozen, same-geometry Gap1 N=50 nominal law: 10-step
MACFlow action sampling, 0.5 m/s speed bound, existing
`CertifiedHardSafetyFilter(HardProjectionConfig())`, and
`BottleneckEnv.step`. The goal tolerance is 0.08 m for **all** agents. No eta,
critic, generator, MAPF, opposing-flow demonstrations, or safety change is used
in the competence repair.

## Diagnosis before retraining

The rejected v4 and v5 checkpoints cross the gate but do not finish the
non-opposing N=50 task. Complete 40,000-step one-way trials gave 45–50/50 gate
crossings but only 0–11/50 final goal occupants, with no collisions. In the
representative failed v4 and v5 one-way traces, 37–44 agents are still pending
beyond the opening at steps 10,000–39,000. At step 39,000, pending goal
distances are 0.41–1.47 m for v4 and 0.51–2.64 m for v5: they are not merely
hovering at the 0.08 m success boundary. Their Flow command has mean speed
around 0.08–0.13 m/s and negligible or negative mean goalward component. The
same accepted safety projection applied to the existing expert's command at
those states retains approximately 0.42–0.50 m/s goalward speed. Thus safety
cannot explain the near-zero model progress in these particular states.

The original train data have exactly 64 L→R and 64 R→L trajectories, 374,150
joint transitions per direction, but only **8 independent physical base seeds**
per direction. Four task modes and two row permutations multiply each base
seed; mirroring supplies the opposite direction. Every original full-density
teacher trajectory has at most **4–5 simultaneous post-gate agents still
outside the goal tolerance**. The rejected Flow rollouts regularly have more
than 30. This is a concrete joint-state coverage gap despite many transitions.

| Original train phase, per direction | Trajectories | Active action-agent steps | Fraction of active task supervision |
|---|---:|---:|---:|
| Pre-gate approach | 64 | 274,572 | 40.0% |
| Gate entry | 64 | 21,232 | 3.1% |
| Inside/crossing gate | 64 | 49,206 | 7.2% |
| Post-gate, goal >1 m | 64 | 248,750 | 36.3% |
| Post-gate, 0.08–1 m | 64 | 71,352 | 10.4% |
| Settled within 0.08 m | 48 | 21,028 | 3.1% |

These fractions use initially task-active agents; ~20,644 moving agent steps
from initially passive agents are excluded. The frozen v4/v5 models' sampled
CFM loss estimates assign 35%/32% of the **actual weighted squared-error
numerator** to the 0.08–1 m phase, despite its smaller population. The near-goal
phase is difficult, not simply ignored by the old loss. The CFM estimates use
2,048 sampled joint frames from each of the uniform, early, and near-goal
sampler strata with fixed stochastic latents and population weighting; they
are not exact full-epoch integrals.
For v4/v5, initially passive plus already-settled agents contribute about
28%/30% of that sampled CFM numerator. This is material but not dominant;
increasing only the existing near-goal sampling fraction would still not
create the missing dense post-gate joint configurations.

The target generator is correct in tested held-out states. The stored
observation matches a fresh `policy_observation_competence` recomputation
exactly; its goal vector and post-gate waypoint point to the actual goal, not
to a stale gate target. `teacher_reference` sets zero only within the true
0.08 m threshold. In held-out single-active states just outside that threshold,
the saved expert safely requests about 0.43 m/s goalward and both old Flows
also request about 0.43 m/s. In complete same-direction expert states the old
Flows are usually goal-directed at larger distances; their collapse appears
in the dense joint states reached by their own closed-loop trajectories.
The set-attention observation has physical position, velocity, relative final
goal, and relative public gate waypoint per agent; row permutations are applied
consistently to observations and actions. v4's all-action normalization gives
action scales (0.0812, 0.0356) m/s, whereas v5's active-action normalization
gives (0.4172, 0.1827) m/s. Both still fail in dense post-gate states, so
normalization alone is not the diagnosed repair. The distinct 200,962- and
26,114-parameter models show the same failure class, so another blind width
increase is not justified by this evidence.

An additional feasibility probe resumed the existing same-direction expert
plus unchanged safety filter from three failed v4 validation states containing
28, 27, and 43 unresolved post-gate agents. It safely reached 50/50 goals
after 4,807, 1,901, and 736 steps respectively, with minimum continuous swept
clearance 0.000668 m, 0.001109 m, and 0.005158 m. These validation states are
**diagnostics only** and are not admitted to training.
From a v5 validation state with 44 unresolved post-gate agents, the same
expert/safety chain also reached 50/50 goals in 901 steps, with positive
0.002886 m minimum swept clearance.

The minimal repair is to add successful same-direction expert continuations
from dense post-gate **training** states produced by the old Flow. Each accepted
L→R continuation is replayed as an exact mirrored R→L continuation. This
increases joint-state coverage and independent base-seed diversity without
showing the model opposing-side negotiation. Existing complete gate-to-goal
demonstrations remain in the dataset. The first new candidate retains the old
architecture and accepted safety controller.
The augmented source provides 96 complete non-opposing train trajectories and
607,599 joint transitions per direction from 24 independent train seeds;
its six independent dev seeds do not overlap training. All 48 predeclared
train-only recovery anchors (three per each of 16 additional full-density
Flow traces) have 7–44 unresolved post-gate agents, median 26.5. The mirror
construction preserves exact L/R counts after successful recovery filtering.

## Evidence and exact entry points

- Old phase counts: `bottleneck_family/audit_n50_dataset.py` and
  `diagnostics/gap_flow_scale_20261007/n50_old_phase_audit_code_20261008.json`.
- Frozen-model phase CFM estimate: `bottleneck_family/audit_n50_cfm_loss.py`
  and `diagnostics/gap_flow_scale_20261007/n50_old_cfm_loss_audit_16frames_20261008.json`.
- Held-out expert/Flow post-gate actions:
  `bottleneck_family/audit_n50_postgate.py` and
  `diagnostics/gap_flow_scale_20261007/n50_postgate_probe_20261008.json`.
- Failed-state expert comparison:
  `bottleneck_family/audit_n50_late_states.py` and
  `diagnostics/gap_flow_scale_20261007/n50_late_state_action_audit_20261008.json`.
- Successful expert recovery probe:
  `bottleneck_family/audit_n50_recovery.py` and
  `diagnostics/gap_flow_scale_20261007/n50_dense_recovery_probe_20261008/`
  plus `n50_dense_late_recovery_probe_20261008/` and
  `n50_v5_late_recovery_probe_20261008/`.
- Frozen nominal execution and goal-stop adapter:
  `bottleneck_family/evaluate_competence.py`; target and waypoint:
  `bottleneck_family/collect_competence.py` and
  `bottleneck_family/observation.py`; simulator:
  `bottleneck_family/environment.py`.

## Balanced competence dataset and first controlled training candidate

All 48 predeclared train-only dense post-gate recovery anchors succeeded under
the accepted expert/safety chain. Each was replayed with reflected geometry and
actions through the same simulator; no recovery candidate was discarded. The
merged dataset reader verified every trajectory shape and stored digest.
TRAIN and DEV use 24 and 6 disjoint independent physical seeds, respectively;
TEST is left out of training and checkpoint selection.

| Split and direction | Complete nominal trajectories | Recovery trajectories | Joint transitions | Independent seeds |
|---|---:|---:|---:|---:|
| TRAIN L→R | 96 | 48 | 745,565 | 24 |
| TRAIN R→L | 96 | 48 | 745,565 | 24 |
| DEV L→R | 30 | 0 | 182,799 | 6 |
| DEV R→L | 30 | 0 | 182,799 | 6 |

Every accepted trajectory ends in actual all-agent simulator success. Each
TRAIN direction has 521,010 active approach, 75,310 entry, 103,404 crossing,
543,789 post-gate >1 m, 188,228 post-gate 0.08–1 m, and 54,111 settled
action-agent steps. The complete source's maximum simultaneous unresolved
post-gate count remains 5; the 48 new recovery trajectories cover 7–44,
median 26.5. Exact L/R trajectory, transition, mode, source and phase counts
match because every recovery is mirrored and replayed. The merged dataset
manifest SHA256 is
`00b9f1b33b042ff45147fab5d02a0042e0900af3e3a7bf2d6c361a7659a854d1`.

One controlled v6 candidate is training from the rejected v5 checkpoint,
retaining its 26,114-parameter one-layer, width-64 set-attention actor,
active-action normalization, 10 Euler flow steps, motion loss weight 20,
permutation augmentation and accepted safety law. The change is the balanced
new dataset plus source-balanced sampling; the near-goal frame fraction is
reduced from 0.5 to 0.3 so training capacity is allocated to the newly
observed dense recovery states. It trains for 30,000 steps with snapshots every
5,000 steps. Snapshot selection will use **non-opposing closed-loop navigation**
and safety compatibility, not opposing traffic or offline CFM loss alone.
The run completed all 30,000 steps in 33.6 s on the allocated GPU shard.
Its fixed sampled train CFM loss changed from 0.3197 to 0.1510; fixed dev
loss was 0.06559 initially, 0.06533 at the offline-best 10,500-step snapshot,
and 0.06739 at the final snapshot. These offline numbers do not decide which
checkpoint is frozen. On 25 held-out dense post-gate one-step validation
states, v5's mean projected goalward speed was 0.278 m/s and the v6 final
snapshot's was 0.341 m/s; the expert's saved executed target was 0.418 m/s.
Complete-horizon non-opposing validation rejected both examined v6 snapshots.
On one held-out physical state per direction, the exact complete-horizon
outcomes were:

| v6 snapshot | Direction | Gate crossings / 50 | Final goal occupants / 50 | Task success | Collision |
|---|---|---:|---:|---:|---:|
| Offline-best | L→R | 49 | 27 | 0/1 | 0/1 |
| Offline-best | R→L | 49 | 19 | 0/1 | 0/1 |
| 30,000-step final | L→R | 23 | 19 | 0/1 | 0/1 |
| 30,000-step final | R→L | 50 | 37 | 0/1 | 0/1 |

All four ran the full 40,000 steps without a collision. For the offline-best
snapshot, the 22 unfinished post-gate agents at step 39,000 receive a mean
Flow goalward command of −0.135 m/s, compared with +0.490 m/s from the same
expert and safety projection at that physical state. A recovery from that
held-out failed state reached all 50 goals safely in 433 steps, confirming
that late goal occupancy is feasible. Across 64 independent Flow latents at
the same held-out states, the mean goalward action remained close to zero or
negative; simply averaging more action samples is not a supported fix.

The first two fresh DEV states for each sparse control completed successfully
under the v6 final snapshot: single-active L→R 2/2, single-active R→L 2/2,
five-active L→R 2/2 and five-active R→L 2/2, all collision-free. Mean joint
projection correction was 0.0021/0.0072 m/s for the single-active categories
and 0.0419/0.0430 m/s for five-active L→R/R→L. These are screening counts,
not the required ≥90% acceptance estimate. Full-density, temporally separated,
and larger fresh held-out cohorts must still finish before freeze.

The two complete fresh full-density final-snapshot trials per direction also
failed after 40,000 steps, collision-free: L→R ended at 21/50 and 36/50
goal occupants; R→L at 29/50 and 24/50. The first temporally separated
even-first control timed out with only 21/50 at goal; the second group was
never released. The second even-first control timed out at 43/50 after group
release. Thus v6 is not
frozen. A second, data-only correction is being prepared from *train-only*
late states reached by the offline-best v6 Flow at predeclared 5,000, 15,000,
25,000, and 29,999-step anchors. These states use eight train-only physical
seeds also present in the v6 nominal source. The **late v6 Flow states** were
absent from v6 training; the seeds themselves were not new.
Successful recovery demonstrations will be mirrored and tagged as a distinct
`late_postgate_recovery` training source, allowing documented source-balanced
sampling to give the rare late joint states meaningful weight without any
direction imbalance or opposing-flow demonstration. No v6 validation state
enters the training set.

The second data collection completed with **31/32** predeclared anchors
accepted per direction. One train-only 5,000-step recovery timed out under
the unchanged expert after 40,000 steps and was excluded, with its failure
retained in the collection report. The other 31 continuations reached every
goal without collision, and each was mirrored and simulator-replayed. New
recovery files received a `v6best_*` prefix before merging because a new
Flow checkpoint can visit a different physical state at the same source
rollout and nominal time as an older checkpoint. The old files were not
overwritten. The production dataset reader verified all relabeled and merged
trajectory digests.

| v7 train source, per direction | Trajectories | Joint transitions |
|---|---:|---:|
| Complete nominal demonstrations | 96 | 607,599 |
| First dense post-gate recoveries | 48 | 137,966 |
| New late post-gate recoveries | 31 | 49,981 |
| **Total TRAIN** | **175** | **795,546** |

The DEV split remains 30 complete trajectories and 182,799 transitions per
direction, from six physical seeds disjoint from the 24 TRAIN seeds. All
TRAIN L→R and R→L counts match exactly, including the accepted late source.
The new late anchors have 11–38 unfinished post-gate agents (median 25),
whereas the original full expert trajectories have at most five. Per TRAIN
direction, the phase audit counts 532,380 active approach, 98,921 entry,
116,006 crossing, 614,146 post-gate-far, 228,496 near-goal, and 61,080
settled action-agent steps. All 175 trajectories per direction terminate in
actual full simulator success. The merged manifest SHA256 is
`0c79e09adda8557d7777cf2d2efe06651293c3e937b8adff1bf177d5f9005706`.

| Fresh v6 final control | Complete success | Collision | Mean final goal occupancy | Mean gate crossing fraction | Mean projection correction (m/s, joint norm) |
|---|---:|---:|---:|---:|---:|
| Single-active L→R | 2/2 | 0/2 | 1.00 | 1.00 | 0.0021 |
| Single-active R→L | 2/2 | 0/2 | 1.00 | 1.00 | 0.0072 |
| Five-active L→R | 2/2 | 0/2 | 1.00 | 1.00 | 0.0419 |
| Five-active R→L | 2/2 | 0/2 | 1.00 | 1.00 | 0.0430 |
| Full one-way L→R | 0/2 | 0/2 | 0.57 | 0.78 | 0.2914 |
| Full one-way R→L | 0/2 | 0/2 | 0.53 | 0.98 | 0.2414 |
| Temporal even-first | 0/2 | 0/2 | 0.64 | 1.00* | 0.1095 |
| Temporal odd-first | 0/2 | 0/2 | 0.81 | 1.00* | 0.1349 |

The temporal control's aggregate gate-crossing fraction includes the staged
group's original side and is not a direct measure of successful final-goal
traversal; complete success and final occupancy are the acceptance measures.
The per-episode metrics and full traces are saved in
`diagnostics/gap_flow_scale_20261007/n50_v6_final_fresh2_complete_metrics.json`
and its eight source evaluation directories.

The second controlled candidate (v7) keeps the same 26,114-parameter Flow,
action normalization, loss, source-balanced/phase sampling fractions,
30,000-step schedule, 10-step action integration, and unchanged safety chain.
It initializes from the v6 offline-best snapshot and changes only the
evidence-based training data. It finished all 30,000 CPU steps without a
numerical error. The fixed sampled train CFM loss changed from 0.16698 to
0.14042; fixed DEV loss from 0.07365 to 0.07350, with its lowest sampled
value 0.06942 at step 24,000. These values are not an acceptance gate.

In 28 one-step held-out dense post-gate states from five DEV expert
recoveries, v6 best averaged 0.317 m/s projected goalward action and the v7
final snapshot 0.356 m/s, against the expert's 0.424 m/s. On the v6-best
failed late-state recovery subset, the model's projected goalward action
rose from 0.224 to 0.402 m/s, against the expert's 0.473 m/s. This supports
the diagnosed coverage repair at those states, but complete closed-loop
navigation remains the decision criterion. The exact per-state comparison is
in `diagnostics/gap_flow_scale_20261007/n50_v7_dense_dev_snapshot_screen.json`.

The first complete-horizon DEV screen used the original 40,000-step limit,
one Flow sample per step, the unchanged hard safety filter, and exactly
mirrored L→R/R→L physical starts and goals:

| v7 snapshot | Direction | Complete success | Steps | Crossed / 50 | Ever reached / 50 | Final at goal / 50 | Collision |
|---|---|---:|---:|---:|---:|---:|---:|
| Offline-best at 24,000 steps | L→R | 1/1 | 13,153 | 50 | 50 | 50 | 0/1 |
| Offline-best at 24,000 steps | R→L | 1/1 | 32,327 | 50 | 50 | 50 | 0/1 |
| Final at 30,000 steps | L→R | 1/1 | 13,621 | 50 | 50 | 50 | 0/1 |
| Final at 30,000 steps | R→L | 0/1 | 40,000 | 50 | 48 | 41 | 0/1 |

This selects the offline-best snapshot for larger **non-opposing** screening;
it does not freeze the model. Its R→L completion is substantially slower
than L→R, and larger fresh cohorts must establish reliability. At the final
snapshot's failed R→L step 39,000, five unfinished post-gate agents receive
mean Flow goalward velocity −0.091 m/s while the existing expert and same
projector provide +0.491 m/s; the model's projection correction is zero at
that sampled step. Thus the final snapshot still exhibits a Flow-side late
goal-navigation failure despite crossing 50/50 gates.

The first two fresh full-density same-direction states for the selected
offline-best snapshot succeeded in each direction, with complete episode
steps L→R 14,701 and 28,429; R→L 15,448 and 21,419. The first temporally
separated odd-first and even-first states also succeeded in 29,387 and
36,541 steps respectively; the latter leaves little margin under the
40,000-step horizon. These are screening outcomes only. A separate
12-physical-state acceptance cohort is running, including single-active,
five-active, full one-way, and temporally released traffic in both orders.

Independent fresh acceptance data exposed a further limitation of v7. It
passed all 12 single-active and all 12 five-active states **per direction**,
but it did not pass the dense or temporally separated gate. The four long
categories were stopped early once two major temporal failures per order
made the desired 11/12 success impossible; their counts are explicitly
partial and are not presented as twelve-case rates:

| Fresh v7 best control | Complete success / cases run | Collisions |
|---|---:|---:|
| Single-active L→R | 12/12 | 0 |
| Single-active R→L | 12/12 | 0 |
| Five-active L→R | 12/12 | 0 |
| Five-active R→L | 12/12 | 0 |
| Full one-way L→R | 7/7 | 0 |
| Full one-way R→L | 2/3 | 0 |
| Temporal release L→R first | 0/2 | 0 |
| Temporal release R→L first | 0/2 | 0 |

The failed full one-way R→L state timed out at 44/50 goal occupants; the
four temporal failures ended at 40/50, 45/50, 38/50, and 27/50. At the
R→L one-way failure's step 39,000, ten
unfinished agents receive mean Flow goalward velocity +0.078 m/s, whereas
the unchanged expert and safety projection provide +0.409 m/s; the Flow
projection correction is only 0.010 m/s in joint norm. In the even-first
temporal failure, the ten unfinished agents receive approximately zero
goalward Flow action while the expert and same projector provide +0.341 m/s.
Both temporal orders are feasible from their held-out failed step-30,000
states: the existing same-direction expert plus the unchanged safety filter
recovered 50/50 agents within 1,776 and 1,414 more steps, with positive
continuous swept clearance 0.001085 and 0.000114 m. These DEV states remain
diagnostics only and cannot enter training. Thus v7 improves some late joint
states but is **rejected**, not frozen, for high-density non-conflicting
navigation. Its safe complete L→R run is retained as a full video, not as
evidence that it meets the N=50 competence gate.
The exact feasibility requests and results are in
`diagnostics/gap_flow_scale_20261007/n50_v7_temporal_expert_feasibility_20261008/`;
the diagnostic code is `bottleneck_family/audit_n50_temporal_feasibility.py`.
The 23,124-frame successful one-way visualization and provenance manifest are
in `diagnostics/gap_flow_scale_20261007/videos/n50_v7_oneway_lr_safety_success.*`.
The representative temporal failure is separately rendered through its
entire 40,000-step horizon, yielding 40,001 MP4 frames, with no collision
and a positive 0.000330 m minimum swept clearance:
`diagnostics/gap_flow_scale_20261007/videos/n50_v7_temporal_postgate_failure.*`.

The recovery-state distribution offers a specific next hypothesis: both
v6/v7 recovery collections originated from **L→R model rollouts** and were
mirrored to balance the labels. A non-reflection-equivariant learned Flow
can reach different R→L failure states; mirrored L→R data do not cover those
actual R→L trajectories. Train-only R→L Flow traces are now being collected
with predeclared anchors; each successful expert recovery will still be
paired with its mirrored L→R replay. No acceptance or opposing-test physical
state is used for this data generation. A second train-only, non-opposing
control starts 25 agents on one side while the other 25 are already parked
at their destination on that same side. It addresses the missing stationary
goal-side context in the temporal control without a release, priority, or
opposing-traffic demonstration. Each successful trajectory is paired with a
mirrored R→L replay. These are hypotheses to test, not claims that the next
controller is already competent.

The train-only R→L model collection is now complete: 16 independent
same-direction physical cases generated 32 predeclared anchors from 5,000,
15,000, 25,000, and 29,999 steps when the trace still existed. Thirty
expert/safety continuations reached all goals and were successfully replayed
in both directions; two expert continuations timed out and were retained in
the collection reports but excluded from training. The accepted set adds
exactly 30 trajectories and 109,523 joint transitions **per direction**.
Those R→L states were visited by the actual v7 Flow, not inferred by
reflecting a L→R Flow trace. Fifteen of the 16 source model traces reached
all goals before 30,000 steps; one remained in a pre-gate queue at that
horizon. The repair therefore broadens actual R→L state coverage but does
not claim that all source traces were failed post-gate states. The four
shards and their immutable preflight, trace, exclusion, and replay evidence
are in `diagnostics/gap_flow_scale_20261007/n50_v7rl_late_dagger/`.

The independent parked-group collection finished with **24/24 TRAIN and
6/6 DEV base cases successful**, each replayed in both directions through
the unchanged simulator. It adds 24 complete TRAIN trajectories and 185,258
joint transitions per direction, plus six DEV trajectories and 45,626 joint
transitions per direction. No teacher failure or mirror failure occurred;
its provenance is
`diagnostics/gap_flow_scale_20261007/n50_parked_context_v8_base/`.
The previously collected R→L Flow-state recoveries were merged only after
checking that every copied parent record was unchanged. The production
dataset reader verified the merged train and DEV splits.

| Final v8 competence dataset, per direction | TRAIN trajectories | TRAIN transitions | DEV trajectories | DEV transitions |
|---|---:|---:|---:|---:|
| Complete nominal / parked-group | 120 | 792,857 | 36 | 228,425 |
| Initial dense recovery | 48 | 137,966 | 0 | 0 |
| v6-Flow late recovery | 31 | 49,981 | 0 | 0 |
| Actual v7 R→L Flow-state recovery, paired | 30 | 109,523 | 0 | 0 |
| **Total** | **229** | **1,090,327** | **36** | **228,425** |

TRAIN has 48 independent physical seeds, DEV has 12, with zero overlap.
Every one of 229 TRAIN and 36 DEV trajectories per direction terminates in
actual 50-agent success. The merged manifest SHA256 is
`0aa368f1572832c7662d66be3475ed1d1bf492e65520cb0598fac58711cddd9c`.
The exact phase audit is
`diagnostics/gap_flow_scale_20261007/n50_goal_occupancy_dataset_v8/phase_audit.json`.
Per TRAIN direction it contains 705,276 approach, 144,009 gate-entry,
170,694 crossing, 892,598 post-gate-far, 305,964 near-goal, and 81,325
settled **moving action-agent steps**. Those six counts and the trajectory
counts match exactly between directions. The corresponding fractions of
moving task supervision are 30.7%, 6.3%, 7.4%, 38.8%, 13.3%, and 3.5%.
The largest accepted trajectory contains 44 simultaneously unresolved
post-gate agents, versus at most five in the original full-density expert
trajectories. The audit's loss-mass column is only a target-magnitude proxy
`1 + 19 I(||a|| > 0.1)`; it is not asserted to be the CFM squared-error
loss. Old model CFM loss fractions are reported above.

One v8 candidate is now training from the non-frozen v7 offline-best
snapshot. It keeps the 26,114-parameter set-attention architecture, active
action normalization, motion loss weighting, source-balanced and phase
sampling, flow integration, and untouched accepted safety law. The only
substantive change is the documented competence data; the 40,000-step
schedule allows exposure to its larger, more diverse train split, with
5,000-step snapshots for non-opposing checkpoint screening. The candidate
will not be frozen based on offline loss. A new fresh 12-state acceptance
cohort uses seed 99171, disjoint from all TRAIN and DEV physical seeds and
from the earlier v7 acceptance cohort. Opposing Gap1 remains unopened.

The first Slurm training attempt had a 24 GB job memory limit and was killed
while loading the larger dataset, before any model update. Its exact log is
`diagnostics/gap_flow_scale_20261007/n50_train_goal_v8_slurm_7727.log`.
The retry reserved 40 GB without changing data, code, hyperparameters, or
initial weights; it completed 40,000 GPU steps in 53.0 s at
`diagnostics/gap_flow_scale_20261007/n50_train_goal_v8_retry40g/`.
The fixed sampled train loss changed 0.15506→0.14503 and independent DEV
loss 0.07291→0.06212; offline-best occurred at step 40,000. The checkpoint
is **not frozen**. In 28 held-out dense post-gate one-step states, mean
projected goalward action changed from v7 best 0.35431 m/s to v8 best
0.37144 m/s, while mean active-agent safety correction changed from
0.02105 to 0.01731 m/s. Expert action averaged 0.42388 m/s.
`diagnostics/gap_flow_scale_20261007/n50_v8_dense_dev_snapshot_screen.json`
contains the per-state comparisons. Complete-horizon non-opposing screens
remain the checkpoint gate.

The v8 best checkpoint's independent two-case complete-horizon screen used
fresh DEV seed 77132, one Flow sample each step, goal-stop, the accepted
hard safety filter, and the original 40,000-step limit. These are screening
cases, not the pre-freeze 12-state acceptance estimate:

| Non-opposing category | Full task success | Collisions | Final goal occupants |
|---|---:|---:|---:|
| Single-active L→R | 2/2 | 0 | 50/50, 50/50 |
| Single-active R→L | 2/2 | 0 | 50/50, 50/50 |
| Five-active L→R | 2/2 | 0 | 50/50, 50/50 |
| Five-active R→L | 2/2 | 0 | 50/50, 50/50 |
| Full one-way L→R | 2/2 | 0 | 50/50, 50/50 |
| Full one-way R→L | 2/2 | 0 | 50/50, 50/50 |
| Temporal release L→R first | **0/2** | 0 | 21/50, 22/50 |
| Temporal release R→L first | 2/2 | 0 | 50/50, 50/50 |

In both even-first temporal failures the first 25-agent L→R group crossed
the opening but did not all reach its final goals, so the externally
scheduled second group was never released. The final 2,000-step average
goalward progress of unfinished agents was only 3.54e-5 and 2.89e-6 m/s.
For the first failure at step 39,000, three first-group agents remained
0.16–0.29 m from goal. Flow's projected goalward commands for them were
−0.018, −0.129, and +0.042 m/s. The expert passed through the **same**
projector yielded +0.447, +0.487, and +0.381 m/s, and safely brought the
whole first group to goal in just ten steps from that physical state.
Wall constraints were absent in the last 2,000 steps of this failure and
mean joint projection correction was only 0.005 m/s. Thus the residual is
a Flow-side final-goal convergence failure, not a safety-induced deadlock.
All saved v8 snapshots gave near-zero or negative goalward action at the
same state; offline checkpoint choice does not explain this failure.
Across 64 independently sampled Flow latents with the evaluator's folded
PRNG convention at that state, the three pending agents' mean goalward
commands were +0.009, +0.011, and −0.054 m/s; only 8/64 samples asked all
three to move goalward. The average over agents and latents was −0.011 m/s.
Thus a single unlucky action sample is not the main explanation either.
The exact probe is in
`diagnostics/gap_flow_scale_20261007/n50_v8_temporal_terminal_latent_audit.json`
and `bottleneck_family/audit_n50_terminal_latents.py`.

`diagnostics/gap_flow_scale_20261007/n50_v8_screen_complete_metrics.json`
contains per-episode full-horizon success, goal occupancy, throughput,
wall/pair activation, correction and clearance. The first-group expert
feasibility plan and result are in
`diagnostics/gap_flow_scale_20261007/n50_v8_temporal_firstgroup_expert_feasibility/`.
The independent, predeclared 12-state temporal acceptance cohort uses seed
99171 and has no TRAIN/DEV seed overlap. Its first two complete 40,000-step
L→R-first trials **both** timed out, collision-free, with 24/50 and 19/50
final goal occupants and no release of the second group. Because even ten
subsequent successes would only give 10/12, and these were major terminal
failures, the remaining ten trials were stopped before running. These are
partial counts, not a 12-case success estimate. v8 is **rejected and not
frozen**. Opposing traffic and eta data remain unopened.
The representative first fresh acceptance failure is rendered from the
actual initial frame through all 40,000 simulator steps at
`diagnostics/gap_flow_scale_20261007/videos/n50_v8_temporal_terminal_failure.mp4`;
its adjacent manifest records 40,001 frames, timeout, no collision, and a
positive 0.000143 m minimum continuous swept clearance.

The most specific remaining training gap is the first-phase 25-agent
same-direction context: the old nominal dataset contained only eight
independent half-traffic TRAIN physical seeds, and neither v7 nor v8 had
Flow-state recoveries from that category. Eight train-only v8 Flow half-
traffic trajectories were therefore sampled at fixed absolute anchors and
at fixed offsets before their own terminal time. All eight source traces
themselves succeeded, but 100 steps before success they typically had
one to three unresolved post-gate agents, with goal distances spanning
0.12–0.9 m. This directly covers the terminal-convergence class observed
in the held-out failure without importing a held-out state. Expert/safety
continuations are being accepted only after full 50-agent success and
mirrored simulator replay; excluded teacher timeouts are recorded. A
separate, mirrored 24-TRAIN/6-DEV collection of new independent half-traffic
initial states broadens that specific non-opposing geometry. Neither
collection contains simultaneous opposing traffic or release priorities.

The train-only v8 half-traffic model traces finished 8/8 task-successfully.
The predeclared recovery protocol used absolute steps 1,000, 3,000, 5,000,
9,000, 15,000, and 25,000 when present, plus offsets 1,500, 500, and 100
before each source trace's terminal step; duplicate or out-of-range anchors
were removed deterministically. Of 47 actual candidate states, 43
expert/safety continuations reached all 50 goals and passed both simulator
replays, while four timed out and were excluded. The accepted set has
43 trajectories and 51,206 joint transitions **per direction**. In 34/43
accepted L→R source states at least one agent was beyond the gate and
0.08–1 m from its goal, with 98 such near-goal agents in total across
the source states. Shard preflight, source model traces, exclusion records,
and mirrored rollouts are retained in
`diagnostics/gap_flow_scale_20261007/n50_v8_half_dagger/`.

The independent half-traffic collection also finished **24/24 TRAIN and
6/6 DEV** base cases successfully, with no expert or mirror failures. Its
new TRAIN trajectories add 170,514 joint transitions per direction; DEV
adds 43,796. `diagnostics/gap_flow_scale_20261007/n50_half_diversity_v9_base/`
retains its preflight, complete trajectories, and exact accepted counts.
The final merged v9 competence dataset is
`diagnostics/gap_flow_scale_20261007/n50_half_terminal_dataset_v9/dataset/`,
manifest SHA256
`42515a620bac810dcbf90a79ce4e85e329134ce2b7ec493d1f6479fcce0b84c0`.

| v9 data, per direction | TRAIN trajectories | TRAIN transitions | DEV trajectories | DEV transitions |
|---|---:|---:|---:|---:|
| v8 audited source | 229 | 1,090,327 | 36 | 228,425 |
| New complete half-traffic | 24 | 170,514 | 6 | 43,796 |
| Successful model-state half recoveries | 43 | 51,206 | 0 | 0 |
| **Total** | **296** | **1,312,047** | **42** | **272,221** |

TRAIN has 72 independent physical seeds and DEV 18, without overlap.
The new fresh screening seed 77134 and acceptance seed 99271 are absent
from both splits and from the earlier v8 acceptance cohort. The production
reader verified every train/dev file and trajectory digest. Every one of
296 TRAIN and 42 DEV trajectories per direction terminates in full 50-agent
success. The exact audit is
`diagnostics/gap_flow_scale_20261007/n50_half_terminal_dataset_v9/phase_audit.json`.
Per TRAIN direction, active moving agent-steps are 838,250 approach,
195,390 gate-entry, 203,320 crossing, 1,073,347 post-gate-far,
352,929 near-goal, and 89,859 settled; all match exactly under reflection.
The corresponding fractions of moving task supervision are 30.4%, 7.1%,
7.4%, 39.0%, 12.8%, and 3.3%. These are phase counts, not fitted CFM
loss percentages.

One controlled v9 candidate now trains from the rejected v8 offline-best
checkpoint. It keeps the same 26,114-parameter set-attention architecture,
active-action normalization, CFM objective and weighting, 10-step Flow
integration, and original hard safety projection. The only substantive
change is the audited half-traffic and terminal-recovery data, with the
same 40,000-update schedule and documented source/near-goal sampler. It
will be selected only on non-opposing navigation and will not be frozen
on offline loss or on opposing traffic.

The v9 Slurm run completed all 40,000 updates in 41.7 s with the same
model semantics. Fixed sampled train loss was 0.15369 initially and
0.14518 at the final step; independent DEV loss was 0.05485 initially,
0.05238 at its offline-best step 36,000, and 0.05475 at the final step.
The offline-best SHA256 is
`f82ab536f14a50fb88cdeeb321cc1b5faeb4d0e72a869cd20aef248992ce2623`;
this is an **unfrozen candidate**, not an accepted controller. Across the
same 28 older held-out dense post-gate one-step states, projected
goalward action was 0.371 m/s for v8 best, 0.374 m/s for v9 best, and
0.378 m/s for v9 final. These small one-step differences are diagnostic
only; the complete-horizon non-opposing screen on fresh seed 77134 is the
next selection gate. Its physical states are absent from TRAIN and DEV.

Opposing Gap1 and eta evaluation remain unopened until a new candidate passes
complete-horizon non-conflicting success, is frozen, and then yields a useful
opposing safety-gridlock regime.

## v9 closed-loop rejection and gate-regression diagnosis

The offline-best v9 checkpoint was screened on fresh physical seed 77134,
which was outside TRAIN and DEV, for **two complete episodes in each mode**.
No collision or numerical failure occurred. Sparse modes reached the final
goals, but full-density and temporally separated modes did not:

| N=50 mode | Complete success | Mean final goal occupancy | Mean gate crossing | Mean joint safety correction (m/s) |
|---|---:|---:|---:|---:|
| Single L→R | 2/2 | 100% | 100% | 0.0084 |
| Single R→L | 2/2 | 100% | 100% | 0.0018 |
| Five active L→R | 2/2 | 100% | 100% | 0.0559 |
| Five active R→L | 2/2 | 100% | 100% | 0.0469 |
| Full-density L→R | **0/2** | **2%** | **3%** | 0.2246 |
| Full-density R→L | 1/2 | 92% | 92% | 0.2908 |
| Temporal, even first | 0/2 | 67% | diagnostic only | 0.1190 |
| Temporal, odd first | 0/2 | 58% | diagnostic only | 0.3170 |

The temporal gate-crossing summary is not a physical second-group throughput
estimate because some agents start beyond the barrier; final occupancy and
phase-release logs are the relevant metrics. Full per-rollout traces and
aggregates are in `diagnostics/gap_flow_scale_20261007/n50_v9_screen_*/`
and `n50_v9_screen_complete_metrics.json`.

The two full-density L→R episodes timed out at 40,000 simulator steps with
0/50 and 2/50 agents at goal, respectively. In the first, **none of the 50
agents crossed the gate**, mean goal distance fell from 11.19 to only 7.10 m,
and the final 2,000 steps made essentially zero net progress. At step 39,000,
the Flow action requested −0.0006 m/s mean radial goalward progress, versus
+0.3075 m/s for the unchanged one-way expert and +0.1212 m/s after the
**same** safety projection. The failure is therefore not explained solely by
the projector blocking a good command. Wall and pair constraints were active
in 89.5% and 100% of the final 2,000 steps; continuous swept clearances
stayed positive. This is a non-opposing navigation failure, **not** a valid
opposing safety deadlock.

On that exact held-out initial state, the earlier v8 candidate reached
50/50 goals in 10,631 steps with the same observation and safety chain.
The v9 final-40,000-update snapshot also timed out on the first L→R and
temporal-even cases; further snapshot trials were stopped after both failed
the pre-freeze gate. The v9 offline-best is **rejected and never frozen**.
Fresh 12-case sparse acceptance on seed 99271 nevertheless gave single
L→R 12/12, single R→L 12/12, five-active L→R 12/12 and five-active R→L
12/12, all collision-free; this does not rescue full-density competence.

The regression is reproducible on the training physical starts themselves:
16 independent v9 Flow `full_LR` traces reached only 0 or 1 gate crossings
by step 1,000 and none reached a final goal by step 2,500. The v9 sampler
did contain about 67.5% full-density *frames* in its estimated mixture, but
its explicit early stratum selected only the first 50 steps of each source
trajectory. The difficult full-density gate encounter occurs roughly at
steps 500–2,000; long post-gate and near-goal frames dominate the rest of
the sample. Thus the dataset has full-density trajectories, while the
training schedule insufficiently rehearsed their decisive gate-approach
phase during 40,000 fine-tuning updates. The old v8 model was competent
on this initial state before that fine-tuning. This is an explanation
supported by the sampling audit and paired rollout, not proof that no other
optimization effect contributed.

One controlled v10 repair is being prepared: retain the v8 checkpoint as
initial weights, retain the balanced complete v9 demonstrations, add
train-only mirrored expert continuations from the v9 Flow's own one-way
gate-congestion states at predeclared steps 1,000 and 2,000, and use the
already-supported early-transition sampler over the first 2,500 steps of
nominal trajectories rather than the first 50. The early fraction is set
to 0.35, near-goal fraction remains 0.30, and source-balanced ordinary
sampling remains enabled. No architecture, safety algorithm, opposing
demonstration, or action semantics changes. All acceptance physical states
remain outside training, and no opposing N=50 evaluation has been opened.

The gate-recovery collection has now closed: 16 independent train-only v9
full-density L→R model traces were sampled at the predeclared 1,000/2,000
steps. Of 32 candidate continuations, **28 succeeded** under the unchanged
one-way queue expert and hard safety projection and were mirrored/replayed
as 28 R→L trajectories; four timed out and were excluded. In two of those
excluded states, even the existing simultaneous one-way expert plus the same
projector failed to finish within a 12,000-step diagnostic horizon, reaching
only 5/50 and 2/50 goals without collision. Thus recovery from every bad
v9 joint state cannot be assumed; the new controller must also avoid entering
those states. Exact inclusion and exclusion records are in
`n50_v9_pregate_recovery_pair*/collection_report.json` and the alternative
expert probe is in `n50_v9_pregate_allteacher_probe/summary.json`.

The merged v10 competence dataset is
`diagnostics/gap_flow_scale_20261007/n50_gate_terminal_dataset_v10/dataset/`,
manifest SHA256
`a56773c257a39124d61c6c1380d14bc0a3151730a95c344a9e773e7aff0f10ed`.
TRAIN has exactly **324 complete trajectories and 1,495,637 joint
transitions per direction**; DEV remains at 42 and 272,221 per direction.
The train/dev independent physical-seed counts are 72/18, with no overlap.
Every accepted recovery has a complete all-50 success terminal state and
positive swept clearance; the four expert timeouts remain audit evidence,
not labels. Train moving action-agent phase fractions per direction are
27.57% approach, 7.16% entry, 7.76% crossing, 40.58% post-gate-far,
13.28% near-goal and 3.65% settled, exactly matched by reflection.
`n50_gate_terminal_dataset_v10/phase_audit.json` contains the full counts.

One v10 training run used the same 26,114-parameter Flow, CFM loss,
normalization, 10-step integration and safety pipeline. It resumed **v8**
weights (SHA256 `bfda7acdc3aeb125d09e67ab3ed544968c5a80878b942764a14cda110666ae85`)
and used the existing sampler with a 35% nominal first-2,500-step stratum,
30% near-goal stratum and source-balanced remainder. The job script is
`n50_train_gate_terminal_v10.sbatch`; exact configuration is
`n50_train_gate_terminal_v10/config.json`. After 30,000 updates, sampled
train loss was 0.11097 initially and 0.10303 final; DEV was 0.05165
initially, 0.04597 at its offline-best step 21,000 and 0.04804 final.
The offline-best file (SHA256
`4f0922c7f1b467b09853c29edc816cb57e8149c83f02f5aa419435ab96e54a55`)
is **not frozen**. Fresh complete-horizon non-opposing validation on seed
77136, plus a paired check on the old v9 failure state, rejected this
offline-best checkpoint. The 10,000-update snapshot (SHA256
`6fb5293b801238ea1e2fb0f05f352344e97a6892ca556ef86d711eb6af746bfa`)
was also rejected. The full 40,000-step safety-chain screen used two fresh
physical states per category:

| Non-opposing category | v10 best complete success | v10 step-10k complete success |
|---|---:|---:|
| Single L→R | 2/2 | 2/2 |
| Single R→L | 2/2 | 2/2 |
| Five active L→R | 1/2 | 2/2 |
| Five active R→L | 2/2 | 2/2 |
| Full one-way L→R | 0/2 | 2/2 |
| Full one-way R→L | 1/2 | **0/2** |
| Temporal even first | 1/2 | 1/2 |
| Temporal odd first | 2/2 | 2/2 |

All these rollouts were collision-free, but 0/2 dense R→L and 1/2 temporal
even first preclude the step-10k snapshot from being frozen. Its failed dense
R→L cases finished with 47/50 and 32/50 goal occupants. The complete
per-rollout results are in `n50_v10_screen_*/summary.json` and
`n50_v10s10_screen_*/summary.json`. Fresh sparse 12-case controls under the
step-10k snapshot completed 12/12 single and five-active episodes in both
directions, all without collision. The first four of a larger dense L→R
cohort also passed, but the remaining eight were stopped after the dense
R→L screen proved the candidate unacceptably asymmetric; **4/4 is not a
12-case result**.

At the failed dense R→L trace's late state, Flow requested almost no
goalward action for 15 agents still before the gate and three beyond it;
wall and pair constraints were active, but the Flow command had already
collapsed. Its mirror L→R state succeeded under the same checkpoint. In
246 train/dev expert frames sampled at steps 500, 1,000, and 1,500, the
step-10k Flow reproduced the expert's moving-agent decisions with at least
0.995 recall and at least 0.974 precision, with positive action alignment
in both directions. This is evidence for closed-loop covariate shift:
expert-state imitation is good, yet a few trajectory deviations can create
unfamiliar collective configurations. It does not by itself establish that
the set architecture or goal-feature scaling is fundamentally defective.

An independent control applied the existing *simultaneous* same-direction
expert action to 50 agents under the same hard safety projection. On the
first paired fresh physical state in each direction, both ran 40,000 steps,
crossed 0/50 agents, reached 0/50 goals, and remained collision-free. The
pairwise and wall constraints were active on nearly every step, with mean
joint projection correction 3.52 m/s. This shows why the ordinary one-way
competence data use a state-only same-direction queue. It is not an
opposing-side yielding demonstration and is not a runtime Flow module.
The results are in `n50_allmoving_expert_probe_*/summary.json`.

A data-only v11 candidate is being assembled from successful expert
continuations of **train-only**, non-opposing Flow states: early dense R→L
queue deviations and late sparse/partially occupied goal configurations.
Every accepted continuation is mirrored and replayed through the unchanged
simulator, so training direction counts remain exactly balanced. Expert
timeouts remain excluded with explicit audit records. No v11 model has been
frozen and opposing N=50 and eta remain unopened.

## v11 balanced data and closed-loop selection

The v11 data merge was sealed **before** any v11 closed-loop evaluation. It
uses five completed early R→L model-state recovery shards, four five-active
terminal shards, and two parked-half terminal shards. Additional predeclared
train-state collection jobs were still running at seal time and therefore
were not selected by observing v11 validation outcomes. All 78 added
continuations per direction are full, collision-free expert successes and
have exact mirrored replays. There are 30 early queue and 48 terminal
continuations per direction. The production dataset reader validated the
merged records and digests. No opposing demonstration was imported.

| v11 split and direction | Trajectories | Joint transitions | Independent physical seeds | Complete all-agent expert successes |
|---|---:|---:|---:|---:|
| TRAIN L→R | 402 | 1,720,116 | 72 | 402 |
| TRAIN R→L | 402 | 1,720,116 | 72 | 402 |
| DEV L→R | 42 | 272,221 | 18 | 42 |
| DEV R→L | 42 | 272,221 | 18 | 42 |

TRAIN phase-active action-agent steps match exactly in both directions:
approach 996,549 (25.32%), entry 309,319 (7.86%), crossing 315,765
(8.02%), post-gate far 1,635,514 (41.55%), near goal 529,555 (13.45%),
settled 149,638 (3.80%). The dataset manifest SHA256 is
`9371ccd1873deb3dbc749883f5eebd739cd12efae49f663aee2b61ed4fe44bb4`.
Exact source and phase counts are in `n50_recovery_dataset_v11/merge_report.json`
and `n50_recovery_dataset_v11/phase_audit.json`.

The v11 controlled run starts from the **rejected but non-opposing-strong**
v10 step-10k snapshot. It retains the 26,114-parameter set actor,
normalization, 10-step CFM integration, motion-loss weight 20, the 35%
first-2,500-step nominal stratum, 30% near-goal stratum, source balance,
permutation augmentation, and unchanged hard safety projection. The only
modeling change is the additional balanced training-state coverage. CPU
training completed 20,000 updates without numerical failure; the exact
training arguments are in `n50_train_recovery_v11.sbatch`, and the actual
CPU result is in `n50_train_recovery_v11_cpu.log`. Fixed sampled DEV loss was
0.05695 initially, reached its minimum 0.05420 at step 12,000, and was
0.05732 finally. These losses do not establish navigation competence;
the complete-horizon non-opposing screens below decide it.

The first complete-horizon v11 screen has **rejected** the 5,000- and
10,000-update snapshots: on one fresh full-density one-way physical state
per direction, final all-agent task success was 0/1 for each direction and
snapshot, with no collision. The 5,000-update checkpoint ended at 8/50
L→R and 35/50 R→L final goal occupants; the 10,000-update checkpoint at
0/50 and 6/50. By contrast, the step-10k sparse controls completed 2/2
single-active and 2/2 five-active episodes in **each** direction. This is a
density-specific regression and cannot be hidden by reporting gate crossing
or sparse success alone.

In the failed step-10k L→R rollout, **0/50** crossed the gate. The Flow itself
requested >0.1 m/s for 6/50 agents at step 0, 35/50 at step 25, and 50/50
at step 75. At step 75 the hard safety projection correction was exactly
zero in the recorded trace; the simultaneous motion request originated in
Flow. The state-only same-direction expert requests meaningful motion for
only the current entrant. By step 5,000, mean goal distance stopped
decreasing, and the final 2,000 steps had nearly zero net goal progress.
On that off-expert step-100 state, **all** tested checkpoints, including the
earlier v10 step-10k, propose substantial motion for all 49 agents that
the queue expert would hold. This is direct evidence that a small early
closed-loop displacement can push the joint state outside the learned
one-way queue policy's reliable region. It also explains why late terminal
recoveries alone did not repair dense traffic.

The next controlled data correction therefore uses only train-split,
non-opposing model traces at predeclared early anchors (1–100 steps) and
retains only full expert/safety successes. A separate `early_queue_recovery`
source and an explicit first-50-step onset sampling stratum will keep the
rare critical queue decisions visible during training, while ordinary
first-2,500-step and near-goal strata remain. No new safety law, opposition
demonstration, or benchmark geometry change is involved. The 12,000-update
offline-best v11 snapshot and temporal diagnostic were completed for the
failure audit; neither redeems the rejected snapshots without independent
complete task success.

The completed v11 offline-best (12,000-update) dense screen also timed out
at 40,000 steps in both directions, with 13/50 final goal occupants each;
it is rejected. The step-10k temporal controls had 0/1 even-first (35/50
goals at timeout) and 1/1 odd-first (full success only at step 38,736).
This mixed temporal result does not overcome the complete one-way failures.

A focused action audit used all nine held-out `full` expert trajectories in
each direction and the first 0, 10, 25, 50, and 100 steps. On the **expert
state manifold**, the v10 step-10k Flow predicted roughly one moving agent
per frame at steps 10–100 in both directions, matching the expert's one;
however, at step 0 it averaged 0.9 moving agents L→R and 3.1 R→L, while
the expert had exactly one in each. The v11 step-10k model averaged 5.0
and 8.0, respectively, at step 0. Once the v11 controller's own rollout
drifted, all tested checkpoints requested motion from 49 expert-held agents
at its step-100 state. Thus the first departures from the queue manifold
and the lack of recovery from them are both supported by direct action
comparisons. `bottleneck_family/audit_n50_onset_actions.py` and
`n50_onset_heldout_action_audit.json` record all per-frame counts. This
does **not** prove that the model architecture is incapable; it identifies
the closed-loop state distribution that the next controlled data pass must
cover.

## v12 early-queue competence correction

The v12 training merge was sealed before opening any v12 closed-loop
validation. It imports 34 train-only, same-direction Flow-state continuations
per direction: six at steps 25/50/100 from two train physical seeds, and
28 at steps 10/25 from 14 other train physical seeds. All 34/34 expert
continuations reached 50/50 goals without collision under the existing
projection, and each was mirrored and replayed. The separate step-1/5/10/20
probe from the first two seeds was not admitted because its collection was
still unfinished at dataset seal time. No test or opposing physical state
entered the merge. The manifest SHA256 is
`ca4f40c428497a0f5f6908fff5577431c297305d366faae19e79d8382d8c9a2f`.

| v12 split and direction | Trajectories | Joint transitions | Independent physical seeds |
|---|---:|---:|---:|
| TRAIN L→R | 436 | 2,186,875 | 72 |
| TRAIN R→L | 436 | 2,186,875 | 72 |
| DEV L→R | 42 | 272,221 | 18 |
| DEV R→L | 42 | 272,221 | 18 |

The training checkpoint starts from the same v10 step-10k weights as v11,
with the same 26,114-parameter architecture, normalization and hard safety
pipeline. It allocates 30% of training frames to the first 2,500 nominal
steps, 25% to near-goal frames, 20% to the first 50 steps of successful
`early_queue_recovery` continuations, and the remaining 25% to
source-balanced ordinary frames. This sampler change addresses the measured
early queue-departure failure while retaining gate and terminal supervision;
it does not change the CFM target or intervene at evaluation time.
The planned 12,000-update run records 1,000-update snapshots for selection
on **non-opposing** closed-loop navigation only. The screen uses fresh
physical seed 77140; the later acceptance cohort, if a snapshot merits it,
uses disjoint fresh seed 99279. Opposing N=50 stays unopened until the
acceptance gate and checkpoint freeze.

The v12 phase audit verified all 436/436 train trajectories per direction
terminate in full success, with **exactly identical** L/R phase counts:
active approach 1,343,544 (27.82%), entry 337,140 (6.98%), crossing
380,303 (7.88%), post-gate far 1,956,231 (40.51%), near goal 629,031
(13.03%), and settled 182,787 (3.79%) action-agent steps. The 34 onset
recoveries per direction add 466,759 joint transitions. The exact report is
`n50_recovery_dataset_v12/phase_audit.json`.

CPU training completed all 12,000 updates without a numerical error in
178.1 s after data loading. Fixed sampled DEV loss was 0.05001 initially,
0.04823 at its offline-best step 1,800, and 0.05098 at the final step.
The offline-best is **not frozen**. A held-out one-step action audit still
finds direction and snapshot sensitivity at expert step 0; for example,
the 3,000-update checkpoint averages 1.22/2.11 false-active waiting agents
for L→R/R→L, while the offline-best averages 0.89/4.78. These numbers only
help prioritize non-opposing full-horizon screens; neither is an acceptance
metric. `n50_v12_onset_action_audit_{early,late}.json` stores the frame-level
results.

## Complete-horizon v12 decision and v13 data diagnosis

The 3,000-update v12 snapshot was tested through the unchanged hard safety
projection for the full 40,000-step limit. On one fresh state, sparse single
and five-active controls succeeded 2/2 per direction, and dense L→R succeeded
50/50 goals at step 19,233. This validates some learned gate throughput, but
is not an acceptance result. The same snapshot's dense R→L state timed out
with 46/50 at goal; both temporal-release orders timed out at 48/50 and
47/50. All were collision-free. The R→L stalled agents at step 39,000 had
mean Flow goalward velocity +0.041 m/s and mean safe velocity +0.041 m/s,
while the unchanged expert and same projector yielded +0.425 m/s; no wall or
pair constraint was active for those agents at that instant. Thus the
terminal failure belongs to Flow rather than an incorrect or blocking
safety projection.

A second independent fresh R→L full state also timed out collision-free after
all 50 crossed, but only 23/50 occupied their goals. At steps 20,000–40,000
all 50 had crossed, while only 18–23 remained within the 0.08 m tolerance;
the remaining distances were mostly 0.1–0.5 m. The second fresh L→R state
failed earlier: 9/50 crossed and 9/50 reached goal at timeout. Its first
unsafe-looking collective drift was a Flow action error, not a projection
error: at steps 0, 10, 25, and 50 Flow requested motion from 1, 11, 19, and
40 agents, respectively; the accepted safety projector changed the joint
command by only 0, 0.017, 0.012, and 0.015 m/s. The resulting crowd jammed
before the gate. The v12 checkpoint is rejected, and no N=50 opposing test
was opened.

The train full-expert data contain 110,778 frames per direction with at least
45 agents at goal, but only 1,930 frames where at least two of the remaining
agents are 0.08–0.4 m from goal, and none where five are in that band. This
misses the terminal occupancy pattern in the R→L failures. The v13 correction
therefore samples train-only successful one-way expert terminal states,
displaces 5, 10, or 20 arrived agents by 0.14–0.32 m subject to exact
geometry and disk separation, then requires the original goal-waypoint
expert plus unchanged safety projection to finish all 50 goals. Every
successful left-to-right continuation is mirrored and replayed right-to-left
through the real simulator. Failed expert continuations and geometry
rejections are reported, never silently admitted. Separately, train-only
v12 Flow traces from 16 independent full one-way states at steps 1 and 5
provide early queue-recovery supervision, because the second L→R fresh
state shows model-driven crowd activation by step 10. The data repair does
not add opposing traffic, eta, a new controller, or a new safety algorithm.
Across those 16 train-only v12 model traces, the median number of moving
agents rises from 1.5 at step 1 to 4 at step 5, 6 at step 10, 12 at step 20,
and 24.5 at step 30, while the successful queue expert initially releases
one entrant. This confirms the early divergence is not limited to one held-out
seed. The train traces are used only for supervised non-opposing recovery;
the DEV full-horizon failures remain excluded from the training merge.

The completed train-only recovery collections contain, per direction,
192 successful 5-agent terminal perturbations (42,337 joint transitions),
48 successful 10-agent perturbations (10,576 transitions), and 48 successful
20-agent perturbations (10,550 transitions). There were no geometry rejections,
expert failures, or simulator collisions in these three predeclared strata.
The 16 train physical seeds × two early anchors produce another 32/32 full
expert successes per direction and 451,143 joint transitions per direction;
all 16 independent shards exited successfully. Every recovery has an exact
mirrored partner and was replayed through `BottleneckEnv.step` with its swept
collision checker. The merged v13 manifest and phase audit are recorded
separately after reader validation; no N=50 opposing state enters this data.

The sealed v13 manifest SHA256 is
`ba2cd93f1a4bc87e652e4f8be6b4f953fff4dd01a394e9db4d2cace9407bd936`.
Its train split has **756 trajectories and 2,701,481 joint transitions per
direction**, from 72 independent physical seeds; the unchanged DEV split has
42 trajectories and 272,221 transitions per direction from 18 disjoint
physical seeds. Every one of the 1,512 train and 84 DEV trajectories has
50/50 final goal occupancy in the exact phase audit. The recovery additions
are 320 trajectories and 514,606 transitions per direction. Per-direction
active action-agent steps now partition into approach 1,681,774 (29.2%),
entry 363,337 (6.3%), crossing 441,033 (7.6%), post-gate far 2,309,829
(40.1%), near goal 755,491 (13.1%), and settled 214,275 (3.7%). The
phase-report `loss_weight_mass_fraction_proxy` is not a measured CFM loss:
it is the fixed action-weight proxy `1 + 19 I(||a|| > 0.1)` before v13's
explicit phase sampling. The phase audit and merged manifest live in
`n50_recovery_dataset_v13/`.

The v13 controlled continuation uses the unchanged 26,114-parameter set
actor, original 10 Euler Flow steps, accepted projection, and v12 step-3,000
initial weights. It trains 8,000 CFM updates with 20% first-50-step onset
recoveries, 20% first-150-step terminal recoveries, 25% early nominal,
20% near-goal, and 15% source-balanced ordinary frames. The fixed DEV loss
reaches 0.05940 at step 7,600; this is not a checkpoint acceptance criterion.
On held-out synthetic terminal states from six DEV full trajectories, the
offline-best v13 snapshot improves mean safe goalward speed from about
0.03–0.05 m/s (v12 step-3,000) to 0.08–0.10 m/s, but the same unchanged
expert/safety chain provides 0.43–0.44 m/s. At expert step 0, v13 step-1,000
falsely activates 3.0/10.44 waiting agents L→R/R→L and the offline-best
activates 7.44/21.78, versus v12 step-3,000's 1.22/2.11. The same late
snapshot falsely activates 4.88/19.25 on TRAIN first frames, so this is
not only held-out distribution shift. However a full 40,000-step one-way
screen on the first fresh R→L physical state succeeds at step 15,884 for
v13 step-1,000 and at step 19,047 for the offline-best, both 50/50 and
collision-free. These full-horizon successes make one-step false-activation
statistics a warning rather than a sufficient failure verdict.

The v14 controlled sampler change explicitly allocates 20% of frames to
the first five frames of nominal demonstrations, because v13's 25% pool
over the first 2,500 steps made the exact first action rare. It uses the
same sealed v13 data, initial v12 checkpoint, capacity and safety chain.
Snapshots after 1,000–8,000 steps still falsely activate about 6–19 waiting
agents at held-out expert step 0; the offline-best artifact is the initial
v12 controller, not a trained improvement. A separate 2,000-update fitting
diagnostic with 80% of frames drawn from the exact first nominal frame still
has 6.28/4.00 false-active TRAIN agents L→R/R→L, worse than the initial
v12's 1.28/1.81. Thus simply increasing first-frame exposure does not fix
the current CFM endpoint behavior. Neither v13 nor v14 is frozen; complete
closed-loop non-opposing results still determine any acceptance.

## Exact reflection audit and non-frozen symmetry candidate

Six untouched DEV full-traffic trajectory pairs give **exact zero** maximum
error when their eight observation features and expert actions are reflected
across the Gap1 vertical axis. The v13 offline-best sampled Flow action mean,
averaged over 64 independent Gaussian latents, differs from its mirrored
counterpart by 0.105 m/s per agent at step 0 (0.003 at steps 10 and 100).
The v13 step-1,000 model has 0.055 at step 0; v12 step-3,000 has 0.016.
The checkpoint normalizer itself commutes with reflection: the x-feature
and x-action means are zero to numerical precision, and scales are shared
across agents. The asymmetry is therefore learned in the Flow field rather
than introduced by the data or observation adapter.

`new_benchmark_common/macflow.py` now has an **opt-in** `reflect_average`
Flow sampling mode, false for every historical checkpoint. With the same
Gaussian latent, it runs the same frozen Flow weights and ten Euler steps on
the physical observation and on its exact x-reflection with reflected latent,
reflects the second output back, and averages the two bounded actions before
the unchanged hard safety projection. It is coordinate symmetrization, with
no goal-side priority, yielding rule, path coordination, eta or new safety
constraint. The candidate at `n50_reflection_candidate_v16/flow.pkl` copies
v13 offline-best weights exactly and stores its source SHA256
`8c7debe5db26d5030b11bfb97b71804cf134ab7d4134703e28265dfa4083f77b`;
the candidate SHA256 is
`de70d3643b0c96c0de68a578f19b866bad8297a8f90e73a3fbcdbe5d31b4edb5`.
It is **not frozen**. Its 64-latent DEV mean reflection error drops to
0.0228 m/s at step 0, with the residual caused by independent finite latent
sampling in the audit. The first fresh non-opposing high-density state
succeeds 50/50 L→R at step 16,016 and 50/50 R→L at step 13,653, both without
collision; four single-active and four few-active fresh controls also all
succeed. These are screening results only, not a ≥90% acceptance audit.

## Fresh complete-horizon v16 acceptance: rejected

The predeclared acceptance controls use 12 fresh DEV physical seeds beginning
at `fresh_seed=99279`, the independent Flow seed `7402`, one action sample per
step, the original 40,000-step horizon, goal-stop and the unchanged accepted
joint hard safety projection. The candidate was selected without viewing any
N=50 opposing run. In the complete acceptance evaluation, single-active
L→R/R→L and five-active L→R/R→L each succeeded **12/12** with zero collision.
High-density L→R succeeded **12/12** and all 600 crossing agents occupied
their final goals. High-density R→L succeeded only **10/12**: fresh states 6
and 11 ended after 40,000 steps with just 10/50 and 15/50 agents crossing and
reaching their goals. Both were collision-free, but mean Flow goalward command
for still-active agents was only about 0.02–0.05 m/s in the late interval;
pair constraints remained active in the one-way queue and executed net
goalward motion stayed near zero. This is a same-direction queue failure, not
an opposing-agent deadlock. The high-density R→L 83.3% success is below the
predeclared ~90% competence gate and exhibits a large L/R asymmetry.

The temporal-release even-first control succeeded **11/12**. Its one failed
state crossed the gate with the first group, but one of those 25 agents stayed
0.523 m from its final goal after 40,000 steps. The other 24 first-group
agents were within 0.08 m; the second group was correctly held because the
first group had not completed. No wall or pair constraint was active at late
sampled times; this is an independent terminal-convergence failure. The
odd-first control also succeeded **11/12**. Its failure released the second
group at step 9,958 but ended with only 29/50 agents within 0.08 m amid
long-lived pair/wall intervention. Across all completed v16 controls there
were zero simulator collisions and
zero numerical failures. The full per-run gate/goal/throughput and safety
metrics are in `n50_v16_accept_sparse_summary.json` and
`n50_v16_accept_dense_summary.json` under the diagnostics root. The
successful L→R and R→L complete-simulation videos, including provenance and
positive swept-collision clearance, are in `diagnostics/gap_flow_scale_20261007/videos/`.
**v16 is not frozen; no opposing test or eta data is allowed from it.**

The two residual defects are precisely what a CFM-only objective can miss:
the generated ten-step endpoint action may be weak even if the conditional
vector field fits its intermediate interpolation targets. A controlled v17
run therefore resumes the same v13 weights and sealed balanced data with one
additional endpoint action-error term, leaving architecture, normalization,
data, direction counts, Flow integration, reflection adapter at screening,
and safety projection unchanged. The objective coefficient is recorded in
`n50_train_endpoint_v17.sbatch`; only non-opposing validation may select its
checkpoint. Its first Slurm attempt (`8679`) was killed by the job's 50 GiB
memory cgroup before producing an update, so the same command was resubmitted
with a 100 GiB limit as job `8680`. This is an execution-resource retry,
not a changed experimental objective.

The v17 endpoint-loss run completed 4,000 updates on the same balanced
dataset. Its fixed DEV objective selected step 1,000, but this is merely a
non-opposing screening candidate. On nine untouched DEV full-traffic frames
per direction at expert step zero, its reflected Flow falsely activated an
average 2.89 L→R and 5.00 R→L waiting agents, versus v16's 22.22 and 24.00.
However a held-out terminal perturbation probe found projected goalward
action dropping from v16's 0.08–0.10 m/s to v17's 0.037–0.054 m/s, while
the unchanged expert provided 0.426–0.443 m/s. This exposes an imbalance
in the direct endpoint penalty: inactive zero-action agents dominate it.
Four complete-horizon v17 diagnostic screens were launched on non-opposing
DEV states only; they do not establish acceptance.

The controlled v18 run changes only the **relative active-agent weight
inside that endpoint term** from 1 to 20, matching the existing CFM motion
weight, while holding data, initial weights, random seed, objective
coefficient, architecture, Flow steps, and safety chain fixed. The new
`endpoint_motion_weight` config defaults to 1, so v17 and historical
checkpoints retain their exact semantics. v18 completed 4,000 updates and
its fixed DEV objective picked step 1,200. Its step-zero false activations
fall between v16 and v17 at 11.67 L→R and 15.56 R→L; held-out terminal
projected goalward action is 0.066–0.090 m/s, also intermediate. Four
matching complete-horizon non-opposing screens are running. These one-step
probes do not replace the complete 40,000-step competence gate.

Both endpoint-loss candidates failed the four predeclared non-opposing
closed-loop screens (one full L→R, the two v16-failed full R→L states, and
the v16-failed even-first temporal state). v17 ended with goal fractions
0.30, 0.12, 0.16, 0.52; v18 with 0.00, 0.02, 0.04, 0.36. All eight ran to
40,000 steps without collision or numerical failure, and none met complete
task success. The candidate comparison is in
`n50_v17_v18_screen_summary.json`. The endpoint-only remedy reduces desired
active commands as well as unwanted waiting-agent commands and worsens
same-direction throughput. It is rejected; no v17/v18 model is frozen or
exposed to opposing traffic.

The next data-focused diagnostic starts from v16's better checkpoint but
collects model rollouts from TRAIN full-density L→R and R→L initial states
only. These are matched to their original train records, run through the
unchanged hard safety chain with rollout preflight, and saved as 5,000-step
non-opposing traces. State-only same-direction queue expert continuations
will be considered only at predeclared early/late anchors and mirrored
exactly to preserve directional balance. The problematic fresh DEV states
that rejected v16 are not used for training. The job array entry point is
`n50_v16_train_trace_array.sbatch`.

Before training on these new recoveries, a cheaper held-out checkpoint
selection check considered the already trained v13 step-1,000 snapshot with
the same reflection-average inference adapter. Its unmodified weight file
SHA256 is `58b36453e20f4b610b331b17863e90ffcce56f6b4750cf1eb96e0389c3e224d1`;
the adapted candidate is `n50_reflection_candidate_v13_s1/flow.pkl`, SHA256
`eab4a7a1d52269ab1b7da217064bd0869de2c7680f9f80fa8a2b179782bc6e9f`.
Its held-out expert-step-zero false-active counts, 10.11 L→R and 13.00
R→L, improve on v16's 22.22/24.00. Terminal one-step goalward commands
are somewhat weaker, so only complete rollouts can decide selection. On the
four diagnostic DEV states used for other candidates it completed L→R at
step 19,946 and both previously failed R→L states at steps 30,363 and
15,520, all 50/50 and collision-free. The difficult even-first temporal
state still timed out at 40,000 with 24/25 first-group agents at goal.
This snapshot is not frozen. Independent fresh-seed (203879) 12-case
acceptance jobs for single/few-active, full one-way both directions and
both temporal release orders are submitted before any opposing evaluation.

The complete train-only trace array produced 12 physical states per
direction, each run for 5,000 steps through the unchanged safety projector.
None collided. At step 5,000, L→R has 7–17/50 final goal occupants (median
12.5) and 8–20/50 gate crossings (median 15.5); R→L has 9–17/50 final
occupants (median 12.5) and 11–23/50 crossings (median 15). These are
deliberate intermediate model states, not task-success labels. They cover
the same early queue and post-gate occupancy regime in which the rejected
DEV R→L runs later stalled. The recovery array
`n50_v16_recovery_array.sbatch` replays expert continuations from fixed
steps 500, 2,000, and 4,500, keeping only actual simulator successes and
mirroring each accepted trajectory exactly. The output will be inspected
before any new model is trained.

The v13 step-1,000 reflection candidate subsequently passed all four sparse
categories at 12/12 each and full L→R at 12/12 on a fresh DEV protocol
(`fresh_seed=203879`, Flow seed `7403`). Its fresh full R→L control acquired
four distinct collision-free same-direction queue timeouts, with only 11/50,
18/50, 11/50, and 19/50 goal occupants. Its exact full R→L result is
**8/12**, below the ~90% gate, versus 12/12 L→R. It is rejected without
freezing. The
scheduled temporal array was cancelled before any rollout began, because
the high-density gate had already failed. No opposing state was opened.

All 12 fixed TRAIN trace shards and their 72 recovery candidates have now
finished. The existing same-direction queue expert succeeded in **62/72**
anchors; ten states timed out and were explicitly excluded. Every success
was reflected and replayed through the simulator, producing exactly **62
complete additional trajectories and 426,518 transitions in each direction**.
The merged v19 dataset has 818 TRAIN complete trajectories and 3,127,999
joint transitions per direction; the untouched DEV split remains 42 and
272,221 per direction. The manifest SHA256 is
`a6d6d34bde27990de2d1db9145ebe69d32e5816e8f135e49e9c23720d1e2088b`.
All TRAIN and DEV trajectory terminal occupancies are 50/50. The v19 phase
audit reports per-direction active action-agent steps: approach 1,856,996
(26.7%), entry 521,547 (7.5%), crossing 545,239 (7.8%), post-gate far
2,843,587 (40.8%), near goal 925,331 (13.3%), settled 272,174 (3.9%).
The new recovery examples add closed-loop model states, but no TEST state,
opposing demonstration, or directional oversampling. Exact merge and phase
reports are in `n50_recovery_dataset_v19/`.

The v19 focused training run resumes the unreflected v13 step-1,000 weights,
keeps the same 26,114-parameter set actor, active normalization and CFM-only
loss, and changes only the balanced non-opposing data coverage. It uses the
v13 phase sampler and 8,000 updates with 1,000-update snapshots. The chosen
start weights are based on non-opposing screening, not opposing behavior.
The job entry point is `n50_train_dagger_v19.sbatch`. Snapshot choice will
still require complete-horizon non-opposing validation; offline loss alone
cannot freeze a controller.

The exact v13 step-1,000 fresh high-density table closed at **12/12 L→R,
8/12 R→L**, with 0/24 collisions and 0/24 numerical failures. In the four
R→L timeouts only 11, 18, 11, and 19 of 50 agents occupied their goals;
pair constraints remained active in the stalled one-way queues. The
completed sparse controls were 12/12 each for single L→R, single R→L,
five-active L→R, and five-active R→L. Its temporal array was cancelled
before starting, as the full-traffic directional gate had already failed.
The exact per-run results are in `n50_v13s1_accept_dense_summary.json` and
`n50_v13s1_accept_sparse_summary.json`. Complete L→R and R→L success videos
with positive continuous swept clearances are saved under `videos/`, but
they do not override the failed full R→L acceptance.

The v19 data-only continuation completed 8,000 CFM updates in 15.8 s on
the allocated GPU. Offline DEV loss fell from 0.06365 to a best 0.06093
at step 2,800, but this is not the selection criterion. On held-out
non-opposing one-step probes, the reflected step-2,800 candidate falsely
activates 3.78/5.78 waiting agents L→R/R→L at expert step zero, while the
step-4,000 candidate activates 12.44/14.33. The latter retains larger
near-goal commands (0.079–0.100 m/s versus 0.045–0.064); both remain below
the expert's 0.426–0.443 m/s. These two snapshots span the measured onset
versus terminal-command tradeoff within one fixed training run. Complete
closed-loop non-opposing diagnostic screens are now running for both; no
N=50 opposing behavior has been viewed.

The step-4,000 candidate completed all four fixed DEV diagnostics with
50/50 final goals and zero collision: full L→R at step 14,481, the two
previously failing R→L states at steps 13,150 and 27,185, and even-first
temporal release at step 28,216. The offline-DEV-loss-best step-2,800
candidate failed the same full L→R state after 40,000 steps with 48/50
goal occupants; its even-first temporal state succeeded only at step
34,659. The remaining two step-2,800 R→L screens are allowed to complete,
but the selection is already determined by these non-opposing outcomes.
**v19 step 4,000 is selected for fresh acceptance, not frozen.** A disjoint
fresh-seed (398201) and Flow seed (7404) drive 12 complete cases per sparse,
dense, and temporal category through the identical safety chain. The
predeclared Slurm entry points are `n50_v19_s4_accept_dense.sbatch`,
`n50_v19_s4_accept_sparse.sbatch`, and
`n50_v19_s4_accept_temporal.sbatch`. No N=50 opposing test has run.

## Fresh v19 acceptance and freeze (2026-10-09)

All predeclared fresh cases finished under the unchanged Flow → speed bound →
`CertifiedHardSafetyFilter(HardProjectionConfig())` → simulator chain, using
physical seed 398201 and stochastic Flow seed 7404. Success means **all 50
agents within 0.08 m of their final goals**, not gate crossing. The same
checkpoint was used in every category; no opposing case was used for
selection. There were zero collisions and zero numerical failures in 96/96
episodes. Wall/pair fractions below count timesteps with at least one active
constraint anywhere in the 50-agent system; correction is the mean **joint**
`||u_safe-u_ref||` in m/s.

| Non-opposing control | Complete success | Mean final occupancy | Mean gate crossing | Wall-active | Pair-active | Mean joint correction |
|---|---:|---:|---:|---:|---:|---:|
| Single active L→R | 12/12 | 1.000 | 1.000 | 0.045 | 0.159 | 0.008 |
| Single active R→L | 12/12 | 1.000 | 1.000 | 0.087 | 0.177 | 0.013 |
| Five active L→R | 12/12 | 1.000 | 1.000 | 0.132 | 0.629 | 0.055 |
| Five active R→L | 12/12 | 1.000 | 1.000 | 0.153 | 0.606 | 0.048 |
| Full one-way L→R | 11/12 | 0.965 | 0.967 | 0.667 | 0.958 | 0.399 |
| Full one-way R→L | 12/12 | 1.000 | 1.000 | 0.649 | 0.944 | 0.384 |
| Temporal even-first | 12/12 | 1.000 | 1.000 | 0.403 | 0.876 | 0.226 |
| Temporal odd-first | 11/12 | 0.997 | 1.000 | 0.429 | 0.896 | 0.235 |

The L→R full one-way failure timed out at 40,000 steps with 30/50 gate
crossings and 29/50 final goals. Pair constraints were active in 99.97% of
steps and wall constraints in 94.25%; the final 2,000-step goalward rate of
pending agents was only 0.00454 m/s. This is an unusually slow **same-direction
queue**, not the old post-gate Flow-goes-nowhere failure. The odd-first
temporal failure had 50/50 gate crossings and 48/50 final goals. Two agents
approached swapped adjacent goals, remained 0.447 m from their own goals,
and lost long-horizon progress under pairwise interaction; the final 2,000
steps averaged −0.000103 m/s goalward for pending agents. Neither timeout is
silently counted as a deadlock or as success. They leave the controller at
the requested approximate ≥90% complete-success standard in *each* core
category, with a real 2/96 non-opposing failure rate to retain as a caveat.

On independent saved expert post-gate states, the target action still points
to the actual final goal and observation reconstruction error is zero. The
new checkpoint improves most held-out action-distance strata, but the dense
R→L just-outside-threshold stratum remains weak: the mean projected
goalward speed is −0.0156 m/s in that single-step probe, versus the expert's
+0.1498 m/s. The complete R→L success rate of 12/12 is stronger operational
evidence, but this local weakness remains in the record; the model is not
claimed perfect. The probe is saved as `n50_v19_s4_postgate_probe.json`.

The frozen controller is
`diagnostics/gap_flow_scale_20261007/frozen/gap1_n50_flow_navigation_v1.pkl`,
SHA256 `9ec0540c877f2d49e048f718b856eff632fbf228293454700f0f0119eee4abf7`.
The byte-identical source is the reflected-inference v19 step-4,000
candidate. `frozen/gap1_n50_flow_navigation_v1_manifest.json` records the
training config, balanced dataset manifest, exact safety/source-code hashes,
fresh acceptance summaries, and the pre-opposing freeze time. TRAIN has
818 complete L→R and 818 complete R→L trajectories, 3,127,999 joint
transitions per direction, including 62 mirrored model-state recoveries per
direction; all TRAIN/DEV demonstrations end at 50/50 goals. The DEV split is
42 complete trajectories and 272,221 transitions per direction. See
`n50_recovery_dataset_v19/phase_audit.json` for per-phase active action-agent
counts and goal-distance quantiles.

After freezing, `evaluate_safety_audit.py` gained only `--archive-start` and
`--archive-count` to shard the eight *unchanged* archived TEST states while
preserving the original per-case Flow RNG index. This does not alter the
controller, observation, projection, dynamics, success rule, or physical
states. The opposing evaluation records the post-freeze evaluator hash
separately from the freeze-time code hash.

## Frozen opposing TEST: N=50 is READY_FOR_ETA

Only after the above checkpoint and manifest were frozen did the original
eight archived N=50 opposing TEST states run. The TEST states, one Flow
sample per physical step, competence-v2 observation, goal-stop rule, 40,000
step horizon, hard projection, and success/collision evaluator are unchanged.
The post-freeze evaluator added an archive-index shard option solely to run
these same eight cases concurrently; it preserves the original per-case Flow
random key. The source entry point is
`n50_v19_frozen_opposing_test.sbatch`; exact rollouts and full traces are
`n50_v19_frozen_opposing_test_i00/` through `i07/`.

| Opposing outcomes | Count |
|---|---:|
| Spontaneous complete success | 0/8 |
| Clean bounded safety-gridlock candidate | 8/8 |
| Ordinary navigation failure before opposing encounter | 0/8 |
| Collision / numerical failure | 0/8 |
| Mixed / unresolved | 0/8 |

These are not timeout-only labels. The predeclared classifier
`bottleneck_family/audit_scale_opposing.py` requires both sides near the gate,
an opposing pair within 0.7 m, a live pair constraint, ≥0.5 m median approach
progress among agents near the gate, a long low-progress tail, and no late
gate crossings. All eight meet it. Opposing encounter occurs at steps
432–694, with 23–31 agents in the near-gate approach set and median approach
progress 3.02–3.75 m. All 8 are collision-free, have **zero gate crossings**,
and retain 0/50 final goals. Minimum continuous swept clearance is positive
(0.00012–0.00034 m). The full per-episode classification is in
`n50_v19_frozen_opposing_test_audit.json`; the independent mechanism
summary is `n50_v19_frozen_opposing_mechanism.json`.

| Mean over eight original TEST episodes | Before opposing encounter | After opposing encounter | Final quarter |
|---|---:|---:|---:|
| Mean goal-progress rate per agent (m/s) | 0.15314 | 0.00020 | approximately 0 |
| Mean executed speed per agent (m/s) | 0.20057 | 0.06360 | 0.06342 |
| Mean Flow goalward component (m/s) | 0.15393 | 0.00742 | 0.00707 |
| Mean safe goalward component (m/s) | 0.15318 | 0.00021 | 0.000006 |
| Mean projection correction per agent (m/s) | 0.02719 | 0.05979 | 0.05955 |
| Timesteps with any active pair constraint | 0.9859 | 1.0000 | 1.0000 |
| Timesteps with any active wall constraint | 0.1099 | 0.9762 | 0.9729 |

The pre-encounter pair-active fraction is already high because 50 agents
queue on their own sides; the relevant *new* event is a physically close
opposing pair at the gate. After that event, Flow itself becomes much less
goalward, and the projection reduces the remaining goalward component nearly
to zero. Both facts matter. The traces support an **opposing-interaction
gridlock under hard safety**, but do not isolate the projection as the sole
cause of the Flow command change. Bounded motion persists; instantaneous
speed need not be zero for long-horizon gridlock. The final quarter changes
mean goal distance by −0.0031 m per agent (an average slight regression).

The frozen checkpoint meets the predeclared approximate ≥90% non-opposing
gate in every category, and the untouched opposing TEST produces eight
reproducible collision-free gridlock states. **N=50 status: READY_FOR_ETA.**
This status does not assert that the existing 3D eta intervention family can
resolve them; that is the next separate dataset question. No frozen Flow
weight or accepted safety projection has been changed after the test.
