# Ring Exchange Stage-I report

## 1. Scenario definition

Four disc agents `A,B,C,D` start at broadly separated locations around a
central forbidden disk and exchange to approximately opposite goals through a
wide annulus.  The central disk has radius 0.90, the outer boundary radius is
3.25, and agent radius is 0.16.  The radial free width is therefore 2.35
(well over six agent radii): this is not a one-at-a-time passage.  Canonical
agent order is `A,B,C,D`.

The deployed Stage-I observation is a 4-by-23 physical local-frame array.
For each agent it contains radius, local velocity, local goal displacement,
other-agent relative position/velocity, and obstacle/outer-boundary geometry.
Actions are four radial/tangential velocity pairs, deterministically decoded
to world velocity at the current physical position.  No CW/CCW, priority,
crossing order, recurrent state, or persistent latent is an input.

## 2. Distinct liveness mechanism and validity

The liveness choice is lateral circulation consistency: compatible CW or CCW
circulation permits simultaneous exchange, while incompatible lateral choices
can create head-on interactions, cyclic blocking, or repeated lateral
conflict.  There is no lane, priority rule, or hidden narrow gate.  The broad
annulus and expert's simultaneous four-agent trajectories verify that the
difficulty is circulation/lateral coordination rather than Give-Way-style
single-resource ordering.

## 3. Initial-state distribution

Each independently seeded train/development/test draw includes a global angle,
per-agent angular offset, radial start/goal offsets, small goal-angle offset,
and initial velocity direction/magnitude.  Splits use distinct seed streams;
test was not constructed from failures.  The frozen v10 manifest contains 120
train, 30 development, and 60 test nominal expert trajectories.

## 4. Centralized expert and multimodality

The centralized joint expert searches CW and CCW circulation hypotheses and
validates candidates in the true swept-collision simulator.  A 30-state pilot
with each hypothesis yielded 60/60 collision-free successful plans.  Both
directions are physically valid on comparable initial states.  The expert's
mode is analysis metadata only.

For recovery near goals, the expert now holds an agent inside goal tolerance
and otherwise takes a direct goal segment whenever that segment is physically
safe with respect to both disk and outer boundary.  This replaces a brittle
angular threshold that could relaunch a nearly completed agent around the
ring.  A regression test forces a CW hypothesis with a safe CCW-side chord and
verifies that the chord is selected.

## 5. Nominal and recovery data

The frozen v10 dataset is
`base_u_v10_local_dataset`.  Its source counts are 210 nominal and 2,269
uniform-recovery trajectories (2,234 train trajectories, 185 development
trajectories, and 60 untouched-test nominal trajectories).  Base-U began with
uniform expert anchors over broad nominal draws.  v8 added 848 successful
development-policy full-trajectory uniform re-queries.  v10 added 687 more
successful re-queries from only v9 development timeouts: dense valid terminal
windows plus bounded-uniform goal-near/non-jointly-converged anchors.  Full
provenance is in `v9_timeout_goal_uniform_report.json`.

The final trainer loaded 37,390 nominal transitions and 188,149 uniform
recovery transitions. It loaded zero targeted wall/obstacle and zero targeted
agent transitions.

No targeted wall/obstacle augmentation is part of the accepted local-frame
v10 lineage: the mature candidate has zero development obstacle and outer-wall
collisions.  Therefore a Base-U+W attribution is not material to this frozen
candidate.  No agent-collision-targeted augmentation was used.

## 6. MACFlow implementation and training

The model is the shared official joint Stage-I MACFlow implementation:
`ActorVectorField`/`ModuleDict`/`TrainState`, conditional flow-matching loss,
Adam, and ten Euler sampling steps, producing one joint 8D action.  v10 used
100,000 steps, batch size 256, seed 3127.  Its early transition sampler gives
50% mass to the first 15 steps of **all** nominal and recovery trajectories.
This is a data-sampling correction, not an architecture change.  Best fixed
development loss was 0.07398 at step 96,250.

## 7. Development support, stability, and teacher diagnostics

The following is the full 30-development-nominal diagnostic.  Rollout,
collision, timeout, and stability values are preserved from
`final_diagnostics_v10_dev/summary.json`; the OOD fields are the pure
post-freeze diagnostic recomputation in
`final_diagnostics_v10_dev_oodfix/summary.json`.  Its selection audit records
zero test archives opened.

| Metric | Development result |
|---|---:|
| Full-task success | 86.67% (26/30) |
| Central obstacle collision | 0% |
| Outer wall collision | 0% |
| Agent collision | 0% |
| Timeout | 13.33% (4/30) |
| Mean episode length | 387.93 |
| Timestep-0 OOD | 10.00% (OOD-fix recomputation) |
| Rollout OOD | 33.33% |
| Teacher-forced mean sampled RMSE | 0.01231 |
| Teacher-forced best-of-8 RMSE | 0.00647 |

OOD is analysis-only.  Timestep-0 OOD uses a train-nominal-t0 p99 threshold
of 1.02112; rollout OOD retains the train-transition nearest-neighbour p99
threshold of 0.5721.  The threshold probe is deterministically capped for
broad recovery corpora.  K-step mean divergence for K=1/5/10/25/50/100 is
0.00128 / 0.00608 / 0.01226 / 0.03664 / 0.24892 / 0.65144.  K=100 collision is
0%.  The remaining failure taxonomy is exclusively timeout, so the residual
issue is late closed-loop convergence/drift, not collision safety.

## 8. Mode statistics and visual diagnostics

Analysis-only expert nominal modes were CW 19 and CCW 11.  Realized policy
development modes were CW 15 and CCW 15.  This confirms both circulation
directions are genuinely used without deploying a mode variable.  Representative
expert/policy SVGs and timeout final-position plot are under
`final_diagnostics_v10_dev/visuals/`.

## 9. Hard-safety compatibility smoke

`hard_safety_smoke.json` records one compatibility smoke only, not a safety
experiment: six unordered pair constraints plus a conservative circumscribed
48-side polygon representation of the central disk.  The smoke was nominally
feasible with a 198-by-8 constraint matrix: six pair constraints and 192
obstacle constraints.  No formal hard-safety comparison was run.

## 10. Frozen untouched-test outcome

The master-authorized one-time frozen evaluation is preserved in
`final_diagnostics_v10_frozen_test/summary.json`.  Its rollout outcomes are
unchanged; the OOD fields below come from the pure post-freeze diagnostic
recomputation `final_diagnostics_v10_frozen_test_oodfix/summary.json`.  It
evaluated all 60 independently sampled frozen nominal states after v10 had
been frozen:

| Metric | Frozen test result |
|---|---:|
| Full-task success | 78.33% (47/60) |
| Central obstacle collision | 0% |
| Outer wall collision | 0% |
| Agent collision | 1.67% (1/60) |
| Timeout | 20.00% (12/60) |
| Mean episode length | 414.38 |
| Timestep-0 OOD | 8.33% (OOD-fix recomputation) |
| Rollout OOD | 36.67% |
| K=100 divergence | 0.67719 |
| K=100 collision | 0% |

Frozen-test policy modes were CW 32/60, CCW 28/60, mixed 0; frozen-test
expert modes were CW 34/60 and CCW 26/60.  Frozen-test trajectory SVGs and a
timeout final-position plot are under `final_diagnostics_v10_frozen_test/visuals/`.
The selection audit in that visual report records exactly 60 opened test
archives.  No frozen-test result has been used to collect recovery data,
retrain, tune, or otherwise adapt v10; no further Ring adaptation is allowed.

## 11. Baseline maturity decision

**`PASS_STAGE1`.**  The frozen v10 candidate has broad nominal/recovery
coverage, low teacher-forced error, meaningful development and frozen-test
success, and an explicit residual timeout/rare-agent-collision population.
Remaining failures are not immediate OOD or a recurring uncovered wall/obstacle
collision region.  The candidate remains frozen; no OrthoFlow3, eta, basin,
G_phi, or formal safety experiment has been run.
