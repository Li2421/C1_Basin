# Controller-conditioned state/geometry transfer v1 — preregistration

Frozen before this task reads any new-panel outcome. UTC/hash evidence is in
`preregistration_receipt.json`. Historical reports and protocols were inspected;
`controller_state_transfer_v1` is an independently started, still-running source
of compatible labels. This is an analysis preregistration for reused data, not a
claim that those labels were generated after this document. No final-test Q is
used to choose the environment, models, features, or calibration.

## Questions and scope
P1: Can the existing entity critic predict and select state-dependent eta on
unseen states AND unseen geometry? P2: Does target calibration repair an eta
regime or transfer conditional structure? P3: Which independently varied
physical factors alter Q? P4: Do label-free mechanism coordinates improve held
geometry/family prediction and selection? Full Q and eta selection are distinct.
All experiments are pure Q generation/training/evaluation; AGENTS video exemption
applies. No controller training, no certification of deadlock resolution or B15.

## Audit and fixed semantics
Reuse the Toy corridor-width source panel (0.41:0.03:0.62), original true-t0
states, frozen Toy Flow endpoint, isotropic/anisotropic FF operators, K8 indices
[0,2,4,5,8,10,12,15], OrthoFlow3, hard projection, dt .05, horizon 850 and swept
collision evaluator. Success remains collision-free arrival before native
termination. Width changes corridor and bay opening together, NOT topology.
Read-only sources: controller_geometry_identifiability_v1,
controller_state_transfer_v1, controller_ff_factorial_repair_v1; snapshot their
scripts/protocols before local extensions. No historical artifact is overwritten.

## Pilot (first gate, outcomes isolated)
Pilot uses source state indices 0..7 ONLY at widths .41,.47,.53,.59. Neither
pilot states nor test states are training examples. Use future seeds 0/1, with
split-half cross-products to remove Bernoulli noise. Factor axes are G,S,C,E.
Report all 15 balanced functional-ANOVA components (state main, SxE, GxE,
CxE, and higher interactions separately), negative noise-corrected estimates
retained. Also report per-(G,C) state×eta energy and half-seed reliability.
A valid strong-state panel requires ALL:
- state-involving signal >=20% of total noise-corrected signal;
- within-(G,C) S×eta energy >=.005, state-bootstrap 95% lower bound >0;
- cross-fitted per-state eta oracle minus per-(G,C) fixed eta >=.05;
  paired 95% CI lower bound >0 (seed 0 selects/1 evaluates AND reverse).
Pilot n=8 is small: if point thresholds pass but uncertainty fails, a single
precision extension to future seeds 2..5 on identical pilot cells is allowed.
If signal fails, one predeclared geometric revision is allowed: bay_top=.84,
widths .41,.47,.53,.59, same pilot states/controllers/eta and seeds0..3. This
increases refuge space without changing tasks, radii, speed, dynamics or success.
If still invalid, report a hard scientific design failure; do not pretend the
strong-state requirement is met or select an environment using final-test Q.

## Phase 1 main split and budget
Training state indices 8..23 (16 states); final-test indices24..31 (8 states),
never used in fitting, calibration, pilot, normalization, early stopping or
selection of variants. Four training geometries .41,.47,.53,.59; held interpolation
.44,.50,.56; extrapolation .62. Primary endpoint is new-state/new-geometry
interpolation; seen-geometry/new-state and .62 extrapolation are separate.
- MANY: each of 4 train widths gets 8 states, cyclic offsets 0,4,8,12 among the
  16 training states. 4*8*2C*8E*2 seeds=1024 labels.
- FEW: widths .41,.59, all16 training states, same1024 labels.
- FEW_x2: same as FEW but seeds0..3, 2048 labels. Generate only1024 missing
  seed2/3 outcomes after exact-cache preflight if gate passes.
- ALL: all4 train widths/all16 states/seeds0/1,2048 labels; finite failure diagnosis.
- EXTRAP: .41,.47,.53/all16 states/seeds0/1; evaluate .59,.62 new states.
Unique source states are identical in MANY and FEW; geometry-state allocation is
explicitly indexed. No validation-tuned checkpoints or hyperparameter search.

Models: existing Critic('state_field_compact'), controller one-hot, wall entities,
6000 steps,batch120,AdamW(8e-4,1e-4),clip5; seeds17,23,41. MLP reference
3x128 SiLU,3000 steps,batch256,AdamW(1e-3,1e-4), train-only standardization.
ENT state-only removes controller input (retains geometry); controller-aware
no-state prior is source mean Q(C,E); global fixed eta is source mean Q(E).
Train state-only and MLP on MANY and ALL; original ENT on all5 regimes.
Save EACH seed's parameters, predictions, losses, and indices plus ensemble.
Finite diagnosis: ALL coverage; ENT+explicit width; one pairwise eta ranking-loss
arm (MANY, weight1 on BCE plus pairwise sigmoid loss using empirical probability
differences, no test tuning). Reuse existing compatible predictions if present.

Metrics: empirical-Q MSE/BCE/correlation and cross-seed noise-corrected MSE;
S×E, C×E and G×E centered pattern correlation and skill; eta pair ordering accuracy
on empirical differences >=.25; eta-only selection for GIVEN controller; joint
(C,eta) selection reported separately, never substituted. Closed-loop success
is the selected cell's held future seeds2/3 (real closed-loop outcomes), with
full-seed estimate supplementary. Oracle selects on seeds0/1,evaluates2/3;
naive max-all-seed Q separately, labelled optimistic. Regret to crossfit oracle
may be negative. Fixed-eta, per-C prior, same-model mean-state choice, state
mismatch (analytical average over all OTHER physical states), random and oracle.
Report each training seed and ensemble; paired physical-state bootstrap2000,
resampling same state across geometries/C/seeds to preserve common randomness.
No treating correlated cells as independent sample size. Per-geometry metrics.

H1 supported only if ENT MANY S×E corr>.3, unbiased skill>0 and matched-minus-
mismatch CI lower>0 on primary unseen states/geometries; otherwise partial or
rejected if reliable signal but no selection gain. H2 geometry-coverage benefit
supported if MANY-FEW success CI lower>0 and S×E corr gain>=.1; test whether
FEW_x2 recovers this. These are local tests, not proof of universal sufficiency.
Methods' major defects cannot be ruled out merely by Q correlation.

## Phase 2 calibration (after Phase 1 reliability audit)
For each target geometry, target states8..23 are a calibration pool, disjoint
from final-test24..31. Use k=0,1,2,4 nested random state subsets,20 predetermined
permutations (RNG20261012), labels seeds0/1 only. Cost=32*k rollouts per target
(2C*8E*2), labels reused where available. Zero-shot ENT MANY; prior-only empirical
target Q(C,E) (k0 source prior); prior+conditional = target prior + centered
source prediction P(s,C,E)-mean_cal P(cal,C,E), clipped[0,1]. Also offset variant
P(s)+mean_cal(Q-P(cal)). No label or source-fit on final-test states. Compare
success/regret, prior-only vs conditional, mismatch gain; paired state CIs and
calibration-draw variation. Repeat on compatible DB historical width and four-
scene datasets where possible; clearly separate retrospective/exploratory arms.
H3 regime repair if calibrated prior improves zero-shot; conditional transfer
requires conditional-prior success CI>0 and matched-mismatch CI>0.

## Phases 3–4: bounded independent-factor and mechanism extension
Freeze detailed factors and implementation in a timestamped addendum BEFORE
new factor outcomes. First exploit prefix-invariant horizon relabelling of
existing complete trajectories (prove action and monitor do not depend on
horizon; earlier-success time is sufficient for success-by-H). Width×horizon
must be fully crossed. Other preferred factors: topology and agent count,
fixed map/radius/dynamics/task for shared agents, shared resources for added
agents. Use the existing bottleneck_family plant if N cannot be varied in
legacy runtimes. No claim that a frozen fixed-N Flow supports different N.
Resource cap: at most8192 new rollouts across all phases,24 CPU slots, GPU via
Slurm with at most4 shards and train-only normalization. Pilot precision and
FEW_x2 consume this cap first. Cost admission uses measured runtime before
submitting, no infinite expansion. One scientific failure does not justify
unbounded search; retain negative evidence. Phase3 must explicitly distinguish
single-family interpolation from genuine cross-family/topology transfer.

Mechanism features defined before fitting and computed ONLY from initial state,
geometry, dynamics/controller: clear passage width / disk diameter (clearance
ratio, NOT measured throughput), shortest collision-free SINGLE-agent path
length and H*dt - max(pathlength/vmax) (optimistic time-margin bound, NOT actual
joint completion margin), connected components/cycle rank of a documented
clearance graph (resolution-dependent proxy, NOT exact topology), static shared
bottleneck demand. Compare ENT,ENT+raw geometric parameters,ENT+mechanism, and
mechanism-only/individual removals if budget allows. Do not call a heuristic a
validated mechanism. Cross-family results require consistent controllers and
fixed evaluator; otherwise explicitly record confounding and inconclusive.

## Reliability, cost and decisions
Unknown/numerical failures retained as unknown, report count and conservative
all-failure/all-success bounds; never replace them with model labels. Missing
planned rollout is an incomplete dataset, never a zero. Assert exact identity,
state/calibration/test disjointness and controller/eta coverage. Evidence index
has SHA256 and source provenance. Report UTC, Git HEAD and dirty-tree baseline,
prereg hashes, config, checkpoint hashes, runtime/environment, CPU/GPU/storage.
Confidence intervals are finite-panel uncertainty, not universal-domain bounds.
All hypotheses use SUPPORTED / PARTIALLY SUPPORTED / REJECTED / INCONCLUSIVE.
Data facts, plausible mechanism interpretation and theory conjectures are kept
separate. A technical/resource boundary yields explicit partial status and
resumable entry points, not fictitious completed phases.
