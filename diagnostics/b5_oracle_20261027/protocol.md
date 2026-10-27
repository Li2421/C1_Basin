# B5 production oracle, 2026-10-27

Base commit: fe3ec0b96901dcc735a7713c7ae14e6746673bf4. Independent branch:
`research/b5-oracle-20261027`. No training and no old experiment modification.
18 CPU affinity (0–17), at most 18 single-CPU workers; BLAS/OpenMP threads = 1.
No GPU needed. Runtime uses the original production Clarabel solver and retries.

## Evidence audit and gaps

Inspected `README.md`, `shared_rollout_db/README.md`, shared B3 implementations,
`diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/README.md`,
its `hetset/run.py`, `run8.py`, `rhr.py`, `accel.py`, scenes and fresh/menu results,
and `scene_search/harness/{runtime_new,scene_environment,scenes}.py` plus backend
`{distributed_safety,correction_revision_20261021,swap_safety,swap_safety_user_fix_20261023,swap_geometry}.py`.

`hetset/fresh.jsonl` has 51,744 closed-loop labels with lengths and costs. The
original 16 legal eta candidates on matching scene/controller/state and seed 0
are reused. It does not contain B5 directions, the new Sobol candidates, or
failure-state trajectory geometry; these are the evidence gaps for new runs.
Historical `sE` candidates often lie outside the requested box: do not count
them as B3-box successes. Archived new-benchmark checkpoint paths in
`new_benchmark_common/safety_eta3.py` no longer exist in the active tree after
the user's prior cleanup. Old-scene historical results are not comparable to
new scenes merely by name. No checkpoint replacement is authorized or made.

The provided-name ZIP and IMPLEMENTATION_SPEC_CN.md were not found in recursive
searches of `/home/zhihan`, `/tmp`, `/mnt`. Therefore its run_checks.py is not
available; repository tests below are not represented as that reference check.

## Real production semantics

Two safety projections per step: bounded nominal -> first safe velocity `a` ->
correction target -> second projection. G is radially bounded goal residual;
P = 3.303687238760696 times the Tikhonov orthogonal flow residual, epsilon=1e-6
times speed. R is mean of individually bounded pair separation vectors, now
gated by a 1.25 safe-distance contact conflict mask. All arrived pairs are
excluded from this contact mask. This gate is retained, unlike the old ungated
shared OrthoFlow3 class. Nominal stops at arrived robots; correction is not
hard-masked there.

Production feasibility is the Cartesian product of independent 2D agent sets,
not a single centralized pair-relative-velocity QP. Neighbor radius=.5;
pair responsibility=.5; gamma=gamma_wall=1; radial speed cones. Existing retry
and restoration (`speed_only`, then `safety_minimax`) are preserved. Report
restoration and original-constraint violations separately; restoration is not
a new B5 safety guarantee. Swept pair/wall collision <=0 ends an episode;
all distances <= goal tolerance means success; horizon exhaustion means timeout.
There is no mathematical deadlock termination predicate in this runtime.

Native optimization controllers (centralized_sampling and gradient_mpc) use
their frozen parameter manifests, no learned checkpoint. Current hetero-set
uses TT recipes; this experiment does not establish legacy FF checkpoint
equivalence. Existing two-stage eta switches at progress theta or 6/7 horizon.

## Frozen finite pools and metric definitions

Development: state index 8 of four specified scenes, both controllers. A has
16 original candidates + 16 scrambled Sobol + 8 box vertices = 40/state.
Budgets 8,16,24,32,40 are prefixes. Failure is Oracle@40 failure, never continuous
impossibility. B and C each contain A as a subset and 96 additions per T recipe:
2 T-only, 26 ordered-role/waypoint configurations times three T coefficients,
16 Sobol eta5. The nearest initial pair is chosen by state-only geometry, both
orders included; the menu implementation supports every ordered pair, but this
budget does not exhaust all pairs. Recipes are reported individually. A pooled
recipe oracle is an explicitly larger search budget, never a fixed method.

Y local forward axis is the frozen yielder-to-passer vector. HOLD asks for zero
velocity only. Back/left/right/forward waypoint displacements are .5,1,2 times
the production safe diameter; velocity is the original bounded goal residual
toward the frozen waypoint. ADD Y = clear-a for yielder only. BLEND interpolates
the entire corrected preprojection yielder input, not merely nominal.
T-cluster uses current contact-connected components, common per-group scale
min(1,speed/max-radius); isolated agents have zero T. T-wall rotates the tightest
relevant original wall CBF normal, ties resolved by row order. T-goal=JG.

D: production two-stage B3 schedule and release of B5 T/Y at horizon/3 for
four best development candidates per recipe/mode. Release uses fixed B3
(.875,0,.375). E deletes columns from successful full candidates without
reoptimization; success preservation is a witness, failure is not proof of
irreducibility. Goal-lower-bound-zero is a separate matched-pool experiment.

Independent test: four external held-out scene definitions, states 16/17,
future seeds 1/2, both controllers. Freeze one candidate per development
scene-family/controller/method before test; no test eta or recipe selection.
Baseline safety and both historical RHR variants retain speed=.35, progress
window 40, threshold .05. A method absent from development is reported missing.
Sample size is small; clustered intervals, if reported, resample scene/state
blocks, not individual selected candidates. Seeds are matched across methods.

Task time is steps*dt. Conditional means exclude failures; paired-success
means use the intersection; truncated time assigns full horizon to non-success,
including collision/unknown, and these categories are also counted separately.
Initial local-progress proxy is time to reduce summed goal distances by .1 m;
this is not a rigorous local deadlock escape time. Failure-state analyses use
separate time and geometric diagnostics. Simulator wall time excludes import
and result compression; end-to-end worker wall time is recorded separately.

Each stage registers exact physical and controller identities and calls the
official preflight. DB eta remains 3D; full eta5, role, waypoint, recipe and
schedule are in the exact controller fingerprint. Workers append journals;
only one merger writes closed-loop rows. Historical labels remain immutable
external provenance and are explicitly recorded as reuse, not new rollouts.

## Completion amendments (recorded after execution)

The pilot was expanded to states8..15 (64 cases), with A40 -> A168 only on
failures, then 96 additions per mode/recipe on the 14 residual cases. Independent
designs remained frozen from the initial state8 pilot, before expansion. The
independent B3 was selected by scene-family/controller; each B5 recipe used a
global pilot template. `INDEPENDENT_FROZEN.json` is authoritative, not the earlier
sentence suggesting per-family B5 selection. No test-outcome recipe selection.

All four original checkpoints were recovered in the sibling flow-field POC
repository. `LEGACY_FROZEN.json` records exact paths and hashes. Its original
TT/FF adapters and joint safety solvers were used for 160 new backtest episodes;
no gated R was backported. `B3_archived` is the stronger history-selected baseline.
The midpoint B3 is only a separate diagnostic baseline. FF_ADD uses original
constant physical field lifting; FF-BLEND is not claimed implemented.

The final local escape metric in analyze.py uses a 40-step low-progress and
low-speed detector, followed by .05m per-agent progress or arrival. The earlier
sum-distance .1m quantity remains raw metadata only. Both are diagnostics, not
strict deadlock predicates. Reported intervals resample eight physical-state
blocks; identical future-seed trajectories are disclosed.

Zero-column behavioral aliases reuse exact simulations without claiming new
runs. All candidate UID/specs are preserved; canonical explicit coordinates and
missing historical eta fields are supplied by candidates_verified.jsonl. During
final verification, the first state insertion was found to omit a required SQL
provenance field; the INSERT OR IGNORE had silently skipped metadata. The original
preflight and ambiguous journals are retained. verify.py reconstructs the missing
48 states from original manifests, uses the official merger, and verifies all
14,694 outcomes without rerollouts. Registration readback assertions now prevent
this failure. Original execution hashes remain in frozen manifests, even where
registration-only delivery code has subsequently changed.

No safety, collision geometry, or success criterion was patched after observing
the wall-margin mismatch. It is documented in safety_geometry_audit.json. The
best independent `blend_cluster` label is actually etaY=etaP=0 (G,R,T only).
The completed report supersedes provisional interpretation, but not frozen
execution identities or raw outcomes. Reference ZIP remains absent.
