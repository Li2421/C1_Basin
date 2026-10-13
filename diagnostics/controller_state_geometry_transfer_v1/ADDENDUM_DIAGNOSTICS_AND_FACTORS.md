# Finite diagnostics and independent-factor specification

Freeze receipt: diagnostics_factor_receipt.json. This is an explicit post-pilot
amendment. Five tested panels/cutoffs did not pass the strong-selection gate.
No broad-main rollout is authorized by that gate; broad_main remains unrun.
Strong-state eta-selection generalization is therefore INCONCLUSIVE, even
though broad-state Q has reliable state×eta signal. Do NOT lower the gate.

To complete finite failure diagnosis and the downstream questions, run the
already-preregistered MANY/FEW/FEW_x2/ALL/EXTRAP and model controls on compatible
LEGACY Toy widths. These are labelled low-headroom diagnostic results, not a
successful strong-state test. Supplement only missing FEW_x2 seed2/3 at .41/.59,
state8..23:1024 labels. Original state24..31 stay test; calibration states8..23
are disjoint from test. Fitting settings and physical-state bootstrap unchanged.
Existing source analysis results are reusable comparisons, with its own distinct
split and no claim its models match this split. Phase2 follows diagnostic audit.

## Factor panel (Phase3, generated only after Phase2 audit)
Use the unmodified bottleneck_family disk plant at fixed map half_length8,
half_height2, radius.16, vmax.5, dt.05, wall radius4/600, wall margin.005,
goal tolerance.08. Fully cross passage widths .50/.65/.80; N2/N4; and ONE gate
vs TWO PARALLEL openings (same barrier x0, thickness.6). One opening at y0;
two at y+/-.6. This changes route topology (a central interior obstacle creates
one hole), NOT merely serial-gate count. All maps remain connected for a single
disk. Native swept collision/arrival evaluator, H850 with H400 exact prefix
labels, no timeout-to-deadlock relabelling and no stall-based termination.

16 state seeds, indices0..7 source training and8..15 final test. Same physical
positions/goals across ALL geometries; N2 is exactly first2 tasks of N4. Generate
all4 starts deterministically with rng([2026101113,i]), alternating left/right,
|x| uniform[1.4,2.4], y uniform[-.5,.5], reject pair distance<=.45. Goals mirror
x and preserve y. Extra agents cross the same barrier and compete for the same
openings; no tiling of disconnected pair problems. Geometry-valid and individual
path checks mandatory. Joint feasibility is witnessed only where actual successful
rollouts exist; unproven cases remain unproven, not geometric deadlocks.

Shared dimension-generic controller: single-agent geometry visibility-path
following at each physical step (no multi-agent priority or expert), desired
speed .85*vmax*tanh(distance_to_waypoint/.5), plus existing .25 repulsion and
.2 velocity damping from REFERENCE. Two endpoint-matched FF matrices isotropic
and anisotropic, unchanged ten-step FF integration, OrthoFlow3 and hard safety
projection. Same program for both topologies and both N. Gaussian initializer
mean0,scale.15 per physical action coordinate, seeds0/1 with common RNG namespace
across G and N (N changes draw shape, hence cannot promise elementwise noise
coupling across N). This is a NEW ANALYTIC controller, NOT evidence of fixed-N
Flow checkpoint transfer. K4 indices0/5/10/15, all episodes fixed eta.
12 configurations*16states*2C*4E*2seeds=3072 new physical rollouts; all factor
contrasts paired on state seed; horizon cutoffs do not multiply sample size.

Source/held tasks (all final evaluation states8..15): WIDTH: train .50/.80 for
both topology/N, test .65; EXTRAP: train .50/.65, test .80; COMBINATION: train
all configurations except(two-openings,N4,.65), test that combination;
FAMILY: train one-opening at all widths/N, test two-opening; reverse too.
Controller independently crossed at every configuration, but both seen at train;
this is controller-conditioned transfer, NOT unseen-controller generalization.

## Phase4 feature definitions, frozen before factor labels/model fitting
All from initial state/map/config, no future actions, success or terminal data:
- aperture clearance ratio = (width-2*(wall_radius+wall_margin))/(2*agent_radius).
  This is a packing/clearance proxy; NOT empirical capacity/throughput.
- optimistic time margin = H*dt-max_i(||goal_i-start_i||/vmax), a lower-bound
  travel-time slack ignoring routing and interactions (NOT joint feasibility).
- route detour ratio = mean_i(visibility-graph witness length/Euclidean length).
  A graph witness upper-bounds a feasible individual route, not exact shortest
  path or a joint plan. Graph resolution is fixed by the existing Geometry.
- route topology = number of isolated interior rectangular obstacle components
  (0 one-gap /1 parallel-gap); passage count=1/2. For this particular controlled
  map these have a direct geometrical meaning; not claimed sufficient generally.
- N/passage_count is demand per opening, assuming equal use; actual load may differ.
- initial approach imbalance = standard deviation of |start_x|/vmax.
- minimum initial wall clearance from actual wall segments.
For Toy diagnostics append clearance ratio, optimistic time margin, absolute
agent approach-time difference, minimum wall clearance, initial pair clearance,
usable bay depth, known connected-component count1 and hole count0. Raw Toy
parameters=[width,bay_top,Hseconds,N]. These definitions are fixed, exploratory.

Compare ENT,ENT+raw geometric parameters,ENT+mechanism,mechanism-only; 3 fit
seeds17/23/41 and exact same data/optimization as Phase1 (MECH uses fixed MLP).
For factor raw parameters=[width,N,Hseconds,number_openings,topology_hole_count].
Add mechanism time-margin removal as the one predetermined ablation in FAMILY
and WIDTH. Success/regret, Q and interaction metrics separate; no feature/model
selected by final-test performance. Cross-topology results are modest two-layout
transfer under one analytic navigation program; NOT all-scene mechanism transfer.

## Resource/continuity admission
Additional labels:2048 depth revision+1024 broadpilot+1024 FEW precision+3072
factor=7168. Target max16 rollout CPU workers plus <=4 GPU shards for fits.
Original8192-label cap suffices; expanded16384 cap will not be used. Persist
unknowns and failures; do not rerun failures to turn them into successes.
No broad-main is generated unless a PREVIOUS registered gate had passed (none).
