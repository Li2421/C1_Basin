# Frozen Toy geometry and local realization certificate

Evidence date: 2026-10-17. Sources are under `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/` (F). See `pre_evaluation_audit.json`, `evaluation_frozen_manifest.json`, and `validation_manifest.json` for full absolute paths, SHA256 values, checkpoint configuration, and imported dependency hashes. No controller, checkpoint, constraint, basis, evaluator, or environment was edited.

## Actual geometry and safety set — VERIFIED

The authoritative runtime is `validation_v2.adapters.OtherRuntime('toy_give_way')`. It loads `F/validation_v2/assets/toy_ckpt.pkl`, SHA256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`, through `source/single_integrator/evaluate.py:load_policy`. Its evaluation configuration has **corridor_half_length=1.3**, not the environment class default 2.5. Goals are `(1.09,0)` and `(-1.09,0)`.

| Segment index / name | Start (m) | End (m) |
|---|---|---|
| 0 left_end | (-1.3,-0.2) | (-1.3,0.2) |
| 1 right_end | (1.3,-0.2) | (1.3,0.2) |
| 2 floor | (-1.3,-0.2) | (1.3,-0.2) |
| 3 left_ceiling | (-1.3,0.2) | (-0.2,0.2) |
| 4 right_ceiling | (0.2,0.2) | (1.3,0.2) |
| 5 bay_left | (-0.2,0.2) | (-0.2,0.68) |
| 6 bay_right | (0.2,0.2) | (0.2,0.68) |
| 7 bay_top | (-0.2,0.68) | (0.2,0.68) |

`wall_layout_scale=1`; corridor width 0.4 m; bay top 0.68 m; agent radius 0.16 m; wall radius 4/600 m; wall collision margin 0.005 m; pair collision margin 0; CBF separation buffer 0.0001 m. Thus

`Rw = 0.17176666666666665 m`, `D = 0.3201 m`.

The evaluator collision thresholds omit the CBF buffer: wall 0.17166666666666666 m and pair 0.32 m. The pair **CBF** uses `h=||p1-p2||²-D²`, not `||p1-p2||-D`. Each wall CBF uses `h=dist(p_i, finite segment)-Rw`, with the exact clamped nearest-point normal. The 17 linear rows comprise the pair row followed by eight wall rows for each agent; two Euclidean speed balls are also enforced. Row `1+8*i+l` means agent index i, segment index l (zero based).

`gamma=gamma_wall=1`, physical `dt=0.05 s`, `max_speed=0.5 m/s`, horizon 850 steps, goal tolerance 0.08 m. Hence `gamma*dt=0.05<=1`. Convex barrier supporting inequalities give the held-action clearance certificate starting in the safe component. Geometry remains frozen during each ten-step pseudo-time solve.

The CBF numerical tolerances are feasibility `1e-9`, speed `1e-10`, optimality `2e-6`; primary SLSQP `ftol=1e-12`, max iterations 200, max speed cuts 32. These are numerical tolerances, not enlarged physical clearances.

## Nonempty local yielding region — VERIFIED geometrically

The bay has center-position width `0.4-2*Rw=0.0564666666666667 m`. Above its mouth, its center can reach `y=0.68-Rw=0.5082333333333334>D`. It therefore has positive room for the proposed lateral clearance; this conclusion uses the actual wall dilation.

An explicit shoulder state is `p1=(-0.30,0)`, `p2=(0.02,0.13)`. `geometry_certificate.json` gives every wall distance, nearest-point parameter, and CBF slack. Its minimum wall h is `0.02136541249161303 m`; pair h is `0.01683599000000001 m²`, with center distance about 0.3453983 m. Per-agent Euclidean perturbations smaller than 0.001 m retain wall h at least 0.0203654 m and pair distance at least 0.3433983 m, strictly above D. This is a nonempty open, collision-free neighborhood inside the actual domain.

The velocity witness `u1=(0.01,0)`, `u2=(-0.01,0.005)` has both desired progress signs. All 16 wall CBF slacks are positive (minimum 0.0282333 m/s), pair CBF slack is 0.00533599, and speed slacks exceed 0.4888 m/s. Strict slack and continuity establish a neighborhood supporting simultaneous small yielding/passing velocities. This is a **feasibility witness**, not an alternative policy.

There is also a fully cleared geometric configuration: `p1=(-0.4,0), p2=(0,0.4)`. A continuous local Yield–Hold–Pass path consists of:

1. Hold p1 at (-0.4,0), move p2 from (0,0) to (0,0.4).
2. Hold p2 at (0,0.4), move p1 from (-0.4,0) to (0.4,0).

On both entire segments every wall distance is at least 0.2 m; pair distance lies in `[0.4,sqrt(0.32)]`. At speed 0.01 m/s all wall CBF slacks are at least 0.0182333 and pair CBF slack at least 0.0462223. The final chiY and chiP both equal 0.0799 m. These are bounds for the whole path, not a sampled collision check. This establishes geometric and CBF compatibility of local maneuvers. It does **not** establish that the frozen policy follows this path, that a task completes within the horizon, or a global joint-planning theorem.

## Original basis — VERIFIED on an open geometric region; PARTIAL on recorded states

The frozen fields are radial bounds of goal displacement and away-from-opponent displacement, plus the archived scaled perpendicular safe-Flow residual. At eta=(0.5,0,0.5), the perpendicular column has coefficient zero. In particular, this witness correction is independent of the Flow anchor, although the executed actions are not.

At the shoulder state, delta2y=0.03591948175783337 and delta1x-delta2x=0.17830288652926893 m/s. Radial projection onto a ball is nonexpansive. In the 0.001 m per-agent neighborhood, each delta_i changes by at most 0.0015 m/s; the two signs retain margins at least 0.0344194 and 0.1753028 m/s. Thus the original fields, without any replacement or rescaling, have the desired raw expressivity on a genuine open region.

This is not universal: recorded state `hard_confirm_toy_1_820013:t0` has delta2y=-0.00029941033. Raw expressivity and executed feasibility must remain separate.

## Assumption audit

| Condition | Status | Evidence / limit |
|---|---|---|
| Actual finite-segment geometry supports nonempty safe yielding/bay space | VERIFIED | Explicit open neighborhood and whole-path bounds above |
| All wall, pair, and speed constraints can permit small positive yielding/passing together | VERIFIED | Strictly feasible velocity witness, all rows checked |
| Original witness has both positive raw margins somewhere in the real geometry | VERIFIED | Positive neighborhood certificate |
| Every difficult Toy state has both raw signs | FAILED | Recorded t0 counterexample above |
| Bay mouth can be modeled using only infinite lines / segment interiors | FAILED | At shoulder p2 the ceiling and bay-side nearest points are mouth endpoints |
| Only a single wall and pair face matter everywhere | FAILED | CSV records multiple wall rows, duplicate endpoint normals, speed activity, and changing active sets |
| Full active constraint matrix automatically satisfies LICQ / unique duals | FAILED as a blanket claim | Adjacent finite segments share endpoint normals; weak/redundant active rows occur |
| Pair contact automatically allows strict horizontal passing | FAILED | At equal heights and distance D, pair row enforces u1x-u2x<=0 |
| Passing always requires y2>D | FAILED as necessity | Historical TT passage uses y1<0; see trajectory report |
| Frozen Flow has the theory's recovery structure on a whole local domain | UNKNOWN | No specified/certified domain or uniform bound; only pointwise diagnostics |
| Every sufficient assumption of an external newly derived theorem holds | UNKNOWN | No complete theorem/hypothesis statement was supplied beyond the prompt; results certify the explicit conditions above |

Nearest-point branches for every agent/segment at every probed state are in the action CSV. Distance to a convex segment is differentiable away from zero distance, including nearest-point transition locations; branch changes can still affect higher derivatives. Projection active-set regularity is a separate issue. No two-face analytical projection formula was forced onto a redundant or different active set.
