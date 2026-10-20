# C1 / OrthoFlow3: theory-to-implementation verdict

**The actual Toy geometry and original basis support local positive yielding and passing. Neither TT nor FF universally preserves both progress signs.** TT can remove a passing component at pair contact; FF can attenuate or reverse progress through Flow feedback and its changing constrained substeps. One historical FF trajectory demonstrates complete Yield–Hold–Pass, but full-episode behavior at the requested exact witness remains unverified.

This is a read-only source/evidence audit plus bounded CPU action probes. It changes no method or existing result. Evidence and scripts are confined to this new directory. No external theorem text was supplied: the verdicts below concern the explicit equations, sign conditions, and named assumptions in the request, not an invented stronger version of a conditional theorem.

## Provenance and evidence audit

Authoritative source root F: `/home/zhihan/research/Basin_C1_flow_field_poc_20261004/`. Audited files include `source/single_integrator/{environment,cbf,evaluate}.py`, `source/shared_control/{hard_projection,basis_families,diagnostic_corrector}.py`, `poc/field_controller.py`, `validation_v2/adapters.py`, `validation_stage2/factorial_controller.py`, the Toy retry helper, actual Flow agent, vendor network code, and protocol. `validation_manifest.json` contains full hashes and runtime configuration; `evaluation_frozen_manifest.json` was saved before the new witness evaluations.

The checkpoint is `F/validation_v2/assets/toy_ckpt.pkl`, SHA256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`. It is the provenance-tagged SI joint Flow BC policy (training seed 0, normalized observations/actions, ten Flow steps, no critic/Q guidance/distillation). Checkpoint training/evaluation metadata and normalization arrays are preserved verbatim in the manifest. The checkpoint loads half-length 1.3 m, corridor width 0.4 m, bay top 0.68 m, layout scale 1, max speed 0.5 m/s, dt 0.05 s, horizon 850.

**Toy backend:** `single_integrator.cbf.project_velocity`, SLSQP convex QPs with adaptive outer speed-ball cuts and acceptance certificates. The unchanged retry wrapper can use tighter identical Clarabel SOCP, then identical nonlinear SLSQP polish. FF additionally has the exact speed-ball projection shortcut when all linear rows are satisfied. `shared_control/hard_projection.py` is an audited, distinct general Clarabel implementation; it is **not** the Toy primary backend. Historical Clarabel-only TT labels were not merged with the present Toy panel.

Historical evidence was searched before probing:

| Evidence | Reuse and compatibility | Gap |
|---|---|---|
| Hard-field `screening_manifest.json`, `confirmation_manifest.json`, `implementation_audit.json`, K16 bank | Four registered source hashes match exactly; same Toy checkpoint and frozen action code | K16 does not contain eta* |
| `diagnostics/placement_theory_property_scan_v1` | Prior branch/activity/derivative evidence; contextual limits retained | Other candidates; not a certificate for eta* |
| `diagnostics/part2_geometry_aligned_20261012` and Part 1/3 implementation reports | Existing warning that Flow feedback, finite corrections, and task outcomes differ; no retraining warranted | Missing requested exact witness and complete per-substep h/r/q |
| Eight saved center trajectory pairs in `placement_boundary_mechanism_audit_v1` | Read-only geometry arithmetic; all noise keys match, exact replay of 13 selected saved actions | Four full candidate-4 episodes; four truncated zero episodes |
| Global SQLite queried with `mode=ro` for exact IEEE-754 eta identity | eta* has 102 rows only under a wholly AMBIGUOUS controller profile | Unknown checkpoint, safety, RNG, scenario; automatic reuse forbidden |

No new rollout was planned or submitted, so no rollout preflight or cache write was needed. The read-only cache coverage audit is saved; it is not mislabeled EXACT_REUSE. Full exact-witness episode outcomes remain UNKNOWN instead of being inferred from incompatible labels.

Noise uses the difficult-state protocol, **future_root=2026100601**, not the stage-2 descriptor's original default root: fold in uint32 of SHA256(base state UID)'s first eight hex digits, then seed index 0, then physical step. All four paired actions share the same state, key, frozen Flow, and basis; eta changes only between zero and (0.5,0,0.5). Midtrajectory probes restore recorded positions **and previous executed velocities**; they do not restore monitors or simulate continuations. Constructed states are labeled explicitly. Exact keys and origin paths are saved.

## Geometry and basis

`geometry_certification.md` lists all eight actual segments, Rw=0.17176666666666665 m and D=0.3201 m, endpoint branches, speed constraints, an open safe neighborhood, and a whole-path CBF feasibility construction. Geometric compatibility is **VERIFIED**. This does not validate the frozen policy's closed-loop behavior.

The basis is unchanged: bounded goal displacement, scaled Tikhonov perpendicular safe-Flow residual (scale 3.303687238760696), and bounded away-from-other displacement. The frozen family-study box is `[.5,1.25] x [-.5,.5] x [0,.75]`: eta* is legal, on the first-coordinate boundary; zero is an auxiliary baseline outside this box. For eta* the perpendicular coefficient is zero. At `p1=(-.30,0), p2=(.02,.13)`, raw delta2y=0.03591948 and delta1x-delta2x=0.17830289 m/s, positive on a certified open neighborhood. The new paired actions are:

| Pipeline | Executed u2y (m/s) | Executed u1x-u2x (m/s) |
|---|---:|---:|
| TT | 0.12013649 | 0.08658147 |
| FF | 0.12013649 | 0.04746591 |

Both signs survive at this state even with multiple walls active. The zero baselines already have positive progress; neither action can be attributed wholly to the correction. Raw witness expressivity on this region is VERIFIED; arbitrary-state expressivity is FAILED. Recorded initial 820013 has negative raw delta2y.

## TT: exact projection is not a progress-preservation theorem

The executable formula `uTT=P_U(a_safe+B eta)` is VERIFIED. It preserves convex safety feasibility to the frozen tolerances; nonexpansiveness alone does not preserve a coordinate's sign.

At constructed state `p1=(-D/2,-.2+Rw), p2=(D/2,-.2+Rw)`, the agents are collision-free with the CBF pair exactly at contact and wall safety clearance at the floor boundary. The raw signs are positive: delta2y=0.00564500119 and delta1x-delta2x=0.1797725196. The TT projection input has both positive signs. But the pair row is

`-2 D (u1x-u2x) >= 0`, hence **u1x-u2x<=0**.

The computed TT gap is 2.78e-17 m/s, numerical zero, and FF gap 3.47e-18. These are not meaningful positive margins. This is an exact geometric counterexample to unconditional positive-gap preservation, not a contradiction to a theorem that excludes this face or requires tilted separation/slack. Adjacent ceiling/side segments also have duplicate active normals at this state; simple independent two-face KKT assumptions cannot be presumed.

At recorded initial 820001, raw delta2y is positive (0.00338643) but the TT input/output u2y is -0.00419612, with no active constraints. The anchor's negative yielding component dominates before projection. This illustrates why **positive correction**, **positive total projection input**, and **positive executed action** are distinct claims.

## FF: propagation, endpoint feedback, and active faces

The frozen recursion is exactly the requested ten-step recursion, with `rho=min(1,0.1/(1-k/10))`. The implementation's min avoids last-step floating cancellation and is mathematically 1/(10-k). Final h10=q9; there is only `assert_safe`, no extra terminal clipping or correction. The new reconstructed witness actions agree with the canonical TT/FF runtime with maximum error **0** on this panel.

Toy world/local transforms are identity. With actual action normalization Sigma and mean mu,

`f(h,s)=Sigma v_theta(obs_normalized, Sigma^-1(h-mu),s)`,
`J_h f = Sigma J_z v_theta Sigma^-1`.

AD is applied to this complete world-coordinate expression, not a normalized-coordinate Jacobian. The CSV records h, f, delta, r, q, h_next, all CBF slacks, active rows, speed slacks, diagnostics, world Jacobian, and `I+(1-s)J` for every substep. At k=9 it also records D9, `I+0.1J`, singular values, and two independent centered-difference checks. Finite-difference absolute Frobenius errors at epsilon 1e-4 have median 2.92e-5/max 1.19e-4; at 3e-5 median 2.62e-6/max 1.07e-5. Their decrease supports the pointwise AD implementation, not a uniform smoothness certificate.

Across 34 sampled endpoints, `||I+0.1J||2` ranges from **0.569929 to 1.484095**. Some points are contraction-like, and others have an expanding local direction. There is no uniform contraction/recovery guarantee. In particular, observing a small endpoint norm does not establish preservation of a signed progress cone.

Two concrete recorded states separate mechanisms:

| State / physical step | Raw (yield, pass) | TT executed (yield, pass) | FF executed (yield, pass) |
|---|---|---|---|
| 820001, FF candidate4 trajectory, t19 | (0.00308500, 0.00009319) | (0.00198105, 0.17024404) | (-0.00066740, 0.17011202) |
| 820013, FF candidate4 trajectory, t215 | (0.01080433, 0.18322005) | (0.08879433, 0.01979909) | (0.08817838, -0.01966930) |

At t19, the FF projections do not contribute a finite paired progress difference. Across ten substeps, yielding direct correction +0.00308500 is opposed by Flow feedback -0.00264957; the final yielding effect is only +0.00043543, insufficient to overcome the negative zero baseline. Passing direct +0.00009319 is opposed by feedback -0.00013410, giving a negative **correction effect**, even while absolute passing velocity stays positive. D9's witness operator norm is 1.07836, while its paired secant ratio along this realized displacement is 0.80501. A sampled directional attenuation is not uniform contraction.

At t215, D9's witness operator norm is 0.78370 and paired secant ratio 0.35560. Yet FF's final r9 passing gap is already -0.00134157; wall projection makes it -0.01966930. TT has active pair row 0 and duplicate wall rows 12/14 and preserves both signs. FF's final rows 12/14 are active, its pair row is inactive, and earlier substeps have other active sets. Over the entire finite paired recursion, passing components sum as direct +0.18322005, Flow feedback -0.03335276, projection contribution -0.13568478, leaving +0.01418252 relative to an already negative FF baseline. These are an exact arithmetic decomposition along two actual pseudo-time paths, **not** an independent causal intervention or a derivative theorem.

The logs include 72 FF substep records with pair and wall rows simultaneously active. Endpoint duplications and weak/nonunique reconstructed multipliers occur. No analytical isolated-face assumption or uniform active-set stability was certified. FF sign preservation is **PARTIAL**, demonstrated for some states, and **FAILED as an unconditional claim**. A progress-recovering Flow structure on an open domain is UNKNOWN.

## Numerical certificates and sample scope

There are 17 state records (15 distinct physical-state/key inputs; two t0 aliases are retained for historical replay), each with four paired outputs: 68 action rows and 340 explicitly logged FF substeps. Twelve state records have both raw witness signs positive; among those TT retains both executed signs in 9 and FF in 7, using 1e-8 m/s only as an analysis threshold for numerical zero. Across all records TT has both signs in 11, FF in 9. This selected, partly duplicated panel is **not** a success-rate or population estimate.

Maximum logged primal violation is 6.62e-11, reconstructed KKT stationarity infinity norm 7.47e-7, and complementarity 3.06e-12, within the frozen tolerances. No numerical failures occurred. In the 408 action/substep diagnostic records (including repeated final FF projections), statuses are: exact speed-ball shortcut 138, primary solved 254, nominal feasible 15, tighter identical SOCP retry 1. Internal retries during canonical replay are not separately instrumented. Solver-native duals are not exposed by the primary interface: saved multipliers are explicitly **independent NNLS reconstructions** against all active CBF rows and active speed-ball unit normals. Redundant rows make individual duals nonunique; a low KKT residual does not certify LICQ or differentiability.

## Claim-by-claim verdicts and paper scope

| Statement | Verdict | Paper-appropriate wording |
|---|---|---|
| Finite-segment Toy geometry admits a collision-free local yielding/passing region | VERIFIED | Explicit geometric and CBF feasibility certificate, with actual dimensions |
| Original eta* expresses both desired raw directions on a nonempty region | VERIFIED | Open-neighborhood certificate; do not extend to all difficult states |
| TT/FF executable formulas, shared-noise pairing, and world-coordinate normalization | VERIFIED | Implementation identity, plus exact witness action replay on this panel |
| Original basis yields positive raw progress at every eligible-looking Toy state | FAILED without explicit eligibility inequalities | Provide actual state inequalities and sign margins |
| Positive raw correction implies positive total/executed action | FAILED | Anchor dominance counterexample; pair-contact annihilation |
| Projection feasibility/nonexpansiveness guarantees positive passing | FAILED | Aligned pair-contact row forbids positive gap |
| A regular active-face KKT formula applies on all tested states | FAILED | Redundant endpoint normals, weak active rows, and changing constraints |
| FF has uniformly contractive D9 on the tested states | FAILED | Pointwise singular norm up to 1.484095 |
| Flow Matching automatically gives progress recovery or cone preservation | UNKNOWN as an architectural property; no such guarantee established | State it as an additional hypothesis, then separately validate it on a specified domain |
| Actual FF preserves both desired progress signs on some states | PARTIAL | State-specific signed margins, Flow endpoint response, and actual active rows |
| Actual FF preserves both signs wherever raw delta has both signs | FAILED | Recorded t19 and t215 counterexamples |
| Local positive first-step signs imply complete Yield–Hold–Pass/task success | UNKNOWN as a conditional theorem; not established by these probes | Require invariant clearance, integrated progress, and event/horizon conditions |
| Complete local Yield–Hold–Pass occurs in the frozen pipeline | PARTIAL | One saved FF candidate4 episode, followed by success |
| Every safe successful pass requires chiY>0 | FAILED | Saved TT 820001 passes with y1<0 and chiY<0 |
| eta* produces complete Yield–Hold–Pass or task success | UNKNOWN | Exact compatible episode data unavailable |
| FF is globally better than TT | UNKNOWN; no supported ordering | The two historical states exhibit opposite success/deadlock outcomes |

A conditional mathematical implication is not refuted when its assumptions fail. For RSS, retain the conditional correction-realization theorem with **explicit** anchor, geometry, active-face/regularity, Flow response, and invariant/event assumptions. Pair it with the geometric certificate and narrowly stated action/trajectory observations. Narrow any claim equating raw correction signs, instantaneous action progress, indicator crossing, and task success. Unconditional progress preservation, universal D9 contraction, and necessary y2>D passage are contradicted by actual implementation evidence. A general Flow recovery theorem, exact-witness full-episode guarantee, and all-assumption certification still need evidence.

## Answers A–F

**A.** Yes for the geometric, feasible-direction, and raw-basis conditions: a nonempty open safe region is certified. Whether every additional assumption of the newly derived theorem, especially a domain-wide Flow/active-face condition, holds is UNKNOWN.

**B.** Yes on the certified region and specific recorded states; no universal sign guarantee exists for the original basis.

**C.** TT preserves both signs in specific cases, including the geometric witness. It does not guarantee them: anchor dominance and exact pair-contact constraints provide separate failures of the unconditional inference.

**D.** FF preserves both signs on some observed branches with actual constraints, but not generally. Feedback can attenuate progress even without projection intervention; active walls can further remove passing progress. Pointwise contraction-like D9 behavior is neither a sign certificate nor a uniform bound.

**E.** Yes in one compatible saved FF candidate4 trajectory: yield step 259, hold 1.65 s with positive chiY, pass step 292, actual success at step 455. For eta* specifically, complete mechanism and task outcome are UNKNOWN.

**F.** Suitable: conditional mathematical statements, actual geometry/basis certificates, executable recursion, and state-specific numerical/trajectory evidence. Require narrower wording: active-face formulas, progress realization, recovery, and mechanism-to-success implications. Contradicted: unconditional sign preservation, uniform D9 contraction on this panel, and chiY>0 as a necessary condition for every safe pass.

## Reproduction and costs

Use `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python` with `JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1`:

```sh
C1_DIAG_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
"$C1_DIAG_PY" diagnostics/tt_ff_correction_realization_20261017_v1/validate.py
"$C1_DIAG_PY" diagnostics/tt_ff_correction_realization_20261017_v1/analyze_saved.py
"$C1_DIAG_PY" diagnostics/tt_ff_correction_realization_20261017_v1/verify_and_manifest.py
```

The pre-evaluation audit is an included frozen input. The first script reads existing sources/checkpoint/trajectories and writes only this directory; the second performs arithmetic-only postprocessing. No env.step is called.

Measured diagnostic run: 3.34247 s wall time (process CPU time in manifest), 64 native Flow samples, 58 FF integrations including canonical checks, 580 FF field evaluations, 340 world-Jacobian evaluations, 544 centered-difference field evaluations, and 701 projector requests. There are 47 canonical action checks (34 witness runtime checks, 13 historical action replays). Native samples each contain ten network steps; Jacobian evaluations are separately counted, not equated to scalar forward calls. Barrier-only state-selection scans and read-only cache queries are included in the measured run. Solver internal iterations are unavailable through the primary interface and are not invented. New rollouts=0, new physical steps=0, training/fits=0, GPU use=0. Eight historical trajectory pairs were reused. Existing code/config/checkpoint hashes were checked unchanged.
