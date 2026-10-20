# Agent B — Yield Reachability

## Scope and evidence audit

This independent analysis uses Agent A's checkpoint-loaded `geometry_contract.json`, frozen root `/home/zhihan/research/Basin_C1_flow_field_poc_20261004`, checkpoint SHA256 `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`. Before new probes it reviewed the existing 68-action/340-substep panel, exact replay records and eight immutable trajectory pairs in `diagnostics/tt_ff_correction_realization_20261017_v1/`. Those records establish some action/trajectory behavior but contain no exact zero-offset/zero-velocity witness panel. Four full historical episodes use candidate4, not eta*. Four zero-correction episodes are truncated. Their outcomes cannot certify exact-witness reachability.

Only 24 deterministic no-`env.step` actions were added after the parent announced their purpose and bounded cost. Nothing changes the method, candidate box, projection, or evaluator. This is neither a rollout nor a new controller. Source hashes, seeds, exact keys, actions, diagnostics and measured cost are in `results.json`, `zero_offset_actions.csv`, `probe_traces.json` and `compute.py`.

## 1. Zero-offset reference reachability: PROVEN

Use the positive-volume entry box, in metres,

`x1∈[-.65,-.35], x2∈[-.025,.025], y1∈[-.02,0], y2∈[-.02,.02]`.

For every state in this box, hold robot 1, and translate robot 2 vertically at `vY=.03 m/s` until `y2=.4`. This includes `y1=y2=0` without a positive lateral seed. It is a reference feasibility certificate, not an instruction to substitute a controller.

Let `s=x1-x2`, `z=y2-y1`, `Ds=.3201`, `Rw=.17176666666666665`. Throughout the path, `|s|≥.325`, `z≥-.02`. The implemented squared pair CBF has slack

`2z vY + gamma(s²+z²-Ds²) ≥ .325²-Ds²-.0012 = .0019609900000000166 m²/s > 0`.

This conservative bound drops the nonnegative z² term. Pair distance is at least `.325 m`, giving distance margin `.0049 m`. Robot 1's zero action satisfies each of its eight wall rows because its clearance is nonnegative.

The potentially restrictive entry corners are the two points `(±.2,.2)`. Horizontal distance from robot 2 is `a≥.175`. Below `.2`, write `t=.2-y2≥0`, `d=√(a²+t²)`. The upward-action wall CBF is `-(t/d)vY ≥ -(d-Rw)`, or `vY≤d(d-Rw)/t`. For `t>0`,

`d(d-Rw) = d(d²-Rw²)/(d+Rw) ≥ (a²-Rw²+t²)/2`,

hence its allowed upward speed is at least `√(a²-Rw²)≥.03348450719694443 m/s > .03` by AM–GM. At `t=0` its normal has zero vertical component and is harmless. This bound covers the finite ceiling endpoint and the coincident bay-wall endpoint; it does not discard duplicate constraints.

Above `.2`, bay sidewall interior normals are horizontal and permit this vertical motion; clearance is at least `.175-Rw=.003233333333333338 m`. Bay-top interior clearance at the final point is `.68-.4-Rw=.10823333333333338 m`, and its CBF slack is at least `.07823333333333338 m/s`. Floor motion points away from the floor. The left and right end segments have horizontal distance at least `1.275`, so below their upper endpoints the same corner bound is much larger; above those endpoints their vertical normal component is nonnegative. All additional walls thus satisfy their actual finite-segment branches. Both speed balls hold (`0,.03 < .5`). The global wall-clearance lower bound `.00323333 m` remains positive.

Thus a continuous collision-free path and a path admitted by all 17 CBF rows and both speed balls exist: **PROVEN**. This path ends in a positive-width bay region, not an infinitesimal boundary witness. It never invokes a different projection or a modified basis.

The required time is `(.4-y2(0))/.03≤14 s`, at most 280 physical `.05 s` steps when the final held action is shortened to land exactly at `.4`. The full horizon is `42.5 s`; with arbitrary prior physical time this requires at least 280 remaining steps. While speed is `.03`, the actual monitor's strict condition `max_speed<.025` is false, so no detector deadlock can accumulate during this reference motion. A shortened final step can be below threshold for at most one step, which cannot create the five-second hold after preceding moving steps. These are properties of the reference path; they do not certify actual controller behavior or the later Pass/Resume phases.

## 2. Original basis and symmetry: PROVEN algebra, limited control evidence

The authoritative basis implementation is `source/shared_control/basis_families.py:129–152`, `diagnostic_corrector.py:14–40`: bounded goal displacement, scaled smooth perpendicular safe-Flow residual, bounded away-from-opponent displacement. There is no extra yield basis. At `y1=y2=0`, with both goals at y=0, `B_goal,2y=B_rel,2y=0`. For eta*=(.5,0,.5), **raw delta2y=0 exactly**, independently of Flow. Thus a theorem requiring strictly positive raw eta* yielding cannot include these states. This does not imply the executed policy has zero yielding: its anchor can contain transverse velocity.

For general legal eta, at these states `B_perp,2y=3.303687238760696*a_safe,2y`; nonzero eta_perp can amplify or attenuate the existing transverse anchor, but has no deterministic positive sign guarantee. If a safe anchor is also purely horizontal, all three columns have zero y component. The real wall geometry is not invariant under vertical reflection, so no symmetry of the full projector is presumed. A concrete mechanism-level reference is p1=(-.4,0), p2=(0,0), with a hypothetical zero safe anchor: eta* produces target ((.05,0),(-.05,0)); its pair-only projection is ((.03595999375,0),(-.03595999375,0)). Every wall row and speed ball accepts this point, so it is also the full projection and has zero transverse action. This proves that the basis and safety constraints alone do not force positive yielding. The zero anchor is hypothetical; it is not claimed as the frozen Flow output.

The actual Toy checkpoint is asymmetric through observation/goals/geometry and noisy Flow sampling. Its Flow can produce transverse action even when physical y offsets are zero. No mathematical stochastic support or uniform noise condition was certified. The two sampled seeds provide neither a domain-wide nor a high-probability startup proof.

### New zero-offset actions: EMPIRICALLY OBSERVED

Three physical states, all with previous velocity zero:

| state | p1 | p2 | pair status |
|---|---|---|---|
| K_zero_offset_0 | (-.4,0) | (0,0) | positive pair clearance |
| K_zero_offset_1 | (-.3201,0) | (0,0) | exact pair CBF contact |
| K_zero_offset_2 | (-.16005,0) | (.16005,0) | exact pair CBF contact |

Each uses seed indices 0 and 1 and the same explicit difficult-state noise construction (root 2026100601; fold SHA256 UID's first eight hex digits, seed, physical step 0). Each state/key is paired across eta=0 and eta*, TT and FF. Synthetic UIDs are explicit; they are not claimed as historical rollout IDs.

All six state/key pairs have strictly positive yielding velocity in both placements and both eta settings: TT range `.0052853063–.0453485844 m/s`, FF range `.0035248821–.0478628257 m/s`. Eta* raw yielding correction remains exactly zero. TT's yielding is identical to its zero baseline at all six pairs. FF's finite path can respond to the horizontal correction, but its positive yield was already present in the zero baseline. Hence **zero physical offset can start yielding under the actual frozen policy on these sampled keys**: EMPIRICALLY OBSERVED. Sustained yield until bay entry, or success from these states, is **UNKNOWN** because no physical step was executed.

At both exact pair-contact states, all passing gaps are numerical zero (absolute <6e-17); the exact pair row requires `u1x-u2x≤0`. Positive yield does not imply immediate positive passing.

The prior panel also supplies a counterexample to universal positive yielding at eta*: recorded initial state 820001 has positive raw delta2y `.00338642678` but TT u2y `-.00419612300` and FF u2y `-.00606923019`. The negative anchor/Flow response dominates. This is **FAILED** for an unconditional witness-yield guarantee, and does not refute a theorem retaining explicit total-action inequalities.

## 3. Quantitative z_seed: what is and is not certified

A universal positive offset independent of x positions, eta, anchor, placement, Flow key and projection cannot be inferred. Over the original legal eta box, eta=(.5,0,0) is allowed: for positive y2 the goal correction pulls downward, so no all-eta positive raw-yield condition follows from `z>0`. Conversely, the new probes show **zero offset can suffice for positive actual instantaneous yielding**. Therefore `z_seed>0` is neither geometrically necessary nor universally necessary for instantaneous actual policy startup.

A useful, explicitly narrower raw-margin certificate is available for the fixed state family

`p1=(-.4,0), p2=(0,z), previous velocities=0, eta*=(.5,0,.5)`.

For `0≤z≤.2`, the relative column is unsaturated (`√(.4²+z²)<.5`) while the goal column is saturated. Consequently

`delta2y(z)=.5 z [1-.5/√(1.09²+z²)]`.

Its derivative is at least `.5-.25/1.09=.2706422018>0`, so each threshold below has a unique minimal positive offset. Values are computed directly from this scalar formula by bracketed root finding; they are numerical roots of a proven monotone formula, not a fitted model or a policy optimization.

| desired raw delta2y (m/s) | minimum z (m) | z/(2h), full channel center height | z/h, this family’s upward range | admissible when y1=0,y2≤h? |
|---:|---:|---:|---:|---|
| .001 | .003694897264 | 6.5435% | 13.0870% | yes |
| .005 | .018472328736 | 32.7137% | 65.4274% | yes |
| .010 | .036931203484 | 65.4035% | 130.8071% | no; exceeds upward allowance h=.02823333 |

These are **NUMERICALLY CERTIFIED for this scalar raw-basis family**, with monotonicity PROVEN. The largest raw margin within the y1=0 main-channel family is `.007643302692 m/s`. They do not certify an executed `.001` or `.005` margin. TT additionally requires the anchor and active projection not to remove progress; FF requires its entire ten-step response. No threshold for actual TT/FF sustained reachability is established: **UNKNOWN**.

A more general sufficient condition for physical startup is the actual state/key inequality `(P_U(a_safe+B eta))_2y≥nuY>0` for TT, or the analogous measured final FF inequality, combined with an invariant admissible tube and enough integrated progress to enter the bay. The first inequality is directly testable; the invariant-tube assumption has not been certified and cannot be silently added as an established fact. The positive-measure entry-box reference certificate shows geometry does not itself require y_e>0; replacing y_e by a policy progress condition still leaves a closed-loop proof obligation.

## 4. B1–B4 verdicts

| Question | Verdict | Evidence/limitation |
|---|---|---|
| Zero initial offset can geometrically Yield? | PROVEN | Full entry-box vertical safe path |
| Original complete Safety QP permits it? | PROVEN | Analytic all-17-row and speed-ball certificate |
| Frozen TT/FF can start from zero offset? | EMPIRICALLY OBSERVED | 6 state/key pairs per placement, eta* and zero; positive first action |
| Frozen TT/FF necessarily enters bay from zero offset? | UNKNOWN | No compatible complete eta* episode or invariant action bound |
| Positive raw yielding at exact zero offset for eta*? | FAILED | Original goal/relative y columns zero and perpendicular coefficient zero |
| Minimum physical offset required? | UNKNOWN as uniform closed-loop threshold; zero can suffice locally | No universal positive necessary offset; raw thresholds above are family-specific |
| Is y_e>0 required by geometry or reference CBF feasibility? | FAILED as a necessity | Explicit zero-offset reference path |
| Can y_e>0 be deleted from an existing TT/FF guaranteed escape proof? | UNKNOWN | It cannot be deleted solely from this reference certificate or finite action samples |

## Reproduction, sources and cost

Run from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/toy_K_quantitative_certification_20261017_v1/agent_b/compute.py
```

Key source locations in the frozen root: `cbf.py:60–93` defines finite segment distances and all 17 rows; `cbf.py:96–162` defines the unchanged cutting-plane projection and certificates; `basis_families.py:129–152` defines the actual basis; `poc/field_controller.py:215–264` defines shared preparation and projected FF integration; `validation_stage2/factorial_controller.py:23–29` routes canonical TT/FF; `environment.py:193–219` defines the progress/deadlock monitor. The implementation's physical CBF/speed constraints are used as-is.

Measured new calculation cost: 1.16638 s within-script wall time, 1.63844 s process CPU time, shell process 1.93 s; 6 native Flow samples (60 network time steps), 12 FF integrations (120 field evaluations), 138 projector requests, 24 action rows, no numerical failures. The CPU run emitted an optional CUDA plugin discovery warning before successfully using the explicitly requested CPU backend; it did not consume GPU or change results. There are zero new rollouts, physical steps, training, fits or policy modifications. Raw scalar roots are geometry/basis arithmetic, not an optimized controller.
