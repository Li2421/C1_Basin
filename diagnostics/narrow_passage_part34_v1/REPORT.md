# C1 Narrow Passage Part 3–4 read-only audit

## Verdict and evidence scope

| Claim | Verdict | Scope |
|---|---|---|
| `PART3_GEOMETRY` | **PARTIAL** | Formal A1/A2/A4 hold in certified local rectangles; A5 is vacuous for two robots. A3 holds at local states in 236/756 traces with a 1 mm wall-CBF margin, but at none of their original t0 states. This does not cover all Toy conflict trajectories. |
| `PERP_ACTION_MECHANISM` | **SUPPORTED** | Exact local pair-CBF derivative and finite frozen-TT single-step comparisons. Other active-set branches are explicitly excluded from the pair-only formula. |
| `TT_FIXED_ETA_ESCAPE` | **SUPPORTED** | A cached fixed-η continuation starts from a state satisfying formal A3, has a measurable Perp contribution, then yields and passes at D=0.3301 and succeeds. This is an existential finite-state witness, not a general escape or full-episode Perp causal guarantee. |

No new rollout, training, GPU computation, controller edit, or label-driven change of definitions was made. There are 19 distinct CPU single-step probes at archived states and keys: 14 from the initial audit and 5 added after the user supplied formal A1–A5. All outputs remain in this directory. The original boundary-level statistics remain in [trajectory_audit.csv](trajectory_audit.csv), [static_probes.json](static_probes.json), [witness.json](witness.json), [provenance.json](provenance.json). Formal follow-up: [protocol](FORMAL_PROTOCOL.md), [geometry results](formal_geometry.json), [state coverage](formal_state_coverage.csv), [local probes](formal_local_probes.json), [formal witness](formal_witness.json).

## Historical evidence and identity

The compatible source is `/home/zhihan/research/Basin_C1_flow_field_poc_20261004`: `validation_stage2/protocol.json`, `validation_stage2/results/*_TT.{json,npz}`, `validation_v2/assets/toy_ckpt.pkl`, `validation_v2/adapters.py`, `poc/field_controller.py`, and the copied `source/` tree. The Stage-2 protocol parent SHA matches the archived V2 protocol. The loaded Toy checkpoint SHA256 is `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`; the Stage-2 TT controller UID is `stage2_ctl_d1dc40505c93b34074e0bb55d90fca2cd635c9338b8d2087a4d05160b7d73e41`. Every read trace has its protocol state, η, seed, start position, goals, and `p[t+1]=p[t]+0.05u[t]` checked. Each probed key matches the frozen SHA256-UID/fold-in rule, and recomputed TT actions match cached actions to at most `1.3e-15` (14 probes). All source hashes are recorded in `provenance.json`.

There are **756 complete Toy TT records out of 816 requested Stage-2 cases** (92.6% coverage), not 816 observed outcomes. Their terminations are 665 success, 64 deadlock, 27 timeout; the 60 absent cases are **unknown**, not failures. The old Part 2 sensitivity files and hard-field reports were inspected for implementation and outcome context, but Stage-2's actual TT traces are the trajectory source because they contain full positions, actions, keys and per-step projector certificates. A change to critic input would not invalidate these control labels. The selected Stage-2 ablation compares placements/η choices, not exact η⊥=0 at fixed other coordinates; no strict full-episode Perp ablation was found. The Stage-2 Toy protocol itself has zero exact η⊥=0 candidates.

## Step 1: geometry and assumptions

Frozen Toy configuration: corridor width `0.400`, radius `0.160`, wall radius `0.0066667`, collision wall margin `0.005`, CBF buffer `0.0001`; central top bay spans `x∈[-0.2,0.2]` at its mouth and reaches `y=0.68`; horizon 850 steps, `dt=0.05`, maximum speed `0.5`. The CBF wall center clearance is `c=0.1717667`. Away from the bay, the center corridor is only `y∈[-0.0282333,0.0282333]`, a `0.0564667` interval. Two opposite robots cannot overlap longitudinally there with the frozen pair-CBF safety distance `D_s=0.3201`. At upper bay height above `y=0.3717667`, its side walls restrict the center to `|x|≤0.0282333`. The eight **finite** wall segments and corner nearest-point gradients are used in the actual CBF.

The formal follow-up fixes an additional geometry-only positive margin `δ=0.01 m`, giving **D=0.3301 m**. It does not relabel the old statistics, which used `0.3201` as the CBF boundary threshold. The formal wall-safe primary region uses `m_w=0.001 m` above the wall-CBF boundary:

`C=[−1.1272333,1.1272333] × [−0.0272333,0.0272333]`,

`Y=[−0.0272333,0.0272333] × [−0.0272333,0.5072333]`.

For every point in both rectangles, distance to every finite wall segment is at least `c+m_w`. For C, the floor/ceiling and end-wall distances give this bound; for Y, the side-wall, corner, floor and top distances do. Thus wall-CBF height is at least 1 mm, with 1.1 mm separation above the evaluator's inclusive wall collision threshold. The rectangles have nonzero interior and are subsets of the actual center free space, despite the nonrectangular bay mouth.

| Formal assumption | Result and proof/evidence |
|---|---|
| A1 | **SUPPORTED mathematically:** `0.32 < W=0.4 < 0.64`; the local main corridor is single file. The widened bay does not violate A1. |
| A2 | **SUPPORTED mathematically:** the C/Y rectangles above have wall-CBF margin 1 mm, strict x ordering, and `H−h−D=0.1499 m>0`. |
| A3 | **PARTIAL finite-state coverage:** 236/756 trajectories (31.2%) contain qualifying local states (8,412 saved state/action pairs): 225 success, 1 deadlock, 10 timeout. None of the three distinct original t0 states qualifies. All 236 primary-region cases use reflected Role L below. |
| A4 | **SUPPORTED as geometric/CBF-feasible continuation:** `a_+−b_+−D=0.7699 m>0`; goals are at ±1.09 inside C. A constructive route is robot 2 vertically to H, robot 1 horizontally past b_++D to its right goal, then robot 2 down and left to its goal. Initial z≥0 makes the first move nondecreasing in pair distance; passing has lateral gap at least H−h=0.48; lowering has longitudinal gap at least 1.09−b_+=1.0627667. Small enough speeds satisfy the actual finite-wall and pair CBF rows and speed balls. This existence argument does not certify the deadlock monitor, horizon, or the frozen TT's choice of that reference route. |
| A5 | **VACUOUS/SUPPORTED for Toy:** no third robot exists. This provides no evidence for moving external robots in an N>2 environment. Participant pair/wall CBF feasibility remains checked separately. |

The geometry-fixed sensitivity family varies the positive wall margin, without choosing it by outcome:

| Wall-CBF margin m | Traces containing A3 | success / deadlock / timeout |
|---|---:|---:|
| 5 mm | 170 | 164 / 1 / 5 |
| 1 mm (primary) | 236 | 225 / 1 / 10 |
| 0.1 mm | 236 | 225 / 1 / 10 |
| 0.01 mm | 243 | 232 / 1 / 10 |
| 0.001 mm | 296 | 285 / 1 / 10 |
| 0.00001 mm | 676 | 665 / 1 / 10 |

The final near-boundary row has only `1e−8 m` wall-CBF margin. It is a geometric coverage diagnostic with negligible practical robustness. It is not used to claim primary-region coverage or to select a success witness. Failure cases inside A3 show that these geometric assumptions alone do not guarantee frozen-TT success. Example `c80c697d313541c7cf1d`, seed 2, satisfies A3 at step 75 and later terminates in deadlock.

The original audit also certified a product of radius-`0.005` Euclidean balls about `p0=(0,-0.015)` and `p1=(0,0.49)`. At the centers, minimum wall-CBF height is `0.0132333`, pair distance is `0.505`, and pair squared barrier is `0.15256099`. By distance's 1-Lipschitz property, throughout that product set wall height remains at least `0.0082333` and pair distance exceeds the formal `D=0.3301` by at least `0.1649`. This is a geometric certificate, not a dynamical reachability theorem. None of the 756 traces enters that small product ball. At the original boundary threshold, the broader observed region `|s|≤0.3201, |z|≥0.3201` occurs in 665 successful traces, 0 deadlocks and 0 timeouts; this finite-state coverage does not certify all states in that region.

Role R uses formal robot 1=physical agent 0 and robot 2=physical agent 1, with the original axes. Role L uses formal robot 1=physical agent 1 and robot 2=physical agent 0, **and reflects x′=−x**. Toy's walls and goals are symmetric under this reflection. Both keep `s=p0x−p1x`, while z is respectively `p1y−p0y` and `p0y−p1y`; formal robot 1 moves right in both charts. The two mappings are evaluated as fixed alternatives, without choosing a role from success. Old threshold-level success counts are 440 for Role R and 225 for Role L. At the formal positive-margin threshold D=0.3301, only 16 Role-R and 225 Role-L successes reach yield then pass; all 241 preserve z≥D between those threshold events. Four timeouts also reach both formal phases, so phase attainment is not episode success.

The user's supplied A1–A5 are now checked literally; the previous missing-definition status is resolved. Some actual bay-mouth states have multiple wall CBF rows active, which limits the pair-only action formula but does not violate A1 or A2. The original seed-5 witness below never satisfies A3 in the primary certified rectangles, even though it passes successfully; it cannot serve as a witness of the complete primary A1–A5 class. Its separate action-mechanism evidence remains valid. The deadlock monitor requires a 2 s progress window, progress `<0.01`, both speeds below `0.025`, and a held candidate for 5 s. Collision, success, deadlock, timeout are evaluated in that order after a swept step. These dynamic conditions are additional to the geometric assumptions.

## Step 2: exact basis, TT, and analytic branch

The frozen chain is `flow → a_safe=Π_U(flow) → B(goal, a_safe, relative) → u_TT=Π_U(a_safe+Bη)`. `B_goal` and `B_rel` use radial row bound `0.5`; `B_perp=3.303687238760696·[a_safe−(a_safe·g)g/(||g||²+(5e−7)²)]` per robot. The fixed η is recomputed through the changing state-dependent basis each physical step. `static_probes.json` gives each of the 14 matrices, raw `(dot s,dot z)` columns, projected actions, active wall/pair/speed rows and certificates. All 14 recomputations succeeded; no numerical unknown was converted to an outcome.

At witness step 30, only pair CBF is active and both speed balls/walls are slack. Raw `(dot s,dot z)` columns are `B_goal=(0.99994872,-0.00385765)`, `B_perp=(-0.00197096,-0.16070977)`, `B_rel=(-0.99988219,0.01534960)`. The actual projected finite derivatives are `∂dot z/∂η_rel=2.6e−14` and `∂dot z/∂η_perp=-0.16070215609`. The exact pair-halfspace derivative is

`b_z − z(s b_s+z b_z)/(s²+z²) = -0.16070215609`;

the proposed approximation `κ_B A_y s²/(s²+z²)` with `κ_B A_y=b_z` is `-0.16067190614`, error `3.03e−5` due to the small nonzero `b_s`. This is a strict local branch calculation plus an independent finite difference; it is not a global formula. The witness's negative η⊥ gives a positive `dot z` difference `+0.00563044 m/s` at this step relative to η⊥=0.

For two robots, `B_rel` is parallel to the pairwise CBF normal in joint action space. Euclidean projection onto the sole active pair halfspace removes that normal component, which proves its zero `dot z` derivative locally when the active branch stays fixed. This proof does not extend across added wall or speed constraints.

At witness step 100, pair plus agent-0 floor and agent-1 ceiling CBFs are active. Holding state, key, ηg and ηr fixed, the η⊥=0 action has `dot z=-0.03560762 m/s`, while the archived full η action has `dot z=+0.00641111 m/s`: **+0.04201873 m/s**. This is an effective local yielding-direction change, with linear certificate minimum `-6.4e−17` and speed-ball slack `0.4655 m/s`. The local derivative at the full η is almost zero because the active set differs across the finite intervention. The pair-only formula is **not applied** there. Other sampled bay-mouth states show Perp completely cancelled by projection; deadlock/timeout states can show positive single-step Perp changes yet still fail.

## Step 3–4: fixed-η trajectories and witness

**Formal A3 witness (supersedes the old witness as evidence for the formal local class):** archived case `024af7eaa6fb5c4f075f`, source state UID `toy_hard_2`, seed 15, physical step 76, Role L. Its diagnostic local UID is `toy_hard_2:024af7eaa6fb5c4f075f:t76:roleL` (a trace identifier, not a registered DB state). In formal coordinates, `p1=(−0.33942349,−0.02201749)` and `p2=(0.02572292,0.02581828)`, so `s0=−0.36514642<0`, `z0=0.04783578∈[0,D)`, with p1∈C and p2∈Y. Actual minimum wall-CBF height is `0.00621584` and pair squared barrier `0.03315616`, both strictly positive.

The **entire archived episode** fixes η=`[0.9508564700113,−0.012613422649824955,0.5149582319564191]`. At that A3 state the original noise key is `[1395625398,3268136300]`; same-state, same-key Perp removal changes formal dot z from `0.01000454` to `0.00801984 m/s`, a measurable **+0.00198470 m/s** Perp contribution. Active constraints are the pair CBF and physical agent-1 floor; both speeds are strictly slack. The full action's minimum linear residual is `8.7e−18`; it exactly reproduces the archived action. The local pair-only analytic formula is not used on this two-row branch. Original and transformed B matrices, η, noise and actions are in `formal_witness.json`.

The same cached continuation reaches formal χY≥0 at episode step 144, χP≥0 at step 175, and succeeds at step 347. Between yield and pass, z≥`0.33121647`, a `0.00111647 m` margin over D=0.3301. It completes 271 steps after the local state, within its 774 remaining steps; no deadlock candidate occurred in the full episode. Recomputed full-episode swept pair surface clearance is `0.00404990 m`; swept wall surface clearance is `0.00510348 m`, `0.00010348` above the evaluator threshold. Suffix CBF residual is at least `−1.34e−15`, with speed excess `9.87e−11` within the unchanged solver tolerance. This is a finite-state fixed-η witness satisfying the formal local geometry, with single-step Perp intervention evidence. No matched Perp-zero full-episode cache is available, so closed-loop Perp necessity remains unknown.

The five added A3-entry probes include three successes, one deadlock and one timeout selected in cached ID order. Only one of the three success entry points has a positive Perp contribution; the others have negative contributions. The deadlock entry has positive total yielding progress but still fails later. All results are retained in `formal_local_probes.json`, so no universal sign or sufficiency is inferred.

Within the primary A3-covered continuations, all 225 successes later yield then pass with formal lateral clearance preserved; 4/10 timeouts also do so, while the deadlock and the other 6 timeouts do not. This is finite-cache evidence and does not establish that satisfying A3 selects a successful frozen η. [verification.json](verification.json) records frozen-source hash agreement, numerical certificates, formal progress counts and script hashes.

The following original statistics and witness use the **CBF boundary threshold 0.3201**, and remain as separate execution evidence rather than formal δ=0.01 certification:

With the physically yielding role chosen from the sign of first `|z|≥D`, 665/665 successful cached traces reach yield then `χP≥0`; 5/27 timeout traces also reach both, and all 64 deadlocks do not. Only 246/670 ordered yield/pass traces keep signed lateral clearance at least `D` continuously between the two threshold events. Lateral clearance can legitimately fall below `D` after longitudinal separation grows; the true requirement is the pair distance plus swept collision checks, not an unconditional `|z|≥D` for the entire episode. The actual cached simulator reports no collisions for the 665 successes. The CSV keeps all failed, missing-phase, timeout and deadlock categories distinct.

**Existential witness:** state UID `toy_hard_2`, case `2f142928c7c061e81d88`, seed `5`, frozen η=`[1.153149541263192, -0.1898878324600946, 0.39028420608603664]`, TT controller UID above. At step 100 the exact noise key is `[4263627212,2403922648]`; its actual `B` matrix, two projected actions and active rows are in `witness.json`. The fixed-η archived trajectory reaches `χY≥0` at step 252 and `χP≥0` at step 279; signed lateral clearance throughout that interval is at least `0.32200010`, margin `0.00190010` over `D`. It finishes with success at step 438, before 850, without a deadlock candidate. Independent swept recomputation gives minimum wall surface clearance `0.00510003` (positive margin `0.00010003` beyond the evaluator's `0.005`) and swept pair surface clearance `0.00016359` (positive). Its minimum saved action-CBF residual is `−1.4e−15`; maximum speed excess `9.72e−11` is within the frozen `1e−10` solver tolerance. All controller actions, seed stream, goals and initial state are the original cache record. The result establishes fixed-η TT passage for this one state/key and an exact single-step Perp contribution. It does **not** establish that the entire episode would fail with Perp removed.

At that step, with robot rows and columns `(goal, perp, rel)`, the actual matrix is `B_goal=[[0.49988516,0.01071550],[-0.49991239,-0.00935966]]`, `B_perp=[[-0.00795134,0.37093531],[0.00099563,-0.05317810]]`, `B_rel=[[-0.33075101,-0.05005556],[0.33075101,0.05005556]]`. Unrounded values and both projected joint actions are in `witness.json`.

Reproduce the original audit with `reproduce.py`. Reproduce the formal follow-up in order: `formal_geometry_audit.py`, `formal_local_probes.py`, then `formal_geometry_audit.py` to assemble its witness; use `/home/zhihan/venvs/gpu/bin/python`, `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=''`, and `PYTHONDONTWRITEBYTECODE=1`. The GPU-installed JAX package may print a CUDA-plugin initialization warning; control computation uses CPU and no physical simulator step is executed. Remaining gaps are full-episode matched Perp ablation, a guarantee over all qualifying states/noise keys, and dynamic N>2 A5 evidence. A1–A5's literal-definition gap is resolved.
