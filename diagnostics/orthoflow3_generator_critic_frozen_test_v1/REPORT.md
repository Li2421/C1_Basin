# Frozen untouched-test evaluation: joint OrthoFlow3 generator + critic

Generalization: **`GENERALIZATION_PARTIAL`**  
Deformation: **`DEFORMATION_OPTIMIZATION_USEFUL`**  
Pipeline: **`PIPELINE_NEEDS_REVISION`**

## 1. Frozen protocol

The generator and critic were frozen before this evaluation. The proposal set is exactly the transformed generator mean plus four deterministic stochastic samples (K=4). The critic ranked only these five points. `tau_critic=0.9375` was frozen from the critic's empirical-Q semantics before test outcomes. The populations are 192 Double-Bottleneck matched rollout-states (the complete two 96-rollout mature hard-safety populations), 60 Four-Way states, and 60 Ring states. No test state entered training or selection.

Proposal manifest SHA-256: `0db0538659490081413cc181c1ffb37c39b317746b5e2c35aaf088632b4a62ac`.

## 2. Results by scenario

### Double-Bottleneck

Per-scenario decision: **`GENERALIZATION_STRONG`**

| Method | Single-seed success | Robust success | Rescue | Break | Mean Q16 | Mean D_inst | Mean D_traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Hard Safety | 131/192 (68.2%) | 19/192 (9.9%; unresolved 0) | 0/173 | 0/19 | 0.650 | 0.00000 | 0.00000 |
| B1 Fixed eta | 192/192 (100.0%) | 192/192 (100.0%; unresolved 0) | 173/173 | 0/19 | 1.000 | 0.12140 | 0.02068 |
| B2 Generator Mean | 185/192 (96.4%) | 172/192 (89.6%; unresolved 0) | 154/173 | 1/19 | 0.962 | 0.12981 | 0.02523 |
| B3 Random Proposal | 168/192 (87.5%) | 149/192 (77.6%; unresolved 0) | 134/173 | 4/19 | 0.870 | 0.13192 | 0.03022 |
| B4 Oracle finite proposals | 190/192 (99.0%) | 185/192 (96.4%; unresolved 0) | 166/173 | 0/19 | 0.988 | 0.12911 | 0.02390 |
| B5 Critic-selected | 188/192 (97.9%) | 181/192 (94.3%; unresolved 0) | 163/173 | 1/19 | 0.980 | 0.12626 | 0.02324 |
| Min-Def Critic-Feasible | 186/192 (96.9%) | 168/192 (87.5%; unresolved 0) | 153/173 | 4/19 | 0.957 | 0.12246 | 0.02422 |
| Oracle Min-Dinst | 186/192 (96.9%) | 185/192 (96.4%; unresolved 0) | 166/173 | 0/19 | 0.984 | 0.11767 | 0.02141 |
| Oracle Min-Dtraj | 187/192 (97.4%) | 185/192 (96.4%; unresolved 0) | 166/173 | 0/19 | 0.985 | 0.11921 | 0.02091 |

B5 changes the robust-state count by +162 versus B0 and -11 versus frozen B1. It is 4 certified robust states below B4. The critic chose the exact oracle proposal on 39/192 states; mean exact-Q16 gap is 0.0078; oracle-robust/critic-nonrobust cases: 4; oracle-robust/critic-unresolved cases: 0; high-prediction (`>=0.9`) / low-actual (`Q16<0.5`) cases: 1.

Among 19 B0-robust states, B5 robust break is 1 and mean B5 D_inst is 0.11706. Min-Def reduces mean D_inst by 3.0% relative to B5; the proposal-set oracle reduction is 6.8%.

B0 -> B5 canonical conversion matrix: `{"success->success": 128, "success->timeout": 3, "timeout->success": 60, "timeout->timeout": 1}`.

Canonical outcome partitions: `{"B0 Hard Safety": {"success": 131, "timeout": 61}, "B1 Fixed eta": {"success": 192}, "B2 Generator Mean": {"success": 185, "timeout": 7}, "B3 Random Proposal": {"success": 168, "timeout": 24}, "B4 Oracle finite proposals": {"success": 190, "timeout": 2}, "B5 Critic-selected": {"success": 188, "timeout": 4}, "Min-Def Critic-Feasible": {"success": 186, "timeout": 6}, "Oracle Min-Dinst": {"success": 186, "timeout": 6}, "Oracle Min-Dtraj": {"success": 187, "timeout": 5}}`.

### Four-Way Intersection

Per-scenario decision: **`GENERALIZATION_STRONG`**

| Method | Single-seed success | Robust success | Rescue | Break | Mean Q16 | Mean D_inst | Mean D_traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Hard Safety | 39/60 (65.0%) | 2/60 (3.3%; unresolved 0) | 0/58 | 0/2 | 0.642 | 0.00000 | 0.00000 |
| B1 Fixed eta | 59/60 (98.3%) | 59/60 (98.3%; unresolved 1) | 58/58 | 0/2 | 1.000 | 1.21802 | 0.23219 |
| B2 Generator Mean | 59/60 (98.3%) | 57/60 (95.0%; unresolved 3) | 55/58 | 0/2 | 1.000 | 1.39222 | 0.24757 |
| B3 Random Proposal | 57/60 (95.0%) | 58/60 (96.7%; unresolved 2) | 56/58 | 0/2 | 1.000 | 1.33822 | 0.25078 |
| B4 Oracle finite proposals | 59/60 (98.3%) | 60/60 (100.0%; unresolved 0) | 58/58 | 0/2 | 1.000 | 1.35488 | 0.24576 |
| B5 Critic-selected | 59/60 (98.3%) | 57/60 (95.0%; unresolved 3) | 55/58 | 0/2 | 1.000 | 1.40012 | 0.24854 |
| Min-Def Critic-Feasible | 60/60 (100.0%) | 60/60 (100.0%; unresolved 0) | 58/58 | 0/2 | 1.000 | 1.07653 | 0.22105 |
| Oracle Min-Dinst | 60/60 (100.0%) | 60/60 (100.0%; unresolved 0) | 58/58 | 0/2 | 1.000 | 1.07653 | 0.22105 |
| Oracle Min-Dtraj | 60/60 (100.0%) | 60/60 (100.0%; unresolved 0) | 58/58 | 0/2 | 1.000 | 1.10432 | 0.21863 |

B5 changes the robust-state count by +55 versus B0 and -2 versus frozen B1. It is 3 certified robust states below B4. The critic chose the exact oracle proposal on 49/60 states; mean exact-Q16 gap is 0.0000; oracle-robust/critic-nonrobust cases: 0; oracle-robust/critic-unresolved cases: 3; high-prediction (`>=0.9`) / low-actual (`Q16<0.5`) cases: 0.

Among 2 B0-robust states, B5 robust break is 0 and mean B5 D_inst is 1.39458. Min-Def reduces mean D_inst by 23.1% relative to B5; the proposal-set oracle reduction is 23.1%.

B0 -> B5 canonical conversion matrix: `{"success->numerical_failure": 1, "success->success": 38, "timeout->success": 21}`.

Canonical outcome partitions: `{"B0 Hard Safety": {"success": 39, "timeout": 21}, "B1 Fixed eta": {"numerical_failure": 1, "success": 59}, "B2 Generator Mean": {"numerical_failure": 1, "success": 59}, "B3 Random Proposal": {"numerical_failure": 3, "success": 57}, "B4 Oracle finite proposals": {"numerical_failure": 1, "success": 59}, "B5 Critic-selected": {"numerical_failure": 1, "success": 59}, "Min-Def Critic-Feasible": {"success": 60}, "Oracle Min-Dinst": {"success": 60}, "Oracle Min-Dtraj": {"success": 60}}`.

### Ring Exchange

Per-scenario decision: **`GENERALIZATION_PARTIAL`**

| Method | Single-seed success | Robust success | Rescue | Break | Mean Q16 | Mean D_inst | Mean D_traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Hard Safety | 47/60 (78.3%) | 37/60 (61.7%; unresolved 0) | 0/23 | 0/37 | 0.768 | 0.00000 | 0.00000 |
| B1 Fixed eta | 58/60 (96.7%) | 58/60 (96.7%; unresolved 0) | 22/23 | 1/37 | 0.982 | 0.01610 | 0.50706 |
| B2 Generator Mean | 20/60 (33.3%) | 2/60 (3.3%; unresolved 0) | 0/23 | 35/37 | 0.325 | 0.01391 | 0.19689 |
| B3 Random Proposal | 35/60 (58.3%) | 28/60 (46.7%; unresolved 0) | 10/23 | 19/37 | 0.584 | 0.01453 | 0.27272 |
| B4 Oracle finite proposals | 53/60 (88.3%) | 47/60 (78.3%; unresolved 0) | 14/23 | 4/37 | 0.934 | 0.01495 | 0.33939 |
| B5 Critic-selected | 51/60 (85.0%) | 40/60 (66.7%; unresolved 0) | 12/23 | 9/37 | 0.852 | 0.01516 | 0.35410 |
| Min-Def Critic-Feasible | 42/60 (70.0%) | 31/60 (51.7%; unresolved 0) | 8/23 | 14/37 | 0.720 | 0.01046 | 0.19261 |
| Oracle Min-Dinst | 57/60 (95.0%) | 51/60 (85.0%; unresolved 0) | 14/23 | 0/37 | 0.907 | 0.01176 | 0.25636 |
| Oracle Min-Dtraj | 57/60 (95.0%) | 51/60 (85.0%; unresolved 0) | 14/23 | 0/37 | 0.904 | 0.01180 | 0.25059 |

B5 changes the robust-state count by +3 versus B0 and -18 versus frozen B1. It is 7 certified robust states below B4. The critic chose the exact oracle proposal on 30/60 states; mean exact-Q16 gap is 0.0823; oracle-robust/critic-nonrobust cases: 7; oracle-robust/critic-unresolved cases: 0; high-prediction (`>=0.9`) / low-actual (`Q16<0.5`) cases: 4.

Among 37 B0-robust states, B5 robust break is 9 and mean B5 D_inst is 0.01477. Min-Def reduces mean D_inst by 31.0% relative to B5; the proposal-set oracle reduction is 22.4%.

B0 -> B5 canonical conversion matrix: `{"success->success": 40, "success->timeout": 7, "timeout->success": 11, "timeout->timeout": 2}`.

Canonical outcome partitions: `{"B0 Hard Safety": {"success": 47, "timeout": 13}, "B1 Fixed eta": {"numerical_failure": 1, "success": 58, "timeout": 1}, "B2 Generator Mean": {"success": 20, "timeout": 40}, "B3 Random Proposal": {"success": 35, "timeout": 25}, "B4 Oracle finite proposals": {"success": 53, "timeout": 7}, "B5 Critic-selected": {"success": 51, "timeout": 9}, "Min-Def Critic-Feasible": {"success": 42, "timeout": 18}, "Oracle Min-Dinst": {"success": 57, "timeout": 3}, "Oracle Min-Dtraj": {"success": 57, "timeout": 3}}`.

## 3. Generator and critic attribution

B2 isolates the transformed generator mean; B3 isolates one pre-registered stochastic draw; B4 is the true-Q oracle over exactly the same five frozen proposals used by B5. Therefore B4-B2 is the generator distribution gain and B4-B5 is the out-of-sample critic ranking gap. No oracle center, eta search, resampling-until-success, or continuous critic optimization was used.

## 4. Zero-sufficient states and gating interpretation
- Double-Bottleneck: correction clearly useful 163/192; correction unnecessary 18/192; ambiguous 11/192.
- Four-Way Intersection: correction clearly useful 55/60; correction unnecessary 2/60; ambiguous 3/60.
- Ring Exchange: correction clearly useful 12/60; correction unnecessary 28/60; ambiguous 20/60.

This is post-hoc interpretation only; no gate was trained.

## 5. Robustness versus deformation

`D_inst` is the squared norm of the actual post-projection instantaneous change from hard-safety action. `D_traj` is the per-rollout time-average of that squared change; integrated correction energy is retained in the per-state evidence. B5 is ROBUST_MAX. Min-Def uses the frozen critic threshold and the same five proposals, falling back to eta=0 when none is feasible. Both oracle Min-Def variants use true Q16 only for analysis.

- Double-Bottleneck: 112 states contain a truly robust lower-D_inst proposal than B5; 89 contain a truly robust lower-D_traj proposal.
- Four-Way Intersection: 57 states contain a truly robust lower-D_inst proposal than B5; 55 contain a truly robust lower-D_traj proposal.
- Ring Exchange: 25 states contain a truly robust lower-D_inst proposal than B5; 28 contain a truly robust lower-D_traj proposal.

Secondary intervention diagnostics (means over states; projection activity is unavailable from the legacy Double-Bottleneck adapter):

| Scenario | Method | Eta norm | Projection activity | Completion steps |
|---|---|---:|---:|---:|
| Double-Bottleneck | B0 Hard Safety | 0.00000 | N/A | 838.98275 |
| Double-Bottleneck | B5 Critic-selected | 0.68045 | N/A | 720.61458 |
| Double-Bottleneck | Min-Def Critic-Feasible | 0.65522 | N/A | 723.67188 |
| Double-Bottleneck | Oracle Min-Dinst | 0.63398 | N/A | 721.93978 |
| Double-Bottleneck | Oracle Min-Dtraj | 0.64381 | N/A | 720.77018 |
| Four-Way Intersection | B0 Hard Safety | 0.00000 | 0.00000 | 925.54062 |
| Four-Way Intersection | B5 Critic-selected | 0.86114 | 0.98418 | 230.13646 |
| Four-Way Intersection | Min-Def Critic-Feasible | 0.76417 | 0.97270 | 243.56771 |
| Four-Way Intersection | Oracle Min-Dinst | 0.76417 | 0.97270 | 243.56771 |
| Four-Way Intersection | Oracle Min-Dtraj | 0.76866 | 0.97378 | 243.41875 |
| Ring Exchange | B0 Hard Safety | 0.00000 | 0.00000 | 435.84375 |
| Ring Exchange | B5 Critic-selected | 0.88460 | 0.91339 | 487.61354 |
| Ring Exchange | Min-Def Critic-Feasible | 0.62923 | 0.67151 | 496.07187 |
| Ring Exchange | Oracle Min-Dinst | 0.66747 | 0.70090 | 439.95521 |
| Ring Exchange | Oracle Min-Dtraj | 0.64779 | 0.69053 | 444.59688 |

## 6. Safety and numerical audit

All nonzero eta continuations retained the mandatory second hard-safety projection. Across 37,128 unique database records, 105 remained `NUMERICAL_SOLVER_FAILURE` after the fixed retry rule and 50 were collision outcomes. These were not folded into timeout or non-robust labels. Double-Bottleneck had no collision/numerical record; Four-Way had 81 numerical and no collision records; Ring had 24 numerical and 50 collision records across all evaluated controller/eta tuples. Importantly, Ring B5 itself had 0 collision seeds, while Ring B0 had 14/960 collision seeds. Thus the learned B5 path is collision-free in this sample, but the blanket claim that every hard-safety rollout was collision-free is false. See `numerical_safety_audit.json`.

## 7. Cache and resumability

Initial exact preflight requested 37,128 seed rollouts, reused 0, and identified 37,128 missing. Final postflight lists 105 non-reusable tuples: these are exactly the unresolved numerical records, not unexecuted work. Every seed-level result was committed immediately to the shared database.

## 8. Explicit answers

1. **Generator generalization is scenario-dependent.** B4 shows useful proposal coverage on Double (185/192) and Four-Way (60/60), but only 47/60 on Ring; Ring B2 mean collapses to 2/60 robust.
2. **Critic generalization is strong on Double/Four but incomplete on Ring.** Ring has seven oracle-robust/critic-nonrobust misselections and a mean exact-Q16 gap of 0.0823; this is genuine misranking, not continuous critic exploitation.
3. **B5 versus B0 robust gain:** Double +162 states (181 vs 19), Four-Way +55 certified states (57 vs 2, with 3 B5 numerical-unresolved), Ring +3 states (40 vs 37).
4. **B5 recovery of B4:** Double 181/185 certified robust states; Four 57/60 certified plus 3 unresolved; Ring 40/47.
5. **Robust rescue:** Double 163/173 B0-nonrobust, Four 55/58, Ring 12/23.
6. **Robust break:** Double 1/19 B0-robust, Four 0/2, Ring 9/37.
7. **Fixed shared eta is at least as good here.** B1 robust counts are 192, 59+1 unresolved, and 58 for Double/Four/Ring, respectively; it exceeds B5 in all three frozen populations.
8. **Some zero-sufficient states are unnecessarily modified.** The most consequential case is Ring: B5 changes all B0-robust states with nonzero mean deformation and breaks 9/37 robust states.
9. **Lower deformation is available, but the deployable threshold selector is not uniformly safe.** Four-Way Min-Def improves certification to 60/60 while reducing D_inst 23.1%; Double and Ring Min-Def lose robust states. Oracle proposal evidence shows lower-deformation robust alternatives remain available.
10. **A future deformation-aware/gating objective is justified**, especially for Ring, but it must also repair proposal/ranking generalization; deformation refinement alone is insufficient.

## 9. Final decisions

- Generalization: `GENERALIZATION_PARTIAL`
- Deformation: `DEFORMATION_OPTIMIZATION_USEFUL`
- Pipeline: `PIPELINE_NEEDS_REVISION`

No model was retrained; no training labels were generated; no test-driven tuning was performed.
