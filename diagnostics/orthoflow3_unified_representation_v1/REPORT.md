# Unified physical representation — final report

UNIFIED_REP_PARTIAL

## Scope and inherited artifacts

Inherited Phase A CRITIC_PARTIAL, Phase B CANONICALIZATION_PARTIAL and CONTRACT_REPAIRED_AND_VALIDATED. R_old is the accepted remaining-time-repaired Phase-B pipeline. No claim that the preceding phases universally succeeded. Audited v2 remains unmodified; all original scientific labels are byte-identical in v3.

## Implementation

One shared physical entity parser/schema, agent encoder, ordered-pair message encoder and obstacle-set encoder replace the learned scenario adapters. Masked mean/max aggregation supports variable N/M. The scene embedding is 128D. Generator and critic have independently trained shared encoders with the same schema. No scene one-hot, arbitrary agent slot or obstacle identity is a neural feature. See REPRESENTATION_SPEC.md, ENTITY_SCHEMA.md and FRAME_CONVENTIONS.md.

The accepted proposal convention remains 16 stochastic proposals plus its existing deterministic location candidate; the location is not a standalone deployed selector. No eta=0 or fixed anchor is added to that set. Eta=0 is evaluated solely as B0. No continuous critic optimization. OrthoFlow3, safety, MACFlow, eta domain and horizon are unchanged.

## Controlled learning comparison

Two matched training seeds (17,23), from scratch, same accepted Gaussian generator family/objective, BCE-Q critic objective, optimizer budget and state/scenario-balanced sampling. Same v2 and existing full-Q proposal-aligned evidence as R_old; no new training labels. Selected checkpoints use validation loss only. The representation architecture changes parameter counts (generator 51,686→57,542; critic 55,585→61,441), so this is not a parameter-count-matched claim. Closed-loop comparison uses the selected checkpoint, not two-seed confidence intervals. Details: old_vs_unified_ablation.json and frozen/training manifests.

| Scenario | Dev old oracle/selected | Dev unified oracle/selected | Unified rescue / break |
|---|---:|---:|---:|
| double_bottleneck | 6/6, 6/6 | 6/6, 6/6 | 6 / 0 |
| four_way_intersection | 6/6, 6/6 | 6/6, 6/6 | 5 / 0 |
| ring_exchange | 6/6, 6/6 | 6/6, 6/6 | 1 / 0 |

## Fresh confirmation, frozen before outcomes

| Scenario | Rotation | States | Old selected | Unified oracle | Unified selected | Selected unresolved | B0 | Rescue | Break | Exploitation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| double_bottleneck | 0 | 3 | 3 | 3 | 3 | 0 | 0 | 3 | 0 | 0 |
| four_way_intersection | 0 | 2 | 2 | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| four_way_intersection | 180 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| four_way_intersection | 270 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| four_way_intersection | 90 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| ring_exchange | 0 | 2 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| ring_exchange | 180 | 2 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| ring_exchange | 270 | 2 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| ring_exchange | 90 | 2 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 0 |

All counts are empirical B15, not proof of population success probability one. Across all requested confirmation candidate/seed slots: collisions=0, unresolved numerical slots=9. Numerical uncertainty is not an ordinary failure; Q bounds and per-state outcomes are preserved in confirmation/results.json. This is a deliberately small confirmation, not a broad generalization claim. No adaptation followed it.

Ring original orientation has a real selection gap: oracle 2/2, selected 1/2, versus old selected 0/2. At 270 degrees one selected tuple is numerically unresolved, not a certified failure. All Ring rotations have exactly identical frozen eta proposals and selected indices. On common valid seeds, selected outcomes agree; seed-0 back-rotated trajectory RMSE is at most 2.93e-6. Thus representation consistency improved without demonstrating complete critic generalization. Four active rotations have 0/2 oracle and selected at each nonzero angle, with six high-score/low-Q flags; these are not solved by an invariant downstream encoder.

## Representation and learned-model metamorphic checks

144 structural tests passed; max embedding error 1.19e-07; mixed padded-batch error 1.34e-07. N=2/3/4/5 and M=0/1/7/20 forward passes passed. These demonstrate architecture capability, not unseen-count policy success. 96 learned-model passive-transform tests: maximum eta discrepancy 2.07e-07, score discrepancy 8.94e-07, top-1 consistency 1.

Active scene transformations are different: Four frozen MACFlow is orientation/slot-sensitive. Policy-frame relations are retained rather than deleted to manufacture invariance. Active Four 90° proposal error decreased .0712→.0404, critic top-1 .75→1 in the targeted cohort, but cyclic-relabel and 270° proposal deviations did not uniformly improve. Ring rotation discrepancy is negligible and cyclic-relabel deviation decreased. See active_metamorphic.json; selected trajectory back-rotation, common-prefix RMSE and outcome agreement are in closed_loop_metamorphic.json.

## Aliasing and shortcut checks

No exact harmful embedding alias was observed in the 245-state corpus. Nearby-state comparisons are restricted to shared eta evidence; no global basin-smoothness claim. The train-only scenario probe reaches 100% validation classification because geometries are physically distinct; bookkeeping-only changes alter no features. This does not imply a hidden scene one-hot. See ALIASING_AUDIT.md and scenario_probe.json.

## Required answers

1. Fixed agent slots in unified h: no; associated physical records are processed by shared weights.
2. Agent list order: invariant within numerical tolerance. Actively reindexing the frozen MACFlow API is not the same operation.
3. Obstacle list order: invariant.
4. Passive global translation: invariant when geometry and policy reference origin transform too.
5. Contract-valid passive/global rotations: invariant; active Four rotations with unchanged upstream policy are not certified equivalences.
6. Vector/action fields: consistently in per-agent goal frames; no Ring world-frame suffix.
7. scene_id in neural conditioning: no. Scenario names remain parser/logging/randomness bookkeeping, not learned routing.
8. Variable N/M: supported structurally without changing model shapes.
9. Exact harmful h alias: none observed, not a universal proof.
10. Near neighbors: substantial common-eta overlap, with finite-evidence limitations documented.
11. Generator oracle coverage: no degradation on matched development; fresh results above.
12. Critic ranking: no matched-development B15 loss or exploitation; fresh results above.
13. Four representation-only rotation/relabel dependence: eliminated to tolerance; active upstream bias remains.
14. Ring mixed-frame shortcut: removed.
15. Suitable foundation for future geometry work: yes structurally, but not authorization or evidence for unrestricted environment-family generalization.

## Provenance, persistence and stop

No rollout was repeated merely to reencode coordinates. Full snapshot identity, not h, keys physical evidence. DB preflight and incremental persistence are recorded in db_cache_report.json. Source-v2 labels/state hashes, trained model hashes and canonical controller artifacts were verified. The only change versus the older Phase-A baseline is the separately accepted, pre-existing FK/raw-import helper fix. Relevant basis regressions, structural tests and syntax checks passed.

Final qualification: the unified representation is implemented and competitive within the tested original-chart scope, but cannot erase a causal frozen-policy orientation dependence. No environment-family experiment or post-confirmation retraining was started.
