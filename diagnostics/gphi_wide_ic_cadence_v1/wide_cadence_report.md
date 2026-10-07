# Frozen WIDE-IC fixed-cadence G_phi efficacy

Final classification: **NONMONOTONIC_CADENCE_TRADEOFF**.

No G_phi retraining, gate, online eta search, success-basin search, or online oracle rollout was used. Every condition was freshly evaluated with matched IC and Flow randomness; corrections act for one physical timestep only.

## Frozen benchmark and integrity

Authoritative asset: `/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/planning/wide_initial_states_200.npz` (SHA256 `30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb`), exact 200-row test cohort. Flow protocol: base seed 42, episode key `fold_in(PRNGKey(42), rollout_id)`, step key `fold_in(episode_key, t)`. Integrity audit: **PASS**; analysis sanity: **PASS**.

IC generation: test seed 2026090902, independent per-agent |x| uniform [0.55, 1.05], y uniform [-0.025, 0.025], no rejection=True; observed ranges |x|=[0.5533372759819031, 1.0497655868530273], y=[-0.024841023609042168, 0.024846605956554413]. Horizon=850 steps at dt=0.05 s.

G_phi source-episode overlap is train/validation/test/unseen = 107/23/23/47. Thus the all-episode result is not an unseen-generalization estimate; partition-stratified rows are included in both success and rescue/break CSVs. FlowBC exact-IC overlap: 0/200.

## Primary outcomes

| condition | success | deadlock | timeout | collision | wall collision | other | Q (Wilson 95% CI) | mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 138 | 11 | 51 | 0 | 0 | 0 | 0.6900 [0.6228, 0.7500] | 0 |
| H=1 | 148 | 4 | 48 | 0 | 0 | 0 | 0.7400 [0.6751, 0.7959] | 0.141813 |
| H=4 | 182 | 0 | 18 | 0 | 0 | 0 | 0.9100 [0.8622, 0.9423] | 0.0429406 |
| H=8 | 187 | 0 | 13 | 0 | 0 | 0 | 0.9350 [0.8920, 0.9616] | 0.0241573 |
| H=16 | 173 | 0 | 27 | 0 | 0 | 0 | 0.8650 [0.8107, 0.9055] | 0.0137437 |

## G_phi source-overlap strata

| partition | N | Safety Q | H=1 Q | H=4 Q | H=8 Q | H=16 Q | H1 rescue/break | H4 rescue/break | H8 rescue/break | H16 rescue/break |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 107 | 0.7944 | 0.6916 | 0.8692 | 0.9065 | 0.8879 | 10/21 | 13/5 | 13/1 | 11/1 |
| validation | 23 | 0.4783 | 0.9130 | 1.0000 | 1.0000 | 0.7391 | 11/1 | 12/0 | 12/0 | 6/0 |
| test | 23 | 0.4348 | 0.8261 | 0.9130 | 0.9130 | 0.7826 | 11/2 | 12/1 | 12/1 | 9/1 |
| unseen | 47 | 0.6809 | 0.7234 | 0.9574 | 0.9787 | 0.9149 | 13/11 | 14/1 | 14/0 | 11/0 |

## Matched efficacy versus Safety

| H | rescue | rescue rate | break | break rate | rescue-break | Delta Q | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| H=1 | 45 | 0.7258 | 35 | 0.2536 | +10 | +0.0500 | 0.3143 |
| H=4 | 51 | 0.8226 | 7 | 0.05072 | +44 | +0.2200 | 2.402e-09 |
| H=8 | 51 | 0.8226 | 2 | 0.01449 | +49 | +0.2450 | 3.18e-13 |
| H=16 | 37 | 0.5968 | 2 | 0.01449 | +35 | +0.1750 | 2.841e-09 |

## Tradeoff (unweighted)

| H | Q | rescue rate | break rate | mean J_def |
|---|---:|---:|---:|---:|
| Safety | 0.6900 | NA | NA | 0 |
| H=1 | 0.7400 | 0.7258 | 0.2536 | 0.141813 |
| H=4 | 0.9100 | 0.8226 | 0.05072 | 0.0429406 |
| H=8 | 0.9350 | 0.8226 | 0.01449 | 0.0241573 |
| H=16 | 0.8650 | 0.5968 | 0.01449 | 0.0137437 |

## Scientific answers

1. Meaningful wide-IC rescue: yes; H=1 45 rescues/35 breaks, H=4 51 rescues/7 breaks, H=8 51 rescues/2 breaks, H=16 37 rescues/2 breaks.
2. Sparse nominal preservation: H=4 break rate 0.05072; H=8 break rate 0.01449; H=16 break rate 0.01449.
3. Highest Q: H=8 (Q=0.9350). Best net rescue-break: H=8 (+49).
4. Lowest J_def with positive efficacy: H=16 (0.0137437).
5. Q over H=1/4/8/16 is nonmonotonic: 0.7400/0.9100/0.9350/0.8650.
6. Rescued Safety failure modes (deadlock/timeout/collision/wall/other): H=1=0/45/0/0/0; H=4=0/51/0/0/0; H=8=0/51/0/0/0; H=16=0/37/0/0/0.
7. H=16 still has positive efficacy: 37 rescues, 2 breaks, Delta Q=+0.1750.
8. Failure-mode interpretation: every successful rescue came from a Safety timeout. None of the 11 Safety deadlocks became success; at H=4/H=8/H=16 all 11 became timeout. Sparse G_phi resolves many timeout/liveness failures, but only changes the terminal category of strict deadlocks rather than resolving them.
9. Cross-cadence evidence for the intermediate optimum: H=8 beats H=4 on 5 paired episodes with 0 losses (one-sided exact p=0.03125; two-sided p=0.0625) and beats H=16 on 14 with 0 losses (one-sided exact p=6.1035e-05).
10. Hard safety: **INTACT** (collision, projection-failure, invalid-action, and execution-error audit).
11. H=32 should **not yet be tested** under the specified efficacy/sparsity rule.
12. Smallest next experiment: Evaluate fixed H=12 on the identical 200 episodes to localize the observed H=8-to-H=16 efficacy drop; change no other component.

## Runtime/resources

Rollout process records: 4; allocated GPU shards: 4 (scheduler-visible GPU identifiers: 0); CPU allocation/limits: allocation=8 cores, BLAS=1/process; memory allocation: 64 GiB; worker processes: 4; rollout wall envelope: 233.651 s; summed episode runtime: 879.852 s. Pre-launch scheduler/GPU/host observations are preserved in `/home/zhihan/research/Basin_C1/diagnostics/gphi_wide_ic_cadence_v1/resource_audit.json`. Analysis launched zero rollouts.
