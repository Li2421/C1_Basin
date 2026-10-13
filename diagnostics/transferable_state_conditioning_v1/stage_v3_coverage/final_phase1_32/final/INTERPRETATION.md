# Physical-only Q(s,η): completed phase-1 confirmation

**Verdict: `CROSS_SCENE_STATE_CONDITIONING_NOT_ESTABLISHED` for the tested
critic and frozen FF pipeline.** Correct physical state has clear causal
selection value in Toy within-scene, but none of the four source-only LOSO
critics shows a reliable same-scene matched-state advantage on independent
held-out states. This does not prove that no future representation or larger,
more diverse source dataset could transfer.

## Protocol and reuse

The complete shared rollout database was audited before simulation. Source
training reused exact-compatible Q4 labels, with 256 Toy, 112 Ring, 48 DB
and 48 Four-Way independent TRAIN states; the eta-only control used the same
labels and sample weighting as the state critic. Concat and eta-prior plus
ranking-residual objectives were fitted; source-validation selection regret,
pairwise ranking and mismatch performance selected checkpoints without
held-out-scene labels. The critic received only physical positions,
velocities, goals, fixed map geometry and 3-D η. Flow, H20, controller
response and scene ID were absent. Robot-permutation and rigid-transform
invariance checks passed.

The confirmation tranche consists of the first 32 eligible, previously
unopened global-distribution TEST physical states in each scene, frozen in
sampler order before any Q16 outcome. Each state has the same frozen
stochastic 16 η candidates and Safety control, plus a separate fixed common
16-η bank. Each state–η has 16 prespecified rollout seeds; the common bank
uses independent discovery seeds 0–7 and evaluation seeds 8–15. The exact
cache preflight found no compatible TEST seed to reuse, so all 67,584
requested seeds were generated and retained. All 528 shard receipts and the
eight per-seed outcome panels completed. Physical state, not rollout, is the
paired-bootstrap unit. Candidate proposals, FF execution, Flow, controller,
safety projection and success evaluator remained frozen.

This is **critic-only LOSO conditional on historical scene-trained proposal
generators**, not end-to-end zero-shot deployment of the whole pipeline.

## Frozen stochastic K16: robust B15 successes / 32 states

| Scene | Within Safety | Within Q(η) | Within Q(s,η) | Within Oracle | LOSO Safety | LOSO Q(η) | LOSO Q(s,η) | LOSO Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Toy | 20 | 27 | **30** | 31 | 20 | 22 | **17** | 31 |
| DB | 4 | 32 | 32 | 32 | 4 | 29 | **31** | 32 |
| Four-Way | 0 | 30 | 30 | 32 | 0 | 31 | **28** | 32 |
| Ring | 22 | 29 | 30 | 32 | 22 | 5 | **6** | 32 |

The decisive causal control is the same-scene state derangement on **the
same physical states, candidate pools, seeds and outcome table**. The
matched-minus-deranged mean rollout-success differences (state-bootstrap
95% CIs) are:

| Scene | Within-scene | LOSO |
|---|---:|---:|
| Toy | **+12.36 pp** [+6.87,+18.62] | **0.00 pp** [0.00,0.00] |
| DB | 0.00 pp [0.00,0.00] | +0.07 pp [−0.27,+0.61] |
| Four-Way | 0.00 pp [0.00,0.00] | −0.13 pp [−0.40,0.00] |
| Ring | +0.10 pp [−0.67,+0.80] | +1.08 pp [−3.80,+5.33] |

Equal-scene-weighted LOSO matched-minus-deranged mean success is +0.25 pp,
95% CI [−0.94,+1.34] across 128 independent states. Its mean-success gain
over Safety is +3.4 pp, CI [−3.3,+10.1], but this aggregate masks a severe
Ring regression: LOSO Ring Q(s,η) loses 51.2 pp of mean success versus Safety
(CI [−66.4,−35.7]). DB/Four-Way gain over Safety comes primarily from
intervention availability and a favorable fixed η, not causal state use.

## Fixed-η and reversal falsification

On all eligible untouched TEST physical-state manifests (not just the 32
Q16 states), every LOSO critic selected a **single η index on the common
fixed bank**, unchanged by 100 same-scene state derangements. For the 32-state
Q16 tranche the indices are Toy 15, DB 15, Four-Way 10 and Ring 10. Hence
their matched-versus-mismatched success difference on that bank is exactly
zero for every possible rollout outcome, and their policy is itself one of
the 16 fixed-η policies. This rules out a common-bank adaptive decision
advantage without relying on a small-sample p-value.

The ground truth nevertheless has 47 strong reversal η pairs in Toy and
eight in Ring, defined on discovery seeds and checked on independent
evaluation seeds. The Toy within-scene critic predicts 46 reversal pairs,
attains 95.3% pair-state switch accuracy versus 51.5% for eta-only, and
correct state beats derangement by **46.5 pp** on the evaluation half
(CI [+39.0,+52.3]). Its common-bank selected success is 96.1% versus 90.2%
for the posthoc best fixed η, but that +5.9 pp fixed-policy difference has a
CI crossing zero with only 32 states. The Toy LOSO critic predicts **zero**
reversal pairs, has no matched-state benefit and scores 44.5% versus the
90.2% best fixed η. Ring LOSO scores 0.4% versus an 82.4% best fixed η,
also with zero selection switching. DB and Four-Way have no strong reversal
pairs in this tranche and a ceiling fixed η reaches 100%, so top-1 state
adaptation is not identifiable there; neither produces a causal mismatch
gain.

## Diagnosis and boundaries

The representation and training code are not simply ignoring state: Toy
within-scene state-effect correlation is 0.707 on fresh states, and correct
state decisively improves rollout selection. The failure arises in transfer
of the **state×η interaction**: LOSO state-effect correlations on the common
bank are −0.066 (Toy), 0.000 (DB), 0.024 (Four-Way) and 0.023 (Ring). Source
validation can reward a scene/global η prior while held-out within-scene
ranking fails. DB/Four-Way source labels are close to a top-1 ceiling; Toy
held-out η proposals have marked source-support shift; the broad-η Ring
source ablation and explicit eta-prior residual changed global preferences
without establishing causal transfer. These are supported contributing
mechanisms, not a uniquely identified single root cause.

No selected method had a collision in the confirmation tranche. Selected
numerical failures were zero in Toy/DB; Four-Way had nine within-scene
eta-only and state failures, eight LOSO eta-only and 16 LOSO state failures;
Ring had one within eta-only, zero within state, two LOSO eta-only and four
LOSO state failures. The B15 counts are confirmed successes, with numerical
failures treated conservatively. The full all-candidate outcome tables,
failure cases and upper/lower bounds are in the machine-readable artifacts.

Earlier 512-state Ring FF evidence remains valid for its own matched
within-scene execution profile. The present formal four-scene Ring source
coverage is only 112 states and uses its frozen LOSO FF profile, so these
results do not negate that earlier finding. The scientifically defensible
paper claim is: *physical state can improve η selection within a scene
(clearly Toy and previously Ring under its matched profile), but the tested
physical-only critic has not learned transferable state-conditioned η
selection across these four scenes*. A larger or more invariant source
training scheme would need a new untouched confirmation set before a
positive cross-scene claim.

The autogenerated [FINAL_REPORT.md](./FINAL_REPORT.md) contains every
within/LOSO table and interval; [consolidated_results.json](./consolidated_results.json)
contains per-panel metrics. `final/{proposal,anchor}/*/outcomes.npz` and the
shared rollout database retain per-state/per-η/per-seed outcomes; `final/`
also contains cache preflights, cases, receipts and safety issue lists.
