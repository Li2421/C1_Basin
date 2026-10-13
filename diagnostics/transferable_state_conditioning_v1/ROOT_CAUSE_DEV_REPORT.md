# Physical-only Q(s,η): source-only development diagnosis

Status: DEVELOPMENT analysis; independent Q16 confirmation is running. These
figures are not the final held-out verdict.

All critics use only deployment-time physical state and 3-D η. The frozen FF
controller, Flow, safety projection, stochastic K16 proposal and evaluator
are unchanged. State and eta-only critics use identical exact-cached Q4
source labels. Held-out-scene labels did not enter model fitting, source
validation, checkpoint selection or normalization. The physical adapter masks
Flow/policy/controller channels and has verified robot-permutation and rigid
transform invariance. It does preserve physical robot/obstacle counts, so a
model may infer a scene prior from geometry without using within-scene state.

## What was tested without new simulation

- Stage v1: 128 Toy, 96 Ring, 48 DB and 48 Four independent TRAIN states.
- Stage v2: add 16 disjoint Ring source states with a frozen broad common η
  bank, using 1,024 exact-cached Q4 seeds.
- Stage v3: 256 Toy and 112 Ring TRAIN states, still exact-cache only. DB and
  Four remain 48 states each because their Q4 source bank is close to ceiling.
- For each source split, fit a head-matched eta-only model and state concat;
  also test an eta-prior-plus-state-residual model with a ranking term. Select
  checkpoints and the state objective on source validation only.
- On each DEVELOPMENT held-out scene, compare 100 same-scene physical-state
  derangements on identical η candidates and identical Q4 rollout labels.

The 16-state source common-bank probe found strong reversal pairs in Toy
(51) and Ring (11), but DB had 0 and Four only 1. A fixed η achieved Q4
success 16/16 states in each of DB and Four, leaving essentially no top-1
headroom there. Source Q4 seed-half preference signs persisted in 98.7% of
Toy and 97.9% of Ring cases, so label noise is not the leading explanation.
The held-out Toy stochastic proposal has substantial η support shift: 60.4%
of target candidate coordinates fall outside source-scene coordinate bounds.

## V3 DEVELOPMENT LOSO, exact cached Q4 mean selected success

| Held-out | matched state | eta-only | same-scene mismatch | common-bank state-effect corr. |
|---|---:|---:|---:|---:|
| Toy | 0.5330 | 0.6788 | 0.5330 | −0.065 |
| DB | 0.9805 | 0.9375 | 0.9871 | 0.000 |
| Four-Way | 0.9727 | 0.9844 | 0.9727 | 0.050 |
| Ring | 0.3482 | 0.2969 | 0.3206 | 0.002 |

The Ring matched-minus-mismatch Q4 difference is +2.77 pp, with a physical-
state bootstrap CI crossing zero. DB's high matched success is lower than
the mismatch control. The source-only residual objective and Ring broad-η
augmentation changed global η preferences but did not produce reliable
within-held-out-scene adaptation. More source-state coverage improved some
target aggregate success yet did not consistently improve causal state use.

## Label-blind untouched TEST falsification

Before opening any independent TEST rollout label, the frozen v3 LOSO
selectors were run on the complete eligible TEST state manifests. On the
same 16 fixed η vectors, every held-out fold selected a single η index for
every state and for every one of 100 same-scene state derangements. Thus its
matched-minus-mismatched decision value on that bank is exactly zero for any
possible outcomes, and it cannot exceed the best fixed η on that bank.

On the stochastic proposal bank, the selection-change fractions under
same-scene derangement were Toy 0.32%, DB 40.92%, Four-Way 3.58%, Ring 9.59%.
These are **upper bounds** on the magnitude of matched-minus-mismatched
mean-success difference (success lies in [0,1]); they do not establish the
sign of any difference. The Q16 confirmation tranche will measure the sign,
Safety, eta-only, oracle, fixed-policy and reversal gaps on untouched states.

The current diagnosis is a source-to-target interaction failure, not a lack
of physical state signal within every scene. Correct state beats state
derangement strongly on Toy source validation and previously on fresh Toy
within-scene states; prior Ring FF 512-state work established within-scene
adaptation under its matched execution profile. In the present four-scene
LOSO pipeline, source-validation selection can favor a scene/global η prior
that does not transfer the state×η effect to a held-out geometry. Distinct
mechanisms, source-label ceiling in DB/Four and Toy target η-support shift
are supported contributing explanations; the data do not isolate one as a
unique root cause.

The prospectively frozen first-32-state-per-scene Q16 tranche and exact
cache preflight are under `stage_v3_coverage/final_phase1_32/`. A positive
or borderline result must be expanded to the remaining untouched states.
