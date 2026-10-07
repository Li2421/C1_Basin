# Source-controller state–eta contrast check

**Verdict:** With the present H20 response representation and source evidence, the critic learns controller-level eta preference but **does not reliably learn state-specific eta ranking**. This is a source-family validation result, not a new strict LOSO success claim. Information sufficiency of the full physical state/controller input remains underresolved.

## Frozen design and database integrity

- Existing 46 source TRAIN families; 16 new, outcome-blind, independent source-family validation states. No overlap with the held-controller or Ring LOSO target UIDs/source groups.
- The same two exact eta probes (indices 10 and 15), chosen using old source-TRAIN Q4 evidence only, were evaluated under the same two frozen source Flow controllers (`alt`, `second`). Q16 was requested for every state × eta × controller cell; success/safety semantics were unchanged.
- Global cache preflight: 3,968 requested continuation keys, 736 partially reusable existing seeds, 3,232 genuinely missing. Those 3,232 were attempted through append-only journals and merged. Four numerical failures remain explicitly recorded, not imputed. Postflight/alignment: 3,964 valid records, four numerical failures, zero unattempted, ambiguous, or collision cases. See `alignment_audit.json` and `cache_postflight.json`.
- All neural models used source FIT families for training and source inner VAL for checkpoint selection. The independent 16-family outcome panel was opened only after checkpoints were frozen. Models were evaluated for seeds 17, 23, and 41.

## Independent validation

The 16 states × two controllers yield 32 controller-state decisions. At least one eta is B15 in 21/32; exactly one is B15 in 16/32. Twenty cases have a true Q contrast of at least 0.25. Three states reverse eta preference when the controller changes; within-controller strong-preference state reversals are present, especially under `second` (six states favor each eta).

| Model | B15 / 32 (three seeds) | B15 / 16 discriminative | Q16 NLL |
| --- | --- | --- | --- |
| eta-only | 18, 18, 18 | 13, 13, 13 | 0.605–0.606 |
| state/controller additive | 18, 18, 18 | 13, 13, 13 | 0.609–0.614 |
| H20 context only | 17, 18, 18 | 12, 13, 13 | 0.621–0.629 |
| full state + H20 context, Q16 training | 17, 17, 18 | 12, 12, 13 | 0.615–0.626 |
| full state + H20 context, original Q4 training | 18, 17, 18 | 13, 12, 13 | 0.593–0.607 |

The Q16 upgrade did not improve B15 selection relative to Q4; the Q4 model even has lower NLL in this particular panel. This does **not** imply Q4 labels are generally preferable. The `full_Q4_weight16` run is mathematically identical to `full_old_Q4` here because all selected source pairs were scaled by the same count; it is not an informative weighting ablation.

Correct versus wrong-*controller* context changes predictions substantially (full-model NLL rises from 0.615–0.626 to 0.840–0.953 with wrong-controller context). But shuffling explicit state input changes NLL by less than 0.00005 and never changes B15 selection. Replacing context with another state's context **under the same controller** has mixed NLL effects and changes discriminative selection by at most one or two cases. Thus the learned context effect is predominantly controller identity/preference, not a reliable state-specific physical response.

The source-family, post-hoc 3-fold crossfit diagnostic uses 10 nominal H20 summaries and 14 eta-response differences. Controller-specific ridge and kNN do not consistently improve Q10−Q15 prediction over a constant controller preference (pooled MAE: constant 0.247; best ridge 0.244; best kNN 0.253). This makes a *neural-only fitting defect* less likely, but cannot establish that the representation contains no useful information.

## Interpretation and next discriminating test

There is genuine state- and controller-dependent Q ranking in the labels, but robust B15 preference for eta15 is rare: on the 46 TRAIN states it is uniquely B15 in only two `alt` and one `second` cases; on the 16 independent states, one `alt` and two `second` cases. Also, under `second`, eta15 has higher mean Q while eta10 has more B15 states. A pure Q likelihood can therefore improve Q probability without learning the rare B15 preference changes needed for selection. The data support **insufficient effective state-dependent supervision and possible Q-versus-B15 objective mismatch**; they do not yet isolate representation insufficiency versus data sparsity.

The smallest next test is to freeze a *new*, outcome-blind source-family panel with a candidate set selected only from source TRAIN evidence to provide more balanced robust-preference reversals, then compare the same frozen models and a source-only state/context predictor. Do not tune on the present independent validation panel or change the generator. A second diagnostic can seek repeated controller interventions on identical physical states to separate controller identity from response variation. No strict LOSO improvement is claimed here.

## Follow-up clarification (2026-10-04)

The subsequent `orthoflow3_state_context_learning_audit_v1` audit qualifies the interpretation above without changing these frozen results. Under conditionally IID Bernoulli success probability p, the B15 probability is p^15(16−15p), increasing in p. A difference between aggregate mean-Q preference and observed B15 prevalence does not by itself demonstrate a defective probability loss: unresolved state heterogeneity and finite-seed uncertainty can produce it. NLL-versus-ranking objective mismatch is therefore not established as a root cause.

The follow-up also found that this two-eta experiment excluded the other 14 historical source eta from training; it is a focused contrast diagnostic rather than a full-data model upgrade. The current authoritative-DB rebuild retains all 16 eta and includes 1,098 additional valid seed outcomes absent from the older wide-data snapshot. Neither restored data nor tested scaling/regularization establishes a robust-selection improvement over the strongest eta-only control. Context may carry state information, so an explicit-h shuffle alone is not evidence that all state dependence is absent. See the follow-up report for joint h/context shuffle, family crossfit, and unresolved numerical-outcome bounds.

Reproduction: `pipeline.py`, `materialize.py`, `train_models.py`, `summarize.py`, `context_swap_diagnostic.py`, `local_signal_diagnostic.py`; all manifests, checkpoints, predictions, and CSV metrics are in this directory.
