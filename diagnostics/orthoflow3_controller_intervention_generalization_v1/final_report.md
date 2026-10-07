# Success Basin C1: controller-intervention generalization

## Decision

**PRINCIPLED_NEGATIVE_RESULT for the current four-scene, fixed-source-support formulation.** No tested deployable (C_{\rm controller}) produced a critic that (i) used context to make reliably better eta choices on independent source families and (ii) stably exceeded eta-only under strict leave-one-scene-out (LOSO). This is not a proof that no richer controller representation or new source geometry could ever transfer.

The mathematical object remains (Q(h,\eta,C_{\rm controller})) and its Success Basin, because the earlier exact controller swap held (h,\eta), first action and seeds constant while 8/64 pairs changed from 16/16 to 0/16. But the present H3/H8/H20/H100 summaries and short function signature do not make that object empirically learnable across these scenes. No minimum sufficient context (C_{\min}) was established.

## Frozen LOSO gate (all methods see identical K16 proposals)

| Held-out scene | Original shared | Source-only eta-only | H8 additive | H8 full | H8 matched-contrast full | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| Toy | 36/48 | 42/48 | 42/48 | 31/48 | 34/48 | 48/48 |
| DB | 13/24 | 14/24 | 24/24 | 23/24 | 13/24 | 24/24 |
| Four-Way | 21/24 + 3 unresolved | 24/24 | 23/24 + 1 unresolved | 23/24 + 1 unresolved | 24/24 | 24/24 |
| Ring | 0/60 | 27/60 | 17/60 | 0/60 | 1/60 | 60/60 |

The additive branch cannot change eta ordering *within a given state* when its context is eta-independent. Its DB 24/24 reflects a different learned global eta score on the fixed candidates, not demonstrated state–eta interaction. Four-Way is essentially saturated. The Ring contrast model still made 59 severe (\hat Q>0.9, Q\le0.5\) top-1 errors. Target context and state shuffles left Ring B15 selection unchanged. These checkpoints were source-selected and frozen before this K16 target outcome was opened. Later experiments in this report are source diagnostics or explicitly post-hoc target adaptation; they do not retroactively alter the LOSO result.

## Intervention supervision and integrity

Three physically compatible Flow conditions were observed at matched source (h,\eta): the canonical future Flow and two different pre-existing frozen checkpoint variants. The initial committed Flow action, eta basis, plant, safety projection, horizon and success rule did not change. Controller identities include the Flow hash; the global cache did not conflate conditions. Base-controller early-stop records remain observed success/failure counts, never fabricated Q16. Four-Way/Ring numerical failures remain unresolved.

The pilot executed 1,152 new continuations; the balanced first variant executed 2,992; the second variant executed 3,072. **Total new continuations: 7,216**, all journaled and merged into `shared_rollout_db/rollout.sqlite`. The second-variant preflight was 3,072 requested / 0 exact / 0 partial / 0 aggregate / 3,072 missing. Its postflight was 3,024 exact / 45 partial / 3 numerical unresolved; no missing *observed* records were silently treated as Q16. The three numerical cases cannot be certified B15 from this evidence.

The second variant created 13 interval-certified canonical→second-controller eta-order reversals in Ring on the same state and same pair of etas. Across controller comparisons, the matched data yielded 45 certified reversal quadruplets in source TRAIN families and 9 in source VAL families. This breaks the *data* version of scene/controller confounding: controller changes can now be supervised at fixed (h,\eta). It did not force the current critic to use context effectively.

## Input use and interaction: source-family validation

H20 eta-conditioned response features were valid for all 253 previously matched pairs. Training the same critic on two conditions did not predict an unseen third Ring controller well: its held-out Ring panel had B15 proposals on 2/4 states, but the H20 nominal critic selected 0–1/4 and the rich critic 0/4. Merely extending the short probe from H8 to H20 changed source VAL controller-flip directions modestly, not deployment ranking reliably.

After training on all three controller conditions, independent source-family Ring B15 totals, aggregated over three seeds and three controller panels, were:

| Model | B15 selected / oracle | Wrong-controller context effect |
|---|---:|---|
| Unrestricted full interaction | 5/24 / 24/24 | 5/24 selected after replacement |
| Correct eta-independent additive (A(h,C(h))+B(\eta)) | 11/24 / 24/24 | 11/24, as structurally expected |
| Bounded interaction residual, logit cap 1 | 6/24 / 24/24 | 7/24 |
| Bounded interaction residual, logit cap 3 | 1/24 / 24/24 | 2/24 |
| Cap 3 + certified controller-eta reversal supervision | 10/24 / 24/24 | 10/24 |

An initial “additive” control accidentally received eta-conditioned response, making it capable of changing eta order. It is retained as an **invalid baseline** and excluded; the table uses the corrected nominal-only input. On the 3 interval-certified held-out Ring reversal cases, unrestricted full got both controller-specific rankings right **0/9** seed-cases; the explicit reversal model got **1/9**. The latter recovered some B15 selection relative to an unregularized residual but did not stably exceed the true additive baseline or make wrong-controller replacement hurt selection. Thus the model is not merely failing because it never saw controller-swapped labels: it sees them, its probability changes, but the learned *eta preference* is still wrong.

A simple local control on the same source TRAIN/VAL evidence reinforced the caution. On Ring's 8 B15-available held-out state×controller groups, same-scene eta-only 1-NN selected 6; eta+H20-context 1-NN selected 5. Exact eta repetition makes this a source diagnostic, not unseen-proposal evidence, but it does not expose a strong missed local context signal.

## Why the candidate context fails

In Ring there are 137 matched-controller comparisons with complete Q16. Thirty differ by at least 0.75 in true Q16. Among those thirty, **24** have normalized H20 rich-response distance below 0.25; distance to the controller response and absolute Q difference have Spearman about **−0.05**. A 100-step, eta-independent nominal physical response (still below 15% of full horizon, and only an information upper bound, not a deployment choice) leaves **23/30** similarly close, with Spearman **−0.12**. A short function-space signature querying raw Flow and safety on an identical common physical trajectory leaves **22/30** close, with Spearman **−0.03**. The analogous Toy controller swap has nine large-Q comparisons, all nine close under the function signature. Numeric separation of some controller conditions does not imply a representation of their *late* feasibility effect.

The Ring held-out H8 controller context also lies outside the three-source support: all 60 frozen Ring states have at least 3 of 10 features outside the source 1–99% envelope, median five. This physical-regime gap and a scene-specific eta prior both matter. Joint four-scene supervision previously achieved 58/60 Ring B15, so the architecture can fit Ring when Ring labels are available; that is capacity evidence, not zero-shot evidence.

## Explicit post-hoc target-label diagnostic, not zero-shot

To separate source-support failure from intrinsic impossibility of Ring prediction, a source-only frozen Ring-fold critic was fine-tuned using outcome-blind hash-selected Ring TRAIN families, with Ring VAL checkpoint selection. The original frozen K16 TEST had already been opened, so these numbers are labeled *post-hoc diagnostics* and cannot support a new strict zero-shot claim.

| Ring TRAIN families | Canonical TRAIN pairs | Fine-tuned state-aware B15 (three seeds) | Ring-supervised eta-only B15 (same families) |
|---:|---:|---:|---:|
| 2 | 519 | 44, 47, 47 /60 | 47, 47, 47 /60 |
| 8 | 2,097 | 44, 44, 44 /60 | 48, 47, 49 /60 |
| 32 | 8,412 | 45, 46, 45 /60 | 50, 48, 54 /60 |
| 64 | 16,865 | 47, 47, 47 /60 | 50, 49, 48 /60 |

Two Ring families already change the transferred critic's source-only **1/60** into roughly **45–47/60**, but eta-only achieves **47/60** with the exact same Ring families. More Ring states improve validation likelihood while state-aware B15 selection largely plateaus. This identifies a substantial **scene-specific eta-preference/support shift**, not a learned transferable state–eta–controller law. It does not imply that state information is useless for every hard state; it means this protocol has not established its added deployment value.

## Answers to the requested questions

1. **Can (Q(h,\eta,C)) currently transfer?** Not credibly under strict fourfold LOSO. The need for controller conditioning is causally established; a sufficient, deployable (C) and a transferable learned interaction are not.
2. **Did controller intervention training solve scene/controller confounding?** It solved the *data-collection confounding* at matched (h,\eta), but the learned critic still mostly responds through controller-wide probability shifts or scene/eta shortcuts; context replacement rarely changes the correct eta decision.
3. **Which inputs are actually used?** Eta preference is used strongly. Physical state and the tested controller summaries can change probabilities in source diagnostics, but neither shows a reliable incremental eta-ranking benefit under independent families and LOSO.
4. **Which inputs can be removed?** H100 nominal and the function signature have no deployment evidence and should not be added. H8/H20 context should not be claimed as the final selector input; no (C_{\min}) can be identified because the success gate was not reached. Keep the physical state representation as a research input rather than interpreting this negative result as proof it is unnecessary.
5. **Does state information change eta preference?** True matched controller evidence contains real ranking reversals, especially Ring. The tested models do not predict them reliably; overall difficulty changes are easier to learn than preference reversals.
6. **Can a structured model preserve eta-only's lower bound and state-aware's potential upper bound?** Bounded residuals were tested. None stably exceeded the corrected additive/eta-only control on independent Ring source families, so this hoped-for combination is not yet validated.
7. **Strict LOSO scores and oracle gap:** The table above is the final clean LOSO gate. No new H20/third-variant model is represented as strict zero-shot after the original target was opened. The Ring gap remains 59/60 for the H8 contrast full critic and 33/60 for source-only eta-only.
8. **Should the generator become (q(\eta\mid h,C_{\min}))?** Not now. The frozen generator already supplies high K16 oracle coverage. The bottleneck is critic selection and source-scene support, and no validated (C_{\min}) exists. No generator architecture or weights were changed.

The most defensible next-generation *hypothesis*, not a validated replacement, is a scenario/controller-adapted eta prior with a constrained state-conditioned residual, evaluated only after adding physically comparable source regimes and a new independent target confirmation cohort. Until then, use the existing frozen generator and do not claim cross-scene zero-shot state–eta feasibility generalization.

## Reproduction and provenance

Entrypoints: `intervention.py`, `balanced_expansion.py`, `second_variant.py`, `rich_intervention_probe.py`, `train_triplet_probe.py`, `triplet_input_audit.py`, `triplet_interaction_audit.py`, `h100_nominal.py`, `controller_signature.py`, and `fewshot_ring.py`. Frozen original LOSO scores are in `loso_results.csv`; per-condition matched outcomes and postflight are in `second_variant/`; source input-use and reversal evidence is in `second_variant/triplet_*`; post-hoc adaptation is in `fewshot_ring/`. Data are indexed by global state, exact eta, controller identity and continuation seed in the global rollout database. No artificial canonical mode IDs, generator retraining, safety-definition change or unrecorded rollout were used.
