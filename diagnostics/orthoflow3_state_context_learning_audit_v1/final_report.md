# State/context learning audit and repairs

The added controller context is used and helps describe controller-specific success probabilities. It has not produced a reliable state-specific B15 selection advantage over a strong eta-only policy. This is not a proof that state is uninformative, that NLL is intrinsically unsuitable, or that cross-scene generalization is impossible.

## Evidence and scope

This follow-up uses 46 existing source families, the same two source controllers, and the frozen eta10/eta15 contrast for evaluation. Three group folds each separate FIT, six inner-validation families, and outer source families; three training seeds are reported. The previously opened independent 16-family panel and all held-controller/LOSO targets are excluded from model choice and retraining. Because the two eta were originally selected using these source labels, these results are **post-hoc source diagnostics**, not a fresh generalization confirmation or strict LOSO result.

There are 92 outer-fold controller-state decisions in total, 47 with a certified B15 proposal and one additional unresolved oracle case. A global eta10 choice already achieves 44 certified B15 cases, so only three certified source decisions can be rescued by state adaptation. The unresolved case is 14 successes, one failure, and one numerical failure for eta10; it remains unresolved rather than a fabricated non-B15 label.

| Model / training data | NLL, mean over seeds | Certified B15, seeds 17/23/41 | Strong Q orderings correct / 45 |
| --- | ---: | --- | --- |
| Narrow eta-only | 0.650 | 44 / 44 / 44, each +1 unresolved | 30 / 30 / 30 |
| Same full neural critic, finer checkpoint cadence | 0.623 | 43 / 43 / 42 | 32 / 35 / 37 |
| Full critic with entity scaling | 0.744 | 43 / 43 / 42 | 31 / 32 / 31 |
| Full critic with entity + context scaling | 0.692 | 43 / 42 / 44 | 32 / 31 / 29 |
| Full raw-input critic, complete current DB, all 16 eta | 0.645 | 43 / 43 / 41 | 35 / 32 / 32 |
| Full scaled critic, complete current DB, all 16 eta | 0.680 | 42 / 35 / 42 | 33 / 27 / 30 |

All rows use the identical two-eta outer evaluation pool. These counts are not comparable numerically to the previous 32-case independent validation or 200-state Toy K16 test. The last two rows retain source trials for every compatible eta; additional eta are training evidence only in this diagnostic.

The NLL advantage of the narrow full model over eta-only is 0.021–0.029 across seeds. Family-bootstrap 95% intervals include zero in every seed (approximately −0.02 to +0.08). Paired B15 rescue/break is 1/2, 0/1, 1/3. Thus even the probability gain should be described as a trend, not established large-sample significance. Complete CSVs include all variants and checkpoints rather than a selected best seed.

## Why meaningful C does not yet imply better B15 selection

1. **Controller information and state-specific ranking are different signals.** Correct context versus the wrong controller raises narrow full-model NLL from 0.623 to 0.918 and reduces certified B15 from a mean 42.7 to 30.3. Explicit-h shuffle leaves results nearly unchanged. Same-controller wrong-state context raises NLL to 0.657 but yields mixed Q-ordering/regret changes. With all compatible data, correct versus wrong-controller NLL is 0.645 versus 0.914; state/context shuffle still does not yield a stable robust-selection effect. The model clearly uses C, but much of the effect is controller-level preference. Since C is itself a function of state and eta, h-only shuffle cannot establish that all state information is unused.

2. **The robust adaptive signal is sparse and selection headroom is small.** In the 46 TRAIN families, eta15 alone is B15 in only two `alt` and one `second` cases; eta10 alone is B15 in 26 and six respectively. Genuine Q ranking changes exist: matched-seed two-sided tests under `second` find five states favoring eta10 and seven favoring eta15 at nominal p<0.05. These exploratory tests are not multiple-testing-adjusted confirmations. Many Q reversals occur where neither candidate is B15, so learning them can improve Q ordering without rescuing B15. A diagnostic known-controller × eta count prior achieves NLL 0.604 and B15 43; lower probability error alone does not show learned state-dependent action choice.

3. **The state residual is statistically difficult and the flexible model overfits.** For the narrow full critic, average FIT NLL falls from 0.525 at selected checkpoints to 0.384 at step 1500 while source inner VAL rises from 0.523 to 0.781; outer crossfit NLL rises to 1.166 at the final checkpoint. Trial upgrades increase precision but do not add independent state families. A strongly regularized compact likelihood model using physical h/context interactions reaches NLL 0.627 but only 35 certified B15; context interaction alone reaches 0.630 and 41 B15. Thus a simple capacity reduction does not repair robust selection. These data do not yet distinguish limited state coverage from insufficient response descriptors.

The H20 context also contains limited dynamic variation: six channels are constant and nominal safety interventions are zero on this source panel. Seven varying channels were strongly attenuated by the old absolute standard-deviation floor. Standardization was tested rather than assumed beneficial; it did not reliably improve performance. H80 had already failed to improve the prior matched experiment, so these observations do not justify blindly extending the probe horizon.

## Repairs completed

- Rebuilt full training evidence from exact controller/state/eta keys in the authoritative SQLite DB. The earlier focused experiment had 184 controller-state-eta pairs; all-eta training now has 1,472. The older wide snapshot also omitted 1,098 subsequently recorded valid continuations across 92 pairs. The corrected table contains 9,186 observed valid continuation outcomes and 14 numerical attempts preserved separately. Compatible observations replace earlier counts; pairs and seeds are not duplicated. Canonical rollout UID provenance is saved in `database_dataset_keys.json`.
- Kept old snapshots and model results as frozen controls. Only the affected wide-data arms were repeated after discovering stale counts.
- Added early checkpoint checks at steps 5/10/25/50/75, complete FIT and VAL trajectories, source-family crossfit, and paired family-bootstrap uncertainty. Checkpoints and normalization use source FIT/inner VAL only.
- Evaluated h shuffle, same-controller state-context shuffle, joint h/context shuffle, and wrong-controller replacement. This corrects the earlier overly broad inference from h-only shuffle.
- Added explicit certified/non-B15/unresolved selection tables. Numerical attempts are excluded from likelihood and are never imputed into exact Q16 or forced negative B15 labels.
- Corrected the interpretation of the probability objective. If continuation outcomes are conditionally IID with success probability p, P(B15)=p^15(16−15p), whose derivative is 240p^14(1−p)≥0. A model that learns exact state-conditioned p ranks candidates consistently with B15 probability. Aggregate Q/B15 discrepancies can arise from state heterogeneity and finite seeds; they are not direct evidence to add ranking loss.
- Tested scaling and small-model hypotheses; neither was promoted as a successful fix. Historical wide-data eta-only fell to 33 B15, so beating that weakened control is not presented as outperforming the stronger 44-B15 eta-only baseline.

## Remaining adjudication

Confirmed: compatible cached labels align, controller context is used, the current flexible model overfits state residuals, and the robust-preference signal in this source panel is highly imbalanced. No preprocessing permutation, incompatible-controller merge, or fake-Q16 construction was found in the audited paths.

Underresolved: whether a richer state-conditioned response representation can predict the rare robust-preference reversals with sufficient new source families. The current evidence does not justify choosing between “insufficient input” and “insufficient diverse supervision,” and does not establish a fundamental learning-theory failure.

The smallest useful next evidence is a source-only audit of the *full deployed candidate pool*: quantify the best global/fixed eta ceiling and naturally occurring robust-preference reversals before collecting anything. If a global eta remains near the oracle, there is no basis to demand a large state-selection gain in that regime. If natural fixed-fail/adaptive-success families exist, freeze independent families and a model-blind shared candidate panel, retain all compatible prior data, and test the frozen model. Do not remove universal candidates merely to manufacture an adaptive advantage. No new deployment model is promoted by this audit.

NEW ROLLOUT = 0. Generator, safety, success definition, eta semantics, and frozen LOSO results are unchanged. Six GPU shards (12 CPU cores) were used for 81 small controlled fits; compact models ran on CPU. There are no remaining jobs for this experiment.

Reproduction: `experiment.py prepare/train/summarize`, `database_dataset.py`, `audit.py`, `compact_probe.py`, `uncertainty.py`, `selection_audit.py`. The frozen dataset/protocol, exact canonical keys, model hashes, checkpoints, fold splits, all decisions, and confidence intervals reside in this directory. Training arrays 0–53 are the initial controls; 54–80 are the authoritative-DB repair arms. Re-running preparation refuses to overwrite frozen artifacts.
