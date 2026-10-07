# Recovery-entry identifiability audit

## Outcome

**Classification: `ENTRY_DECISION_NOT_NEEDED_ON_THIS_DISTRIBUTION`.**

The current generic queried-state distribution contains some recovery-beneficial and recovery-harmful states, but selective entry has little attainable value. Persistent recovery is non-harmful within the predeclared ±0.05 descriptive band on 95.7% of states, and the hindsight statewise upper bound improves over Always-Recovery by only 0.0139 absolute success probability. Neither the source-held-out linear nor small-MLP probe beats Always-Recovery.

This is a distribution-specific identifiability judgment, not a claim that entry selection is universally unnecessary and not a production full-episode result.

## Protocol and evidence

- Generated 120 new development-only WIDE roots before outcome inspection; 112 roots produced 184 generically sampled nonterminal states.
- State sampling used two deterministic uniform-over-visited-timestep requests per source, independent of future failure, failure type, recovery outcome, spatial region, or terminal-relative time.
- Each state received 16 matched future streams for Safety and Recovery. A frozen one-shot uncertainty rule added streams 16--31 for 135 states; 49 remained at 16.
- Final cost: 5,104 matched N/R pairs and 10,208 branch continuations, below the 12,000 limit.
- Recovery was the frozen one-shot persistent structured-eta option plus the authoritative frozen learned exit and Safety-after-exit. No controller, exit rule, eta model, or feature was changed.

## Statewise continuation values

| Quantity | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Q Safety | 0.5715 | 0.9375 | 0.0000 | 1.0000 |
| Q Recovery | 0.6467 | 0.9844 | 0.0000 | 1.0000 |
| Delta Q | 0.0752 | 0.0000 | -0.5625 | 1.0000 |
| Rescue | 0.0895 | 0.0000 | 0.0000 | 1.0000 |
| Break | 0.0143 | 0.0000 | 0.0000 | 0.6250 |

Descriptive sensitivity fractions:

- `DeltaQ > +0.10`: 14.13%
- `DeltaQ > +0.05`: 16.85%
- `|DeltaQ| <= 0.05`: 78.80%
- `DeltaQ < -0.05`: 4.35%
- `DeltaQ < -0.10`: 3.80%

At the diagnostic ±0.05 cuts there were 31 beneficial, 8 harmful, and 145 ambiguous states. At exact zero there were 39 positive, 10 negative, and 135 tied states.

## Trivial policies and hindsight bound

| Statewise decision rule | Expected continuation success |
|---|---:|
| Always Safety | 0.5715 |
| Always Recovery | 0.6467 |
| Hindsight statewise oracle | 0.6607 |

The non-deployable hindsight gain over the best trivial policy is only 0.0139. Thus a selective gate has little room to add value on this sampled state distribution even before accounting for prediction error.

## Source-held-out probes

Primary evaluation used three repeats of five-fold source-grouped cross-validation. Normalization, ridge strength, MLP checkpoint, and policy thresholds were fit inside each training/validation fold.

| Probe | Pearson | Spearman | MAE | RMSE | AUROC | Balanced accuracy | Probe-policy value | Entry fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 0.0007 | 0.0236 | 0.1579 | 0.2714 | 0.5435 | 0.4992 | 0.6311 | 0.7011 |
| 214-64-64-1 MLP | 0.1860 | 0.1500 | 0.1948 | 0.3093 | 0.5495 | 0.5198 | 0.6254 | 0.6902 |

Both learned policies are worse than Always-Recovery (0.6467). Binary metrics are also fragile because only eight harmful states exist. The cost-sensitive linear utility ranking has mean Spearman 0.0236, -0.0258, and -0.0995 for break multipliers 1, 2, and 4.

## Shortcut, neighborhood, and feature audits

- Random-state versus grouped Spearman was 0.3056 versus 0.0236 for ridge and 0.2629 versus 0.1500 for MLP. This indicates source shortcut sensitivity, but random-split performance is not strong enough to establish a useful entry rule.
- Across confidently beneficial/harmful held-out states, mean normalized nearest-same distance was 0.317 and nearest-opposite distance 0.477; mean opposite-label frequency among 5-NN was 10.6%, while ambiguous neighbors were 57.8%. There is overlap, but ambiguity rather than clean class separation dominates.
- Leave-one-feature-block-out ridge results stayed weak; no existing block carried decisive entry information.
- The only nonconstant deployment-available saved information omitted from h was real goal-error history older than the 41-sample tail. Adding 12 causal summaries of that older history did not help: Pearson 0.0202, Spearman 0.0342, MAE 0.1660, RMSE 0.2761.

## Uncertainty and integrity

After confirmation, 160/184 paired 90% intervals still crossed at least one predeclared cut and 157 crossed zero. This limits confident per-state labeling, but does not create additional achievable policy value: the point-estimate hindsight upper bound is already only 1.39 percentage points above Always-Recovery.

All 10,208 records were complete and paired; `DeltaQ = rescue - break` held exactly. There were zero agent collisions, wall collisions, invalid actions, NaN/Inf events, or projection failures. No production gate was trained or saved.

## Decision

Do **not** build a recovery-entry gate from this dataset. The next action justified by this audit is: **do not build an entry gate because selective entry has little value on the sampled distribution**. If a future deployment distribution creates a materially larger predeclared hindsight gap, repeat this identifiability audit there before changing representation or history.
