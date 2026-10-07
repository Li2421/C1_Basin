# Source-shortcut audit

## Conclusion

**MODERATE_SOURCE_SHORTCUT_EVIDENCE**

Source identity is recoverable from every existing semantic feature block at
more than twice the 12-group chance rate, and the three robust false-positive
zero states resemble intervention-heavy training source patterns across all
feature blocks.  The evidence is nevertheless heterogeneous: broad
oracle-stable labels remain strongly source-held-out separable in geometry,
history, and projection features, and the three robust false-negative states
do not consistently resemble opposite-class source patterns.  Thus this audit
does not support the stronger claim that the gate is globally dominated by a
source identifier.

## Population and safeguards

- Primary analysis used one centroid per independent augmented state: 277
  oracle-stable states (157 zero, 120 nonzero).  All 47 oracle-ambiguous states
  were excluded.
- Source-association comparisons used the 172 stable states in the 12 source
  groups having at least two states and both oracle classes.  This avoids the
  tautological ANOVA inflation caused by treating 105 singleton groups as
  estimable source distributions.
- Every stable state had exactly 64 saved Flow variants.  No variant was
  counted as an independent state.
- The diagnostic source decoder was a fixed-regularization linear ridge
  decoder with deterministic within-source held-out folds.  Label diagnostics
  used fixed-L2 logistic regression with complete leave-one-source-group-out
  testing and train-only normalization.  They are probes, not replacement gate
  training.

## Source signal versus transferable label signal

| Existing feature group | Source decoder accuracy | Chance | Oracle-label LOSO BA | Oracle-label LOSO AUROC | Median source eta-squared |
|---|---:|---:|---:|---:|---:|
| history / monitor | 0.407 | 0.083 | 0.963 | 0.984 | 0.034 |
| episode / time | 0.360 | 0.083 | 0.661 | 0.702 | 0.434 |
| geometry / observation | 0.233 | 0.083 | 0.873 | 0.965 | 0.073 |
| projection / safety diagnostics | 0.192 | 0.083 | 0.866 | 0.969 | 0.063 |
| current control (`u_Flow`, `u_safe`, `B_goal`, `B_rel`) | 0.186 | 0.083 | 0.788 | 0.706 | 0.112 |

Episode/time is the clearest compact source fingerprint and has the weakest
transferable label discrimination after current control.  History is the best
multivariate source decoder, but it also carries the strongest transferable
oracle-label signal; source decodability alone therefore does not make history
a spurious feature.

The largest label-adjusted source effects occur in physical/normalized time,
`history_start_step`, `max_stuck_timer`, selected wall/position/goal-relative
coordinates, `u_flow`, `u_safe`, and projection residuals.  These are
associations, not causal attributions.

## Cross-group correlation changes

Raw positive and negative oracle-label associations occur across groups in all
8 current-control dimensions, 22/67 geometry dimensions, 22/92 history
dimensions, and 16/43 projection dimensions.  After requiring a within-group
standardized effect magnitude of at least 0.2, sign disagreement is
concentrated in current control (median 0.174; 3/8 dimensions at least 0.25),
geometry (15/67 dimensions at least 0.25), and projection diagnostics (10/36
nonconstant/informative dimensions at least 0.25).  Episode/time has no such
sign reversal.  Small per-group counts make these descriptive rather than
confirmatory statistics.

## Six robust OOF mistakes

The three robust false-positive zero states show a coherent local shortcut
pattern:

- `RBV_Q_pair228_m080_s95401003_p030`
- `RB_Q_pair226_m080_s95400802_p073`
- `RB_Q_pair228_m080_s95401001_p050`

For each, the state is closer to the nonzero fold-training centroid in all
five semantic groups and full 214-D space; every group's nearest eligible
training-source centroid has a nonzero-majority label; and the OOF probability
is closer to the held-out group's nonzero score median.  The first and third
map most closely to `anchor_D4_pair227` in geometry, history, projection, and
full space.  This is direct local evidence that unseen stable-zero states can
inherit an intervention-like source pattern.

The three false-negative nonzero states are different.  Their nearest
training-source centroids are nonzero-majority in all feature blocks.  Two
(`RB_P_r175...p144`, `RB_Q_pair225...p166`) nevertheless have OOF probabilities
closer to their held-out zero-class score median; `RB_P_r175...p132` retains a
score closer to the nonzero distribution but falls below its fold threshold.
This part is not explained by a simple source-majority shortcut.

## Cross-check with ranking attribution (Sub-agent A)

The attribution audit independently labels Pair 1 `MULTI_GROUP_SHORTCUT`.
Its dominant wrong-direction input effects are geometry/observation
(-0.733 logit), control/projection (-0.295), history/monitor (-0.210),
`u_safe` (-0.184), inter-agent relative (-0.154), and goal-relative (-0.154).
Those blocks are source-decodable here at 0.186--0.407 versus 0.083 chance,
and current-control, geometry, and projection dimensions show the most
meaningful cross-group sign changes.  The two diagnostics therefore agree that
multiple source-correlated blocks elevate the Pair-1 zero state.

The attribution audit also establishes an essential qualification: the
reported Pair-1 OOF reversal compares two different fold models.  Both models
order the pair correctly when each scores both clouds; a -1.913 cross-fold
model offset overwhelms a +0.611 within-model feature effect.  Consequently,
the numerical cross-fold reversal itself is mainly a fold-score offset, while
the shortcut evidence explains why the zero state has an intervention-like
absolute score under its own fold.

## Interpretation

The strongest supported statement is local and asymmetric: source-correlated
geometry/history/control patterns explain the three persistent false-positive
zero cases, but source shortcut does not by itself explain all six robust
errors.  Broad stable-state label information remains highly transferable,
so the result is **moderate**, not strong, source-shortcut evidence.  The
separate calibration audit should determine how much of the remaining failure
is fold-dependent score offset and threshold drift.

