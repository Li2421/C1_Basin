# STATE_BALANCED_ONLY condition

The frozen seven-fold OOF protocol completed: 21 MLP runs (three seeds per
fold), 21 finite checkpoints, and 130 unique frozen held-out state means.
No state, label, feature, normalization, threshold rule, or architecture was
changed.

Each epoch used exactly its fold's original number of training-sample draws,
with replacement, by uniformly sampling a train augmented state and then one
of that state's 64 saved Flow variants.  Every train state has exactly 64
variants.  Consequently this sampler has the *same expected state and source
group weights* as sample-uniform training; it changes only epoch-level
resampling noise.  In the seven folds the largest source group had 24--31
states while a singleton group had one, so both sample-uniform and this
condition retain a 24--31x source-group expected-weight range.

| OOF subset (seed mean) | States | BAcc | FPR | FNR | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen held-out stable union | 130 | 0.85875 | 0.12000 | 0.16250 | 0.85385 |
| Difficult stable boundary | 14 | 0.43750 | 0.50000 | 0.62500 | 0.42857 |
| Recovery stable | 95 | 0.80667 | 0.12000 | 0.26667 | 0.81053 |

These classification outcomes are identical to the strictly reproduced
sample-uniform baseline.  Across all 520 per-seed/ensemble OOF predictions,
the mean absolute probability difference from baseline was 0.00153 (maximum
0.01149); no thresholded state prediction changed.  The three robust
false-positive zero clouds remained 64/64 intervention predictions:

| State | Mean fold-relative logit margin | Mean probability |
| --- | ---: | ---: |
| `RBV_Q_pair228_m080_s95401003_p030` | +0.83920 | 0.90230 |
| `RB_Q_pair226_m080_s95400802_p073` | +0.62213 | 0.75044 |
| `RB_Q_pair228_m080_s95401001_p050` | +1.00573 | 0.91614 |

This optional control establishes that repeated Flow variants do not create a
separate state-weighting confound in Dataset V4: all stable states already
have equal variant cardinality.  Source-group-balanced sampling is therefore
the condition that tests the stated source-frequency hypothesis.
