# Calibration shift audit

All scores come from frozen, exactly reproduced OOF MLP checkpoints. No model was trained and no test threshold was changed.

## Cross-group drift

- `anchor_D2_pair228`: zero/nonzero logit shifts +2.490/-0.142; common offset +1.174, differential -2.632, separation ratio 0.649; **CLASS_DEPENDENT_SHIFT**.
- `anchor_D4_pair227`: zero/nonzero logit shifts -1.067/+0.114; common offset -0.476, differential +1.181, separation ratio 1.433; **CLASS_DEPENDENT_SHIFT**.
- `qual_pair226`: zero/nonzero logit shifts -0.240/-0.014; common offset -0.127, differential +0.226, separation ratio 1.064; **MOSTLY_ADDITIVE_SHIFT**.
- `anchor_D1_pair231`: zero/nonzero logit shifts -3.042/-0.253; common offset -1.648, differential +2.789, separation ratio 1.657; **CLASS_DEPENDENT_SHIFT**.

The shared affine centroid fit is diagnostic only: test_logit ≈ -0.276 + 0.966·validation_logit, R²=0.732, RMSE=1.419. An additive-only fit has shift -0.269 and RMSE=1.422. Across the 101 stable states in these four held-out groups, raw/additive-frame/affine-frame diagnostic error counts are 14/17/16; the corresponding FPR/FNR are 0.146/0.133, 0.220/0.133, and 0.220/0.117. The correction fits used held-out labels and are explanatory checks only. Their large centroid residuals and failure to uniformly remove errors do not support a universal post-hoc correction.

## Relation to Pair 1's cross-fold reversal

Agent A's exact decomposition gives a cross-fold model/score offset of -1.913 logits. The fold-wide common calibration offsets differ by -1.650 logits (D4 minus D2), which accounts for 86.3% of that offset magnitude with the correct sign. Using the relevant label-conditional shifts gives -2.376 logits (D4 nonzero minus D2 zero), within 0.462 logits of the exact model offset. Thus the apparent Pair-1 cross-model ranking reversal is substantially explained by source/fold-dependent score calibration, especially the upward zero-class shift in the D2 fold. It is not a same-model ranking reversal.

## Pair 2

The zero/nonzero cloud means are 1.104/1.665 logits (margin +0.562, AUC 1.000). Their independently selected OOF thresholds are 0.480/1.338. The local midpoint is 1.384; relative to the zero fold threshold, this is a +0.905 displacement. Because the states use different OOF models, this is a score-coordinate diagnostic rather than a deployable common threshold.

## Clean control pair

Both control states use the same OOF checkpoint. Their means are 2.401/3.212, margin +0.810, AUC 0.999268. The validation threshold is 1.388, while the local mean midpoint is 2.806, a +1.419 shift. A clean all-variant common threshold does not exist because the narrow cloud supports overlap.

Final sub-agent conclusion: **CLASS_DEPENDENT_SHIFT**.

The calibration failure is not explained by one universal scalar threshold offset alone when class-specific shifts and between-fold scale changes are material. All counterfactual shifts above are diagnostic; no test threshold was retuned.
