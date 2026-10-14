# Frozen analysis clarification (before any new success label)

H_cov_E primary representation is R5-PCA2. H_switch primary representation is R5-PCA2 ⊕ R6-Delta, using its E and t* predictions for both G_E and G_EL in the paired comparison. Other representations remain descriptive. H_cov_L uses either R6-Delta or the concatenation as specified.

Aggregate CV Spearman and RMSE pool all out-of-fold controller predictions, one per held-out controller per fold. Source-mean baseline is recomputed within each fold and pooled likewise. Paired fold bootstrap resamples the seven recipe folds and computes the pooled difference, with 10,000 draws. ICC confidence intervals resample recipes with replacement. If fewer than four complete training-seed waves exist, all k=4 hypothesis gates are INCONCLUSIVE; k<=3 results are descriptive.

For the first-paragraph descriptive regime answer, classify recipe dominance when the point ICC for N exceeds 0.5; otherwise classify seed dominance. Always report the ICC CI and discordant-sign count so this point classification is not mistaken for a significance test. If fewer than four waves are complete, mark the four-training-seed regime question INCONCLUSIVE.
