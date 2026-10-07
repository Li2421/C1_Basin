# Why the continuous Basin critic trained poorly

This audit used only `pair_table.parquet`, the frozen split records, training summaries, and model evaluations. No rollout or retraining was performed.

## Primary cause: the database is not a crossed state × eta design

The nominal pair count overstates the independent supervision available for learning an interaction field.

- Toy TRAIN has 3,146 pairs from 128 independent states and 493 exact eta values. The median exact eta occurs in only one state.
- DB TRAIN has 512 pairs, but they are exactly eight eta values repeated over 64 states.
- None of the eight DB TRAIN eta values changes from failure to robust success across states. Their labels are almost entirely eta-specific, so DB supplies essentially no evidence from which to learn a state-dependent feasibility interaction.
- Toy contains more interaction evidence, but it is concentrated: only 75/493 TRAIN eta values exhibit both clear failure and robust success across states.

This explains why eta-only is the strongest DB model and why the discrete aligned codebook experiment was much easier: that experiment deliberately constructed a dense state × shared-mode matrix, while the historical continuous eta data are mainly adaptive, state-specific probes.

## Simultaneous eta holdout crosses sharp unsupported regions

Eta components within normalized distance 0.05 were kept together. Test eta therefore do not merely differ by exact ID: the test-to-train nearest distance has median 0.110.

For Toy simultaneous holdout pairs, a nearest-TRAIN-eta mean-Q baseline has MAE 0.623 and negative rank correlation (-0.095); 63.1% of pairs flip between a robust and a failure extreme relative to their nearest TRAIN eta. Thus proximity at the available sampling scale does not provide a smooth interpolation target. This is consistent with the previously observed sharp Basin intrusions.

The problem is not ordinary range extrapolation: 99.6% of Toy simultaneous-holdout eta lie inside the coordinate-wise TRAIN range. The missing information is local boundary coverage inside that range.

## Probability calibration is weakly identified

TRAIN labels are strongly bimodal:

- Toy: 2,220 robust, 787 clear failure, 137 intermediate, and only 2 near the 15/16 boundary.
- DB: 439 robust, 55 clear failure, 18 intermediate, and zero near-boundary pairs.

Consequently the dataset can teach broad success/failure regions but barely constrains intermediate success probabilities or the robust boundary. Q64 provides precise estimates at sampled points, but precision at isolated points does not replace geometric coverage between them.

## DB suffers a severe distribution and feature shift

- DB mean TRAIN Q is 0.903; simultaneous-holdout mean Q is 0.498.
- The joint model consequently predicts mean Q=0.855 on that panel and is badly overconfident.
- DB test-state nearest-TRAIN-state RMS-normalized feature distance has median 0.836, versus 0.292 for Toy.
- DB simultaneous holdout comprises two all-failure eta groups and one nearly all-success group. Eta identity alone separates them, explaining eta-only NLL 0.428 versus joint NLL 0.887.

The DB simultaneous panel contains only 64 pairs, so its precise metric values have limited resolution, but the direction of the failure is unambiguous.

## State information is present, but it does not transfer uniformly across eta

Toy h is not simply useless. For TRAIN eta observed on at least ten states, nearest-h prediction at the same eta has MAE 0.067, versus 0.205 for the eta-wide mean. The state representation therefore contains locally useful feasibility information.

However, state-feasibility signatures differ across eta: among comparable Toy eta pairs, the median cross-eta state-ranking correlation is only 0.358 and 36.3% are negative. A state feature that predicts feasibility for one eta often does not transfer cleanly to an unseen eta. Simultaneously holding out both variables therefore asks the network to infer an interaction that the data only sparsely identify.

## Optimization symptoms are consequences, not the leading cause

All three joint seeds and all three DB-only seeds selected the first evaluated checkpoint (step 50); later optimization never improved held-out NLL. Eta-only models continued improving to steps 350–1,500. This reproducible pattern indicates immediate fitting of train-specific state interactions followed by worse holdout behavior, rather than one unlucky initialization.

The joint model has roughly 75k parameters but only 192 independent TRAIN states across both scenarios. Thousands of repeated eta pairs do not create thousands of independent state contexts. More weight decay or a smaller network may reduce variance, but cannot manufacture the missing crossed interaction evidence.

## Cross-scenario sharing is also weakly supported by the sampled eta domains

Toy TRAIN contains 493 unique eta values, DB only eight, with zero exact overlap. DB eta are moderately near a small part of Toy support, but only 0.6% of Toy TRAIN eta lie within 0.10 normalized distance of a DB TRAIN eta. A shared eta encoder therefore sees highly asymmetric scenario support. This is consistent with positive joint transfer on Toy but negative joint transfer on DB.

## Bottom line

The failure is primarily a data-identifiability problem:

1. continuous historical probes are not a dense crossed state × eta panel;
2. DB TRAIN contains essentially no state-dependent mode variation;
3. local boundary/intermediate-Q coverage is nearly absent;
4. simultaneous holdout traverses sharp internal support gaps;
5. state effects are eta-specific rather than governed by one simple transferable ordering.

The result does **not** show that h contains no useful Basin information. It shows that the current cached evidence is well suited to aligned discrete-mode feasibility learning, but insufficient to identify a globally general continuous `F(h, eta)` over both unseen states and unseen eta.
