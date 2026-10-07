# OrthoFlow3 basin-margin learning v1

## Outcome

**BASIN_MARGIN_LEARNING_UNDERRESOLVED**

The verified-ball dataset was successfully completed, but neural training was not started. Label construction used **21,128 new continuations**, which exceeds the protocol's explicit 20,000-continuation label-stage stop threshold. The overall experiment caps were still respected (5,480,321 physical steps; fewer than 25,000 continuations).

## Frozen verified-ball dataset

| split | labels | independent source families | true t=0 | intermediate/temporal |
|---|---:|---:|---:|---:|
| TRAIN | 25 | 5 | 0 | 25 |
| VAL | 10 | 2 | 0 | 10 |
| TEST | 9 | 2 | 0 | 9 |

There is zero source-family leakage. The dataset contains 9 expensive-anchor labels and 35 verified transfer labels. Ball radii have mean 0.299103, median 0.265625, min 0.233750, and max 0.353621.

Seven new expensive anchors were attempted in frozen outcome-blind order. Four passed the frozen `r >= 0.20` usability criterion; three geometrically valid balls were rejected as too small (`S_r065_p24`: 0.127500, `S_r006_p06`: 0.021250, `S_r021_p21`: 0.170000). The four eligible anchors produced 16/16 full-radius (`s=1.00`) transferred labels. All transfer centers were 64/64 and all 64 mandatory interior points were 64/64; confirmed internal false inclusions were zero.

No label in this dataset is a true episode-start state (`absolute_step == 0`). It is therefore a state-conditioned diagnostic dataset, not a t=0 training dataset. Any later fresh-WIDE evaluation would be a genuine distribution/generalization test.

## Frozen reference and decisions

The frozen G_LOWJ checkpoint hash was verified as `bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7` and was not retrained.

- Decision A: **POINT_TARGET_COMPARISON_UNDERRESOLVED**.
- Decision B: **BASIN_MARGIN_LEARNING_UNDERRESOLVED**.

No evidence about G_CENTER versus G_MARGIN has been generated in this run. In particular, verified set-valued supervision has not yet been shown to outperform center MSE; normalized error-to-radius, perturbation tolerance, J_def tradeoffs, and fresh-WIDE outcomes remain unevaluated.

## Runtime and next step

Label generation used 56 exact cached continuations and 21,128 new continuations (5,480,321 physical steps). Critical-path rollout wall time was 46.53 minutes with at most 6 GPU shards, 12 concurrent CPU threads, and 48 GiB scheduled RAM. GPU peak memory was not captured. Optional outside-shell work was skipped because it is not required for label usability and the stop threshold had already fired.

The smallest justified next step is a separately approved training/evaluation continuation using this now-frozen 25/10/9 dataset: train only G_CENTER and G_MARGIN with the predeclared three seeds, then run the frozen VAL/TEST/fresh-WIDE comparison. No Q, J, selector, or gate experiment should be added.
