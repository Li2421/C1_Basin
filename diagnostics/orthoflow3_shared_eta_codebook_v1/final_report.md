# Shared eta codebook feasibility

Classification: **COMMON_ROBUST_MODE_DOMINATES**

Codebook M=12; source-isolated split 128/32/32. Matrix cost: 24,576 TRAIN + 12,288 VAL + 24,576 TEST mode continuations, plus 1,024/2,048 matched VAL/TEST Safety continuations. Cached reuse reduced new matrix work to 58,504.

TEST oracle coverage: 100.0%; oracle mean Q64: 1.000. Selector: 32/32 B63, mean Q64 1.000, mean regret 0.000.

Selected mlp_seed17; temperature 1.117 (used); tau=0.95.

## Codebook

|Mode|eta|TRAIN exact-B63 support|
|---:|---|---:|
|0|(0.7421875, 0.46875, 0.7265625)|22|
|1|(1.23046875, -0.171875, 0.01171875)|10|
|2|(0.8984375, -0.15625, 0.4453125)|14|
|3|(1.09375, 0.375, 0.09375)|13|
|4|(0.68359375, 0.265625, 0.15234375)|12|
|5|(0.9765625, 0.28125, 0.4921875)|13|
|6|(0.625, 0, 0.375)|13|
|7|(0.788671494, 0.166046143, 0.421131134)|12|
|8|(0.91796875, 0.078125, 0.57421875)|12|
|9|(0.56640625, -0.140625, 0.31640625)|10|
|10|(0.859375, 0.3125, 0.328125)|8|
|11|(0.0465416908, 0.000412839465, 0.0209529677)|1|

## Model comparison

|Model|TEST NLL|TEST Brier|Top-1 empirical Q|Rank correlation|
|---|---:|---:|---:|---:|
|global_prior|0.529|0.133|0.956|0.415|
|nearest_neighbor|0.457|0.045|0.969|0.790|
|linear|0.243|0.029|0.999|0.768|
|mlp_seed17|0.213|0.021|1.000|0.846|
|mlp_seed23|0.204|0.020|1.000|0.834|
|mlp_seed41|0.198|0.016|1.000|0.847|

## Closed-loop TEST

|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|J_def|
|---|---:|---:|---:|---:|---:|---:|---:|
|Safety|13/32|0.638|1307/2048|95|646|0|2.272083348969705e-17|
|G_POINT_T0|19/32|0.782|1601/2048|247|200|0|0.6556189241773068|
|G_LOWJ|17/32|0.711|1456/2048|408|184|0|0.38325322863103833|
|CODEBOOK_SELECTOR|32/32|1.000|2048/2048|0|0|0|0.5488797835550874|
|CODEBOOK_ORACLE|32/32|1.000|2048/2048|0|0|0|0.5303682246464909|
|TRAIN_PRIOR_COMMON_MODE|29/32|0.956|1957/2048|71|20|0|0.5349169244575659|

Selector vs Safety rescue/break: 741/0; selector vs G_POINT_T0: 447/0.
TRAIN-prior common mode: 29/32 B63, mean Q64 0.956. The best single-mode B63 fraction was 90.6%.

## Interpretation

Shared discrete codebook sufficient: YES.
Can h0 predict feasible modes: YES.
Did aligned supervision solve arbitrary-target failure: YES. It exactly matched the oracle, although a broad common mode already explains most states.

Next: freeze one new source-diverse cohort and compare the fixed common mode with this frozen selector before adding local margins.
