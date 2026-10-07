# ORTHOFLOW3_NLL_WEIGHTING_ABLATION_V1

Classification: **CURRENT_NLL_WEIGHTING_NOT_MAIN_CAUSE**

NEW ROLLOUT = **0**.

## Primary structured-only TEST

| Weight | NLL | MAE | Spearman | K8 regret/B15/top3 | K16 regret/B15/top3 | K32 regret/B15/top3 |
|---|---:|---:|---:|---|---|---|
| W0 | 0.7290 | 0.2351 | 0.6068 | 0.335/0.667/0.956 | 0.228/0.771/0.896 | 0.176/0.812/0.917 |
| W1 | 0.8095 | 0.2438 | 0.5824 | 0.439/0.556/0.956 | 0.354/0.646/0.896 | 0.283/0.708/0.917 |
| W2 | 0.8095 | 0.2438 | 0.5824 | 0.439/0.556/0.956 | 0.354/0.646/0.896 | 0.283/0.708/0.917 |
| W3 | 0.7754 | 0.2419 | 0.5917 | 0.397/0.600/0.956 | 0.311/0.688/0.896 | 0.260/0.729/0.917 |

## Secondary structured + wide TEST

| Weight | NLL | MAE | Spearman | Mean regret | Mean B15 top-1 |
|---|---:|---:|---:|---:|---:|
| W0 | 0.4411 | 0.1334 | 0.8227 | 0.0495 | 0.9366 |
| W1 | 0.3997 | 0.1259 | 0.8328 | 0.0417 | 0.9509 |
| W2 | 0.4045 | 0.1271 | 0.8290 | 0.0842 | 0.9079 |
| W3 | 0.3854 | 0.1274 | 0.8292 | 0.0825 | 0.9079 |

## State domination

- primary_structured: W0 max/median=1.057, W1 max/median=1.000, W2 max/median=1.000, W3 max/median=1.000
- secondary_combined: W0 max/median=10.649, W1 max/median=8.020, W2 max/median=1.000, W3 max/median=1.000

## NLL trajectory versus K8 regret

- primary_structured W0: Pearson=0.399, Spearman=0.203
- primary_structured W1: Pearson=0.331, Spearman=0.275
- primary_structured W2: Pearson=0.333, Spearman=0.299
- primary_structured W3: Pearson=0.489, Spearman=0.378
- secondary_combined W0: Pearson=0.874, Spearman=0.706
- secondary_combined W1: Pearson=0.780, Spearman=0.540
- secondary_combined W2: Pearson=0.805, Spearman=0.511
- secondary_combined W3: Pearson=0.855, Spearman=0.707

## Comparison with ranking-aware critic

| Source | Model | NLL | Mean regret | Mean B15 top-1 |
|---|---|---:|---:|---:|
| primary_structured | pure_nll_W0 | 0.7290 | 0.2461 | 0.7500 |
| secondary_combined | pure_nll_W1 | 0.3997 | 0.0417 | 0.9509 |
| ranking_aware_v1 | A | 0.4299 | 0.0521 | 0.9361 |
| ranking_aware_v1 | B | 0.4281 | 0.0694 | 0.9222 |
| ranking_aware_v1 | C | 0.4533 | 0.0738 | 0.9153 |
| ranking_aware_v1 | D | 0.4299 | 0.1207 | 0.8782 |

## Interpretation

Primary best weighting: **W0**. Relative to W0, mean regret gain=0.0000, mean B15-selection gain=0.0000.

