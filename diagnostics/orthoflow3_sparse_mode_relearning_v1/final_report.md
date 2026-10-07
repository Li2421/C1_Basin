# Sparse mode relearning stress test

Classification: **SPARSE_MODE_RELEARNING_SUPPORTED**

Frozen sparse codebook: original modes [3, 9]; selected from TRAIN only. Removed modes: [0, 1, 2, 4, 5, 6, 7, 8, 10, 11].

Important limitation: all 12 original modes exceeded 35% TRAIN strong-feasible prevalence. No subset could meet the requested 10--30% per-mode range. The Pareto pair is therefore a weaker sparsity stress test.

## Difficulty

|Split|Mode 3 prevalence|Mode 9 prevalence|Union/oracle coverage|Oracle mean Q|
|---|---:|---:|---:|---:|
|train|47.7%|59.4%|75.0%|0.834|
|val|53.1%|53.1%|78.1%|0.869|
|test|37.5%|75.0%|90.6%|0.935|

## TEST controller result

|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|
|---|---:|---:|---:|---:|---:|---:|
|BEST_FIXED_SPARSE_MODE|24/32|0.792|1623/2048|351|74|0|
|SPARSE_SELECTOR|30/32|0.974|1994/2048|37|17|0|
|SPARSE_ORACLE|29/32|0.935|1914/2048|29|105|0|
|SAFETY|13/32|0.638|1307/2048|95|646|0|

On 29 oracle-coverable TEST states, the selector chose a B63 sparse mode on 96.6%; fixed mode success was 82.8%.
Mode choices: {'3': 11, '9': 17}, normalized entropy 0.967; mean mode regret 0.015.

## Predictor comparison

|Predictor|TEST NLL|Brier|Top-1 empirical Q|Rank correlation|
|---|---:|---:|---:|---:|
|global_prior|0.629|0.153|0.792|0.360|
|linear|0.418|0.072|0.899|0.680|
|nearest_neighbor|0.774|0.072|0.849|0.652|
|mlp_seed23|0.258|0.015|0.919|0.840|
|mlp_seed23_calibrated|0.256|0.014|0.919|0.840|

## Fresh replication

Fresh 64 states: fixed 32/64 B63, mean Q64 0.600; selector 56/64 B63, mean Q64 0.966.
Selector vs fixed rescue/break: 1546/50. Failure diagnosis: {'SELECTOR_PREDICTION_FAILURE': 2, 'SPARSE_CODEBOOK_COVERAGE_FAILURE': 6}.

|Frozen action|States|Selector B63|Selector mean Q64|Fixed B63 on same states|
|---|---:|---:|---:|---:|
|3|27|24/27|0.963|12/27|
|9|19|18/19|0.989|18/19|
|safety|18|14/18|0.944|2/18|

## Answers

The dominant original mode 0 was removed: **YES**.
The relearned selector genuinely used multiple sparse modes: **YES**.
Did the original selector depend critically on the dominant mode: **NO**.
This result does not establish the requested 10--30% mode-prevalence regime because the frozen 12-mode dictionary contains no such modes.
