# Fresh selector versus fixed common mode

Classification: **SELECTOR_ADVANTAGE_REPLICATED**

Fresh source-isolated states: 64. Frozen codebook/model/temperature/threshold/fixed eta were unchanged.

|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|J_def|Episode length|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|safety|38/64|0.7476|3062/4096|73|961|0|0.0000|650.0|
|fixed|57/64|0.9805|4016/4096|65|15|0|0.5334|429.7|
|selector|64/64|0.9995|4094/4096|2|0|0|0.5579|370.6|

Selector vs fixed: rescue 79, break 1, net 78 matched continuations. Mean-Q64 gain 0.0190, paired state-bootstrap 95% CI [0.0034, 0.0447].
Selector vs Safety: rescue 1034, break 2, net 1032.

Mode selections: {0: 15, 1: 18, 2: 8, 4: 16, 8: 1, 9: 5, 11: 1}. Selector chose fixed mode on 15/64 states and switched on 49.
Among switched states, selector was B63 on 49/49. All 7 fixed-mode non-B63 exception states were rescued to B63; switching produced 79 rescue and 1 break continuations.

Conditional audit: 0 selector-prediction failures; 0 codebook-coverage failures.

State-conditioned mode selection necessary: **YES**.
