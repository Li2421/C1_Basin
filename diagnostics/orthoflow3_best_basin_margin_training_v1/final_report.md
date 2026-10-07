# Best available Basin margin training

Classification: **MARGIN_SUPERVISION_FAILS**

Selected family: `evidence_capped_superbody_cut3_p4`. It was Pareto-selected using known false inclusion, unsupported sparse-evidence expansion risk, usable state count, extent, witness separation, diagnostic recall, and complexity; neural outcomes were not used. Fresh validation had 6/48 non-B63 points (12.5%), hence quality **C**. The final training label was globally eroded to retained gamma=0.50 and inner gamma=0.40.

Train retained diameter median: 0.208; VAL median: 0.208. Train/VAL median volume proxies: 0.000750/0.000750; median robust witnesses are 1/1. Independent-B63 retained recall for the selected candidate is diagnostic only: 0.125.

|Controller|B63 states|Mean Q64|Success /512|Deadlock|Timeout|Collision|Mean successful J_def|
|---|---:|---:|---:|---:|---:|---:|---:|
|Safety|3|0.721|369|32|111|0|1.0917694621703692e-16|
|G_POINT_T0|2|0.557|285|130|97|0|0.6208820947986655|
|G_LOWJ|4|0.795|407|74|31|0|0.35267627286569914|
|G_MARGIN|2|0.377|193|221|98|0|0.6135834035489894|

G_MARGIN rescue/break relative to Safety: 39/215. None of the 8 held-out predictions was inside the post-hoc TEST diagnostic inner or retained set.

The conservative region was not large/reliable enough to improve held-out control in this first trial. Full-Basin recall remains diagnostic only. Same mathematical body-plus-cuts form remains meaningful across compatible scenarios as a geometric hypothesis, but previous cross-scenario gates did not establish a deployable shared label and Double-Bottleneck features were not combined into neural training.

See `basin_size_vs_q64.csv` for state-level size/membership versus control results.
