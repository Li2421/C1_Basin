# Stage-1 intervention-window audit — final handoff

Stage 1 is complete by explicit user instruction. Stage 2 gate training was
not prepared or run.

## Scope and integrity

- Oracle-stable states: **277** across **117** source groups.
- Coarse delays: `[0, 4, 8, 16, 32, 64]` steps; adaptive extension/refinement used matched randomness.
- Adaptive protocol: transition ambiguity advanced **64 -> 128 -> 256** only; terminal ambiguous points at the cap: **28**.
- Downstream limitation: frozen `eta_best` recovery approximation, not arbitrary-state every-step oracle re-query.
- Stage 2: **not run**; no gate or G_phi was trained.

## Empirical windows

- Window categories: `{'SHORT_WINDOW': 35, 'IMMEDIATE': 28, 'LONG_WINDOW': 183, 'MEDIUM_WINDOW': 28, 'UNRESOLVED': 3}`.
- `d_last_safe` distribution: `{0: 42, 4: 7, 8: 10, 16: 9, 20: 1, 28: 2, 32: 8, 40: 2, 48: 1, 64: 12, 128: 183}`.
- `d_first_bad` distribution: `{'20': 4, '8': 4, '14': 4, '24': 5, '4': 28, 'RIGHT_CENSORED': 186, '128': 12, '32': 6, '64': 8, '48': 5, '40': 4, '28': 1, '12': 3, '13': 1, '50': 2, '16': 1, '25': 1, '6': 1, '56': 1}`.
- First aggregate statistically supported recoverability degradation: **d=4 (0.20 s)**.
- Among old `y_long=1` states: **28/120** degrade at the first tested positive delay (`d=4`); `y_long` remains diagnostic only.

| Delay | Seconds | Macro Q | Macro Q0-Qd | Paired-state bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.998223 | 0.000000 | [0.000000, 0.000000] |
| 4 | 0.20 | 0.991638 | 0.006473 | [0.003878, 0.009392] |
| 8 | 0.40 | 0.985122 | 0.012946 | [0.008236, 0.018276] |
| 16 | 0.80 | 0.968186 | 0.029967 | [0.019348, 0.041657] |
| 32 | 1.60 | 0.908915 | 0.089181 | [0.062444, 0.118259] |
| 64 | 3.20 | 0.834753 | 0.163357 | [0.127186, 0.201744] |

## Deformation and tradeoff

- Aggregate J_def directions across full-pool nonzero delays: `{'DECREASE': 5}`.
- Recoverability-compatible, nonzero-delay Pareto points: **1008**.

| Delay | Macro J_def | Macro Jd-J0 | Paired-state bootstrap 95% CI |
|---:|---:|---:|---:|
| 0 | 0.087834 | 0.000000 | [0.000000, 0.000000] |
| 4 | 0.086728 | -0.001077 | [-0.001694, -0.000557] |
| 8 | 0.085613 | -0.002131 | [-0.003119, -0.001269] |
| 16 | 0.082046 | -0.005732 | [-0.008126, -0.003599] |
| 32 | 0.069844 | -0.017920 | [-0.023961, -0.012410] |
| 64 | 0.051620 | -0.036148 | [-0.044738, -0.028090] |

Q and J_def are reported separately; no scalarization was used.

## Stage-1-frozen candidate horizons (not trained)

Candidates: **[4, 8, 16]**. They were selected only
from Stage-1 evidence and frozen before any Stage-2 model result existed.

- H=4 (0.20 s): resolved zero/one=234/28, ambiguous=15, strict-LOGO-feasible=True
- H=8 (0.40 s): resolved zero/one=228/24, ambiguous=25, strict-LOGO-feasible=True
- H=16 (0.80 s): resolved zero/one=218/35, ambiguous=24, strict-LOGO-feasible=True

## Terminal interpretation

This run establishes the empirical intervention-window target evidence only.
Whether any frozen `y_H` target is source-generalizable remains **untested**
because Stage 2 was explicitly deferred. The smallest next experiment, if
resumed later, is the already specified strict source-group LOGO target-validity
probe on the frozen candidate horizons; correction G_phi should remain deferred
until that probe succeeds.
