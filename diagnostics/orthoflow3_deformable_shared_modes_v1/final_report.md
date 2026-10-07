# Deformable shared modes

Classification: **FIXED_MODES_ALREADY_SUFFICIENT**

Usable local sets: 158/159; selected seed 23.

|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|J_def|
|---|---:|---:|---:|---:|---:|---:|---:|
|fixed|64/64|0.9995|4094/4096|2|0|0|0.5579|
|deformable|64/64|1.0000|4096/4096|0|0|0|0.5570|

Moving vs fixed: rescue 2, break 0; mean Q64 difference 0.0005, bootstrap 95% CI [0.0000, 0.0012].
Mean paired J_def difference -0.0009, 95% CI [-0.0042, 0.0024]. This is not a statistically resolved deformation reduction.
Mean normalized residual magnitude was 0.0584; raw eta coefficient norm changed by -0.0263 on average (proxy only).

Local-set labels: 158/159 usable; 2388 positive, 192 negative, and 70 ambiguous eta observations.
Residual learnability: TRAIN/VAL membership proxy 100.0%/100.0%; mean distance 0.0209/0.0207.
The learned residual is state-dependent (variance 0.000869); neighbor h/delta Spearman 0.354.
Cross-scenario affine+per-mode alignment is descriptively promising: residual median 0.058, <=0.20 fraction 100.0%. The affine condition number 63.4 makes this evidence ill-conditioned rather than confirmatory.

The moving mode preserved robustness, but it did not show a stable advantage over the already saturated fixed selector. Mode centers can move predictably with state, yet that motion is not currently necessary for Toy closed-loop success.
Upgrade to a conditional multimodal generator: **NO_NOT_YET**.
