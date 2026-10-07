# OrthoFlow3 t0 basin completion and phase-mismatch audit v1

## Decision

**CONSERVATIVE_BALL_TOO_SMALL_TO_DIAGNOSE**

The fixed eight-state completion finished without changing any state, center, controller, OrthoFlow3, or safety semantic. Geometry strongly differs from the intermediate dataset, but the conservative balls are too small to make controller rho a causal diagnostic: all 24 frozen-controller predictions are outside their t0 balls, while 15/24 are nevertheless B63.

## Completed geometry

- Balls resolved and independently validated: 8/8.
- Centers: 8/8 at 64/64.
- Independent screening: 96/96 at 8/8.
- Mandatory interior validation: 32/32 B63.
- Confirmed internal false inclusion: 0.
- Margin-usable `r>=0.20`: 2/8.
- Radii: mean 0.125508, median 0.127500, min 0.021250, max 0.249688.
- Center distance: mean 0.261719, median 0.407583, max 0.475159.
- Center-distance / mean-radius: median 2.148.
- Full-ball all-intersection: False; maximum overlap 4/8.
- 0.8r all-intersection: False; maximum overlap 4/8.

The intermediate dataset instead has 44/44 centers at `(0.625,0,0.375)` with radii 0.23375–0.35362 and a universal retained-region intersection. The completed t0 centers have normalized distances 0–0.407583 from that common center, and six t0 radii are below 0.20. Thus t0 geometry is narrower and more state-dependent.

## Q64 table

| state | zero | common eta | LOWJ | CENTER | MARGIN |
|---|---:|---:|---:|---:|---:|
| T0_WIDE_perm00_ep0082 | 46/64 | 55/64 | 0/64 | 64/64 B63 | 0/64 |
| T0_WIDE_perm01_ep0217 | 64/64 B63 | 64/64 B63 | 64/64 B63 | 64/64 B63 | 64/64 B63 |
| T0_WIDE_perm02_ep0182 | 0/64 | 64/64 B63 | 64/64 B63 | 0/64 | 64/64 B63 |
| T0_WIDE_perm03_ep0205 | 47/64 | 64/64 B63 | 64/64 B63 | 0/64 | 7/64 |
| T0_WIDE_perm04_ep0195 | 64/64 B63 | 45/64 | 40/64 | 64/64 B63 | 0/64 |
| T0_WIDE_perm05_ep0179 | 51/64 | 59/64 | 0/64 | 64/64 B63 | 64/64 B63 |
| T0_WIDE_perm06_ep0074 | 59/64 | 61/64 | 30/64 | 64/64 B63 | 64/64 B63 |
| T0_WIDE_perm07_ep0139 | 34/64 | 64/64 B63 | 64/64 B63 | 64/64 B63 | 64/64 B63 |


- ZERO_B63: 2/8.
- Intermediate common eta B63: 4/8; it is **not** universally robust at t0.
- G_LOWJ: 4/8 B63, mean Q64 0.637.
- G_CENTER: 6/8 B63, mean Q64 0.750.
- G_MARGIN: 5/8 B63, mean Q64 0.639.
- Collisions and numerical failures: 0.

Safety-equivalent zero is B63 while CENTER fails in 0 states and while MARGIN fails in 1 state. Conversely, zero is non-B63 while LOWJ/CENTER/MARGIN rescue 3/4/4 states. The only direct audited Safety-to-learned break is MARGIN on ep0195; no such CENTER case appears in this n=8 cohort.

## Rho and causal interpretation

| controller | inside ball | B63 outside ball | mean rho | Spearman(rho,Q64) |
|---|---:|---:|---:|---:|
| G_LOWJ | 0/8 | 4/8 | 15.793 | -0.613 |
| G_CENTER | 0/8 | 6/8 | 9.147 | 0.126 |
| G_MARGIN | 0/8 | 5/8 | 8.590 | -0.866 |


Large rho sometimes coincides with failure, but it is not generally sufficient: every prediction is outside and most CENTER/MARGIN predictions remain B63. The verified balls certify small inner regions, not the full t0 success sets. Therefore the completed experiment supports geometric phase mismatch but does **not** establish it as the dominant causal explanation for the 300-episode fresh-WIDE deficit.

## Explicit answers

- Is the old universal intermediate eta robust at true t0? **No.** It is B63 on only 4/8 states.
- Do intermediate-trained controllers fail when badly mismatched? **Some do, but not uniquely:** all predictions are geometrically mismatched, while 15/24 remain B63.
- Does phase mismatch explain CENTER/MARGIN falling below Safety? **Not causally from this audit.** It plausibly contributes, especially the MARGIN break at ep0195, but the conservative balls are too narrow for rho to distinguish success from failure.

## Smallest next scientific step

On these same eight states only, perform a sparse model-free outward coverage audit around the frozen controller eta values to determine whether their successful points belong to broad t0 success regions outside the conservative balls. Do not train a t0 learner until the target-set coverage issue is resolved.

## Runtime

Reused exact tuple inventory: 17,904; new continuations: 4,400; new physical steps: 2,082,469; wall time 23.1 min; max 6 GPU shards; max 12 CPU threads; 48 GiB allocated RAM ceiling. GPU memory and observed RAM peak were unavailable because Slurm accounting is disabled.
