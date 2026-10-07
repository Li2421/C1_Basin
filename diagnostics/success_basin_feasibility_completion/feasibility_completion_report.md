# D2/D4 provisional 63/64 feasibility completion

## Result

Both states now have a non-empty empirical `B_63` under the provisional matched-64 rule. No targeted local search was needed: all 160 newly added, previously missing tuples succeeded. Existing tuples were reused and no tuple was rerun.

### B_63

- D2: (0.40625, -0.5, 0.0), (0.4375, -0.53125, 0.0), (1.0, 0.0, 0.25)
- D4: (0.375, -0.4375, -0.0625), (0.40625, -0.4375, -0.0625), (1.0, 0.0, 0.25)

| state | eta | success | mean J_def (successful runs) | std | mean steps | E_near |
|---|---|---:|---:|---:|---:|---:|
| D2 | (0.40625, -0.5, 0.0) | 64/64 | 0.206189 | 0.012316 | 287.20 | yes |
| D2 | (0.4375, -0.53125, 0.0) | 64/64 | 0.208330 | 0.011357 | 271.77 | yes |
| D2 | (1.0, 0.0, 0.25) | 64/64 | 0.476949 | 0.011197 | 223.81 | no |
| D4 | (0.375, -0.4375, -0.0625) | 64/64 | 0.194341 | 0.013095 | 321.16 | yes |
| D4 | (0.40625, -0.4375, -0.0625) | 64/64 | 0.197543 | 0.013713 | 288.53 | yes |
| D4 | (1.0, 0.0, 0.25) | 64/64 | 0.474817 | 0.009824 | 224.02 | no |

The strict empirical minima are D2 `(0.40625, -0.5, 0.0)` with mean `J_def=0.206189`, and D4 `(0.375, -0.4375, -0.0625)` with mean `J_def=0.194341`. In each state the adjacent low-cost candidate is statistically indistinguishable by the predeclared paired 95% CI, so each `E_near` contains two eta values. The anchor is feasible but clearly not near-optimal.

## First-step executed-correction stability

The audited supervision quantity is the four-dimensional joint correction `g_exec,t = u_exec,t - u_safe,t` after the second hard projection, with the same state and Flow seed held fixed.

| state | classification | E_near size | mean same-seed distance | max | mean/vmax | mean/typical correction | eta variation/Flow variation |
|---|---|---:|---:|---:|---:|---:|---:|
| D2 | LABEL_MILDLY_AMBIGUOUS | 2 | 0.00240393 | 0.00257297 | 0.00481 | 0.06917 | 0.83322 |
| D4 | LABEL_MILDLY_AMBIGUOUS | 2 | 0.00193958 | 0.00193958 | 0.00388 | 0.05991 | 0.79471 |

Thus deterministic `G_phi(z, xi) -> g_t` first-step supervision is supported for D2 and D4 at the resolution of this diagnostic. This claim is local to the sampled `E_near`, the provisional 63/64 rule, and the current diagnostic family; it is not a claim of a unique global eta.

## Integrity and runtime

- New rollouts: 160 (96 screening/completion stage 1 + 64 completion stage 2), all successful; CPU batches ran D2/D4 concurrently.
- Frozen corrector, environment, both hard-projection implementation inputs, retry solver, checkpoint, and prior directional manifest hashes agree before/after.
- Maximum new-run J_def reconstruction error: 0.000e+00.
- Maximum corrector reconstruction error: 0.000e+00; maximum second-projection replay error: 0.000e+00.
- Same-state/same-seed first-step `u_safe` mismatch across near-optimal eta: 0.000e+00.

The 63/64 rule remains provisional and is not a confidence-calibrated theorem.
