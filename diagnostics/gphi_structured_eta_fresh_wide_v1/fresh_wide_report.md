# Structured eta fresh WIDE full-episode audit

## Frozen protocol

- Manifest frozen before rollout: `2026-09-25T16:04:35.751414+00:00`; file SHA256 `274a1c5968c9fe957d58575c0d388099c6707b90a6e421c835378e67d9f95b67`.
- Authoritative WIDE generator: `numpy.default_rng(seed); abs_x~U(0.55,1.05),(N,2); y~U(-0.025,0.025),(N,2); signs [-,+]; final float32` with IC root `2026092601` and Flow root `2026092602`.
- Fresh overlap audit: **PASS**, exact overlap **0** against 1606 prior records (1108 unique ICs).
- Safety, direct-g H8 one-step, and one-shot persistent structured eta used identical ICs and Flow streams.
- No training, eta search, oracle query, gate, cadence/model selection, or post-hoc replacement was performed.

## Primary outcome (N=200)

| Controller | Success | Strict deadlock | Timeout | Collision | Q (95% Wilson CI) | Mean J_def | Median completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safety | 143 | 7 | 50 | 0 | 0.715 [0.649, 0.773] | 0.0000 | 28.55 |
| Direct-g H8 | 188 | 0 | 12 | 0 | 0.940 [0.898, 0.965] | 0.0242 | 27.45 |
| Structured eta | 90 | 100 | 10 | 0 | 0.450 [0.383, 0.519] | 0.2383 | 19.05 |

## Safety-referenced efficacy

| Learned controller | Rescue | Rescue rate | Break | Break rate | Net rescue | Delta Q (paired 95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct-g H8 | 51 | 89.5% | 6 | 4.2% | 45 | +0.225 [+0.160, +0.295] | 5.68e-10 |
| Structured eta | 33 | 57.9% | 86 | 60.1% | -53 | -0.265 [-0.365, -0.165] | 1.27e-06 |

## Failure-mode rescue

| Method | Safety failure | Count | Rescued | Rescue rate |
|---|---|---:|---:|---:|
| Direct-g H8 | timeout | 50 | 50 | 100.0% |
| Direct-g H8 | deadlock | 7 | 1 | 14.3% |
| Structured eta | timeout | 50 | 28 | 56.0% |
| Structured eta | deadlock | 7 | 5 | 71.4% |


## Direct-g versus structured eta

- BOTH_SUCCESS: 81; ETA_ONLY_SUCCESS: 9; DIRECT_ONLY_SUCCESS: 107; BOTH_FAIL: 3.
- Paired Delta Q (eta - direct): -0.490, 95% paired bootstrap CI [-0.570, -0.410].
- Structured-eta nominal break rate among Safety successes: 60.1%; direct-g: 4.2%.

## One-shot eta and safety

- Eta clipping: 41/200 (20.5%); near-zero (norm <= 0.05): 0/200.
- Eta norm mean/median/P95: 0.612 / 0.563 / 1.195.
- Eta norm by outcome (mean): success 0.709, deadlock 0.535, timeout 0.502. Successful episodes tended to receive larger modes; this is descriptive, not a tuned threshold.
- Hard safety intact: **True**. No collision, invalid action, NaN/Inf, or projection solver failure was accepted.

## Conclusion

**STRUCTURED_ETA_STRICT_ONLY**

The frozen test result is reported without model or protocol changes. Detailed paired cells, deformation by outcome, completion-time comparisons, eta-outcome associations, and integrity checks are in the machine-readable artifacts.
