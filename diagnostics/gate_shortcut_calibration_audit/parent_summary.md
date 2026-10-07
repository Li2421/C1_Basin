# Gate shortcut and calibration audit — parent synthesis

## Decision

**BOTH_SHORTCUT_AND_CALIBRATION**

The three independent diagnostics agree that the hard OOF failures are not caused by missing 214-D information or Flow-variant noise. They arise from two related but empirically distinct source-shift effects:

1. local multi-group shortcuts raise unseen zero states to intervention-like absolute logits;
2. fold/source-dependent, class-dependent score calibration makes validation thresholds non-transferable.

## Pair 1: apparent ranking reversal

The reported OOF margin compares two different valid fold models. Its `-1.302027` logit reversal is therefore not evidence that one common gate ranks the clouds backwards.

- Symmetric same-model feature/path effect: `+0.611240` (correct direction).
- Cross-fold model/score-frame effect: `-1.913268`.
- D2 gate scores zero/nonzero as `2.380337 / 3.489386` (`+1.109050`, correct).
- D4 gate scores zero/nonzero as `0.964878 / 1.078309` (`+0.113431`, correct).
- The D4-minus-D2 common calibration offset is `-1.650` logits and explains 86.3% of the cross-fold offset with the correct sign.

The zero state's abnormally high score remains a genuine local error. Relative to the train-mean reference, the largest wrong-direction group contributions are:

| Existing feature group | Mean logit contribution |
|---|---:|
| geometry / observation | -0.733323 |
| control / projection | -0.295081 |
| history / monitor | -0.210197 |
| `u_safe` | -0.184390 |
| inter-agent relative | -0.153746 |
| goal-relative | -0.153734 |

Episode/time contributes `+0.485480` in the correct direction and `u_Flow` contributes `+0.018882`.

Conclusion: Pair 1's numerical cross-fold reversal is predominantly a calibration-frame artifact, while its zero-state false positive is supported by a **MULTI_GROUP_SHORTCUT**.

## Pair 2 and control: threshold failure

Pair 2 preserves ordering: zero/nonzero means are `1.104 / 1.665` logits, margin `+0.562`, AUC `1.0`. Its fold thresholds are `0.480 / 1.338`; the local midpoint is `1.384`, placing the zero-fold threshold `0.905` logits below that diagnostic midpoint.

The same-model control pair has logits `2.401 / 3.212`, margin `+0.810`, and AUC `0.999268`, but its validation threshold is `1.388` versus midpoint `2.806`, a `+1.419` displacement. This confirms threshold transfer failure even without cross-model ranking ambiguity.

Drift is **class-dependent**, not a universal additive or scale correction:

- D2 zero/nonzero shifts: `+2.490 / -0.142` logits;
- D4: `-1.067 / +0.114`;
- qual226: `-0.240 / -0.014` (the closest to additive);
- D1: `-3.042 / -0.253`.

A global affine diagnostic (`test_logit ≈ -0.276 + 0.966·validation_logit`) has RMSE `1.419` and changes diagnostic errors from 14 raw to 16; additive-only changes them to 17. No universal scalar recalibration is supported.

## Source-shortcut evidence

Evidence is **moderate**, local, and asymmetric:

- source decoding exceeds 1/12 chance in every block: history `0.407`, time `0.360`, geometry `0.233`, projection `0.192`, current control `0.186`;
- geometry, current-control, and projection dimensions show the most meaningful cross-group label-correlation sign changes;
- all three robust false-positive zero states are closer to nonzero training centroids in every semantic block and full 214-D space;
- all three also inherit intervention-like OOF scores, while false-negative nonzero cases do not follow one uniform source-majority pattern.

Attribution and shortcut diagnostics agree on geometry, history, and control/projection. History also carries strong transferable label information, so source decodability does not make the whole block intrinsically spurious.

## Mechanism and next experiment

One broad cause—source-group distribution shift—links both failures, but the manifestations differ. Pair 1 combines local shortcut-driven zero-score inflation with a large cross-fold score-frame offset. Pair 2 retains correct local ranking but fails because class-dependent calibration shifts the threshold frame.

The single smallest justified training experiment is: **repeat the unchanged 214→64→64→1 gate, BCE, labels, and OOF protocol with training batches weighted equally by independent source group, changing nothing else**. Compare the same fixed OOF states for absolute-logit stability, threshold transfer, and shortcut-sensitive false positives. This isolates whether unequal source correlations are driving both effects before considering architecture or feature changes.

No frozen-checkpoint masking ablation was run: the three independent audits already converged, and synthetic masked inputs were unnecessary for the mechanism decision.
