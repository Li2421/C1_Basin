# Startup / warm Pareto checkpoint audit

## Result

**CHECKPOINT_SELECTION_PROBLEM**

The original early-stopping run ended at epochs 190--220 and selected epochs
25/45/15. Continuing the unchanged unified optimization to the pre-existing
maximum of 1200 epochs exposed substantially better joint validation
checkpoints. In particular, seed 23 epoch 1171 improves both validation
slices relative to the previously selected startup-complete checkpoint:

| Checkpoint | Startup validation mean L2 | Warm validation mean L2 | All validation mean L2 |
|---|---:|---:|---:|
| Previous seed 17, epoch 25 | 0.079695 | 0.070070 | 0.072520 |
| New seed 23, epoch 1171 | 0.075760 | 0.032243 | 0.043320 |
| Old V3 reference | N/A | 0.034798 | N/A |

Thus the new balanced checkpoint reduces startup validation error by 4.94%
and warm validation error by 53.99% relative to the previous merged-model
checkpoint. Its warm error is also 7.34% below the fixed old-V3 reference.
Seed 23 epoch 1156 gives the minimum warm error among checkpoints that still
improve startup over the previous checkpoint: startup `0.076564`, warm
`0.032001`, so there is no warm degradation (an 8.04% improvement over V3).

The paired state bootstrap is appropriately cautious: the 14-state startup
slice is small, and the validation startup improvement CI crosses zero. On
the held-out startup test slice, the new primary is `0.088858` versus
`0.079115` for the previous checkpoint, also with a CI spanning zero. The
audit therefore demonstrates removal of the catastrophic warm regression,
not a statistically proven startup improvement.

## Frozen cohorts

| Split | Startup states / samples | Warm states / samples |
|---|---:|---:|
| Train | 91 / 5,824 | 203 / 12,992 |
| Validation | 14 / 896 | 41 / 2,624 |
| Test | 14 / 896 | 50 / 3,200 |

Warm means the exact immutable V3 prefix, not a timestep-derived proxy. All
states have 64 Flow variants. Startup and warm are evaluation slices only;
training used one unified sample-uniform dataset and ordinary MSE.

## Epoch minima and Pareto selections

| Seed | Best startup epoch / error | Best warm epoch / error | Best aggregate epoch / error | Balanced Pareto epoch | Pareto points |
|---:|---:|---:|---:|---:|---:|
| 17 | 25 / 0.079361 | 698 / 0.035788 | 884 / 0.050589 | 884 | 15 |
| 23 | 1171 / 0.075760 | 1156 / 0.032001 | 1171 / 0.043320 | 1171 | 2 |
| 41 | 1195 / 0.072308 | 757 / 0.035593 | 1195 / 0.046215 | 1176 | 9 |

The pre-registered validation-only selection minimizes worst multiplicative
regret to each cohort optimum over the Pareto frontier. Test labels and the
old V3 reference do not enter selection.

## Matched test results

| Validation-selected checkpoint | Startup test mean L2 | Warm test mean L2 | All test mean L2 |
|---|---:|---:|---:|
| Seed 17, epoch 884 | 0.082764 | 0.015116 | 0.029914 |
| Seed 23, epoch 1171 (primary) | 0.088858 | 0.012075 | 0.028871 |
| Seed 41, epoch 1176 | 0.087393 | 0.014765 | 0.030653 |
| Previous startup-complete | 0.079115 | 0.047916 | 0.054741 |
| Old V3 reference | N/A | 0.015413 | N/A |

The primary checkpoint restores warm-history capability and is better than
old V3 on the matched warm test. Startup generalization remains noisy but is
finite and of the same order as the previous model. A paired state bootstrap
for primary-minus-reference gives:

- startup validation vs previous: `-0.003893`, 95% CI
  `[-0.038907, 0.033202]`;
- warm validation vs old V3: `-0.002552`, 95% CI
  `[-0.007118, 0.001940]`;
- startup test vs previous: `+0.009743`, 95% CI
  `[-0.011320, 0.034810]`;
- warm test vs old V3: `-0.003338`, 95% CI
  `[-0.006237, -0.000583]`.

## Integrity and readiness

All 3,600 epoch checkpoints were saved. Frozen data, split, normalization,
architecture, optimizer, learning rate, weight decay, MSE, and minibatch rule
were unchanged. The old selected epochs reproduced within 0.23% validation
MSE (GPU multi-process floating-point execution was not bitwise identical).
No closed-loop rollout was run.

Offline projection replay of the primary checkpoint had zero solver failures.
Test executed-action mean error was `0.026815`, and mean second-projection
rewrite was `0.004703`.

The primary seed-23 epoch-1171 checkpoint is
**READY_FOR_FULL_EPISODE_PILOT**. Readiness means the prior warm-retention
blocker is removed; it does not claim that startup accuracy has improved.

The single next experiment is the already specified fresh, matched
Safety-versus-learned full-episode pilot using the frozen seed-23 epoch-1171
checkpoint, without online oracle, eta search, or a gate.
