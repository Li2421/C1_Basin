# Fixed-cadence G_phi closed-loop ablation

This is a causal ablation of correction frequency, not a proposed final controller. It uses the frozen checkpoint, frozen controller stack, and exact first 128 matched pilot episodes. Safety and H=1 are reused byte-for-byte from the pilot; no G_phi retraining, gate, eta search, success-basin search, or online oracle rollout was used.

| condition | success | deadlock | timeout | collision | wall collision | other | Q (Wilson 95% CI) | mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 128 | 0 | 0 | 0 | 0 | 0 | 1.000000 [0.970863, 1.000000] | 0 |
| H=1 | 79 | 1 | 48 | 0 | 0 | 0 | 0.617188 [0.530732, 0.696814] | 0.160329 |
| H=4 | 128 | 0 | 0 | 0 | 0 | 0 | 1.000000 [0.970863, 1.000000] | 0.0272677 |
| H=8 | 128 | 0 | 0 | 0 | 0 | 0 | 1.000000 [0.970863, 1.000000] | 0.0153997 |
| H=16 | 128 | 0 | 0 | 0 | 0 | 0 | 1.000000 [0.970863, 1.000000] | 0.00849018 |

## Paired outcome changes versus H=1

- H=4: H=1 fail -> success 49; H=1 success -> fail 0; Delta Q=+0.382812, paired bootstrap 95% CI [+0.296875, +0.468750], exact McNemar p=3.55271e-15.
- H=8: H=1 fail -> success 49; H=1 success -> fail 0; Delta Q=+0.382812, paired bootstrap 95% CI [+0.296875, +0.468750], exact McNemar p=3.55271e-15.
- H=16: H=1 fail -> success 49; H=1 success -> fail 0; Delta Q=+0.382812, paired bootstrap 95% CI [+0.296875, +0.468750], exact McNemar p=3.55271e-15.

## Diagnosis

- Highest Q: H=4, H=8, H=16 (Q=1.000000).
- Lowest mean J_def while preserving Q within 0.020 of the best: H=16 (Q=1.000000, mean J_def=0.00849018).
- Q is monotonic nondecreasing as cadence becomes sparser (H=1/4/8/16: 0.617188/1.000000/1.000000/1.000000).
- Timeout counts H=1/4/8/16: 48/0/0/0.
- Relative to H=1, the deformation reduction for H=16 is 0.0243328 in steps 0-40 and 0.127506 in steps >=41; the larger contribution is later persistent warm/recovery correction.
- Scheduled queries fall by 94.7% for H=16 versus H=1. A count-only scaling using H=1 per-query executed-correction energy predicts mean J_def=0.00857346, versus 0.00849018 observed (ratio 0.990); reduced query count is the dominant deformation mechanism.
- Per-scheduled-query deformation energy is 0.000237225 at H=1 and 0.000234921 at H=16.
- Per-query projection behavior does change materially under the declared 25% relative-mean criterion. Thus the stronger claim that projection behavior is invariant across H is not supported. Mean rewrite norm/fraction above tolerance change from 0.0298404/0.619 at H=1 to 0.00987126/0.382 at H=16.
- Visited-state OOD changes materially across cadence conditions; remaining failures are not strongly OOD-associated by the secondary criterion.
- Final classification: **PERSISTENT_OVERCORRECTION_CONFIRMED**.
- H=32 should not be tested under the preregistered rule; Q plateaued before H=16 rather than continuing to improve.
- Smallest justified next experiment: Rerun one H=16 condition on the same 128 episodes with phase offset 8 (apply when t mod 16 = 8) to test whether the result is cadence-density or t=0 phase specific; change no other component.

## Integrity and runtime

Sanity status: **PASS**. Checkpoint SHA256: `c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e`. The output cohort contains 128 exact seed/IC records. Analysis started zero rollouts. New runner process records: 6; observed physical GPU identifiers: 0; allocation: 6 GPU shards, 12 CPU cores, 80G memory; summed new episode compute time: 307.732 s; parallel rollout wall time approximately 56.355 s. At job start the GPU reported 2 MiB used and 0% utilization; the pre-submit scheduler queue was empty.
