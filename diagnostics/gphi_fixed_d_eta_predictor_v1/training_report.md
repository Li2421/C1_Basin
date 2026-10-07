# Fixed-D structured eta training

A single 214 -> 128 -> 128 -> 3 SiLU network (44,419 parameters) was trained from scratch with normalized eta MSE on the exact 27,136-sample coverage dataset.  The direct-g comparator has 44,548 parameters.  No state, target, split, weighting, or optimizer setting was changed.

## Validation-only selection

- Seed: 17
- Epoch: 777
- Checkpoint: `/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz`
- SHA256: `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`
- Startup validation normalized-eta mean L2: 0.10464021
- Warm validation normalized-eta mean L2: 0.12638554
- Aggregate validation normalized-eta mean L2: 0.12085036

The checkpoint was frozen before test, historical-11, fresh-6, or closed-loop evaluation.
