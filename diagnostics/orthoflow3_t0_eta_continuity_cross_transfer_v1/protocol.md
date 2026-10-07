# OrthoFlow3 true-t0 eta continuity and cross-transfer protocol

Frozen inputs: 40 source-isolated true-t0 states from `orthoflow3_true_t0_point_learning_v1`; authoritative OrthoFlow3 SHA256 `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.

Distances use the exact TRAIN-frozen 214-D normalization and frozen normalized eta geometry. Target jumps are frozen at `d_h <= 5.50094204` (25th percentile of 1-NN distances) and `d_eta >= 0.893650445` (75th percentile of rank-1/2/3/5 neighbor displacements). Forty unordered pairs are frozen before rollout and tested symmetrically, totaling at most 80 new directed Q64 transfers after exact cache reuse. No model training or basin reconstruction is permitted.
