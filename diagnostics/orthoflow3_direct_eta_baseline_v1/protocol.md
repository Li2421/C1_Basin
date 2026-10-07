# OrthoFlow3 Direct-eta baseline v1 protocol

## Scope

This diagnostic trains only a deterministic `G(h) -> eta` baseline. It does
not train or use Q, J, Direct-g, a zero/active gate, a mixture/flow, or any
recovery controller.  The frozen OrthoFlow3 implementation is
`diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py`, SHA256
`51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.

## Frozen population and targets

The experiment reuses the exact Q-v2 source-group-held-out 80/20/20 split,
its exact 214-D conditioned deployment features, and its exact 24-point eta
cloud (zero plus the first 23 authoritative Sobol points).  Source groups do
not cross splits.

For every state, eta=0 is extended to 64 matched futures.  If it reaches B63,
the target is exactly zero. Otherwise, nonzero cloud candidates are ordered
without the learned Q network by decreasing existing empirical success,
then increasing mean successful full-horizon J_def, then frozen cloud index.
Candidates are promoted to 64 matched futures until two B63 candidates are
found or the cloud is exhausted. The active target is the confirmed B63
candidate with lowest mean successful J_def. This target is named
`eta*_24-robust-lowJ`; it is not claimed to be the global continuous canonical
eta.

## Model and training

The sole learned model is `214 -> 128 -> 128 -> 3` with SiLU hidden
activations and a smooth bounded eta output. Inputs use TRAIN-only feature
normalization. Eta error uses authoritative coordinate scales. The loss is
only normalized eta MSE, with no rebalancing or auxiliary loss. Seeds are
17, 23, and 41; validation MSE alone selects the checkpoint and early stop.

## Frozen evaluation

After checkpoint freeze, every resolved TEST prediction is evaluated with
64 matched fixed-eta OrthoFlow3 continuations, without snapping, search, Q,
or J. Zero-target and active-target outcomes are reported separately. Only
after these outcomes are frozen may Q-v2 be loaded read-only as a diagnostic.

A fresh outcome-blind 200-episode WIDE cohort is frozen before outcomes. Each
episode compares matched Safety-only and Direct-eta. Direct-eta computes the
deployment feature at step 0, predicts eta once, latches it for the episode,
recomputes OrthoFlow3 fields each physical step, and retains the authoritative
second safety projection. No cadence, gate, entry/exit, or online search is
used.

## Resources and stopping rules

Exact compatible continuations are reused. Automatic execution is capped at
15,000 new continuations, 8,000,000 physical steps, and approximately 60
minutes rollout wall time. Six GPU shards are permitted only after confirming
the server is essentially idle; no shard allocation is changed inside a live
batch. Training uses one GPU shard.
