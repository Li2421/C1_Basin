# Double-Bottleneck joint MACFlow pilot diagnosis

## Scope

This is the authoritative pilot for the Toy-source-equivalent four-agent
MACFlow Stage-I behavior-cloning baseline.  It is a fixed-geometry, true joint
policy over the canonical order `(A1, A2, B1, B2)`; it does not use the old two
frozen-dyad compatibility controller.

The policy consumes the flattened four-agent observation `(4, 18) -> 72` and
samples the full joint action `(4, 2) -> 8`.  It directly reuses the same
official MACFlow `ActorVectorField`, `ModuleDict`, `TrainState`, conditional
flow-matching loss, Adam update, and 10-step Euler sampler as Toy Give-Way.
Training examples are sampled uniformly over transitions, matching Toy's
`GiveWayDataset.sample`.  The only policy-dimensional extension is observation
`20 -> 72` and action `4 -> 8`.

## Training result

- Dataset: 24 validated expert episodes, with 18 train and 6 validation
  episodes (12,907 and 4,303 transitions).
- Parameters: 154,632; hidden dimensions `(256, 256, 256)`.
- Fixed train loss: `2.01111 -> 0.18097`.
- Fixed validation loss: `1.96742 -> 0.17909`.
- Best checkpoint: step 2,000; exact reload difference `0.0`.
- Device: CPU; training time: 2.57 s.

## Raw closed-loop result

Twelve unprojected pilot rollouts were evaluated across clearly asymmetric,
weakly asymmetric, and near-symmetric regimes with four sampling seeds per
regime.

- Success: `0/12`.
- Wall collision: `12/12`.
- Agent collision, deadlock, timeout: `0/12` each.
- Mean termination step: `75.17` (3.76 s at `dt=0.05`).
- Minimum observed pairwise surface distance: `0.1969`.
- Minimum wall clearance: `-0.00642`.
- Teacher-forced mean sampled-action RMSE: `0.01904`.
- Teacher-forced best-of-8 action RMSE: `0.01167`.
- Closed-loop action total variation: `0.02693`, versus expert `0.00337`
  (about 8.0 times larger).
- Mean nearest-expert-phase state RMSE: `2.6372`.

## Interpretation and gate decision

The implementation/source-parity question is resolved: this pilot uses the
same MACFlow primitives and Stage-I algorithm as Toy Give-Way.  Source parity
does not by itself make the learned closed loop valid.  Small action errors and
resampling variation compound rapidly near narrow walls, and all raw rollouts
leave the valid corridor before exhibiting a successful coordination mode.

Therefore Gate D is **not passed**.  This checkpoint is useful as a diagnostic
pilot, but it is not yet a validated scientific four-agent Flow-BC baseline.
No safety-baseline or eta/basin conclusion should be drawn from it, and no
`G_phi` training was performed.

Authoritative machine-readable artifacts are `training_summary.json` and
`evaluation/summary.json`; the latter also records the checkpoint hash and all
per-rollout terminal data.
