# Eight-mode joint MACFlow pilot diagnosis

## Dataset result

All eight centralized-expert hypotheses were solved from each of the twelve
pilot initial Markov states.  Diversity was accepted only from executed
bottleneck-crossing events, not from hypothesis labels.

- 96/96 successful expert rollouts.
- 0 collisions, deadlocks, or timeouts.
- 70,502 transitions.
- Eight distinct crossing signatures, exactly 12 trajectories per signature.
- Every one of the 12 identical-start groups realizes all eight signatures.
- Family-grouped split: 72 train episodes and 24 validation episodes.  All
  eight trajectories from one initial state remain in the same split.

The source dataset and two-mode dataset have the same fixed-environment
fingerprint.  The earlier dataset and checkpoints were not overwritten.

## Training result

The policy is the Toy-source-equivalent MACFlow Stage-I joint model.  Training
used uniform transition sampling, seed 0, batch size 256, and 2,000 updates.

- Fixed train loss: `1.99768 -> 0.19599`.
- Fixed validation loss: `1.98817 -> 0.19608`.
- Best step: 2,000.
- Exact checkpoint reload difference: `0.0`.
- Teacher-forced mean sampled-action RMSE: `0.01928`.
- Teacher-forced best-of-8 action RMSE: `0.01183`.

## Raw closed-loop result

The evaluator follows Toy's RNG protocol: it folds the absolute step into the
episode key and therefore draws a fresh flow noise sample at every control
step.  Twelve raw, speed-bounded rollouts used four seeds for each of the three
initial-condition regimes.

- Success: `0/12`.
- Wall collision: `12/12`.
- Agent collision, deadlock, timeout: `0/12` each.
- Mean terminal step: `40.33`, versus `75.17` for the earlier two-mode pilot.
- Minimum wall clearance: `-0.01658`.
- Closed-loop action total variation: `0.02413`, versus expert `0.00329`.
- No successful coordination signature was completed.

## Interpretation

Training all eight modes is scientifically justified—the physical task really
does realize all eight—but it does not fix the raw closed loop.  With eight
valid actions or action phases compatible with nearby observations, independent
per-step flow sampling creates a stronger risk of temporal mode switching.  The
present measurements establish the failure, but do not by themselves prove
that mode switching is its sole cause.

Gate B and dataset Gate C pass for the eight-mode pilot.  Gate D remains
failed.  A next diagnostic should test temporal mode consistency explicitly
(for example an episode-level latent or short joint action-horizon model)
against the unchanged Toy-style per-step sampler.  This should be treated as a
model diagnostic, not as an eta/basin experiment.  No eta experiment or
`G_phi` training was performed.
