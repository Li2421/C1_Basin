# Exact Q conditioning semantics

## Code-level resolution

`FeatureBuilder.build` in
`diagnostics/gphi_training_dataset_v2/finalize_dataset.py:219` consumes the
query-step `u_flow` and `u_safe`. They occupy feature offsets 40--43 and 44--47;
the feature also contains the first-projection delta and constraint activity.
Consequently the 214-D input is genuinely `h(z, xi_0)`, not a state-only `h(z)`.

The historical oracle runner in
`diagnostics/gphi_training_dataset_v2/adaptive_run.py:82-110` derives a Flow key
from each continuation seed and uses it from the first queried physical step.
Aggregating those seeds therefore integrates over different `xi_0` values and
estimates `Q(z, eta)`. Such aggregate labels cannot be duplicated across the 64
Flow-conditioned h variants.

The repository already defines the required conditional replay mechanism in
`diagnostics/recovery_entry_identifiability_v1/run_paired_branches.py:73-88`:
reuse the saved current Flow key at the decision step, then use a separately
folded future stream. The authoritative step kernel consumes the supplied key
directly at `diagnostics/single_segment_recovery_training_v1/state_machine.py:209-229`.
The present runner uses exactly this separation.

## Randomness decomposition

1. Fixed by h: complete augmented state, current observation/history/monitor,
   query-step Flow key `xi_0`, resulting `u_flow`, `u_safe`, and first-projection
   diagnostics.
2. Fixed by candidate: eta, held constant for the full continuation.
3. Resampled across trials: Flow keys only after the first executed transition.
4. Matched within a state/future index: all eta values see the same future Flow
   key schedule.

The query-step key is reconstructed exactly as the original dataset did:
`fold_in(fold_in(PRNGKey(flow_seed), rng_namespace), absolute_step)`. Feature
replay is a mandatory runtime assertion at tolerance 1e-10. The smoke test gave
maximum absolute replay error 2.22e-16.

## Archived evidence and variants

The 424-state manifest contains 27,136 feature samples, normally 64 variants per
state. Multiple variants from one physical state may not share a binomial Q
label because they condition on different query-step Flow draws. This audit
selects one variant per selected state before outcomes and estimates its own
conditional Q with future-only resampling.

Eleven historical strict-deadlock states have feature variants but no retained
`rng_namespace`; their exact current key cannot be proven from the current
authoritative metadata. They are excluded rather than assigned an invented key.

Conclusion: `Q_CONDITIONING_SEMANTICS_AMBIGUOUS` is **not** triggered. The
mathematical target is exact and auditable, while incompatible historical
state-level Q aggregates are explicitly rejected from label reuse.
