# Shared conservative success-set family protocol

The scientific object is a nontrivial retained analytic subset of the exact-Q64
robust set, not a reconstruction of the complete Basin.  The mathematical form,
fitting algorithm, global hyperparameters, and erosion rule are shared across
ToyGiveWay_2A and DoubleBottleneck_4A; only continuous per-instance parameters
may vary.  No controller, critic, implicit classifier, or parameter predictor is
trained.

The authoritative Q64 inventory is deduplicated by exact float64 `(state,eta)`.
The 12 adequately sampled states are the frozen eight Toy true-t0 dense states
and four compatible fixed-current Double-Bottleneck true-t0 states.  Three
state-level outer folds each contain both scenarios.  For a held-out state,
theta is fit by the same frozen limited-oracle procedure from nested 16/24/32
exact labels; the remaining exact eta outcomes are evaluation evidence.  Global
family choices are made only from the development states of each outer fold.

The initial label-blind global-maximin simulator was rejected before rollout:
it selected 15 negatives and only one B63 witness among the first 16 labels on
every Double-Bottleneck state.  That cannot test a set geometry from a robust
anchor.  The frozen V1 acquisition simulator uses the same sequential rule in
both scenarios: one known robust anchor, nearest/farthest known boundary
negatives, then two nearest-to-observed-robust queries followed by one global
maximin query.  Fixed provenance bonuses apply to cross-transfer, independent
interpolation, and conditional-extent candidates.  V0 files remain archived.

Retained sets use one globally selected homothetic erosion factor about a robust
interior anchor.  This gives a shared, explicit erosion for curved bodies and
cuts.  Candidate selection is lexicographic: zero fit-time retained negatives,
then robust-witness and cross-transfer coverage, then retained extent, then
lower complexity.  Full-Basin recall is diagnostic only.

After offline state-level CV, only a frozen candidate meeting cached reliability,
size, moderate recall, multimodal support, and both-scenario usability proceeds
to six new exact-Q64 retained points per usable state.  No new rollouts are used
to rescue a failed family.  Fresh retained precision must be 64-seed exact and
zero-false separately in both scenarios.
