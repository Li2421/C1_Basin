# Target construction semantics

The deterministic regression target is `eta*_24-robust-lowJ`.

For each state from the frozen Q-v2 80/20/20 source-group split, the exact
Q-v2 conditioned state and its fixed 24-point eta cloud are used. Eta zero is
first evaluated with 64 matched futures. When it passes B63 (at least 63/64
successes), the target is zero because its authoritative deformation cost is
zero.

Otherwise, nonzero candidates are ordered using only real Q-v2 rollout
evidence: empirical success fraction descending, mean successful
full-horizon J_def ascending, and frozen eta-cloud index ascending. Candidate
rollouts are extended to the same 64 futures. Promotion continues until two
distinct B63 candidates are confirmed or all 23 nonzero cloud points are
exhausted. The target is the confirmed robust candidate with the smaller mean
successful full-horizon J_def. A sole confirmed candidate is retained and
explicitly marked `TARGET_SINGLE_ROBUST`; no confirmed candidate is marked
`TARGET_UNRESOLVED`.

The learned Q network is not consulted for target generation, promotion,
ordering, or training. Because the search is restricted to a frozen generic
24-point cloud, this target is not an exact global continuous canonical eta.
