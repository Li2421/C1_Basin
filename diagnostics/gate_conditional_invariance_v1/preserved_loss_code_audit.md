# Preserved BCE and conditional-invariance audit

The source-balanced BCE loss line remains in
`conditional_invariance_gate_runner.py` under the explicit marker
`BASELINE_BCE_PRESERVED`; it is commented, not deleted.  The
`source_balanced_bce` / lambda=0 branch executes the same original ordinary
per-example BCE after the frozen source-group→state→Flow-variant sampler.

For positive lambda, each full training epoch computes group/class mean logits
over the complete training fold.  It adds the variance of these means across
nonempty **training** source groups separately for oracle class 0 and oracle
class 1.  Empty cells are skipped, and a class with fewer than two training
source cells does not enter the penalty.  There is intentionally no penalty
between the two class means, no class-unconditional source statistic, and no
validation or outer-test statistic in either the regularizer or its eligibility
calculation.
