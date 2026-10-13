# Numerical-failure handling audit

The factor collection completed all3072 planned attempts:1665 success,1406
timeout,1 numerical failure,0 collisions. The unknown is training state6,
parallel topology,N2,width.8,isotropic,eta10,seed1, step72, original
CBFSolverError (identical_nonlinear_polish_failed). It is retained in raw JSON,
trace, shared DB and unknown mask. No retry and no controller/solver change.
The first factor-fit entry correctly stopped on the unknown instead of treating
it as0; its failure log is retained. Following the original preregistered unknown
policy, fit Bernoulli likelihood only to OBSERVED successes/failures, omitting
the unobserved outcome from likelihood, with exact per-fit counts saved. This is
not deletion of the row. Test states8..15 have ZERO unknowns, so their selection
and Q metrics need no imputation and are unchanged by this handling choice.

All-state factor descriptive statistics require an explicit sensitivity check.
Use three monotone completions for this single censored physical trajectory:
(success_by400,success_by850)=(0,0),(0,1),(1,1). Save the primary all-failure
completion and min/max point-estimate ranges over all three. These are labelled
sensitivity quantities, never measured successes. Horizon labels are correlated.
Source training labels, main data arrays and the shared DB retain NaN/unknown.

Historical four-scene calibration also retains51 observed numerical unknowns.
Calibration means use known outcomes and source-prior fallback if a cell has no
known outcomes. For the SAME resulting decisions, report exact success lower
and upper bounds on test unknowns. Oracle/regret under the lower completion are
explicitly sensitivity estimates for those panels, not identified true regrets.
