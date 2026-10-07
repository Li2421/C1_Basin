# Oracle Boundary Confidence Audit

## Decision

**MIXED_ORACLE_AND_FEATURE_PROBLEM**

This audit ran only frozen `eta=(0,0,0)` continuations. It changed no oracle threshold, controller, feature, dataset, or learned model.

## Cohort and sampling

- Primary difficult/boundary states: **34**; controls: 8; total audited: **42**.
- Every state reached at least 256 continuations; **22** were adaptively extended to 512.
- Existing tuples reused: 1955; new tuples: 14429; duplicate tuples: 0.

## Stability evidence

- Difficult states with >=95% probability of reproducing the original label under a new random 64 draw: **13/34**.
- Modal random-64 label reversals: **8**; direct fresh-64 flips: **9**.
- Matched pair classifications: {'OVERLAPPING_AMBIGUOUS': 6, 'CLEARLY_SEPARATED': 9}; point-estimate order reversals: 2.
- Failures: {'timeout': 407, 'deadlock': 1479}.

Old hard states: R_D4_s95101008_p134: 506/512 Q0=0.9883 [0.9747,0.9946] flip=0.173; R_D4_s95101014_p123: 506/512 Q0=0.9883 [0.9747,0.9946] flip=0.173; R_D4_s95105001_p116: 256/256 Q0=1.0000 [0.9852,1.0000] flip=0.000; R_D4_s95105004_p114: 256/256 Q0=1.0000 [0.9852,1.0000] flip=0.000.

Smallest justified next experiment: Stratify a small feature-information audit by oracle-stable versus oracle-ambiguous pairs, keeping both strata separate.
