# Evidence reused before contract tests

| Question | Existing evidence | Reuse |
|---|---|---|
| Corrected Ring safety | ring_revision_v1/RING_SAFETY_AUDIT.md; current v2 and K16 evidence | No safety re-audit |
| Exact physical equivariance | ring_fourway_symmetry_conditioning_audit/REPORT.md | No full physical suite repeated |
| K16 coverage and critic ranking | ring_k16 diagnostic; completed Phase-A confirmation | No K sweep or loss search |
| Current labels, leakage, normalization | orthoflow3_data_hygiene_v2/REPORT.md | Physical v2 evidence retained |
| Numerical Q reporting | orthoflow3_autonomous_redteam_v1/REPORT.md | Unknowns/bounds preserved |
| Canonical local frames | Phase-B passive 1,280-case tests and active model probes | Reuse exact chart tests |
| Active Four rotations | Phase-B new rotation cohort: 4/4 original,0/4 each rotated | Demonstrates full-kernel asymmetry; not reused for adaptive training |
| Ring active rotations | Phase-B new cohort:4/4 each orientation,trajectory RMSE3.42e-6 | Supports tested Ring rotational equivalence |

The local experiment ledger and red-team ALREADY_AUDITED table identify old
ranking-loss, LCB and NLL-weighting experiments as completed/superseded.
This audit preserves the current squashed Gaussian and empirical-Q BCE.

New questions: remaining-time information in h, reference versus executed
Flow timing, fixed-eta schedule, and faithful valid-snapshot restoration.
