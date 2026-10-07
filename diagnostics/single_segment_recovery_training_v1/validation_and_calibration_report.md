<!-- generated_by: finalize_pilot_report.py -->
# Validation and calibration report

Policy-iteration alignment: **CURRENT** — validation report matches both policy-iteration head hashes.
The numerical table is preserved as development evidence, but stale evidence is not used to accept the iterated policy.

| Split | Sources | Safety Q | Learned Q | Rescue | Break | Mean J_def |
|---|---:|---:|---:|---:|---:|---:|
| Validation selected | 20 | 0.800 | 0.950 | 3 | 0 | 0.016921 |
| Calibration | 20 | 0.700 | 0.850 | 3 | 0 | 0.026159 |
| Final test | 200 | 0.715 | 0.905 | 38 | 0 | 0.024862 |

Selected policy hash in the authoritative report: `f6a2c5727e0de07dafc5b3fb3e5166273fbf8550502bf62df4bcd8b0bd8ed878`.

Validation segment evidence:

- Entered: 20/20
- Exited: 20/20
- Successful finite segment followed by Safety: 19
- Returned to Safety then failed: 1
- Mean J_def: 0.016921330807130076

Calibration audit:

- Observed break: 0 (rate 0.0)
- 95% break interval: [0.0, 0.21531080273763575]
- Pilot point budget: 0.05
- Support sufficient for certification: False
- No certification is claimed.

Frozen final-test evidence:

- Safety Q: 0.715; learned Q: 0.905; paired delta: 0.19000000000000006 with CI [0.11121457272449328, 0.25947394271299346].
- Rescue/break: 38/0.
- Timeout rescue: 37/54; strict-deadlock rescue: 1/3.
- Successful finite segment then Safety: 181/200.
- Entry-almost-everywhere degeneracy: True. This is an explicit interpretability/selectivity limitation.
