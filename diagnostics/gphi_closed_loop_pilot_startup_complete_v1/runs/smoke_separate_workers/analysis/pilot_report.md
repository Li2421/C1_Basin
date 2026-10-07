# Startup-complete learned G_phi closed-loop pilot

The evaluation used 1 fresh, matched full episodes per controller.  It made no online eta, basin, oracle, or gate call.

| controller | success | deadlock | timeout | collision | other | Q (Wilson 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Safety | 1 | 0 | 0 | 0 | 0 | 1.000000 [0.206549, 1.000000] |
| Learned G_phi | 1 | 0 | 0 | 0 | 0 | 1.000000 [0.206549, 1.000000] |

Paired success improvement is +0.000000, with paired-bootstrap 95% CI [+0.000000, +0.000000].  Learned deformation mean/median/P95 is 0.123856 / 0.123856 / 0.123856.  Hard-safety integrity: **True**.

Classification: **GPHI_CLOSED_LOOP_FAIL**.  If refinement is required, the single descriptively dominant logged axis is **projection interaction**.  Timing remains diagnostic only and did not alter control.
