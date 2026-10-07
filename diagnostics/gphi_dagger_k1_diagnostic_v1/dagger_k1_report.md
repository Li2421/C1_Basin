# Minimal k=1 DAgger diagnostic

## Result

**K1_FIXES_LOCAL_ERROR_NOT_CLOSED_LOOP**

704 historical learner-visited k=1 states were generated; the frozen source eta remained valid for 704/704. The immutable dataset added 704 states/samples with no weighting or oversampling. Selected checkpoint: seed 23, epoch 992, `83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a`.

| metric | old coverage | new k1 |
|---|---:|---:|
| historical k1 executed L2 | 0.235206 | 0.002181 |
| fresh-6 k1 executed L2 | 0.266562 | 0.034464 |
| dense strict-deadlock success | 841/1088 | 298/1088 |
| B63 states | 0/17 | 0/17 |
| exact strict-deadlock | 15/17 | 4/17 |

## Teacher-error growth

| step | old L2 | new L2 | old cosine | new cosine |
|---:|---:|---:|---:|---:|
| 0 | 0.012644 | 0.013349 | 0.998695 | 0.997782 |
| 1 | 0.246272 | 0.013830 | 0.570635 | 0.998897 |
| 2 | 0.212428 | 0.015916 | 0.782640 | 0.998006 |
| 4 | 0.204430 | 0.021056 | 0.832371 | 0.990499 |
| 8 | 0.187603 | 0.032824 | 0.826530 | 0.949307 |
| 16 | 0.163908 | 0.046089 | 0.848984 | 0.918472 |
| 32 | 0.121602 | 0.078226 | 0.667882 | 0.864053 |

## WIDE development regression

New H8: 187/200 (Q=0.935); timeout rescue 59/59; strict-deadlock rescue 1/6; breaks 8; collisions 0; mean J_def 0.029548. References: original checkpoint 188/200, coverage checkpoint 176/200.

Broader on-policy aggregation justified: **TRUE**, but one-layer k1 aggregation is not sufficient and caused strict-deadlock closed-loop regression. The measured error front did not jump at k2; it grew gradually and was largest at k32 among the registered points. Smallest next diagnostic: without training, substitute the frozen eta teacher only at k32 on the historical dense rollouts and then return immediately to the frozen k1 G_phi. This tests whether the later mismatch is causally important before collecting another training layer; fresh-6 remains untouched.
