<!-- generated_by: finalize_pilot_report.py -->
# First semantically aligned single-segment recovery pilot

Report status: **FINAL**.
Classification: **SINGLE_SEGMENT_RECOVERY_SUPPORTED**.
Reason: frozen 200-episode test improves task success with positive paired CI, zero observed breaks, finite recovery-to-Safety completions, and intact hard safety.

## Completed stages

| Stage | Status |
|---|---|
| Stage A primitive audit | COMPLETE |
| Generic source collection | COMPLETE |
| Bootstrap/second-pass entry and exit labels | COMPLETE |
| First head training and validation/calibration | COMPLETE |
| Required bounded policy-improvement round | POLICY_ITERATION_FINAL_TEST_COMPLETE |
| Post-iteration validation/calibration | COMPLETE |
| Frozen final 200-episode test | COMPLETE |

## Recovery candidates

- System G (dense Direct-g): Stage A found weak/inconsistent rescue; it was retained as an integrity reference and did not consume head-label training budget.
- System ETA (one-shot persistent eta): Stage A established credible mid/late strict-deadlock takeover capacity and it advanced to entry/exit learning.

## Decision semantics

- Entry input: 214-D deployment feature `h_t`.
- ETA exit input: 217-D concatenation `[h_t, eta_latched]`.
- Both heads use `input -> 64 -> 64 -> 1` with SiLU.
- Success loss: `r*softplus(-s) + (1+lambda_break)*b*softplus(s)`; scores are decision scores, not success probabilities.
- WAIT executes one Safety transition and then follows an incumbent policy that can enter later in temporally aligned rounds; it does not forbid future recovery.
- EXIT executes Safety immediately and Safety remains in control to terminal; paired labels therefore include actual downstream Safety behavior.
- Recovery is one contiguous segment. Eta is predicted once on entry and never re-predicted within the segment.

## Frozen full-loop evidence

| Split | Sources | Safety Q | Learned Q | Rescue | Break | Mean J_def |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 20 | 0.800 | 0.950 | 3 | 0 | 0.016921 |
| Calibration | 20 | 0.700 | 0.850 | 3 | 0 | 0.026159 |
| Final test | 200 | 0.715 | 0.905 | 38 | 0 | 0.024862 |

These metrics use policy `f6a2c5727e0de07dafc5b3fb3e5166273fbf8550502bf62df4bcd8b0bd8ed878` and are **aligned to the iterated heads**.
Validation observed 19 successful finite recovery segments followed by Safety, and 1 exits followed by failure.
Calibration observed Q=0.85 versus Safety Q=0.7, rescue=3, break=0; its break interval was [0.0, 0.21531080273763575], so support was insufficient for certification.
Final test: Safety 143/200 (Q=0.715); hierarchy 181/200 (Q=0.905); rescue=38, break=0, paired delta=0.19000000000000006 CI=[0.11121457272449328, 0.25947394271299346].
Failure slices: timeout rescue 37/54; strict-deadlock rescue 1/3.
All 200 entered and 200 exited; 181 completed after a finite recovery segment returned to Safety; 19 returned then failed.
Mean J_def=0.024861752285720327; hard safety intact=True.

## Critical limitation: entry-almost-everywhere

The frozen final controller entered recovery on 200/200 episodes, and the degeneracy audit reports `enter_almost_everywhere=True`. Therefore the experiment supports the one-segment recovery system's task performance and learned exit, but does **not** establish a selective or semantically discriminating entry rule. In effect, the entry head nearly always initiates one recovery segment, after which the learned exit returns to Safety. This limitation must be addressed before claiming that entry identifies only recovery-needing states.

## Data and budget

- Safety source trajectories: 80 across train/validation/calibration.
- Generic materialized decision states: 97 from 57 roots.
- Frozen independent root-source groups: 40 train + 20 validation + 20 calibration + 200 final test = 280 roots.
- Original merged entry decisions: 36; original merged exit decisions: 36.
- Policy-iteration decisions: 12 entry + 6 exit.
- Total supervised decision-state records: 90; paired branch continuations: 5760.
- Recorded new continuations: 6704/12000.
- Recorded physical steps: 2262171/6000000.
- Observed runtime timestamp window: 5625.263326 s; summed recorded stage spans: 4969.437341 s. These exclude uninstrumented training/report time.
- Maximum recorded simultaneous shard count: 2; recorded thread settings: [2].

## Safety and integrity

Integrity status: **PASS**. Observed hard-safety counts: `{"agent_collisions": 0, "invalid_actions": 0, "nan_inf": 0, "projection_solver_failures": 0, "wall_collisions": 0}`.
Final-test episode outcome files observed: 200 (runtime JSON files excluded).

## Remaining uncertainty

Sparse paired decision evidence left many inputs unresolved; non-significance was not treated as equivalence. Calibration remained too small for standalone certification, although the frozen final test subsequently showed zero observed breaks with a 95% upper bound below 5%. The dominant remaining semantic limitation is entry-almost-everywhere, not task success or learned exit feasibility.

This pilot does not solve repeated recovery, exit/re-entry, a different scenario, or controller retraining.
