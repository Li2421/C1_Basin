# Recovery takeover primitive audit

## Frozen development protocol

- New authoritative WIDE development cohort: IC root `2026092701`, Flow root `2026092702`, 200 episodes; manifest frozen at `2026-09-26T03:31:38.742912+00:00` before outcomes.
- Safety outcomes: **138 success / 54 timeout / 8 strict deadlock**.
- Exact augmented-state restoration maximum error: `0.0`. All 248 failure states and 248 time-matched nominal controls were available.
- No H8, bursts, hybrid, gate, exit rule, online oracle, eta search, training, or controller modification was used. Recovery remained active until termination.

## Recovery and false-trigger windows

| Controller | Offset | Timeout rescue | Strict-deadlock rescue | All failure rescue | False-trigger break | Mean post-takeover J_def |
|---|---:|---:|---:|---:|---:|---:|
| Dense Direct-g | -8 s | 3/54 (5.6%) | 3/8 (37.5%) | 6/62 (9.7%) | 1/62 (1.6%) | 0.0352 |
| Dense Direct-g | -4 s | 0/54 (0.0%) | 0/8 (0.0%) | 0/62 (0.0%) | 0/62 (0.0%) | 0.0358 |
| Dense Direct-g | -2 s | 1/54 (1.9%) | 2/8 (25.0%) | 3/62 (4.8%) | 0/62 (0.0%) | 0.0249 |
| Dense Direct-g | -1 s | 1/54 (1.9%) | 0/8 (0.0%) | 1/62 (1.6%) | 0/62 (0.0%) | 0.0365 |
| Persistent structured eta | -8 s | 1/54 (1.9%) | 6/8 (75.0%) | 7/62 (11.3%) | 2/62 (3.2%) | 0.0864 |
| Persistent structured eta | -4 s | 1/54 (1.9%) | 7/8 (87.5%) | 8/62 (12.9%) | 0/62 (0.0%) | 0.0716 |
| Persistent structured eta | -2 s | 0/54 (0.0%) | 6/8 (75.0%) | 6/62 (9.7%) | 0/62 (0.0%) | 0.0597 |
| Persistent structured eta | -1 s | 1/54 (1.9%) | 5/8 (62.5%) | 6/62 (9.7%) | 0/62 (0.0%) | 0.0455 |

## Interpretation

- **Dense Direct-g — NOT_A_USEFUL_RECOVERY_PRIMITIVE**. Across all four windows it rescued 5/216 timeout branches and 5/32 strict-deadlock branches; false takeover broke 1/248 nominal branches.
- **Persistent structured eta — TRUE_RECOVERY_PRIMITIVE**. Across all four windows it rescued 3/216 timeout branches and 24/32 strict-deadlock branches; false takeover broke 2/248 nominal branches.

True late-recovery evidence: Dense Direct-g late (-2/-1 s) rescue 3/62 and 1/62; Persistent structured eta late (-2/-1 s) rescue 6/62 and 6/62. Hard safety intact: **True**.


Nominal matching used 42 unique Safety-success episodes and 56 unique augmented states; reuse was allowed by the frozen rule. Raw break rates and source-cluster bootstrap intervals are both reported.

Important limitation: there is deliberately no exit-back-to-Safety rule. Success establishes takeover capacity; failures and false-trigger breakage can include harm caused by remaining in recovery mode after the maneuver is no longer needed.

## Classification

- Dense Direct-g: **NOT_A_USEFUL_RECOVERY_PRIMITIVE**
- Persistent structured eta: **TRUE_RECOVERY_PRIMITIVE**

The structured-eta classification is specifically supported for **strict-deadlock recovery**, not timeout recovery. The smallest justified next experiment is a validation-controlled strict-deadlock entry-trigger audit that switches Safety directly to persistent structured eta and keeps eta active until termination; Direct-g H8 must not be integrated into that experiment.
