# G_phi extended-horizon timeout audit

The original 850-step benchmark remains official. This diagnostic changes only the environment termination horizon from 850 to 1700 and replays only the 157 original timeout tuples.

To satisfy exact prefix identity, the deployed FeatureBuilder retains its frozen 850-step time coordinate. Consequently, after step 850 its normalized time features extrapolate beyond their original range; changing them to a 1700-step normalization would alter G_phi actions before step 850 and invalidate this pure-horizon comparison.

## Integrity

Prefix audit: **PASS**. All controller/plant/RNG/feature/action/deadlock-monitor fields for physical step indices 0–849 are bitwise identical (maximum numeric error 0). The sole expected event-field change is the old horizon marker `timeout` becoming `running` at index 849; all event fields through index 848 are exact.

## Timeout transitions

| condition | original timeout | late success | late deadlock | persistent timeout | other | classification |
|---|---:|---:|---:|---:|---:|---|
| Safety | 51 | 20 | 0 | 31 | 0 | MIXED_HORIZON_AND_LIVENESS |
| H=1 | 48 | 0 | 0 | 48 | 0 | MOSTLY_TRUE_LIVENESS_FAILURE |
| H=4 | 18 | 0 | 0 | 18 | 0 | MOSTLY_TRUE_LIVENESS_FAILURE |
| H=8 | 13 | 0 | 0 | 13 | 0 | MOSTLY_TRUE_LIVENESS_FAILURE |
| H=16 | 27 | 2 | 0 | 25 | 0 | MOSTLY_TRUE_LIVENESS_FAILURE |

## Q versus horizon

| condition | Q850 | Q950 | Q1050 | Q1200 | Q1400 | Q1700 |
|---|---:|---:|---:|---:|---:|---:|
| Safety | 0.6900 | 0.7400 | 0.7600 | 0.7750 | 0.7850 | 0.7900 |
| H=1 | 0.7400 | 0.7400 | 0.7400 | 0.7400 | 0.7400 | 0.7400 |
| H=4 | 0.9100 | 0.9100 | 0.9100 | 0.9100 | 0.9100 | 0.9100 |
| H=8 | 0.9350 | 0.9350 | 0.9350 | 0.9350 | 0.9350 | 0.9350 |
| H=16 | 0.8650 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 |

## Late-success delay beyond 42.5 s

| condition | median s | mean s | P90 s | max s |
|---|---:|---:|---:|---:|
| Safety | 5.550 | 8.463 | 22.820 | 30.250 |
| H=1 | NA | NA | NA | NA |
| H=4 | NA | NA | NA | NA |
| H=8 | NA | NA | NA | NA |
| H=16 | 2.325 | 2.325 | 3.265 | 3.500 |

## Controller interpretations

- Safety: 20 of 51 original timeouts were finite-horizon late successes.
- H=8: 0 late successes, 0 late deadlocks, and 13 persistent timeouts among its 13 cases; case-level rows are in `h8_timeout_case_table.csv`.
- H=16: 2 of 27 timeouts caught up by 1700. Among 14 episodes where H=8 succeeded by 850 but H=16 timed out, 2 later succeeded under H=16.
- H=1: 0 of 48 timeouts later succeeded; its classification is `MOSTLY_TRUE_LIVENESS_FAILURE`.

## Additional deformation for late success

- H=1: mean additional J_def NA, median NA, P95 NA.
- H=4: mean additional J_def NA, median NA, P95 NA.
- H=8: mean additional J_def NA, median NA, P95 NA.
- H=16: mean additional J_def 0.001754550254070837, median 0.001754550254070837, P95 0.0033113554436261564.

## Persistent-timeout movement diagnostics

The table below reports observed motion without redefining deadlock. A persistent timeout remains a timeout unless the frozen detector fired.

| condition | N | median goal-error reduction | median displacement sum | median agent speed | median candidate fraction |
|---|---:|---:|---:|---:|---:|
| Safety | 31 | 0.000538441 | 0.00230843 | 0.00886701 | 0.811765 |
| H=1 | 48 | -0.685534 | 1.12507 | 0.0563722 | 0.222353 |
| H=4 | 18 | -0.706426 | 1.05814 | 0.0471554 | 0.0211765 |
| H=8 | 13 | 1.13882 | 1.35833 | 0.068794 | 0.135294 |
| H=16 | 25 | -0.0017851 | 0.327635 | 0.0296063 | 0.52 |

Safety persistent timeouts are effectively stationary by displacement and show the frozen deadlock-candidate condition during most extension steps, although the hold criterion never reaches terminal deadlock. H=1/H=4/H=8 persistent cases continue substantial motion; H=1 and H=4 worsen goal error on median, while H=8 often makes partial progress without task completion. H=16 is heterogeneous, mixing near-stationary and moving unresolved cases.

## Paired completion-time summaries

```json
{
  "H=16_minus_H=8": {
    "max": 18.5,
    "mean": 0.8431428571428572,
    "median": 0.05,
    "p25": 0.0,
    "p75": 0.125,
    "p90": 3.43,
    "p95": 4.6599999999999975,
    "paired_successes": 175,
    "std": 2.346771775616639
  },
  "H=16_minus_Safety": {
    "max": 2.9000000000000004,
    "mean": -3.8971153846153848,
    "median": -0.1,
    "p25": -4.9625,
    "p75": 0.0,
    "p90": 0.0,
    "p95": 0.0,
    "paired_successes": 156,
    "std": 7.370722904942174
  },
  "H=1_minus_Safety": {
    "max": 8.55,
    "mean": -4.639406779661018,
    "median": -0.9,
    "p25": -6.9,
    "p75": 0.05,
    "p90": 1.6949999999999996,
    "p95": 2.8299999999999987,
    "paired_successes": 118,
    "std": 9.000120729709984
  },
  "H=4_minus_Safety": {
    "max": 3.5,
    "mean": -4.830463576158939,
    "median": -0.45,
    "p25": -7.35,
    "p75": -0.1,
    "p90": 0.0,
    "p95": 0.05,
    "paired_successes": 151,
    "std": 8.067479931820786
  },
  "H=8_minus_Safety": {
    "max": 2.5,
    "mean": -4.446794871794872,
    "median": -0.2,
    "p25": -6.25,
    "p75": -0.05,
    "p90": 0.0,
    "p95": 0.05,
    "paired_successes": 156,
    "std": 7.831307474339835
  }
}
```

## Most important conclusion

The learned-cadence timeout failures are not principally a 42.5-second cutoff artifact: H=1/H=4/H=8 recover 0 additional cases, and H=16 recovers only 2. H=8 remains at Q=0.935 through 85 seconds, while H=16 reaches only Q=0.875 and catches up on only 2/14 of the cases H=8 had already solved. The H=8-versus-H=16 gap therefore reflects genuine cadence-dependent recovery loss under this frozen controller, not merely slower completion. Safety is different: 20/51 timeouts are late successes, so its horizon and liveness effects are mixed.

## Runtime/resources

Rollout wall envelope: 201.875045 s using 2 GPU shards and 4 CPU cores. Analysis wall time: 0.405 s.
