# OrthoFlow3 global rollout database v1

The persistent SQLite cache is operational at `shared_rollout_db/rollout.sqlite`. Migration scanned 132 historical experiment directories and indexed 923,092 unique seed-level rollout records, of which 551,419 have sufficiently complete compatible fingerprints for automatic exact reuse. Another 371,673 remain searchable but `AMBIGUOUS` and therefore cannot be reused automatically.

The aggregate table contains 48,353 unique records: 5,607 Q16, 702 Q32, 12,578 Q64, and 29,466 other trial counts. There are 14,552 aggregate records that mathematically certify B15; 9,348 of these are 63/64-or-better Q64 records. Seed-level evidence additionally contains 3,696 state–eta–controller groups with at least 64 trials and at most one failure.

## Coverage

| scenario | compatible state–eta pairs | pairs with ≥16 compatible seeds | reuse coverage | observed B15 pairs |
|---|---:|---:|---:|---:|
| Toy Give-Way | 20,283 | 6,644 | 32.8% | 5,456 |
| Double-Bottleneck | 13,916 | 3,393 | 24.4% | 2,698 |

Exact eta identity uses IEEE-754 float64 bytes; nearby eta is never accepted for cache reuse. State, controller, conditioning, RNG and scenario fingerprints are independently checked. The 2,082 contradictory rollout keys and two source-file mutation events are quarantined. In addition, 272 otherwise exact-profile rollout rows involved in conflicts are blocked from automatic reuse.

## Dry runs

| historical experiment | requested | exact/partial records reusable | genuinely missing | saved |
|---|---:|---:|---:|---:|
| Toy shared eta codebook | 68,528 | 68,528 | 0 | 68,528 |
| DB shared-mode transfer | 53,069 | 53,068 | 1 | 53,068 |
| Toy–DB conditional generator | 27,616 | 27,616 | 0 | 27,616 |

The one DB miss is deliberately retained because its key is conflict-quarantined. No rollout was executed during database construction or validation.

## Integration

The common Toy Oracle and Double-Bottleneck rollout executor now perform a read-only exact-cache preflight and materialize compatible cached rows before physical execution. New results continue to be written to experiment artifacts, then to append-only shard journals; only the single merger writes SQLite. Both hooks were tested against one historical exact request and returned the cached record without rollout. Physical control, safety projection, horizon and RNG logic were not modified.

Run a preflight with:

```bash
cd /home/zhihan/research/Basin_C1
python -m shared_rollout_db.plan \
  --manifest diagnostics/EXPERIMENT/planned_rollouts.json \
  --output diagnostics/EXPERIMENT/cache_preflight.json
```

The report returns exact reusable, partial reusable, aggregate reusable, ambiguous/incompatible, and genuinely missing seed counts. Only the final category may be submitted.
