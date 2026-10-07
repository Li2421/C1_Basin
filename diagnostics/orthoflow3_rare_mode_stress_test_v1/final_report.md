# Rare-mode stress-test cache audit

Classification: **RARE_MODES_NOT_AVAILABLE_FROM_CACHE**

Deduplicated compatible evidence: 4,610 state × eta records; 904 unique TRAIN eta coordinates; 236 robust-witness clusters after the 0.05 merge.
Prevalence-identifiable candidates: 12. Identifiable 10--30% rare candidates: **0**.

The only eta with complete systematic 128-state TRAIN coverage are the original aligned modes. Their strong prevalence range is 46.9%--94.5%; all exceed 35%.
Non-codebook historical eta were evaluated on adaptively selected states, typically only a small part of TRAIN. Treating unknown states as failures would manufacture artificial rarity; treating the observed subset as representative would create selection bias.

Even the three most rare-looking partial coordinates would require approximately 15,184 continuations to complete TRAIN/VAL/TEST at 16/32/64 seeds.

## Requested answers

1. Reliably identifiable rare eta candidates: **0**. There are 236 robust-witness coordinate clusters, but all non-codebook clusters lack outcome-blind 128-state prevalence coverage.
2. Final codebook size: **0**; the first stop condition fired.
3. Per-mode TRAIN prevalence: not applicable for a rare codebook. The 12 identifiable historical modes span 46.9%--94.5% and are listed in `mode_train_prevalence.csv`.
4. Best fixed TEST coverage: not inspected; codebook was not frozen.
5. Union-oracle TEST coverage: not inspected.
6. Selector TEST coverage: no selector was trained.
7. Oracle-coverable selector accuracy: not applicable.
8. Multi-mode selection: not applicable.
9. Fresh replication: not executed.
10. New rollout cost: **0 eta evaluations / 0 continuations**.
11. Target pattern (individual 10--30%, union 70--80%): **not achieved**.
12. Reason: cached non-codebook eta were acquired adaptively on too few states; certifying only three candidates would cost about 15,184 continuations, above the 8,000 continuation cost gate.
13. CONTINUE TO DOUBLE-BOTTLENECK = **YES** (not started automatically).

No codebook was frozen, no selector was trained, no fresh replication was run, and no new rollout was generated.
