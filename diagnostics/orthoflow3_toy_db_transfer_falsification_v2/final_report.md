# OrthoFlow3 Toy→DB transfer falsification V2

## Stage A — leave-mode-out deformation

Each fold fit 8 canonical correspondences and predicted 4 unseen modes. Held-out coordinates and outcomes were excluded from fitting and family selection. Geometry filtering preceded all control validation; there was no local eta optimization.

| family | median held-out residual | median distance to independent robust evidence | promising / 12 | tested modes with majority B15 support | mean tested B15 prevalence | max condition |
|---|---:|---:|---:|---:|---:|---:|
| raw_toy | 0.6619 | 0.6372 | 0/12 | 0/0 | - | 1.00 |
| translation | 0.4943 | 0.5197 | 0/12 | 0/0 | - | 1.00 |
| diagonal | 0.1656 | 0.0493 | 5/12 | 5/5 | 0.925 | 5.75 |
| rotation_anisotropic | 0.1886 | 0.0697 | 5/12 | 4/5 | 0.775 | 8.64 |
| regularized_affine | 0.1381 | 0.0641 | 5/12 | 5/5 | 0.925 | 10.00 |

Best held-out coordinate family: **regularized_affine**. Its median residual improves over raw Toy eta by 79.1%. Stage A: **MODE_DEFORMATION = PARTIAL**.

Stage-A cache accounting: requested 1,920; exact reused 0; partial reused 0; aggregate reused 0; newly executed 1,920. Every new record was journaled and atomically merged into the global rollout database.

The identical post-run manifest now resolves as 1,920/1,920 `EXACT_REUSE` with zero missing continuations, verifying that future experiments will not rerun this batch.

## Stage B — mode-ID permutation control

The Toy semantic trunk/head was frozen. Only an 80→64 DB adapter was trained at 25% and 50% DB data. Correct identity and 10 fixed derangements used identical subsets, optimizer, epochs, and seeds 17/23/41.

| DB data | correct Q | shuffled median Q [IQR] | correct B15 | shuffled median B15 [IQR] | correct regret | shuffled median regret |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 1.0000 | 0.9917 [0.9268, 1.0000] | 16/16 | 15/16 [14, 16] | 0.0000 | 0.0083 |
| 50% | 1.0000 | 1.0000 [0.9971, 1.0000] | 16/16 | 16/16 [16, 16] | 0.0000 | 0.0000 |

Stage B: **MODE_SEMANTIC_TRANSFER = GENERIC_ONLY**. Correct correspondence did not obtain a stable, out-of-distribution advantage over shuffled identities at both data fractions; the observed joint-training benefit therefore cannot be specifically attributed to shared output-column semantics.

## Joint conclusion

**CURRENT_CROSS_SCENARIO_CLAIM_WEAKENED**. Action on the previous Toy→DB claim: **WEAKEN**. The evidence supports at most partial low-complexity eta-space deformation; it does not currently isolate shared mode semantics from generic multitask/adapter regularization.
