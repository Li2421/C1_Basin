# Independent OrthoFlow3 analytical-min-def replication

**ANALYTIC_FAILURE_INDEPENDENTLY_REPLICATED**. 9 state-identity-disjoint ACTIVE-required states were resolved; 9 have >=3 and 9 have >=4 B63 candidates.

| Proxy | mean / median Spearman | mean / median Kendall | median J ratio | <=10% |
|---|---:|---:|---:|---:|
| R_eta_raw | 0.422 / 0.400 | 0.296 / 0.333 | 1.000 | 67% |
| R_eta_norm | 0.422 / 0.400 | 0.296 / 0.333 | 1.000 | 67% |
| R_gram | -0.378 / -0.800 | -0.407 / -0.667 | 1.663 | 22% |
| R_exec1 | -0.111 / -0.200 | -0.074 / 0.000 | 1.656 | 33% |

Inversions: 20 across 7 states; worst ratio 2.066.
New execution: 2216 continuation rollouts and 588742 physical steps. No model was trained.
