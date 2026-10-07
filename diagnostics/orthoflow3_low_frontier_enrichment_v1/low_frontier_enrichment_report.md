# OrthoFlow3 low-frontier enrichment audit

**MIXED_ANALYTIC_MINDEF_EVIDENCE**. Exactly 10 frozen ACTIVE-required states were reused. 10/10 have >=3 and 10/10 have >=4 confirmed B63 candidates.

| Proxy | mean Spearman | mean Kendall | median J ratio | ≤5% | ≤10% | ≤20% |
|---|---:|---:|---:|---:|---:|---:|
| R_eta_raw | 0.460 | 0.433 | 1.174 | 30% | 40% | 50% |
| R_eta_norm | 0.460 | 0.433 | 1.174 | 30% | 40% | 50% |
| R_gram | -0.860 | -0.800 | 1.795 | 0% | 0% | 0% |
| R_exec1 | -0.680 | -0.600 | 1.541 | 0% | 0% | 10% |

Promotions were selected before B63 and without true J: lowest raw eta norm and lowest start-Gram energy among distinct 8/8 candidates. The full-horizon J comparison uses only B63-feasible candidates and authoritative mean successful `J_def`.

New branch work: 2400 continuations, 876123 physical steps; maximum completed-shard wall 173.0s. Low-proxy/high-true-J inversions: 30.
