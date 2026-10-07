# ORTHOFLOW3_TOY_Q_DB_TRANSFER_V1

Classification: **POSITIVE_TRANSFER**

NEW ROLLOUT = **0**.

## Frozen DB unseen-state + unseen-eta probability

| Model | NLL | MAE | Spearman | B15 AUROC | B15 AUPRC | B15 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| eta_only | 0.4366 | 0.3013 | 0.9000 | 1.0000 | 1.0000 | 0.5000 |
| db_scratch | 0.6599 | 0.3596 | 0.7477 | 0.9518 | 0.9493 | 0.5312 |
| toy_state_adapter | 0.5872 | 0.2885 | 0.6372 | 0.8763 | 0.8670 | 0.7344 |
| toy_state_eta_adapter | 0.4400 | 0.2609 | 0.7049 | 0.9196 | 0.9063 | 0.6927 |
| toy_full_finetune | 0.1614 | 0.1159 | 0.8500 | 1.0000 | 1.0000 | 0.8490 |

## Finite-candidate ranking on frozen DB TEST states

| Model | K | Eligible | Selected Q | Oracle Q | Regret | B15 selection | Top-3 B15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eta_only | 8 | 16 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| eta_only | 16 | 16 | 0.7002 | 1.0000 | 0.2998 | 0.1250 | 1.0000 |
| eta_only | 32 | 10 | 0.7063 | 1.0000 | 0.2938 | 0.1000 | 1.0000 |
| db_scratch | 8 | 16 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| db_scratch | 16 | 16 | 0.7002 | 1.0000 | 0.2998 | 0.1250 | 1.0000 |
| db_scratch | 32 | 10 | 0.7063 | 1.0000 | 0.2938 | 0.1000 | 1.0000 |
| toy_state_adapter | 8 | 16 | 0.8958 | 1.0000 | 0.1042 | 0.8958 | 1.0000 |
| toy_state_adapter | 16 | 16 | 0.7292 | 1.0000 | 0.2708 | 0.7292 | 0.9375 |
| toy_state_adapter | 32 | 10 | 0.6667 | 1.0000 | 0.3333 | 0.6667 | 0.9333 |
| toy_state_eta_adapter | 8 | 16 | 0.9375 | 1.0000 | 0.0625 | 0.9375 | 1.0000 |
| toy_state_eta_adapter | 16 | 16 | 0.7897 | 1.0000 | 0.2103 | 0.7917 | 0.9792 |
| toy_state_eta_adapter | 32 | 10 | 0.7000 | 1.0000 | 0.3000 | 0.7000 | 0.9667 |
| toy_full_finetune | 8 | 16 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| toy_full_finetune | 16 | 16 | 0.9941 | 1.0000 | 0.0059 | 0.9583 | 1.0000 |
| toy_full_finetune | 32 | 10 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Transfer gains

- toy_state_adapter: NLL gain vs DB scratch=0.0728; MAE gain=0.0710; Spearman gain=-0.1104; NLL gain vs eta-only=-0.1505.
- toy_state_eta_adapter: NLL gain vs DB scratch=0.2199; MAE gain=0.0986; Spearman gain=-0.0428; NLL gain vs eta-only=-0.0034.
- toy_full_finetune: NLL gain vs DB scratch=0.4986; MAE gain=0.2437; Spearman gain=0.1023; NLL gain vs eta-only=0.2753.

## Toy-source eta overlap audit

Of the four frozen DB D-test eta values, 1 appeared exactly in the Toy source. On the remaining three globally unseen eta values:

| Model | NLL | MAE | Spearman |
|---|---:|---:|---:|
| eta_only | 0.3111 | 0.2289 | 0.8814 |
| db_scratch | 0.4659 | 0.2731 | 0.7087 |
| toy_state_adapter | 0.4337 | 0.2361 | 0.5346 |
| toy_state_eta_adapter | 0.4112 | 0.2525 | 0.5758 |
| toy_full_finetune | 0.1990 | 0.1400 | 0.7796 |

Best transfer model: **toy_full_finetune**.

