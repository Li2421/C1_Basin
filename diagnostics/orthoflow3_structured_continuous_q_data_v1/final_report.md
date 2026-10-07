# ORTHOFLOW3_STRUCTURED_CONTINUOUS_Q_DATA_V1

## Decision

- Toy: **CONTINUOUS_ETA_GENERALIZATION_STILL_FAILS**
- DB transfer diagnostic: **NOT_DEMONSTRATED**
- New Toy continuations: **45,516**; new DB continuations: **0**.
- Global DB postflight exact reuse: **True**.

## Frozen design

States: 48/12/16 TRAIN/VAL/TEST with zero source-group overlap. Eta: 40/12/32; TRAIN types global/transition/boundary = 18/14/8. TEST-to-TRAIN normalized eta distance min/median = 0.0846/0.1954.

Preflight requested 56,064: exact 9,860, partial 688, aggregate 0, truly missing 45,516.

## Toy simultaneous unseen state + unseen eta

| Model | NLL | MAE | Spearman | B15 AUROC | B15 accuracy |
|---|---:|---:|---:|---:|---:|
| Structured full | 0.7325 | 0.2480 | 0.5866 | 0.8354 | 0.7402 |
| Structured matched basis | 0.6884 | 0.2099 | 0.6550 | 0.8780 | 0.8105 |
| Pair/Q-bin/state-matched sparse | 0.4628 | 0.1703 | 0.7696 | 0.9178 | 0.8320 |
| Structured eta-count matched basis | 0.8883 | 0.2536 | 0.5587 | 0.8199 | 0.7676 |
| Eta-count/Q-bin/pair-matched sparse | 0.7207 | 0.3812 | 0.2614 | 0.6802 | 0.4512 |
| Old critic | 0.5580 | 0.3690 | 0.0150 | — | — |

Full structured improvement over old: NLL -0.1745, MAE +0.1210, Spearman +0.5716. Pair/Q-bin/state matched structured improvement over sparse: NLL -0.2256, MAE -0.0396, Spearman -0.1146. Eta-count/pair/Q-bin/state matched structured improvement over sparse: NLL -0.1676, MAE +0.1275, Spearman +0.2973.

## Finite-candidate ranking

| K | Regret | B15 selection | Top-3 B15 |
|---:|---:|---:|---:|
| 8 | 0.3477 | 0.6667 | 0.9333 |
| 16 | 0.2500 | 0.7500 | 0.8750 |
| 32 | 0.2930 | 0.6875 | 0.8750 |

## Density ablation

| Density | Train pairs | NLL | MAE | Spearman |
|---:|---:|---:|---:|---:|
| 25% | 480 | 1.1054 | 0.2143 | 0.6312 |
| 50% | 960 | 0.9583 | 0.2322 | 0.6158 |
| 100% | 1920 | 0.7325 | 0.2480 | 0.5866 |

Monotonic NLL/MAE improvement: **False**.

## Frozen DB diagnostic

| Model | NLL | MAE | Spearman | B15 accuracy |
|---|---:|---:|---:|---:|
| eta_only | 0.3393 | 0.2670 | 0.9000 | 0.5000 |
| db_only | 0.7493 | 0.4176 | 0.8358 | 0.5312 |
| toy_pretrained_db_adapter | 1.5002 | 0.5057 | -0.2828 | 0.4062 |
| joint_finetune | 0.8116 | 0.3642 | 0.3711 | 0.5000 |

Toy-pretrained gain versus DB-only: NLL -0.7509, MAE -0.0882, Spearman -1.1186. Joint-finetune gain: NLL -0.0623, MAE +0.0534, Spearman -0.4647. A null transfer result cannot falsify shared continuous structure because no DB rollout or DB cross-matrix enrichment was permitted.
