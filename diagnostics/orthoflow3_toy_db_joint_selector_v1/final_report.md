# Toy–Double-Bottleneck joint selector

Classification: **JOINT_SELECTOR_STRONGLY_SUPPORTED**

Toy and DB inputs are not directly compatible: Toy uses 214-D h0, while DB uses 80-D observation+Flow features. The frozen joint model therefore uses scenario-specific 64-D adapters and a shared 64-D trunk plus shared 12-mode head. No feature was dropped and no rollout was generated.

## TEST comparison

| Scenario / model | NLL | Brier | top-1 Q64 | B63 | regret | selection entropy |
|---|---:|---:|---:|---:|---:|---:|
| Toy-only | 0.2134 | 0.0211 | 1.0000 | 32/32 | 0.0000 | 0.549 |
| Joint (Toy) | 0.1997 | 0.0170 | 1.0000 | 32/32 | 0.0000 | 0.630 |
| DB-only | 0.1272 | 0.0021 | 1.0000 | 16/16 | 0.0000 | 0.623 |
| Joint (DB) | 0.1282 | 0.0032 | 1.0000 | 16/16 | 0.0000 | 0.436 |

## DB label efficiency

| DB TRAIN fraction | Joint NLL | DB-only NLL | Joint top-1 Q | DB-only top-1 Q | Joint B63 | DB-only B63 |
|---:|---:|---:|---:|---:|---:|---:|
| 25% (16) | 0.1544 | 0.1488 | 1.0000 | 0.9893 | 1.000 | 0.875 |
| 50% (32) | 0.1372 | 0.1340 | 1.0000 | 0.9971 | 1.000 | 0.938 |
| 100% (64) | 0.1282 | 0.1272 | 1.0000 | 1.0000 | 1.000 | 1.000 |


Toy performance retention: 1.000; DB retention: 1.000. Joint-25% DB is near DB-only-100% on the preregistered top-1/B63 criterion. Material 25%-DB efficiency gain over DB-only-25%: True.

Shared-latent same-mode prototype cosine is 0.750 versus cross-mode 0.749; this is diagnostic only.

The prototype diagnostic does **not** independently support mode-specific latent alignment (same-mode top-1 retrieval 0.167). Predictive/data-efficiency transfer is therefore stronger evidence than latent clustering, and should not be overinterpreted as a scenario-invariant state geometry.

**Answer:** State-to-mode feasibility can be shared through a common trunk/head. Scenario-specific input adapters remain necessary because the h schemas are different.
