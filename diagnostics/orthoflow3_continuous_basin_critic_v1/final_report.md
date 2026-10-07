# OrthoFlow3 continuous Basin critic V1

## Dataset and split

Training used 3,146 Toy and 512 DB canonical aggregate pairs. No rollout was generated. Source-group overlap is zero. Spatial eta grouping gives a test→train nearest-distance median of 0.110 normalized units (minimum 0.050).

## Simultaneous unseen-state / unseen-eta

| scenario | pairs | model | NLL | Brier | Q MAE | Q Spearman |
|---|---:|---|---:|---:|---:|---:|
| Toy | 268 | joint | 0.558 | 0.167 | 0.369 | 0.015 |
| Toy | 268 | eta-only | 1.005 | 0.341 | 0.526 | -0.101 |
| DB | 64 | joint | 0.887 | 0.327 | 0.444 | 0.800 |
| DB | 64 | eta-only | 0.428 | 0.143 | 0.280 | 0.900 |

Toy joint improves absolute error over eta-only but loses essentially all rank information on simultaneous holdout. In DB, eta-only is materially better than the state-aware joint model; the 64-pair DB panel contains only a limited number of held-out eta components.

Joint versus scenario-only is mixed: positive transfer on Toy (NLL 0.558 vs 0.928) and negative transfer on DB (0.887 vs 0.686).

## Finite-candidate ranking

| scenario | K | eligible states | regret | B15 selection | top-3 oracle hit | status |
|---|---:|---:|---:|---:|---:|---|
| Toy | 8 | 32 | 0.236 | 0.719 | 0.969 | OK |
| Toy | 16 | 32 | 0.266 | 0.688 | 0.969 | OK |
| Toy | 32 | 2 | 0.000 | 1.000 | 1.000 | DATA_COVERAGE_INSUFFICIENT |
| DB | 8 | 16 | 0.000 | 1.000 | 1.000 | OK |
| DB | 16 | 16 | 0.113 | 0.562 | 1.000 | OK |
| DB | 32 | 10 | 0.119 | 0.500 | 1.000 | OK |

Toy K=8/16 regret is large despite high top-3 coverage. DB K=8 is trivial/perfect on the highly robust anchors, but K=16/32 degrades sharply. Toy K=32 has only two eligible states and is not considered evidence.

## Old codebook sanity

- Toy: critic robust selection 0.938, legacy selector 1.000, mode agreement 0.188, rank Spearman 0.331.
- DB: both achieve robust selection 1.000, but exact mode agreement is zero because many anchors tie at Q=1; rank Spearman is 0.763.

## Decision

**CONTINUOUS_FIELD_DOES_NOT_GENERALIZE**.

The continuous critic learns useful partial structure, but the joint state-aware field fails the decisive simultaneous holdout: Toy ranking collapses, DB is dominated by eta-only, and finite-candidate regret rises with candidate diversity. It is not ready for eta selection or critic-guided optimization.

**NEW ROLLOUT = 0**.
