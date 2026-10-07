# Double-Bottleneck shared-mode transfer

Classification: **SHARED_MODES_TRANSFER_STRONGLY_SUPPORTED**

All 12 frozen Toy mode IDs received one-to-one DB correspondences. The selected scenario transform is `anisotropic` in normalized eta space, with condition number 7.22, median residual 0.114, and maximum residual 0.203. Exact A, b, transformed anchors, and correspondence are in `selected_transform.json` and `mode_correspondence.csv`.

Raw Toy codebook on held-out DB: 6/16 oracle-coverable, mean best Q64 0.869.
Transformed codebook oracle: 16/16 B63, mean best Q64 1.000.
TRAIN-prior best fixed transformed mode: 16/16 B63, mean Q64 1.000.
DB selector: 16/16 B63, mean Q64 1.000; coverable-state B63 selection accuracy 1.0, mean regret 0.000. It selected six distinct modes on TEST (normalized entropy 0.607), so the network did not collapse to one output.

Toy versus DB mean feasible-mode count: 7.12 versus 9.12; best-single coverage 90.6% versus 100.0%. DB is not uniformly sparser/harder on both preregistered measures.
DB-native oracle diagnostic: not needed because transformed coverage was adequate.
Fresh replication: safety 0/32 B63, mean Q64 0.695; fixed 32/32 B63, mean Q64 1.000; selector 32/32 B63, mean Q64 1.000.
The fresh fixed mode and selector were exactly tied at 32/32 B63 and Q64=1.000. Thus the experiment strongly supports the cross-scenario transformed mode structure, but it does **not** establish that state-conditioned mode selection is needed on this DB cohort; the TRAIN-prior fixed transformed mode was already sufficient.

## Direct answers

1. Correspondence: 12/12 Toy modes.
2. T_DB: anisotropic; condition number 7.22.
3. Raw Toy DB coverage: 6/16, mean best Q64 0.869.
4. Transformed DB oracle coverage: 16/16, mean best Q64 1.000.
5. Best fixed transformed: 16/16, mean Q64 1.000.
6. DB selector: 16/16, mean Q64 1.000.
7. Oracle-coverable selector accuracy: 1.0.
8. DB harder than Toy: NO on the two preregistered sparsity measures.
9. DB-native clearly better: not evaluated because unnecessary.
10. Fresh replication: safety 0/32 B63, mean Q64 0.695; fixed 32/32 B63, mean Q64 1.000; selector 32/32 B63, mean Q64 1.000.
11. Shared mode identity + scenario deformation: SUPPORTED; selector necessity on these DB states: NOT DEMONSTRATED.


## Frozen design and transform

The source-isolated DB split was 64 TRAIN / 16 VAL / 16 TEST with 96 unique source groups. The DB learner input was the exact true-t0 80-D vector `obs[4,18] + current raw Flow action[4,2]`; it was not forced into the Toy 214-D schema.

With `z=(eta-[0.875,0,0.375])/[0.75,1,0.75]`, the frozen transform is `z_DB=A z_Toy+b`, where:

```text
[[ 0.449369428  0.521099611 -0.218526119]
 [ 0.169040086 -0.050630166  0.226874877]
 [ 0.051641908 -0.066932941 -0.053414369]]
[-0.542748424  0.389883298 -0.432917915]
```

The experiment generated 53,069 rollout records. One projection-solver result was scientifically invalid; its entire matched VAL future block was deterministically replaced before aggregation (`numerical_repair_manifest.json`).

## Held-out TEST controller comparison

| Controller | B63 | mean Q64 | success/trials | deadlock | timeout | collision | J_def | episode length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 0/16 | 0.700 | 717/1024 | 0 | 307 | 0 | 0.0000 | 837.7 |
| best_fixed_transformed | 16/16 | 1.000 | 1024/1024 | 0 | 0 | 0 | 0.7457 | 717.2 |
| DB_selector | 16/16 | 1.000 | 1024/1024 | 0 | 0 | 0 | 0.6920 | 723.9 |
| transformed_oracle | 16/16 | 1.000 | 1024/1024 | 0 | 0 | 0 | 0.7457 | 717.2 |

Selector versus Safety on matched TEST continuations: 307 rescues, 0 breaks.

## Feasibility-model diagnostics

| Model | TEST NLL | TEST Brier | top-1 empirical Q | within-state rank correlation |
|---|---:|---:|---:|---:|
| global_prior | 0.1303 | 0.0053 | 1.0000 | 0.908 |
| linear | 0.1369 | 0.0023 | 0.9971 | 0.885 |
| nearest_neighbor | 0.1426 | 0.0037 | 1.0000 | 0.916 |
| mlp_seed23 | 0.1272 | 0.0021 | 1.0000 | 0.722 |

## Fresh 32-state replication

| Controller | B63 | mean Q64 | success/trials | timeout | collision | J_def | episode length |
|---|---:|---:|---:|---:|---:|---:|---:|
| safety | 0/32 | 0.695 | 1423/2048 | 625 | 0 | 0.0000 | 835.7 |
| fixed | 32/32 | 1.000 | 2048/2048 | 0 | 0 | 0.7252 | 711.3 |
| selector | 32/32 | 1.000 | 2048/2048 | 0 | 0 | 0.6140 | 719.8 |

Selector vs fixed: 0 rescues / 0 breaks. Selector vs Safety: 625 rescues / 0 breaks.
Fresh selector used modes {"0": 1, "5": 8, "6": 3, "7": 1, "8": 4, "9": 15}; normalized selection entropy 0.563.

No subsequent stage was started.
