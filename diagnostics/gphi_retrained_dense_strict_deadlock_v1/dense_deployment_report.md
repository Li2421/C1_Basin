# Retrained G_phi dense strict-deadlock audit

## Result

**PARTIAL_CLOSED_LOOP_RECOVERY**

Checkpoint `340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700` (seed 41, epoch 1177) was frozen. No training, fine-tuning, eta search, gate, or online oracle control was used. Only dense H1 and the frozen H8-trigger/L8 burst were evaluated. The fixed source eta was evaluated on learned-policy states only as a diagnostic teacher.

## Starting-state approximation

| cohort | executed L2 mean | median | cosine | norm ratio |
|---|---:|---:|---:|---:|
| historical 11 | 0.002766 | 0.002596 | 0.999802 | 0.999872 |
| fresh diagnostic 6 | 0.030753 | 0.005790 | 0.996664 | 0.926247 |

## Closed-loop result

| condition | exact historical | exact fresh | exact combined | robust success | B63 states |
|---|---:|---:|---:|---:|---:|
| dense H1 | 9/11 | 6/6 | 15/17 | 841/1088 | 0/17 |
| H8 + L8 | 9/11 | 6/6 | 15/17 | 840/1088 | 0/17 |

References: oracle H1/L8 = 1088/1088 and 17/17 B63; old G_phi H1/L8 = 45/1088 and 0/17 B63.

## Dense teacher-error growth

| dense step | executed L2 mean | median | cosine | norm ratio |
|---:|---:|---:|---:|---:|
| 0 | 0.012644 | 0.003108 | 0.998695 | 0.973887 |
| 1 | 0.246272 | 0.252384 | 0.570635 | 0.205318 |
| 2 | 0.212428 | 0.219405 | 0.782640 | 0.281586 |
| 4 | 0.204430 | 0.211883 | 0.832371 | 0.294662 |
| 8 | 0.187603 | 0.192121 | 0.826530 | 0.323191 |


## Interpretation

Historical robust H1 rate: 0.7841; fresh diagnostic robust H1 rate: 0.7526. DAgger/on-policy supervision justified: **TRUE**.

Smallest next experiment: Collect fixed-eta teacher labels on the first eight states visited by dense retrained G_phi, then perform one frozen-config on-policy augmentation retrain.

## Runtime/resources

1170 retained executed rollouts, 674433 physical steps; max resumed-shard wall time 148.2 s. Final allocation: 6 GPU shards, 12 CPU cores, 100 GB requested memory, 12% GPU memory cap per process (72% total).
