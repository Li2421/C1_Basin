# Toy–DB conditional generator

Classification: **CROSS_SCENARIO_GENERATOR_TRANSFER_SUPPORTED**

The 12-mode selector, anchors, transform, splits, and normalizations were frozen. The generator dataset contains 3,610 cached state–eta records. Toy has genuine local off-anchor support; the DB panel has anchor outcomes only, so its mode-local labels are weak/unsupported and no continuous targets were fabricated. Twenty-one joint/single-scenario models were trained. VAL closed-loop selection froze joint seed 17 before TEST.

## TEST

| Scenario/controller | B63 | mean Q64 | J_def | collision |
|---|---:|---:|---:|---:|
| Toy anchor | 32/32 | 1.0000 | 0.4937 | 0 |
| Toy generator mean | 32/32 | 1.0000 | 0.5006 | 0 |
| DB anchor | 16/16 | 1.0000 | 0.6152 | 0 |
| DB generator mean | 16/16 | 1.0000 | 0.6111 | 0 |

Median generated mean distance from anchor is 0.0259; fraction beyond 0.05 is 0.104. The sampled distribution is nontrivial: 18/48 promoted samples are B63 and farther than 0.05 from every frozen mode (maximum anchor distance 0.1462).

## Cross-scenario/data efficiency

The joint generator achieves 16/16 DB B63 at 25%, 50%, and 100% DB TRAIN data. DB-only achieves 16/16, 15/16, and 15/16 respectively. Joint-25% therefore matches DB-only-full, but DB-only-25% also reaches 16/16 and Q64=1.0: a causal Toy label-efficiency gain is **not demonstrated**.

The generator transfers across scenarios and preserves robustness, but fixed anchors already saturate both TEST panels. Toy J_def increases by 0.0069; DB J_def decreases by 0.0041; aggregate improvement is absent. Therefore the frozen selector + anchors remains the primary controller; the generator is a validated optional diversity/capacity layer rather than the recommended default.
