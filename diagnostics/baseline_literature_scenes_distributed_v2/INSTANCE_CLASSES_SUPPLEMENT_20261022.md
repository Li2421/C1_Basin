# 补充修订实例归类

只用原封存文件中的九格计数重算，原 INSTANCE_CLASSES_REVISED 不改。安全层全局失败多数判据合并逐机器人滤波器平衡点死锁和被已到达者阻挡；死锁证据计数只用前者。其余候选阈值、严格多数、墙前陷阱优先顺序、W=5s/τ=.05 和九格不变。timeout 不等于死锁。

| 场景 | 层 | 合格实例 | controller 级失败 | 安全层全局失败 | 墙前陷阱 | 有真正死锁类别且可作证据的实例 |
|---|---|---:|---:|---:|---:|---:|
| lit_narrow_passage | central | 44 | 0 | 0 | 0 | 0 |
| lit_narrow_passage | distributed | 44 | 0 | 44 | 0 | 43 |
| lit_crowded_random | central | 44 | 0 | 0 | 0 | 0 |
| lit_crowded_random | distributed | 44 | 0 | 0 | 0 | 40 |
| lit_circle_five | central | 45 | 3 | 0 | 0 | 12 |
| lit_circle_five | distributed | 45 | 3 | 0 | 0 | 42 |
| lit_doorway | central | 45 | 5 | 0 | 0 | 0 |
| lit_doorway | distributed | 45 | 5 | 0 | 0 | 1 |

主归类改变 2 个实例；完整变化、原始互斥计数、九格归类和另一层旁证见 JSON。新 rollout=0，新训练=0。
