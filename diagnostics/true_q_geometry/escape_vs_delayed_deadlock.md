# TRUE_ESCAPE、DELAYED_DEADLOCK 与 timeout 分析

本分析只比较相同增广起点、相同 Flow 随机种子下的完整闭环策略。`G_phi` 在每个物理步从当前状态重新计算，并持续到 collision、success、strict deadlock 或 timeout；没有在短时域后关闭 corrector。

## 结果

在 `p0` 最终严格死锁的 160 个同状态/同种子策略比较中：

| 分类 | 数量 |
|---|---:|
| `TRUE_ESCAPE` | 0 |
| `DELAYED_DEADLOCK` | 22 |
| `TIMEOUT` | 47 |
| `NO_EFFECT` | 91 |

因此，本研究没有发现任何“原本严格死锁、换闭环策略后成功”的样本。降低 `Q_D` 的主要机制是让 episode 活到 H 并 timeout，而不是完成任务。该事实不改变 `Q_D` 的数学定义——timeout 的 deadlock value 确实为 0——但它禁止把低 `Q_D` 解释为任务成功或真正逃逸。

## 代表性延迟死锁

`D6_pair226`、seed `51003`：

- `p0` 在 94 步（4.70 s）后严格死锁；
- `goal_m1` 在 578 步（28.90 s）后仍然严格死锁；
- 两条轨迹第一步就出现大于 `1e-3` 的执行动作和状态差异；
- 共同前 94 步的执行动作 RMS 差为约 `0.0384`，状态 RMS 差为约 `0.1020`；
- active-set signature 在共同前缀的分歧比例为 1.0。

这是明确的 `DELAYED_DEADLOCK`，不能算风险消除。

## 代表性 timeout 替代

`D1_pair231`、seed `51003`：

- `p0` 在 20 步（1.00 s）后严格死锁；
- `rel_p1` 运行至剩余 horizon 的 367 步（18.35 s）并 timeout；
- 共同 20 步内状态轨迹 RMS 差只有约 `0.0191`，但执行动作从第一步起已经不同；
- 没有 success，因此这是 `TIMEOUT` 而不是 `TRUE_ESCAPE`。

## 最早物理差异

所有代表性的 delayed-deadlock/timeout 分支都在第一个物理步（0.05 s）出现可测的执行动作与状态差异，并伴随 active-set 变化。可见结果差异不是“只改第一步”的实验伪影，而是闭环状态反馈被投影后持续积累的结果。

但由于 `TRUE_ESCAPE=0`，本数据无法回答“真正逃逸相对延迟死锁的最早独有征兆”。能观察到的是：

- 100 步持久状态位移对 `Q_D` 有强排序关系；
- task progress 的符号在不同起点并不稳定，因为 timeout 策略可以持续移动但不接近目标；
- timer 延迟本身明显不充分，22 个样本只是把严格死锁推迟。

完整逐对数据位于 `raw/escape_delay_summary.json`，投影后的轨迹差异位于 `projection_role.json`。

