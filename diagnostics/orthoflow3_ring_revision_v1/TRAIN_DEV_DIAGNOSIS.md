# Ring 训练/开发诊断与冻结决策

所有决策只使用 64 个训练状态和 16 个开发状态；旧/新测试状态均未参与。

## eta 几何

- 24/80 状态有一个观测稳健连通分量，56/80 有多个分量。
- 多分量质心距离中位数为 1.2673（归一化 eta 坐标）。
- 多数第二分量是与非零修正云分离的 `eta=0` no-op 点，而不是可解释为两个显式协调模式的非零分量。
- generator mean 到最近已验证稳健 eta 的平均距离为 0.1139；标签空间代理下 mean 稳健率为 78/80。

所以没有证据支持“均值落在两个非零模式之间”是主要成因，也没有引入 CW/CCW 或任何模式标签。

## K 缩放（mean + K 个随机 proposal）

| K | oracle 稳健覆盖代理 | 每状态稳健 proposal 均值 | proposal 两两距离均值 | 相对评分成本 |
|---:|---:|---:|---:|---:|
| 1 | 79/80 | 1.9375 | 0.3262 | 2 |
| 4 | 80/80 | 4.6875 | 0.4458 | 5 |
| 8 | 80/80 | 8.4625 | 0.4782 | 9 |
| 16 | 80/80 | 15.9375 | 0.4967 | 17 |

边界饱和率均为 0。按“最小已饱和 K”规则，冻结 `K_FINAL=4`。

## eta=0 候选消融

加入 eta=0 后，冻结 critic 在 train 和 validation 都选择 eta=0 共 0 次；稳健覆盖、rescue、break 和平均 eta 范数均未变化。因此它没有达到“显著降低 break 且不损失 rescue”的预注册采用门槛，最终 proposal set 不加入 eta=0。

## critic

- eta 标签：21,520
- Q MAE：0.09674
- 同状态 pairwise ranking accuracy（Q 差至少 0.25）：0.97216
- train/dev finite-set top-1 accuracy：1.000
- 平均 regret：0
- robust recall at 0.9375：0.92857
- 平均分数：robust 0.9745、boundary 0.5639、clear non-robust 0.1007

全标签中有 449 个 high-score/low-Q 点，显示全局边界/稀疏区校准并不完美；但它们没有造成 train/dev 冻结有限候选集的 top-1 regret。旧测试中的 7 次 oracle/critic miss 因而属于未触发 train/dev 修复门槛的跨分布排序失配，不能用已查看测试标签反向训练。

## 最终冻结决定

- eta=0：不采用
- `K_FINAL=4`
- generator retraining：否
- critic retraining：否
- 新训练/开发标签：0
- 模型 family：不变

这严格执行 R0→R1→R2→R3→R4 的最小修改顺序；train/dev 已在 R0、K=4 饱和，因此没有科学依据用旧测试失败触发 R3/R4。
