# Ring Exchange K=16 提议预算诊断

## 范围与冻结条件

本实验只把随机生成器提议数从 K=4 增至 K=16。候选集严格为“生成器均值 + 16 个随机提议”；未加入 eta=0、固定 eta 或历史 basin 中心。生成器、critic、OrthoFlow3、eta 域、Ring MACFlow、环境和修正后的外边界硬安全适配器均未改变。该 60 状态集合已被查看，因此结果仅是诊断，不构成新的 untouched generalization claim。

- P4 ⊂ P16 严格逐值检查：PASS（60/60）
- 提议在查询任何 rollout outcome 前冻结：PASS
- 生成器哈希：`a7bb0476cf25b7e0046a5d977d73f91aa478191c3f59ca73a5c487d4bf5d9965`
- critic 哈希：`f0a2dd358a477590cf434c7803d6ddcc307509157a9698c7dbfaafcf302e02ed`
- Ring safety 哈希：`18419e7a15d3dbeb9b49bacefbaf6e376a7e3551833a89bdfec52c3bab1861be`
- 初始缓存：4767 / 16320 seed tuples 复用，新增缺失 11553
- 结束缓存缺失：0
- 实际执行：11871 次物理 rollout（含数值未认证 tuple 的 3 次额外同条件重试）

## 核心结果

| 指标 | 结果 |
|---|---:|
| B0 hard safety | 35/60 |
| B1-FAIR | 57/60 |
| B4 oracle K=4 | 45/60 |
| B4 oracle K=16 | 60/60 |
| B5 critic K=4 | 39/60 + 1 unresolved |
| B5 critic K=16 | 53/60 |
| NEW_HIT_5_TO_16 | 15 states |
| K=16 oracle-vs-critic miss | 7 |
| K=16 critic exploitation | 4 |

随机提议自身的命中分类：HIT_AT_K4=45，NEW_HIT_5_TO_16=15，NO_HIT_K16=0。均值仅参与 canonical B4/B5，不参与这三个随机提议命中类别。

## 嵌套覆盖曲线

同一批已冻结的 16 个随机提议，canonical oracle 候选始终包含均值：

| 随机 K | Robust states |
|---:|---:|
| 1 | 18/60 |
| 2 | 29/60 |
| 4 | 45/60 |
| 8 | 57/60 |
| 12 | 59/60 |
| 16 | 60/60 |

## Rescue / break

以 B0 的 35 robust / 25 non-robust 状态为基准：

- B4 K=16 rescue：25/25；break：0/35。
- B5 K=16 rescue：23/25；break：5/35。
- 旧 B5 K=4：rescue 16/25；break 12/35。

K=16 critic 相对旧 K=4：旧 robust 变 non-robust 3 个，旧失败/未决变 robust 17 个。平均 critic regret（Q_oracle-Q_selected）为 0.0802。

## 生成器分布诊断

- 每状态 robust 随机提议数：均值 5.267，中位数 5.000。
- 至少一个 robust 随机提议的状态比例：100.0%。
- 归一化 eta 的平均协方差 trace：0.1536。
- 平均两两距离：0.5002；状态中位两两距离的总体中位数：0.4753。
- eta 域边界饱和比例：0.00%。

这些组件完全由 eta 空间和 Q16 证据定义，没有使用 CW/CCW 或其他模式标签。

## Critic distractor / exploitation

K=16 oracle 有 robust 候选而 critic 未选中的状态有 7 个。高分低 Q 的 exploitation 预注册定义为 critic score ≥ 0.9375 且实际 Q16 < 0.5；计数为 4。新增提议既可能提供新 robust 候选，也可能成为 critic distractor；逐状态详情见 `critic_distractor_analysis.json` 和 `per_state_results.json`。

实际结果是两者同时发生：B5 robust 数相对旧 K=4 净增 14 个，但有 3 个旧 K=4 robust 状态因新增候选改变 critic 选择后变为 non-robust。故更大的提议预算明显帮助 critic，同时也引入了可测的 distractor。

## 安全与数值

- B5 K=16 collision seeds：0（obstacle=0，outer-boundary=0，agent=0）。
- B5 K=16 numerical-failure seeds：1。
- B5 K=16 numerical unresolved states：0。
- 全部 mean+16 候选的 collision seeds：0；numerical-failure seeds：106。
- DB integrity：`ok`；foreign-key violations=0。

## 结论

- 生成器提议行为：`K4_WAS_TOO_SMALL`
- Critic 效应：`CRITIC_BENEFITS_FROM_K16`

B4 K=16 比 B1-FAIR 多 3 个 robust 状态；B5 K=16 比 B1-FAIR 少 4 个状态。K=16 把 oracle 从 45/60 提高到 60/60，说明是 `LOW_PROBABILITY_BASIN`，而不是 generator distribution 完全漏掉 robust basin。本诊断到此停止；未进行重训、调参、数据生成或新测试集评估。
