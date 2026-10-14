# Round16 controller critic report

通用 critic 在独立训练的新 controller 上：Q1 REJECTED，A1 相对 A0 的遗憾差 -0.28pp（95% CI -0.61 至 0.00）；Q2 REJECTED，A1 遗憾 1.51pp、G_E 0.89pp、A1 最大遗憾 7.81pp；逐 state 价值 Q3 REJECTED，遗憾差 -0.28pp（95% CI -0.61 至 0.00）。

全部结论为 DEVELOPMENT。T1 是冻结模型的外部零样本测试；T1 标签此前已由 round14A 打开，不是全新前瞻确认。

## 事实

| 测试集 | A0 state 遗憾 | A1 state 遗憾 | A1 fixed 遗憾 | A3 state 遗憾 | G_E 遗憾 | A1 最大遗憾 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 1.79pp | 1.51pp | 1.79pp | 0.67pp | 0.89pp | 7.81pp |
| T2 | 1.20pp | 1.09pp | 1.03pp | 0.89pp | 0.64pp | 5.47pp |
| T3 | 3.91pp | 3.52pp | 2.00pp | 1.32pp | 1.42pp | 16.80pp |

Q4（描述性）：T1 上 A3 state 与 A1 state 的遗憾分别为 0.67pp / 1.51pp；灾难 controller 数分别为 0 / 0。
T2 的 A2 state 遗憾 1.17pp（冻结后发现兼容 R6 特征，描述性补充）。

## 源数据配方交叉验证

| 模型 | state 遗憾 | 最大遗憾 | NLL | Brier |
|---|---:|---:|---:|---:|
| A0 | 2.96pp | 25.00pp | 0.4451 | 0.1396 |
| A1 | 2.78pp | 25.00pp | 0.3858 | 0.1177 |
| A2 | 2.82pp | 25.00pp | 0.3860 | 0.1176 |
| A3 | 1.45pp | 7.03pp | 0.3720 | 0.1121 |

Q 预测质量、校准、E/L 相关、unknown 上界敏感性、每 controller 选择和全部 CV 折见 `results.json`；图为 `REGRET_BY_TEST.svg`、`CONTROLLER_CHOICES.svg` 和 `CHOICES_BY_CONTROLLER.svg`。

## 外部 Q 质量与 E/L 推出值（描述性）

| 测试集 | 模型 | NLL | Brier | E Spearman | L Spearman |
|---|---|---:|---:|---:|---:|
| T1 | A0 | 0.5237 | 0.1709 | 未定义 | 未定义 |
| T1 | A1 | 0.4682 | 0.1476 | 0.681 | 0.792 |
| T1 | A3 | 0.4820 | 0.1498 | 0.665 | 0.638 |
| T2 | A0 | 0.4740 | 0.1526 | 未定义 | 未定义 |
| T2 | A1 | 0.4247 | 0.1314 | 0.881 | 0.930 |
| T2 | A2 | 0.4247 | 0.1318 | 0.807 | 0.952 |
| T2 | A3 | 0.4352 | 0.1315 | 0.811 | 0.956 |
| T3 | A0 | 0.5284 | 0.1722 | 未定义 | 未定义 |
| T3 | A1 | 0.4744 | 0.1474 | 0.929 | 0.755 |
| T3 | A3 | 0.4449 | 0.1320 | 0.929 | 0.934 |

A0 没有 controller 输入，同一 state 的 E/L 推出值为常数，相关系数未定义。十档校准和每档样本数在 `results.json`。这些 Q 指标不能替代选择收益的检验。

## 判定解释

Q1 未通过严格 CI 上界 < 0：A1 比 A0 的 T1 平均遗憾低 0.28pp，但配对区间触及 0。Q2 的两个门槛均未通过：A1 比本轮冻结的 Round15 源训练 G_E 高 0.61pp，最大遗憾 7.81pp。Q3 的 state 选择改善同样只有 0.28pp，区间触及 0；逐 state 的额外价值未获支持。

Q4 为描述性：T1 的 A3 遗憾 0.67pp，优于 A1 的 1.51pp，两者均无相对 F10 低至少 10pp 的 controller。旧动作空间中“无约束通路有害”的现象在本轮数据上未复现；本轮无 Q4 推断门槛。T3 的 A1 有 1 个相对 F10 的灾难 controller，A3 为 0。

G_E 是本轮只用 Round15 训练标签拟合并在外部标签开放前冻结的 R5-PCA2 Ê ridge；它与 round14A 历史报告中的 G_E 训练来源不同，故报告遗憾不能直接与历史 0.61pp 对比。T2 有 11 个 controller×action 标签位不可用，选择菜单逐 controller 限于可用动作；Freg 仅入 oracle。unknown 按预注册下界计失败，上界敏感性在 `results.json`，T1/T2/T3 unknown 数分别为 183/131/59。

## 解释与范围

Q1 只检验 controller 表示对选择遗憾是否有增益；NLL/Brier 不作为其替代证据。Q2 按预注册的 0.5pp 与 5pp 双门槛判定。Q3 用配对 state 簇 bootstrap。T2/T3 为补充外部数据；T2 的 A2 预测使用事后发现的 Round11 冻结 R6 数组，仅作描述性补充，见 `T2_A2_EXTENSION_PLAN.receipt.json`。T1/T3 仍无兼容 R6。

训练和评估完全复用历史 rollout 标签及冻结探测特征；未新增 rollout、探测或共享 DB 写入。模型与规则冻结收据见 `MODEL_FREEZE.receipt.json`。
