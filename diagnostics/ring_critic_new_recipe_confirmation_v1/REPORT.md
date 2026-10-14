# Round17 new-recipe critic confirmation

冻结的通用 critic A3 在新训练配方 controller 上：K1 SUPPORTED，相对 A0 固定规则的平均遗憾差 -4.23pp，95% CI [-5.4466342630931655, -2.957332494765258]；K2 REJECTED，A3/G_E 平均遗憾 1.70/5.45pp、A3 最大遗憾 6.36pp；K3 SUPPORTED，F10 低至少10pp 的新 controller 数 0。

按预注册的联合门槛，本轮尚不能宣称三项全通过的前瞻确认。K1 支持可迁移的 controller 依赖，K3 未发现预设的 10pp 灾难。K2 **仅因最大遗憾门槛**失败：`base_u_v3_macflow__arch512` 的 A3 遗憾为 6.36pp，超过 5pp；A3 的平均遗憾 1.70pp 低于 G_E 的 5.45pp，G_E 最大遗憾为 48.44pp。因此，预注册结论映射中“K1 通过、K2 不通过 ⇒ 结构化规则更好”的文字在本次数据下不能按字面作性能结论；实际证据支持平均选择收益，但未满足 A3 的最坏 controller 约束。

## 事实

全部新 controller、48个新 state 与 Q4 标签为本轮前瞻采集。N2 共享原训练数据集，仅为新架构配方；N1 为新数据集配方，四个 controller 的均值比较只作描述。

| 群体 | controller | A3 fixed 遗憾 | A0 fixed 遗憾 | G_E 遗憾 | A3-all fixed 遗憾 | A3 state 遗憾 | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 4 | 0.00pp | 0.00pp | 0.65pp | 0.00pp | 0.00pp | 15 |
| N2 | 13 | 1.70pp | 5.93pp | 5.45pp | 2.00pp | 1.42pp | 152 |

N2 K1 bootstrap、逐 controller 已知 seed 数、各动作成功率、N1 描述、E/L Pearson/Spearman 及所有策略详值见 `results.json`。图见 `METHOD_REGRET.svg`、`CONTROLLER_CHOICES.svg`、`E_L_SCATTER.svg`。

## 解释

K1/K2 的主判定只针对 N2；K3 根据预标签澄清检查所有 17 个可用新 controller。unknown 不计成功或失败且未重跑；timeout 是单独终止原因，未当作 deadlock。
三项未全通过，故不称完整前瞻确认。K2 的失败原因和规则间实际表现见上文；Q 预测质量与 E/L 相关不作为选择收益的替代证据。

## 证据与执行

历史训练包字节和 N1 1,792 文件校验、17 checkpoint 新颖性/兼容性、R5 冻结、A3/A0/G_E/A3-all 选择冻结、48 state 去重、32例一致性门、DB preflight/单进程 merge/postflight 都在本目录留有审计与收据。纯 Q 实验按 AGENTS.md 例外不运行视频入口。
