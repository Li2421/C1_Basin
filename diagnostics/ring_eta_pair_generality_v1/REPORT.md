# Round22 eta pair generality

P1 INCONCLUSIVE（5/7 对达支配门），P2 INCONCLUSIVE（6/7 对达方向共享门），P3 INCONCLUSIVE（3/7 对达严格收益 CI 门）。unknown导致至少一项判定未定，不能声称一般性。

14 controller ×32 新state ×26去重臂 ×2seed =23296 条；unknown 329保留并未重跑；32/32一致性门通过，DB/raw/trace postflight通过。

| η pair | dominance | direction agreement | shared direction | gain (pp) | 95% CI (pp) | schedule ties |
| --- | --- | --- | --- | ---: | --- | --- |
| [13, 10] | 14/14 | 0.929 | S13to10 | 28.24 | [15.29, 41.96] | 0 |
| [4, 7] | 12/14 | 0.857 | S7to4 | 43.30 | [22.32, 63.06] | 0 |
| [5, 9] | 14/14 | 1.000 | S5to9 | 0.00 | [0.0, 0.0] | 14 |
| [7, 15] | 13/14 | 1.000 | S7to15 | 2.12 | [-0.45, 6.03] | 10 |
| [6, 11] | 13/14 | 0.929 | S11to6 | 37.39 | [22.32, 51.79] | 1 |
| [3, 8] | 8/14 | 0.643 | S3to8 | -3.46 | [-7.7, 0.33] | 2 |
| [2, 6] | 9/14 | 0.929 | S2to6 | 5.25 | [-1.56, 13.17] | 2 |

锚点复现：{'mean_gain': 0.28236607142857145, 'CI': [0.15290178571428573, 0.41964285714285715], 'direction': 'S13to10', 'replicated': True}。随机对最佳时间表相对锚点的描述性平均差 -2.68pp。

事实与解释：P1 的“好方向”逐controller取两个时间表最佳，这是oracle描述；P2 依冻结并列规则计方向，平局很多时方向一致率会高估严格排序证据。P3要求严格正收益，尤其用于排除两固定端点都失败且时间表也失败的空支配。固定偏好符号、方向一致率差及CI见results.json。不得把低成功率的支配率单独写成部署价值。

去重范围：对抽取时全部13161个DB state核对UID与content_hash；对可恢复positions/velocities/goals的state另核对物理签名。4607个历史state缺完整可恢复物理三元组，因此无法额外证明它们与本轮在跨namespace物理表示上不同；该缺口见STATE_DEDUP_AUDIT.json。

范围仅Ring、已有14配方、随机6对及锚点、t500/Q2；纯Q数据生成，无视频入口。所有新文件仅本目录，共享DB只执行授权生命周期，未修改历史资产/代码，未做commit。

## Unknown sensitivity and failure timing

下界点判：P1 5/7、P2 6/7、P3 3/7；unknown兼容的达门对数范围分别为3–6、4–7、3–7。因此最终三项均为INCONCLUSIVE；下界点判不能替代最终判定。

未知共329条（1.412%），错误类型{'CBFSolverError': 329}；首个错误步范围37–698，切换前246条。所有已知非成功均为原生timeout@700，不直接称为deadlock。逐例见FAILURE_RECORDS.json与UNKNOWN_CASES.json。

| η pair | best single direction mean gain pp | two direction means minus anchor pp |
| --- | ---: | --- |
| [13, 10] | 28.24 | [0.0, -49.0] |
| [4, 7] | 43.30 | [-61.38, -6.7] |
| [5, 9] | 0.00 | [-89.62, -89.62] |
| [7, 15] | 2.12 | [-82.37, -86.16] |
| [6, 11] | 37.39 | [-84.49, -42.08] |
| [3, 8] | -3.46 | [-69.53, -73.66] |
| [2, 6] | 5.25 | [-60.49, -81.25] |

上述最大平均收益是在两个已注册方向中取较大均值；results.json另保留逐controller最大收益。共享方向与其CI依下界点判选择，属于注册的样本内多数方向统计；不是部署时已知的方向。

官方只读cache postflight：22967条已知标签可复用（exact+partial）；planner排除numerical_failure，故报告329条missing。逐身份核对这些missing恰好是329条已留存unknown，无真实记录缺口；ambiguous=0，绝不据此重跑。
