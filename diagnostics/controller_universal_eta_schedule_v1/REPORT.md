# Round10 universal Ring eta schedule

**H_schedule 未获普适支持；按预注册映射标为INCONCLUSIVE，G_universal=未通过，held-out组=INCONCLUSIVE。** 全14个controller平均增益26.09pp，95%未知稳健区间[12.02, 40.26]pp；held-out10个平均增益30.16pp，区间[12.66, 47.62]pp。普适点门的明确缺口是v9_local：S500为234/256=91.41%，F10为247/256=96.48%，低5.08pp、无unknown，超过预定5pp阈值。Held-out的v3低于Freg点率5.86pp，但unknown可行差值为−7.42至−3.91pp，跨越5pp界，所以held-out未通过点门且未知稳健判定INCONCLUSIVE。均值收益成立，逐controller普适不劣未建立；不能把controller regime差异全部归因于固定η的动作空间限制。前瞻终点在首次读取新state结果前已预注册，机制解释为DEVELOPMENT。数值完成，正式视频仍缺['deadlock_show', 'deadlock_resolution']；状态NUMERICAL_COMPLETE_VIDEO_INCOMPLETE。

18,688主请求+32一致性门；主有效18557，unknown 131，missing/quarantined均0，18,688份真实轨迹hash已核验。unknown保留、不重跑、不当作失败，区间保留所有请求分母；timeout保持timeout。本轮只检验锁定的η13→η10，主500步，次300/400步，没有看到结果后换主时间表。

预注册原种子2026160000..0063与round4 loco_confirmation64状态全部重叠，原文件和失败审计完整保留；没有用这些状态作前瞻终点。在任何新rollout/终点读取之前，PREREGISTRATION_V2重新锁定2026190000..0063并留UTC/SHA，时间表、臂、阈值全部不变。修订64个状态的全DB UID/content和可恢复物理坐标均无重叠。旧DB缺物理坐标的记录在审计中单列，不能把它们假称已完成绝对坐标核验。新结果在全cohort结束、ENDPOINT_OPENING.receipt写出后才由merger/分析打开，WATCH_HISTORY只有文件数量/资源，没有中期结局分析。

| 组 | N controller | S500−oracle平均pp | 跨state/controller bootstrap95%未知稳健pp | >5pp亏损controller | E3 vsF10 | 点门 | 未知稳健门 |
|---|---|---|---|---|---|---|---|
| all | 14 | 26.09 | [12.024274553571429, 40.26297433035713] | ['base_u_v3_macflow', 'base_u_v9_local_macflow'] | 0 | False | FAIL |
| dev | 4 | 15.92 | [-2.34375, 35.842285156249964] | ['base_u_v9_local_macflow'] | 0 | False | FAIL |
| held_out | 10 | 30.16 | [12.65625, 47.6171875] | ['base_u_v3_macflow'] | 0 | False | INCONCLUSIVE |

Held-out门按要求只强制整体正收益和逐controller5pp点门，另报告E3；all/dev门同时要求E3=0。控制器等权，state64成簇、controller层独立重抽，10000次，oracle在每次bootstrap中重新取max。逐controller CI只重抽state。点率为known-success下界；结论采用unknown可行范围检查，不猜填unknown。max oracle仅限F10/F13及三项预注册Freg，不是完整K16最优η。

G_universal的FAIL不依赖v3的unknown：v9的必要点门已经失败。H_schedule保留INCONCLUSIVE是遵守预注册的结论映射：dev=FAIL、held-out=未知可翻转的混合情形，不事后重写成“两组都不通过”。这不表示本批数据的普适门可能通过，也不把点估计低5.08pp误写成总体真实损失必然超过5pp的95%证明；v9的配对CI为−8.59至−1.95pp，跨越−5pp边界。

| controller | 组 | oracle固定臂 | F10 | F13 | Freg | S500 | S500−oracle pp / 95%未知稳健pp |
|---|---|---|---|---|---|---|---|
| base_u_plus_w_v4_wide_macflow | dev | F10 | 61.72–63.28% | 21.88–21.88% | — | 94.53–94.53% | 32.81 / [21.09375, 42.96875] |
| base_u_macflow | dev | F13 | 26.56–27.73% | 53.12–54.69% | — | 87.89–89.45% | 34.77 / [24.609375, 45.3125] |
| base_u_v4_macflow | held_out | F10 | 38.67–39.06% | 26.95–27.34% | — | 87.50–87.89% | 48.83 / [35.9375, 60.9375] |
| base_u_plus_w_v6_macflow | held_out | F10 | 36.33–36.72% | 5.86–6.25% | — | 90.23–90.62% | 53.91 / [44.53125, 63.671875] |
| base_u_plus_w_v3_macflow | held_out | Freg | 83.20–85.16% | 60.94–61.72% | 92.97–94.14% | 95.31–96.09% | 2.34 / [-2.34375, 6.640625] |
| base_u_plus_w_v4_train_targeted_macflow | held_out | F10 | 39.06–39.06% | 6.25–7.42% | — | 77.73–78.91% | 38.67 / [26.953125, 51.5625] |
| base_u_v3_macflow | held_out | Freg | 66.41–69.53% | 54.30–56.64% | 88.28–89.84% | 82.42–84.38% | -5.86 / [-13.28125, 1.5625] |
| base_u_v2_macflow | held_out | F10 | 37.11–37.50% | 23.44–25.00% | — | 86.72–88.67% | 49.61 / [39.453125, 61.328125] |
| base_u_v10_local_macflow | dev | F13 | 71.48–71.48% | 94.14–94.14% | — | 95.31–95.31% | 1.17 / [0.0, 2.734375] |
| base_u_plus_w_v5_macflow | held_out | F10 | 30.47–31.25% | 12.50–12.89% | — | 92.97–93.75% | 62.50 / [50.78125, 73.828125] |
| base_u_v7_local_macflow | held_out | F10 | 97.27–97.27% | 38.67–38.67% | — | 97.27–97.27% | 0.00 / [-3.125, 3.515625] |
| base_u_plus_w_v4_macflow | held_out | F10 | 41.80–42.58% | 4.30–4.69% | — | 93.75–94.14% | 51.95 / [39.453125, 64.0625] |
| base_u_v8_local_macflow | held_out | F10 | 100.00–100.00% | 34.38–34.38% | 100.00–100.00% | 99.61–99.61% | -0.39 / [-1.171875, 0.0] |
| base_u_v9_local_macflow | dev | F10 | 96.48–96.48% | 73.44–73.44% | — | 91.41–91.41% | -5.08 / [-8.59375, -1.953125] |

![14 Flow arm comparison](ARM_COMPARISON_all.png)

![held-out Flow comparison](ARM_COMPARISON_held_out.png)

S300/S400为次要臂，完整同样门指标保存在results.group_metrics，不能替代失败的S500主终点。

- S300：all=INCONCLUSIVE，held-out=INCONCLUSIVE，all平均增益22.96pp；held-out平均26.21pp。
- S400：all=INCONCLUSIVE，held-out=INCONCLUSIVE，all平均增益25.14pp；held-out平均28.87pp。

时间表增益与F10−F13的Pearson=-0.09540470526171456，CI=[-0.6143703601298113, 0.44985925770862323]；Spearman=-0.0945054945054945，CI=[-0.6454616795760294, 0.47982511210762324]。这是预定14点相关描述，不能证明regime由某个失败阶段造成；unknown点下界和小controller样本的限制保留。

| controller / arm | 结局计数 | success/timeout等真实发生步数摘要 |
|---|---|---|
| base_u_plus_w_v4_wide_macflow/F10 | {'success': 158, 'timeout': 94, 'numerical_failure': 4} | {'success': {'n': 158, 'min': 375, 'median': 566.0, 'max': 698}, 'timeout': {'n': 94, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 4, 'min': 298, 'median': 466.0, 'max': 616}} |
| base_u_plus_w_v4_wide_macflow/F13 | {'timeout': 200, 'success': 56} | {'timeout': {'n': 200, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 56, 'min': 281, 'median': 522.0, 'max': 699}} |
| base_u_plus_w_v4_wide_macflow/S300 | {'success': 238, 'timeout': 18} | {'success': {'n': 238, 'min': 281, 'median': 345.5, 'max': 696}, 'timeout': {'n': 18, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v4_wide_macflow/S400 | {'success': 240, 'timeout': 16} | {'success': {'n': 240, 'min': 281, 'median': 426.0, 'max': 684}, 'timeout': {'n': 16, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v4_wide_macflow/S500 | {'success': 242, 'timeout': 14} | {'success': {'n': 242, 'min': 281, 'median': 520.0, 'max': 684}, 'timeout': {'n': 14, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_macflow/F10 | {'success': 68, 'timeout': 185, 'numerical_failure': 3} | {'success': {'n': 68, 'min': 510, 'median': 612.5, 'max': 700}, 'timeout': {'n': 185, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 3, 'min': 544, 'median': 621.0, 'max': 678}} |
| base_u_macflow/F13 | {'success': 136, 'timeout': 116, 'numerical_failure': 4} | {'success': {'n': 136, 'min': 311, 'median': 514.0, 'max': 700}, 'timeout': {'n': 116, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 4, 'min': 211, 'median': 243.0, 'max': 284}} |
| base_u_macflow/S300 | {'success': 209, 'timeout': 42, 'numerical_failure': 5} | {'success': {'n': 209, 'min': 308, 'median': 433.0, 'max': 691}, 'timeout': {'n': 42, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 5, 'min': 211, 'median': 259.0, 'max': 679}} |
| base_u_macflow/S400 | {'success': 218, 'timeout': 34, 'numerical_failure': 4} | {'success': {'n': 218, 'min': 311, 'median': 437.0, 'max': 699}, 'timeout': {'n': 34, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 4, 'min': 211, 'median': 243.0, 'max': 284}} |
| base_u_macflow/S500 | {'success': 225, 'timeout': 27, 'numerical_failure': 4} | {'success': {'n': 225, 'min': 311, 'median': 511.0, 'max': 694}, 'timeout': {'n': 27, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 4, 'min': 211, 'median': 243.0, 'max': 284}} |
| base_u_v4_macflow/F10 | {'timeout': 156, 'success': 99, 'numerical_failure': 1} | {'timeout': {'n': 156, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 99, 'min': 374, 'median': 601.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 39, 'median': 39.0, 'max': 39}} |
| base_u_v4_macflow/F13 | {'timeout': 186, 'success': 69, 'numerical_failure': 1} | {'timeout': {'n': 186, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 69, 'min': 322, 'median': 562.0, 'max': 693}, 'numerical_failure': {'n': 1, 'min': 230, 'median': 230.0, 'max': 230}} |
| base_u_v4_macflow/S300 | {'success': 206, 'timeout': 48, 'numerical_failure': 2} | {'success': {'n': 206, 'min': 310, 'median': 405.0, 'max': 697}, 'timeout': {'n': 48, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 230, 'median': 425.5, 'max': 621}} |
| base_u_v4_macflow/S400 | {'success': 216, 'timeout': 38, 'numerical_failure': 2} | {'success': {'n': 216, 'min': 322, 'median': 443.0, 'max': 700}, 'timeout': {'n': 38, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 230, 'median': 396.0, 'max': 562}} |
| base_u_v4_macflow/S500 | {'success': 224, 'timeout': 31, 'numerical_failure': 1} | {'success': {'n': 224, 'min': 322, 'median': 522.0, 'max': 700}, 'timeout': {'n': 31, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 230, 'median': 230.0, 'max': 230}} |
| base_u_plus_w_v6_macflow/F10 | {'timeout': 162, 'success': 93, 'numerical_failure': 1} | {'timeout': {'n': 162, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 93, 'min': 382, 'median': 589.0, 'max': 697}, 'numerical_failure': {'n': 1, 'min': 261, 'median': 261.0, 'max': 261}} |
| base_u_plus_w_v6_macflow/F13 | {'timeout': 240, 'success': 15, 'numerical_failure': 1} | {'timeout': {'n': 240, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 15, 'min': 411, 'median': 540.0, 'max': 654}, 'numerical_failure': {'n': 1, 'min': 307, 'median': 307.0, 'max': 307}} |
| base_u_plus_w_v6_macflow/S300 | {'success': 207, 'timeout': 49} | {'success': {'n': 207, 'min': 302, 'median': 384.0, 'max': 684}, 'timeout': {'n': 49, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v6_macflow/S400 | {'success': 222, 'timeout': 33, 'numerical_failure': 1} | {'success': {'n': 222, 'min': 408, 'median': 443.0, 'max': 686}, 'timeout': {'n': 33, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 307, 'median': 307.0, 'max': 307}} |
| base_u_plus_w_v6_macflow/S500 | {'success': 231, 'timeout': 24, 'numerical_failure': 1} | {'success': {'n': 231, 'min': 411, 'median': 530.0, 'max': 697}, 'timeout': {'n': 24, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 307, 'median': 307.0, 'max': 307}} |
| base_u_plus_w_v3_macflow/F10 | {'success': 213, 'timeout': 38, 'numerical_failure': 5} | {'success': {'n': 213, 'min': 304, 'median': 430.0, 'max': 696}, 'timeout': {'n': 38, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 5, 'min': 104, 'median': 198.0, 'max': 375}} |
| base_u_plus_w_v3_macflow/F13 | {'success': 156, 'timeout': 98, 'numerical_failure': 2} | {'success': {'n': 156, 'min': 279, 'median': 481.5, 'max': 697}, 'timeout': {'n': 98, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 199, 'median': 220.0, 'max': 241}} |
| base_u_plus_w_v3_macflow/Freg | {'success': 238, 'numerical_failure': 3, 'timeout': 15} | {'success': {'n': 238, 'min': 278, 'median': 391.0, 'max': 693}, 'numerical_failure': {'n': 3, 'min': 46, 'median': 60.0, 'max': 102}, 'timeout': {'n': 15, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v3_macflow/S300 | {'success': 239, 'timeout': 15, 'numerical_failure': 2} | {'success': {'n': 239, 'min': 279, 'median': 362.0, 'max': 695}, 'timeout': {'n': 15, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 199, 'median': 220.0, 'max': 241}} |
| base_u_plus_w_v3_macflow/S400 | {'success': 243, 'numerical_failure': 2, 'timeout': 11} | {'success': {'n': 243, 'min': 279, 'median': 420.0, 'max': 696}, 'numerical_failure': {'n': 2, 'min': 199, 'median': 220.0, 'max': 241}, 'timeout': {'n': 11, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v3_macflow/S500 | {'success': 244, 'numerical_failure': 2, 'timeout': 10} | {'success': {'n': 244, 'min': 279, 'median': 511.0, 'max': 677}, 'numerical_failure': {'n': 2, 'min': 199, 'median': 220.0, 'max': 241}, 'timeout': {'n': 10, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v4_train_targeted_macflow/F10 | {'success': 100, 'timeout': 156} | {'success': {'n': 100, 'min': 440, 'median': 606.0, 'max': 697}, 'timeout': {'n': 156, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v4_train_targeted_macflow/F13 | {'timeout': 237, 'success': 16, 'numerical_failure': 3} | {'timeout': {'n': 237, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 16, 'min': 300, 'median': 574.0, 'max': 698}, 'numerical_failure': {'n': 3, 'min': 114, 'median': 285.0, 'max': 285}} |
| base_u_plus_w_v4_train_targeted_macflow/S300 | {'timeout': 67, 'success': 184, 'numerical_failure': 5} | {'timeout': {'n': 67, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 184, 'min': 300, 'median': 408.5, 'max': 696}, 'numerical_failure': {'n': 5, 'min': 114, 'median': 285.0, 'max': 494}} |
| base_u_plus_w_v4_train_targeted_macflow/S400 | {'timeout': 59, 'success': 194, 'numerical_failure': 3} | {'timeout': {'n': 59, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 194, 'min': 300, 'median': 449.0, 'max': 696}, 'numerical_failure': {'n': 3, 'min': 114, 'median': 285.0, 'max': 285}} |
| base_u_plus_w_v4_train_targeted_macflow/S500 | {'timeout': 54, 'success': 199, 'numerical_failure': 3} | {'timeout': {'n': 54, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 199, 'min': 300, 'median': 530.0, 'max': 692}, 'numerical_failure': {'n': 3, 'min': 114, 'median': 285.0, 'max': 285}} |
| base_u_v3_macflow/F10 | {'success': 170, 'timeout': 78, 'numerical_failure': 8} | {'success': {'n': 170, 'min': 289, 'median': 417.5, 'max': 698}, 'timeout': {'n': 78, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 8, 'min': 48, 'median': 97.0, 'max': 354}} |
| base_u_v3_macflow/F13 | {'success': 139, 'timeout': 111, 'numerical_failure': 6} | {'success': {'n': 139, 'min': 265, 'median': 457.0, 'max': 695}, 'timeout': {'n': 111, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 6, 'min': 86, 'median': 108.5, 'max': 597}} |
| base_u_v3_macflow/Freg | {'success': 226, 'timeout': 26, 'numerical_failure': 4} | {'success': {'n': 226, 'min': 268, 'median': 369.5, 'max': 697}, 'timeout': {'n': 26, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 4, 'min': 71, 'median': 170.0, 'max': 293}} |
| base_u_v3_macflow/S300 | {'success': 213, 'numerical_failure': 5, 'timeout': 38} | {'success': {'n': 213, 'min': 265, 'median': 366.0, 'max': 693}, 'numerical_failure': {'n': 5, 'min': 86, 'median': 103.0, 'max': 182}, 'timeout': {'n': 38, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v3_macflow/S400 | {'success': 211, 'numerical_failure': 6, 'timeout': 39} | {'success': {'n': 211, 'min': 265, 'median': 425.0, 'max': 681}, 'numerical_failure': {'n': 6, 'min': 86, 'median': 108.5, 'max': 499}, 'timeout': {'n': 39, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v3_macflow/S500 | {'success': 211, 'numerical_failure': 5, 'timeout': 40} | {'success': {'n': 211, 'min': 265, 'median': 513.0, 'max': 696}, 'numerical_failure': {'n': 5, 'min': 86, 'median': 103.0, 'max': 182}, 'timeout': {'n': 40, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v2_macflow/F10 | {'success': 95, 'timeout': 160, 'numerical_failure': 1} | {'success': {'n': 95, 'min': 413, 'median': 593.0, 'max': 700}, 'timeout': {'n': 160, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 525, 'median': 525.0, 'max': 525}} |
| base_u_v2_macflow/F13 | {'success': 60, 'timeout': 192, 'numerical_failure': 4} | {'success': {'n': 60, 'min': 264, 'median': 504.5, 'max': 698}, 'timeout': {'n': 192, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 4, 'min': 116, 'median': 260.5, 'max': 497}} |
| base_u_v2_macflow/S300 | {'success': 205, 'timeout': 48, 'numerical_failure': 3} | {'success': {'n': 205, 'min': 264, 'median': 363.0, 'max': 684}, 'timeout': {'n': 48, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 3, 'min': 116, 'median': 236.0, 'max': 285}} |
| base_u_v2_macflow/S400 | {'success': 216, 'timeout': 37, 'numerical_failure': 3} | {'success': {'n': 216, 'min': 264, 'median': 416.0, 'max': 681}, 'timeout': {'n': 37, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 3, 'min': 116, 'median': 236.0, 'max': 285}} |
| base_u_v2_macflow/S500 | {'success': 222, 'timeout': 29, 'numerical_failure': 5} | {'success': {'n': 222, 'min': 264, 'median': 512.0, 'max': 691}, 'timeout': {'n': 29, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 5, 'min': 116, 'median': 285.0, 'max': 521}} |
| base_u_v10_local_macflow/F10 | {'success': 183, 'timeout': 73} | {'success': {'n': 183, 'min': 378, 'median': 552.0, 'max': 698}, 'timeout': {'n': 73, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v10_local_macflow/F13 | {'success': 241, 'timeout': 15} | {'success': {'n': 241, 'min': 276, 'median': 317.0, 'max': 613}, 'timeout': {'n': 15, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v10_local_macflow/S300 | {'success': 245, 'timeout': 11} | {'success': {'n': 245, 'min': 276, 'median': 307.0, 'max': 517}, 'timeout': {'n': 11, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v10_local_macflow/S400 | {'success': 245, 'timeout': 11} | {'success': {'n': 245, 'min': 276, 'median': 317.0, 'max': 486}, 'timeout': {'n': 11, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v10_local_macflow/S500 | {'success': 244, 'timeout': 12} | {'success': {'n': 244, 'min': 276, 'median': 317.0, 'max': 688}, 'timeout': {'n': 12, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v5_macflow/F10 | {'success': 78, 'timeout': 176, 'numerical_failure': 2} | {'success': {'n': 78, 'min': 376, 'median': 610.0, 'max': 699}, 'timeout': {'n': 176, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 422, 'median': 439.5, 'max': 457}} |
| base_u_plus_w_v5_macflow/F13 | {'timeout': 223, 'numerical_failure': 1, 'success': 32} | {'timeout': {'n': 223, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 404, 'median': 404.0, 'max': 404}, 'success': {'n': 32, 'min': 389, 'median': 549.5, 'max': 688}} |
| base_u_plus_w_v5_macflow/S300 | {'success': 221, 'timeout': 32, 'numerical_failure': 3} | {'success': {'n': 221, 'min': 309, 'median': 398.0, 'max': 699}, 'timeout': {'n': 32, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 3, 'min': 314, 'median': 335.0, 'max': 452}} |
| base_u_plus_w_v5_macflow/S400 | {'success': 235, 'timeout': 20, 'numerical_failure': 1} | {'success': {'n': 235, 'min': 389, 'median': 436.0, 'max': 693}, 'timeout': {'n': 20, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 400, 'median': 400.0, 'max': 400}} |
| base_u_plus_w_v5_macflow/S500 | {'success': 238, 'timeout': 16, 'numerical_failure': 2} | {'success': {'n': 238, 'min': 389, 'median': 527.0, 'max': 697}, 'timeout': {'n': 16, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 404, 'median': 453.5, 'max': 503}} |
| base_u_v7_local_macflow/F10 | {'success': 249, 'timeout': 7} | {'success': {'n': 249, 'min': 282, 'median': 398.0, 'max': 692}, 'timeout': {'n': 7, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v7_local_macflow/F13 | {'success': 99, 'timeout': 157} | {'success': {'n': 99, 'min': 266, 'median': 415.0, 'max': 699}, 'timeout': {'n': 157, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v7_local_macflow/S300 | {'success': 255, 'timeout': 1} | {'success': {'n': 255, 'min': 266, 'median': 330.0, 'max': 694}, 'timeout': {'n': 1, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v7_local_macflow/S400 | {'success': 256} | {'success': {'n': 256, 'min': 266, 'median': 421.0, 'max': 694}} |
| base_u_v7_local_macflow/S500 | {'success': 249, 'timeout': 7} | {'success': {'n': 249, 'min': 266, 'median': 519.0, 'max': 694}, 'timeout': {'n': 7, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_plus_w_v4_macflow/F10 | {'success': 107, 'timeout': 147, 'numerical_failure': 2} | {'success': {'n': 107, 'min': 439, 'median': 568.0, 'max': 696}, 'timeout': {'n': 147, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 458, 'median': 510.0, 'max': 562}} |
| base_u_plus_w_v4_macflow/F13 | {'timeout': 244, 'success': 11, 'numerical_failure': 1} | {'timeout': {'n': 244, 'min': 700, 'median': 700.0, 'max': 700}, 'success': {'n': 11, 'min': 308, 'median': 405.0, 'max': 574}, 'numerical_failure': {'n': 1, 'min': 94, 'median': 94.0, 'max': 94}} |
| base_u_plus_w_v4_macflow/S300 | {'success': 226, 'timeout': 29, 'numerical_failure': 1} | {'success': {'n': 226, 'min': 307, 'median': 375.0, 'max': 688}, 'timeout': {'n': 29, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 94, 'median': 94.0, 'max': 94}} |
| base_u_plus_w_v4_macflow/S400 | {'success': 231, 'timeout': 23, 'numerical_failure': 2} | {'success': {'n': 231, 'min': 308, 'median': 446.0, 'max': 696}, 'timeout': {'n': 23, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 2, 'min': 94, 'median': 313.5, 'max': 533}} |
| base_u_plus_w_v4_macflow/S500 | {'success': 240, 'timeout': 15, 'numerical_failure': 1} | {'success': {'n': 240, 'min': 308, 'median': 526.0, 'max': 700}, 'timeout': {'n': 15, 'min': 700, 'median': 700.0, 'max': 700}, 'numerical_failure': {'n': 1, 'min': 94, 'median': 94.0, 'max': 94}} |
| base_u_v8_local_macflow/F10 | {'success': 256} | {'success': {'n': 256, 'min': 278, 'median': 330.0, 'max': 654}} |
| base_u_v8_local_macflow/F13 | {'success': 88, 'timeout': 168} | {'success': {'n': 88, 'min': 287, 'median': 466.5, 'max': 699}, 'timeout': {'n': 168, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v8_local_macflow/Freg | {'success': 256} | {'success': {'n': 256, 'min': 290, 'median': 367.5, 'max': 684}} |
| base_u_v8_local_macflow/S300 | {'success': 256} | {'success': {'n': 256, 'min': 287, 'median': 317.0, 'max': 510}} |
| base_u_v8_local_macflow/S400 | {'success': 256} | {'success': {'n': 256, 'min': 287, 'median': 414.0, 'max': 507}} |
| base_u_v8_local_macflow/S500 | {'success': 255, 'timeout': 1} | {'success': {'n': 255, 'min': 287, 'median': 514.0, 'max': 608}, 'timeout': {'n': 1, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v9_local_macflow/F10 | {'success': 247, 'timeout': 9} | {'success': {'n': 247, 'min': 299, 'median': 387.0, 'max': 634}, 'timeout': {'n': 9, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v9_local_macflow/F13 | {'success': 188, 'timeout': 68} | {'success': {'n': 188, 'min': 266, 'median': 334.5, 'max': 687}, 'timeout': {'n': 68, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v9_local_macflow/S300 | {'success': 242, 'timeout': 14} | {'success': {'n': 242, 'min': 266, 'median': 312.0, 'max': 674}, 'timeout': {'n': 14, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v9_local_macflow/S400 | {'success': 241, 'timeout': 15} | {'success': {'n': 241, 'min': 266, 'median': 367.0, 'max': 693}, 'timeout': {'n': 15, 'min': 700, 'median': 700.0, 'max': 700}} |
| base_u_v9_local_macflow/S500 | {'success': 234, 'timeout': 22} | {'success': {'n': 234, 'min': 266, 'median': 364.0, 'max': 669}, 'timeout': {'n': 22, 'min': 700, 'median': 700.0, 'max': 700}} |

逐seed的完整step数组、配对F10/F13结局转移见NUMERICAL_RESULTS/results.outcome_summaries。数值异常用原error step，成功按simulator first_success_step，timeout按原真实终止步；没有新增死锁判定。

解释（DEVELOPMENT）：本轮没有建立S500对所有14个Flow的通用不劣；失败控制器和门余量见表。若dev通过而held-out失败，属于预注册的部分过拟合门模式；仅因均值CI宽而未通过，并不证明某个机制或所有时间表都不可行。不能用次要时刻救回主命题，也不能宣称固定η动作空间已解释全部controller依赖。

视频适用：本轮是Ring大型评价，明确适用视频要求；没有使用纯Q豁免。videos.py在新目录中仅把统一入口场景范围缩到用户要求的ring_exchange，保留原roles/trace验证/renderer源码和完整帧数检查。已检索历史视频manifest，未发现有已记录真实deadlock criterion的兼容Ring案例；本次冻结runtime也无deadlock event。统一严格入口返回非零，VIDEO_GATE和videos/manifest.complete=false完整记录缺口。

- [safety_success](videos/ring_exchange/safety_success.mp4)：306帧，dt=0.05，video SHA=a7afe41fb314df2147293489e1e0f002ca326e83474790d175c19321391d2984；formal_role=True。
- [timeout_to_success_1](videos/ring_exchange/timeout_to_success_1.mp4)：701帧，dt=0.05，video SHA=4767beaa0d1db41e274cf9e6fd6495cad4f2f83d1f8e4a4577448847c45a743c；formal_role=False。
- [timeout_to_success_2](videos/ring_exchange/timeout_to_success_2.mp4)：701帧，dt=0.05，video SHA=54d63a3ca109b51760169ef9c59a444a5c21cdee37a6f596b210dbcd2d0c366c；formal_role=False。

两个timeout-to-success视频若存在，只是全程真实比较，清楚标TIMEOUT，不填deadlock_show/deadlock_resolution。独立safety_success保留simulator连续运动净余量、完整augmented初态（原true-t0 reset精确hash重建，零仿真步）、seed/config/checkpoint/trace/video来源。缺死锁案例意味着batch视频交付未完成；没有追加rollout来制造案例或更改阈值。

执行：round9原run.py/core数值实现和FF控制链复用，run.py仅隔离IO并转换journal numpy。14纯固定profiles和已有dev schedule UIDs沿用，其他10 schedule按相同η hex/time/code fingerprints生成。先design/preflight，32gatePASS再提交主missing；worker仅journal，官方单进程merger以局部exact现代seed/timestep适配，postflight核验全部raw/DB/trace。只写本实验目录和明确授权的新DB/本experiment journals生命周期，不修改历史资产、不commit。资源用户任务CPU<=18/GPU<=3，17workers+1编排；ming任务未计入zhihan额度。
