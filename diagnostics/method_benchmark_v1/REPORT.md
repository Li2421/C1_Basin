# Round26 method benchmark

| method | T1 regret pp | T2 | T3 | T4 | T5 | pooled | pooled95%CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M0 | 31.81 | 34.49 | 30.27 | 26.04 | 26.62 | 29.67 | [24.41, 35.51] |
| M1 | 31.81 | 34.49 | 30.27 | 26.04 | 26.62 | 29.67 | [24.41, 35.51] |
| M2 | 1.79 | 0.84 | 4.20 | 6.97 | 3.65 | 3.38 | [2.07, 5.83] |
| M3_fixed | 30.02 | 32.03 | 29.30 | 23.96 | 16.03 | 25.42 | [20.03, 31.57] |
| M3_full | 0.95 | 2.40 | 5.86 | 1.52 | 2.78 | 2.44 | [1.42, 4.64] |
| M4_fixed | 31.81 | 32.76 | 30.27 | 23.96 | 26.62 | 28.91 | [23.76, 34.6] |
| M4_full | 0.78 | 0.61 | 0.29 | 1.92 | 4.05 | 1.79 | [1.28, 3.13] |
| M5 | 0.95 | 0.39 | 1.37 | 5.93 | 2.84 | 2.36 | [1.33, 4.73] |
| M6_E_fixed_k16 | 30.29 | 29.70 | 25.43 | 21.80 | 14.55 | 23.71 | [18.36, 29.77] |
| M6_E_full_k16 | 1.72 | 1.92 | 2.39 | 3.04 | 3.85 | 2.67 | [2.39, 3.74] |
| M6_Thompson_fixed_k16 | 32.16 | 30.96 | 26.59 | 22.93 | 15.33 | 24.93 | [19.51, 31.03] |
| M6_Thompson_full_k16 | 2.93 | 2.46 | 3.15 | 3.78 | 4.13 | 3.35 | [3.05, 4.42] |
| M7_fixed_k16 | 30.04 | 30.47 | 26.79 | 23.86 | 20.96 | 26.10 | [20.95, 31.95] |
| M7_full_k16 | 1.54 | 1.56 | 2.17 | 2.77 | 3.53 | 2.39 | [2.05, 3.53] |

| pooled method | success % | max regret pp | catastrophes | labels/controller min–max |
| --- | ---: | ---: | ---: | --- |
| M0 | 62.31 | 74.22 | 0 | 0–0 |
| M1 | 62.31 | 74.22 | 0 | 0–0 |
| M2 | 88.60 | 52.08 | 4 | 0–0 |
| M3_fixed | 66.56 | 68.75 | 1 | 0–0 |
| M3_full | 89.54 | 39.84 | 0 | 0–0 |
| M4_fixed | 63.07 | 74.22 | 0 | 0–0 |
| M4_full | 90.19 | 14.58 | 0 | 0–0 |
| M5 | 89.62 | 52.08 | 1 | 0–0 |
| M6_E_fixed_k16 | 68.27 | 69.22 | 0 | 32–64 |
| M6_E_full_k16 | 89.31 | 9.93 | 0 | 80–128 |
| M6_Thompson_fixed_k16 | 67.05 | 70.61 | 0 | 32–64 |
| M6_Thompson_full_k16 | 88.63 | 10.47 | 0 | 80–128 |
| M7_fixed_k16 | 65.87 | 68.81 | 0 | 32–48 |
| M7_full_k16 | 89.59 | 12.35 | 0 | 80–128 |

B1 **REJECTED**；下界点估计配对CI全通过=False。不满足严格优于M4比较的基线：['M3_full', 'M5']。逐测试集及unknown敏感性见results.json。

M0固定参数；M1源平均固定策略；M2源平均增益调度；M3静态任务描述条件价值函数；M4交互上下文辨识式critic（类似PEARL的范式类比，不宣称实现PEARL）；M5结构化手工辨识规则；M6逐实例经验调参/后验采样；M7模型先验加快速残差适应。

## 动作空间与方法（B2）

| test | M3 full−fixed pp | M4 full−fixed pp | fixed M4−M1 pp |
| --- | --- | --- | --- |
| T1 | -29.07 [-42.13, -15.74] | -31.03 [-45.31, -16.74] | 0.00 [0.0, 0.0] |
| T2 | -29.63 [-43.75, -15.79] | -32.14 [-43.86, -20.15] | -1.73 [-6.08, 0.0] |
| T3 | -23.44 [-46.29, -3.41] | -29.98 [-50.29, -9.86] | 0.00 [0.0, 0.0] |
| T4 | -22.44 [-32.53, -12.1] | -22.04 [-32.13, -11.62] | -2.08 [-7.05, 0.56] |
| T5 | -13.25 [-22.97, -4.17] | -22.57 [-31.08, -14.64] | 0.00 [0.0, 0.0] |
| POOLED | -22.98 [-29.06, -16.94] | -27.12 [-32.68, -21.52] | -0.77 [-2.04, 0.06] |

合并结果中，M4完整菜单相对固定菜单的遗憾差为-27.12pp，95%CI [-32.68, -21.52]；固定菜单内M4相对M1为-0.77pp，CI [-2.04, 0.06]。动作菜单的改善幅度明显大于固定菜单内方法差异；这是描述性比较，不是因果可加分解。

## 达到M4零样本水平的最小标签预算（B3）

| test | method/menu | k | total target labels min–max |
| --- | --- | ---: | --- |
| T1 | M6_E_fixed | >16 | >16/arm; last tested [64, 64] |
| T1 | M6_E_full | >16 | >16/arm; last tested [96, 96] |
| T1 | M6_Thompson_fixed | >16 | >16/arm; last tested [64, 64] |
| T1 | M6_Thompson_full | >16 | >16/arm; last tested [96, 96] |
| T1 | M7_fixed | >16 | >16/arm; last tested [48, 48] |
| T1 | M7_full | >16 | >16/arm; last tested [80, 80] |
| T2 | M6_E_fixed | >16 | >16/arm; last tested [32, 48] |
| T2 | M6_E_full | >16 | >16/arm; last tested [80, 96] |
| T2 | M6_Thompson_fixed | >16 | >16/arm; last tested [32, 48] |
| T2 | M6_Thompson_full | >16 | >16/arm; last tested [80, 96] |
| T2 | M7_fixed | >16 | >16/arm; last tested [32, 48] |
| T2 | M7_full | >16 | >16/arm; last tested [80, 96] |
| T3 | M6_E_fixed | >16 | >16/arm; last tested [64, 64] |
| T3 | M6_E_full | >16 | >16/arm; last tested [96, 96] |
| T3 | M6_Thompson_fixed | >16 | >16/arm; last tested [64, 64] |
| T3 | M6_Thompson_full | >16 | >16/arm; last tested [96, 96] |
| T3 | M7_fixed | >16 | >16/arm; last tested [48, 48] |
| T3 | M7_full | >16 | >16/arm; last tested [80, 80] |
| T4 | M6_E_fixed | >16 | >16/arm; last tested [48, 48] |
| T4 | M6_E_full | >16 | >16/arm; last tested [128, 128] |
| T4 | M6_Thompson_fixed | >16 | >16/arm; last tested [48, 48] |
| T4 | M6_Thompson_full | >16 | >16/arm; last tested [128, 128] |
| T4 | M7_fixed | >16 | >16/arm; last tested [48, 48] |
| T4 | M7_full | >16 | >16/arm; last tested [128, 128] |
| T5 | M6_E_fixed | >16 | >16/arm; last tested [48, 48] |
| T5 | M6_E_full | 16 | 128–128 |
| T5 | M6_Thompson_fixed | >16 | >16/arm; last tested [48, 48] |
| T5 | M6_Thompson_full | >16 | >16/arm; last tested [128, 128] |
| T5 | M7_fixed | >16 | >16/arm; last tested [48, 48] |
| T5 | M7_full | 1 | 8–8 |
| POOLED | M6_E_fixed | >16 | >16/arm; last tested [32, 64] |
| POOLED | M6_E_full | >16 | >16/arm; last tested [80, 128] |
| POOLED | M6_Thompson_fixed | >16 | >16/arm; last tested [32, 64] |
| POOLED | M6_Thompson_full | >16 | >16/arm; last tested [80, 128] |
| POOLED | M7_fixed | >16 | >16/arm; last tested [32, 48] |
| POOLED | M7_full | >16 | >16/arm; last tested [80, 128] |

逐测试集未能以95%CI证明M4严格优于的基线：
T1: M2, M3_full, M5
T2: M2, M3_full, M5
T3: M2, M3_full, M5
T4: M2, M3_full, M5
T5: M2, M3_full, M5

**T2输入身份修正：** R10的FRESH_STATES.json是旧64state抽样，与实际main_CASES/protocol的64state集合不相交。最初盲预测与原R16加载逻辑使用了该旧输入；R16 DATA_AUDIT中记录的FRESH文件hash与当前旧文件一致。统一比较的身份门拦截此错配后，本轮保持M3/R16模型与所有超参数不变，仅对实际标签对应state重做前向推理，保留原冻结预测及T2_ALIGNMENT_CORRECTION收据。此输入修正发生在重建已读取标签之后，但不使用成功/unknown决定输入、拟合或规则。历史T2逐state指标需要这一限定；本轮T2数值使用正确输入。

证据等级DEVELOPMENT：所有标签此前均已打开，本轮不构成新的前瞻确认。M3及方法超参数、R3和初版盲预测先冻结，再读取测试标签并复现R21全部14项重建门；未新增rollout、探测或共享DB写入。

各测试controller按state_uid前半校准/后半评估；零样本方法只读输入，M6/M7只读校准标签。CI对controller和评估state分层配对重抽5000次，条件于既有校准池和1000次推荐分布，不估计新校准池总体不确定性。不同配方/同run兄弟作为controller单位，CI不能解释为独立配方总体区间。

T2的Freg仅3个controller有标签：两个实际为η2，已按执行身份映射至冻结F2预测头；另一个为η4，R15没有该训练头，仅保留在完整oracle及M6菜单，明确限制了此处菜单完全同构性。所有实际预算和可用臂见逐controller表。Thompson严格使用每臂k条平衡采集后做后验采样推荐，不是自适应分配的bandit轨迹。

事实与解释分开：B2的配对差给出动作空间贡献与固定菜单内方法贡献的相对尺度；B3只是在已测离散预算中达到同一评估集M4点遗憾的最小预算，不是新部署保证。unknown上下界、最大遗憾、灾难数和标签预算均保留于主CSV/JSON。
