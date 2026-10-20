# 学生蒸馏与离线检查

14 个学生均完成预注册的80,000步训练；只按固定8批×256个dev state的蒸馏MSE选择checkpoint。6个选择了初始化step0，不能把它们解释为成功学到了教师动作。没有闭环选checkpoint、调参、替换或重训。

学生输入统一为自身local_observation行、id one-hot和自身2维noise；目标为各教师native_fn的物理速度raw。输出chart按教师实际frame执行原to_world；10个global教师raw为world，4个local教师raw为径向/切向。仅径向观测与world目标之间存在坐标信息差，本轮没有分离这一因素；不能把全部代价归因于分散化本身。

| Teacher | Raw frame | Selected step | Fixed dev MSE | Independent dev RMSE (m/s/component) | Mean cosine | Error ≤0.05m/s | Info tests |
|---|---|---:|---:|---:|---:|---:|---|
| base_u_plus_w_v4_wide_macflow | global | 500 | 0.0728313 | 0.268364 | 0.2255 | 19.80% | PASS |
| base_u_macflow | global | 0 | 0.0933944 | 0.306151 | -0.0128 | 7.19% | PASS |
| base_u_v4_macflow | global | 500 | 0.08086 | 0.288814 | 0.1035 | 14.34% | PASS |
| base_u_plus_w_v6_macflow | global | 75500 | 0.0615118 | 0.253568 | 0.4788 | 25.44% | PASS |
| base_u_plus_w_v3_macflow | global | 0 | 0.0895511 | 0.2984 | 0.0261 | 10.08% | PASS |
| base_u_plus_w_v4_train_targeted_macflow | global | 0 | 0.080926 | 0.286956 | 0.0498 | 16.81% | PASS |
| base_u_v3_macflow | global | 0 | 0.0898601 | 0.299103 | 0.0276 | 11.79% | PASS |
| base_u_v2_macflow | global | 0 | 0.0920521 | 0.303586 | 0.0127 | 9.35% | PASS |
| base_u_v10_local_macflow | local | 75000 | 9.24048e-05 | 0.0105154 | 0.9883 | 98.79% | PASS |
| base_u_plus_w_v5_macflow | global | 0 | 0.082696 | 0.290334 | -0.0351 | 18.64% | PASS |
| base_u_v7_local_macflow | local | 77000 | 0.00017458 | 0.0120529 | 0.9830 | 98.21% | PASS |
| base_u_plus_w_v4_macflow | global | 500 | 0.0730823 | 0.270711 | 0.1951 | 20.70% | PASS |
| base_u_v8_local_macflow | local | 71500 | 0.000121274 | 0.0120763 | 0.9851 | 98.30% | PASS |
| base_u_v9_local_macflow | local | 67000 | 0.000121319 | 0.0112846 | 0.9826 | 98.55% | PASS |

MSE是官方蒸馏损失的逐动作分量平均。独立dev检查为2,048个固定state/noise样本，不含闭环标签。≤0.05m/s列以二维动作误差范数计算；cosine的零范数分母下限为1e-12。全部14个学生通过其他机器人目标不变性、只使用自身noise、给定观测/noise确定性及nominal-only逐位一致测试。

训练超参数及14个seed见TRAINING_CONFIG.json，数据来源/hash见TEACHERS.json，各文件观测重建差异见data/*/DATA_AUDIT.json。初始速度来自initial_state_json，后续速度由保存位置差/dt重建；不以专家actions作为训练目标。teacher observation重新按runtime计算，记录与存储observations的差异。

[训练曲线](figures/training_curves.png)；机器可读曲线见training_curves.csv。

初始化checkpoint：base_u_macflow, base_u_plus_w_v3_macflow, base_u_plus_w_v4_train_targeted_macflow, base_u_v3_macflow, base_u_v2_macflow, base_u_plus_w_v5_macflow
