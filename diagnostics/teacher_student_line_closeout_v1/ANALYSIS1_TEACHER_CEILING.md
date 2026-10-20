每个教师/学生对应32个state×2seed，n=64；全部DEVELOPMENT。逐教师差值及state簇CI见SECONDARY_ANALYSES.json。

| 教师全名 | 坐标系 | 只跑教师 | 教师只加分布式安全层 | 原学生只跑 | 闭环重标注学生只跑 | 共享随机种子学生只跑 |
|---|---|---|---|---|---|---|
| 冻结的联合MACFlow教师（base_u_plus_w_v4_wide_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_v4_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_plus_w_v6_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_plus_w_v3_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_plus_w_v4_train_targeted_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_v3_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_v2_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_v10_local_macflow） | local | 76.56% | 76.56% | 92.19% | 96.88% | 95.31% |
| 冻结的联合MACFlow教师（base_u_plus_w_v5_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_v7_local_macflow） | local | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_plus_w_v4_macflow） | global | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 冻结的联合MACFlow教师（base_u_v8_local_macflow） | local | 1.56% | 1.56% | 0.00% | 0.00% | 3.12% |
| 冻结的联合MACFlow教师（base_u_v9_local_macflow） | local | 39.06% | 39.06% | 71.88% | 53.12% | 53.12% |
