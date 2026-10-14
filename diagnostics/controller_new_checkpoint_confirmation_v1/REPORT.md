# Round12：8 个新参数 final 的前瞻确认

C1 **SUPPORTED**（字面点估计门 通过）；C3 **SUPPORTED**。C2 的 headroom 门 满足：3/8 个真实下界点 E<5pp（unknown 可取范围 [3, 3]），C2 **SUPPORTED**。这次只确认 8 个参数不同的 final；6 个参数完全相同的 final 排除。全部 12,288 条主 rollout 与 256 次修复后无标签探测已完成。批次状态 **NUMERICAL_COMPLETE_VIDEO_INCOMPLETE**；视频严格门未通过时不宣称整批交付完成。

## 固定门槛与结果

主方案每个 final 排除同 run 的 best，使用 13 个历史 best；次方案 14 源仅作敏感性分析。155D R5、源侧均值填补/zscore/PCA2、ridge α=1、Ê<5pp 选 F10 否则 S500 均未改变。5000 次共享 state 成簇 bootstrap、64 state、每簇保留 4 seeds，不重采样 controller。

主 G_E 平均遗憾 1.416pp，95% CI [0.781pp, 2.637pp]；unknown bounds [1.270pp, 1.514pp]。最大遗憾 7.031pp；C1b **REJECTED**。相对 F10 平均收益 26.807pp，95% CI [23.340pp, 30.322pp]，unknown envelope [22.705pp, 30.811pp]。C1 仍要求该下界>0、平均遗憾≤2pp、每 controller 损失≤10pp；n=8 不放宽。

C2 相对固定 S500 的配对改善 2.487pp，95% CI [1.416pp, 3.564pp]；Paired state-cluster improvement CI; oracle cancels exactly。菜单内含 F10/F13/S300/S500，F2/F4 仅参加六臂 oracle。

C3 主 Spearman=0.8333333333333335（门≥0.6），RMSE=14.029pp，13 源均值基线 RMSE=31.182pp，比值=0.44989664163073123（门≤0.7）。次方案 C1=SUPPORTED，C2=SUPPORTED，C3=SUPPORTED；不替换主方案。n 从14降为8使 controller 关系估计和 CI 的精度降低，预期不确定性更宽；各实际 state CI 仍由数据决定。没有因样本数降低放宽任何门槛。

C4 **INCONCLUSIVE / 描述性**：E>0 6/8，L>0 8/8，E和L均≥10pp 3/8。历史 best 参照分别11/14、14/14、10/14，必须比较分母/比例；无额外二元阈值。N=E−L 恒等式审计通过。

PROSPECTIVE_NUMERICAL_CONFIRMATION: schedule plus R5-predicted early component transfers to these new Ring final siblings; not independent training-run or cross-scene confirmation.

## 每个新 controller

| controller | G_E | Ê | E | L | oracle | regret | gain vs F10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_u_v4_macflow__final | S500 | 52.026pp | 50.391pp | 73.828pp | S500 | 0.000pp | 50.391pp |
| base_u_plus_w_v6_macflow__final | S500 | 38.539pp | 53.906pp | 91.016pp | S500 | 0.000pp | 53.906pp |
| base_u_plus_w_v3_macflow__final | S500 | 25.660pp | 10.938pp | 5.469pp | F2 | 3.125pp | 10.938pp |
| base_u_plus_w_v4_train_targeted_macflow__final | S500 | 51.620pp | 62.500pp | 73.047pp | S500 | 0.000pp | 62.500pp |
| base_u_v10_local_macflow__final | S500 | 11.056pp | 36.719pp | 5.469pp | S300 | 7.031pp | 36.719pp |
| base_u_v7_local_macflow__final | F10 | -1.901pp | -19.922pp | 52.734pp | F10 | 0.000pp | 0.000pp |
| base_u_v8_local_macflow__final | F10 | -2.980pp | 1.172pp | 62.891pp | S300 | 1.172pp | 0.000pp |
| base_u_v9_local_macflow__final | F10 | -1.180pp | -1.172pp | 9.766pp | F10 | 0.000pp | 0.000pp |

超出5pp遗憾的 controller：base_u_v10_local_macflow__final。全部失败类型、发生动作步及原始 rollout UID 保存在 FAILURE_RECORDS.json；timeout 始终为 timeout，数值失败为 unknown，不当作已观测失败，也不重跑。

base_u_v10_local_macflow__final：六臂 oracle=S300，可描述为该臂偏好型；G_E=S500，遗憾 7.031pp。事件汇总：{('F10', 'timeout', None): 123, ('F10', 'numerical_failure', 'CBFSolverError'): 2, ('F13', 'timeout', None): 45, ('S300', 'timeout', None): 13, ('S500', 'timeout', None): 31, ('F2', 'timeout', None): 170, ('F4', 'timeout', None): 14}。逐事件实际发生步请见 FAILURE_RECORDS.json，不将终止步推断为死锁发生步。

## 兄弟差异与六个排除项

| final | 参数L2 | 相对L2 | R5 raw L2 | R5源标准化L2 |
| --- | --- | --- | --- | --- |
| base_u_v4_macflow__final | 4.262272735498019 | 0.09571150525052255 | 1.8696444405561945 | 5.060088716578176 |
| base_u_plus_w_v6_macflow__final | 1.1456795301318345 | 0.023420098486215978 | 0.9751555451335457 | 3.035806659092203 |
| base_u_plus_w_v3_macflow__final | 6.380014216351151 | 0.1452484308768261 | 2.2413035679077358 | 8.60049385877579 |
| base_u_plus_w_v4_train_targeted_macflow__final | 4.36033571389343 | 0.09741096226116783 | 1.3268436617928776 | 3.2500182274987224 |
| base_u_v10_local_macflow__final | 2.4243602763986187 | 0.047259758403888096 | 1.5692810301712328 | 6.467222387323132 |
| base_u_v7_local_macflow__final | 0.9639123643682977 | 0.021048907286004365 | 1.181482578164205 | 4.982825428376517 |
| base_u_v8_local_macflow__final | 0.8187542053931758 | 0.01739114324769387 | 2.3339801379022806 | 8.49824463500565 |
| base_u_v9_local_macflow__final | 1.8290465149782462 | 0.036257995309323846 | 0.789002396417839 | 3.2609302387660026 |

| 排除的run | best参数SHA256 | final参数SHA256 | 排除原因 |
| --- | --- | --- | --- |
| base_u_macflow | 00ef1ec70ef7a5a6a7a3e642ddcd5def84e47c9da5fa9018a046a2a3b4e5924e | 00ef1ec70ef7a5a6a7a3e642ddcd5def84e47c9da5fa9018a046a2a3b4e5924e | 参数逐字节相同，已有best评估；不运行final |
| base_u_plus_w_v4_macflow | eee44b071ad7abf28492ac2f91618f54e9f1f73746f19c037eb721bf3a92146f | eee44b071ad7abf28492ac2f91618f54e9f1f73746f19c037eb721bf3a92146f | 参数逐字节相同，已有best评估；不运行final |
| base_u_plus_w_v4_wide_macflow | 83729a1e20abd0079ea49577e4e153cfaffbd6dca8b6aed3bea4f4970cb58047 | 83729a1e20abd0079ea49577e4e153cfaffbd6dca8b6aed3bea4f4970cb58047 | 参数逐字节相同，已有best评估；不运行final |
| base_u_plus_w_v5_macflow | b5e7324a6156761fbf0c0bc85196729941b5210111b8a30e4c321c11fe70b5c7 | b5e7324a6156761fbf0c0bc85196729941b5210111b8a30e4c321c11fe70b5c7 | 参数逐字节相同，已有best评估；不运行final |
| base_u_v2_macflow | c84acd55a8351fe3f0f4cf967fe87215fc24b1b3cbca4c488883d667cf27d1e9 | c84acd55a8351fe3f0f4cf967fe87215fc24b1b3cbca4c488883d667cf27d1e9 | 参数逐字节相同，已有best评估；不运行final |
| base_u_v3_macflow | 04904c45e6f5849a1c66aff9092521b32ecd3e9efc40318c58218815f22b10a7 | 04904c45e6f5849a1c66aff9092521b32ecd3e9efc40318c58218815f22b10a7 | 参数逐字节相同，已有best评估；不运行final |

8 个纳入项与全部14历史best参数哈希均不同；未为六个排除 final 运行新的探测或主实验。原精度错误阶段对14 final的无标签探测作为事故证据保留，不能算作本轮有效确认。

## 精度事故、拦截与冻结时序

首次探测入口在导入前设置 FP32，但 poc/field_controller.py 导入时把 jax_enable_x64 改为 True。此前448个无标签探测、目标R5特征、Ê/G_E选择均无效，原始文件、protocol、收据完整保留，见 INVALID_PRECISION_EVIDENCE.json。初次一致性 worker 被 x64 断言在物理步之前拦截，未产生 final 成功标签。文件哈希不同的14 final也经参数审计纠正为8个真实新策略。

用户批准后先冻结 N8/FP32 补充，然后在全部数值/controller导入之后显式恢复 FP32；检查网络参数与采样输出 float32，并在每次动作入口断言 x64False。模拟物理 numpy float64 和投影算法不变。8个零步兼容检查通过，256次重新探测统计独立重算通过；原固定源侧拟合参数逐值相同，仅修复目标特征与由其产生的预测/选择。重新冻结R5和选择之后运行历史best32例门，outcome、episode_length、first-action float64 bytes对Round9及Round4均精确相同；之后才运行8 final主实验。

| 收据 | UTC | 收据SHA256 |
| --- | --- | --- |
| PREREGISTRATION.receipt.json | 2026-10-13T11:39:38.859844+00:00 | b21f346ac851dee2b3ce1fae860deb82f8262d7e2290510a18e03a185a054a81 |
| PREREGISTRATION_ADDENDUM_N8_FP32.receipt.json | 2026-10-13T12:57:02.915031+00:00 | 718c5ae70868996739330b0be6d2563e8f877f82c509a9e0c1817a3fc351e176 |
| fp32_n8/R5_FROZEN.receipt.json | 2026-10-13T13:04:18.372196+00:00 | 7a4dfa8ca8495e71ca4b2838525c60eedb60be8220bd96021c8053297c398d8c |
| fp32_n8/SELECTOR_FROZEN.receipt.json | 2026-10-13T13:04:18.450475+00:00 | 8ebad943ea5e8002364d4d194e8f027f92e51b69bf2e1eb68008f1cce6254eed |
| fp32_n8/ENDPOINT_OPENING.receipt.json | 2026-10-13T13:56:36.706391+00:00 | b403b374ac733e8f68909833769ed93b61034069cf1cb16c214db6f2e24c3792 |

每个 worker START_AUTHORIZATION 在新 selector 冻结之后；最早 final raw UTC=2026-10-13T13:05:19.524630+00:00。主标签只在全部12288 raw到齐后打开，见 ENDPOINT_OPENING.receipt.json。原64状态逐字节复用并与原去重凭证相连，没有事后重抽。

## 缓存、unknown 与视频

主 preflight、32例 gate、单进程官方 merger/本地精确身份 adapter、主 POSTFLIGHT 均保存。主 unknown=59；每个 unknown保留原记录，不重跑、不零填成功标签；所有 trace/raw SHA及DB flags已核对，无quarantine。64个历史资产和907个错误精度证据文件hash保持不变；DB正确新实验共12320条（主12288+gate32），无标签probe成功表0行。

统一视频入口已实际执行，退出码 2；完整视频 3 个，缺失正式角色 ['deadlock_show', 'deadlock_resolution']。Ring原生runtime没有deadlock判据，不能把timeout冒充deadlock；两条timeout-to-success视频如存在，仅为完整诊断对照。safety素材验证来自实际连续运动碰撞检查。数值终点可报告，整批状态保持 NUMERICAL_COMPLETE_VIDEO_INCOMPLETE，不生成完成标记。

资源补充：用户授权无人使用时利用全部空闲CPU；计算17→23 worker，另留1CPU编排。RESOURCE_BOOST_PLAN/receipt记录明确非证据预留，仅转移尚未开始的尾部case；原在途case继续，不中止或重跑unknown。预留标记不含成功字段，原worker仅跳过，owner用相同冻结数值循环生成真实记录并原子替换；合并跳过经验证的操作标记，终点等待两组job完成。规模、物理、选择与门槛均未变。

合并阶段曾因操作标记统计键未初始化报 KeyError，已按 MERGER_COUNTER_REPAIR.receipt.json 修复计数器，只重跑合并与后处理，仿真0条重跑。原失败日志保留；实际seed身份与标签解析规则未变。

## 使用范围与文件

这是Ring、同run兄弟checkpoint、全新64state上的前瞻数值确认，不是独立训练run或跨场景泛化；历史证据仍为DEVELOPMENT。未知结局的lower点及upper边界/CI全部报告，不能把lower点解释为所有unknown失败。图见 six_arm_success.png/pdf、Ehat_vs_E.png/pdf。机器结果 results.json；逐臂 ARM_RATES.csv；效应 E_L_N.csv；选择/遗憾 ACTIONS_REGRETS.csv；冻结模型 SELECTOR_MODELS.json；失败事件 FAILURE_RECORDS.json；视频 videos/manifest.json。

Active evidence directory: fp32_n8/. All referenced numerical artifacts in that directory.
