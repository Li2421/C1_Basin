# Round19b Four controller axis v2

等价门：G0 **PASS 8/8**、G1a **PASS 32/32**、G1b **MISMATCH 3/4**（条件性、非阻断）、G2 **PASS 25/25**。Four 跨 controller 的固定 η regime 差异 **SUPPORTED**：多数最佳臂 F10，5 个 controller 的非多数最佳臂领先至少 10pp。阶段压缩 H_compress_Four **REJECTED**；由冻结探测学习 E **REJECTED**。数值研究已完成；正式视频交付 **INCOMPLETE**。

## 事实与门控

补充预注册 UTC `2026-10-14T10:39:42.955993+00:00`，SHA256 `a5d6a1759ceba09f34e05cfd3c24b1d0f7344240aabdec3ba4bcebbc129f0b8e`。v1 因历史首动作字段缺失而在 Stage 0 停止，本轮只修改 G1 与目录。G1a 事前冻结 32 条历史干预记录，v12 与 v10-wide 各 success 8、timeout 8；outcome、episode_length、两项 float64 最小距离全部逐位一致。G0 用新适配层复现 E8 固定 8 例；G2 的 25 个候选结构、配置、FP32 和同 key 前向均合格。G1b 的 1 条分叉从首步动作开始，见 `G1B_REPORT.json`；这项按补充预注册不阻断后续实验，不能称为全通过。

16 个新 probe state 和 32 个新 census state 经共享库可恢复物理初态去重，无碰撞。1,200 次 200 步无标签探测产生 25×240 R5，7 次不完整；列均值插补、population zscore 与 PCA-2 在任何新成功标签前冻结，收据 SHA256 `803ae073ec67ae5fb7767ec282b0628da702eb91637d45996c97d4db36640e13`。探测只记录进展、投影比例、最小两机器人距离、平均速度、低速步数；不读取终止标签。

固定臂 25 controller × 32 state × 3 臂 × 2 seed = **4,800** 条；unknown **36**，成功率一律为成功数 / 已知种子数。多数最佳臂为 **F10**；条件(a)达到 **5 ≥ 3**，条件(b)为 **0**，故按预注册进入时间表阶段。

| 达到(a)的 controller | 最佳臂 | 该臂成功率 | 相对多数臂优势 |
|---|---|---:|---:|
| `base_u_v5_outer_buffer_macflow__best` | F15 | 100.00% | 70.97 pp |
| `base_u_v5_outer_buffer_macflow__final` | F15 | 100.00% | 81.25 pp |
| `base_u_v9_world_goal_recovery_macflow__best` | F15 | 100.00% | 20.31 pp |
| `base_u_v9_world_goal_recovery_macflow__final` | F15 | 100.00% | 62.90 pp |
| `base_u_v9_world_goal_recovery_source_balanced_macflow__final` | F15 | 100.00% | 34.38 pp |

时间表 25 × 32 × 5 × 2 = **8,000** 条；unknown **57**。所有 E=S−F_late、L=S−F_early、N=F_early−F_late 见 `E_L_N.csv`。`S15to10_786` 在 25 个 controller 上的已知种子合计 **1597/1597** 成功，另有 3 个 unknown；不能把这 3 个算作成功。同 run 的 best/final 一起留出，固定动作集的平均/最大 oracle 遗憾为 **1.01/9.68 pp**，扩展动作集为 **0.00/0.00 pp**。同时满足 E、L≥10pp 的 controller 有 **0/25**；预注册 H_compress_Four 判定 **REJECTED**。

E 学习：所选时间表 `S15to10_786`；留一 Spearman -0.4103392444358914, RMSE 27.89 pp，源均值基线 25.68 pp，RMSE 比 1.086。按同 run 留一、仅用冻结 R5 PCA-2、源折标准化与 ridge α=1，预注册判定 **REJECTED**；逐折系数、预测、真值见 `SCHEDULE_RESULTS.json`。

## 证据链与限制

固定臂和时间表均先序列化请求，运行官方 `shared_rollout_db.plan` preflight（分别 4,800 与 8,000 全部 genuinely_missing、ambiguous=0），worker 只追加本目录 shard journals，单进程官方 merger 使用本目录的精确 seed 身份适配器，然后运行官方 postflight。数值 unknown 保留、未重跑。新执行身份由 checkpoint SHA、归一化器所用适配层及时间表定义确定；旧 intervention 标签只用于 G1a 验证，不代替新 census 标签。

探测前完成 state 去重，但**没有在提交短探测前留存官方 probe preflight**。这违反了共享库工作流的收据要求，已在 `PROBE_EVIDENCE_AUDIT.json` 明确登记；不能事后伪称事前 preflight，也不重复有效探测。所有探测输出为本目录的无标签源证据，共享 DB 中这 16 个 probe state 的成功行数为 0。

视频：3 个来自不同真实完整安全成功轨迹的 MP4 已导出，正式 `safety_success` 通过素材校验。统一 `new_benchmark_common.batch_videos` 入口已运行，返回 2；Four 无已记录的正式死锁判据，`deadlock_show` 与 `deadlock_resolution` 缺失，manifest `complete=false`。timeout 未标为 deadlock；因此大 batch 视频交付仍未完成。

解释：Four 的固定 η 差异已由此队列证明；阶段压缩与 E 可学习性按各自预注册数值门单独判定。结论仅针对同环境 fingerprint 的 25 个 best/final checkpoint、这些新 development state 与指定臂/时间表；不同环境版本的 9 个 run 未纳入。

![固定臂成功率](figures/fixed_rates.png)

![冻结 R5 PCA](figures/r5_pca.png)

![时间表 E 与 L](figures/schedule_effects.png)
