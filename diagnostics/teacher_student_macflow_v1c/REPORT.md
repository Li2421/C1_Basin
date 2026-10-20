完整性审计通过，Basin_C1 shared_rollout_db与默认journals路径断言通过。8个学生中8个通过修正门，主分析共10个学生。主分析：TS1=PASS，TS2=NOT_PASS，TS3a=PASS，TS3b=INCONCLUSIVE，TS6=PASS，TS6b=PASS；仅新增敏感性：TS1=PASS，TS2=NOT_PASS，TS3a=PASS，TS3b=INCONCLUSIVE，TS6=PASS，TS6b=PASS；合并DEVELOPMENT：TS1=PASS，TS2=NOT_PASS，TS3a=NOT_PASS，TS3b=INCONCLUSIVE，TS6=PASS，TS6b=PASS。调参后RHR选择速度0.1m/s、触发1s；主分析S500−B-RHR-tuned=94.53%，95%CI=[0.9015625, 0.9828125]。

# Round30c — regate and tuned right-hand baseline

本轮没有训练或更换任何checkpoint。8名学生的门在计算新指标之前预注册，目标、噪声、2048固定dev state/noise与round30b相同，唯一改动是仅在teacher动作norm≥.10m/s的样本算余弦。step非0与全样本误差≤.05m/s占比≥80%不变。已有6名学生维持原有效性，不用描述性新指标改变其判定。REGATE.md/JSON及dev_predictions/包含全部14人的.05/.10/.20敏感性、样本数、覆盖率、RMSE、原平均余弦与完整预测。近零动作的方向不能由全样本平均余弦稳健反映，本轮改变的是门定义，不是网络。

主分析由round30b已评估的2名加本轮修正门及能力门通过者组成；如实披露旧2人标签已打开，并在修正门后补入学生，所有人仍用同一组原预注册终点。敏感性只用本轮新增者；DEVELOPMENT追加旧4名local，其标签此前已打开。主分析少于8则全部终点IC；TS3b各分析集独立检查至少3名各偏好F10/F13≥10pp的前提。FROZEN_CHOICES.json只保存本轮新学生；旧6人的冻结选择按原收据复用，在analysis合并而不重选。

## RHR tuning and fairness

调参为16个新的state，固定seed区间2026101503000–2026101503999按升序取前16个，与DB uid/content/physical、round28 eval和round30 pilot去重。所有Ring DB state具备positions/velocities/goals，RING_PHYSICAL_SCHEMA_AUDIT.json说明非Ring异形state导致通用signature缺失的范围；不存在未解析Ring物理state的去重遗漏。12组speed(.10,.15,.25,.35)×trigger(20,40,80步)，低速阈值.01、顺时针、未到目标、D-local安全层保持原定义。不添加OrthoFlow3。每个有效学生每点16个state、seed0，按学生成功率等权平均最高选一个全局参数；平局速度更小、时长更短。unknown以下界参与选择，另报上界，不称为已知失败。参数冻结先于任何tuned主评估。全部网格结果及逐学生计数见RHR_GRID_RESULTS.json、调参raw和DB_ROWS，图见rhr_tuning_grid.png。该RHR用了学生调参标签，而η在联合教师上确定，未针对学生调整，因此比较偏向RHR。

| Speed m/s | Trigger s | Tuning success lower–upper | Unknown |
|---|---:|---:|---:|
| 0.1 | 1 | 12.05%–12.05% | 0 |
| 0.1 | 2 | 12.05%–12.05% | 0 |
| 0.1 | 4 | 12.05%–12.05% | 0 |
| 0.15 | 1 | 12.05%–12.05% | 0 |
| 0.15 | 2 | 12.05%–12.05% | 0 |
| 0.15 | 4 | 12.05%–12.05% | 0 |
| 0.25 | 1 | 12.05%–12.05% | 0 |
| 0.25 | 2 | 12.05%–12.05% | 0 |
| 0.25 | 4 | 12.05%–12.05% | 0 |
| 0.35 | 1 | 12.05%–12.05% | 0 |
| 0.35 | 2 | 12.05%–12.05% | 0 |
| 0.35 | 4 | 12.05%–12.05% | 0 |

## Primary, sensitivity and development tables

成功率下界/上界保留unknown；controller后state成簇配对bootstrap5000次，seed2026103032，两个seed成簇保留，每次重抽中重选最佳fixed/schedule/oracle。四臂oracle仅F10/F13/S300/S500。所有碰撞与超时率另列；timeout不等于deadlock。

### CONFIRMATORY

| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |
|---|---:|---|---:|---:|---:|
| nominal only | 0.00% | [0.00%, 0.00%] | 0.00% | 49.38% | 50.62% |
| safety baseline | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| F10 | 25.16% | [19.84%, 30.78%] | 25.31% | 0.00% | 74.69% |
| F13 | 87.66% | [73.44%, 97.03%] | 87.66% | 0.00% | 12.34% |
| S300 | 91.41% | [85.94%, 96.41%] | 91.41% | 0.00% | 8.59% |
| S500 | 94.53% | [90.16%, 98.28%] | 94.53% | 0.00% | 5.47% |
| B-RHR | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| B-RHR-tuned | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| A3 zero-shot | 94.53% | [90.16%, 98.28%] | 94.53% | 0.00% | 5.47% |
| always S500 | 94.53% | [90.16%, 98.28%] | 94.53% | 0.00% | 5.47% |
| oracle | 94.53% | [90.16%, 98.28%] | 94.53% | 0.00% | 5.47% |

- TS1: {"mean": 0.9453125, "CI95": [0.9015625, 0.9828125], "identification_interval": [0.9453125, 0.9453125], "CI_unknown_envelope": [0.9015625, 0.9828125], "passed": true, "status": "PASS"}
- TS2: {"mean": 0.06875, "CI95": [0.0, 0.203125], "coverage": 1.0, "passed": false, "CI_unknown_envelope": [0.0, 0.203125], "status": "NOT_PASS"}
- TS3a: {"regret": {"mean": 0.0, "CI95": [0.0, 0.0]}, "gain_over_safety": {"mean": 0.9453125, "CI95": [0.9015625, 0.9828125]}, "passed": true, "comparisons": {"A0_full": {"mean": 0.0, "CI95": [0.0, 0.0]}, "G_E": {"mean": 0.0, "CI95": [0.0, 0.0]}, "S500": {"mean": 0.0, "CI95": [0.0, 0.0]}, "F10": {"mean": 0.69375, "CI95": [0.621875, 0.759375]}}, "gain_CI_unknown_envelope": [0.9015625, 0.9828125], "regret_identification_interval": [0.0, 0.0], "status": "PASS"}
- TS3b: {"preference_F10_count": 0, "preference_F13_count": 9, "regret_A3_minus_A0": {"mean": -0.259375, "CI95": [-0.46878906249999996, -0.0671875]}, "status": "INCONCLUSIVE"}
- TS6: {"mean": 0.9453125, "CI95": [0.9015625, 0.9828125], "identification_interval": [0.9453125, 0.9453125], "CI_unknown_envelope": [0.9015625, 0.9828125], "passed": true, "status": "PASS"}
- TS6b: {"mean": 0.9453125, "CI95": [0.9015625, 0.9828125], "identification_interval": [0.9453125, 0.9453125], "CI_unknown_envelope": [0.9015625, 0.9828125], "passed": true, "status": "PASS"}
- always S500−A3: {"mean": 0.0, "CI95": [0.0, 0.0]}
- 安全层臂碰撞数（包含tuned）：0；unknown：1。

### SENSITIVITY

| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |
|---|---:|---|---:|---:|---:|
| nominal only | 0.00% | [0.00%, 0.00%] | 0.00% | 44.73% | 55.27% |
| safety baseline | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| F10 | 26.76% | [20.90%, 32.82%] | 26.95% | 0.00% | 73.05% |
| F13 | 86.72% | [69.53%, 97.85%] | 86.72% | 0.00% | 13.28% |
| S300 | 92.58% | [85.93%, 98.05%] | 92.58% | 0.00% | 7.42% |
| S500 | 95.31% | [90.43%, 99.22%] | 95.31% | 0.00% | 4.69% |
| B-RHR | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| B-RHR-tuned | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| A3 zero-shot | 95.31% | [90.43%, 99.22%] | 95.31% | 0.00% | 4.69% |
| always S500 | 95.31% | [90.43%, 99.22%] | 95.31% | 0.00% | 4.69% |
| oracle | 95.31% | [90.43%, 99.22%] | 95.31% | 0.00% | 4.69% |

- TS1: {"mean": 0.953125, "CI95": [0.904296875, 0.9921875], "identification_interval": [0.953125, 0.953125], "CI_unknown_envelope": [0.904296875, 0.9921875], "passed": true, "status": "PASS"}
- TS2: {"mean": 0.0859375, "CI95": [0.0, 0.25390625], "coverage": 1.0, "passed": false, "CI_unknown_envelope": [0.0, 0.25390625], "status": "NOT_PASS"}
- TS3a: {"regret": {"mean": 0.0, "CI95": [0.0, 0.0]}, "gain_over_safety": {"mean": 0.953125, "CI95": [0.904296875, 0.9921875]}, "passed": true, "comparisons": {"A0_full": {"mean": 0.0, "CI95": [0.0, 0.0]}, "G_E": {"mean": 0.0, "CI95": [0.0, 0.0]}, "S500": {"mean": 0.0, "CI95": [0.0, 0.0]}, "F10": {"mean": 0.685546875, "CI95": [0.603515625, 0.759765625]}}, "gain_CI_unknown_envelope": [0.904296875, 0.9921875], "regret_identification_interval": [0.0, 0.0], "status": "PASS"}
- TS3b: {"preference_F10_count": 0, "preference_F13_count": 7, "regret_A3_minus_A0": {"mean": -0.32421875, "CI95": [-0.556640625, -0.0859375]}, "status": "INCONCLUSIVE"}
- TS6: {"mean": 0.953125, "CI95": [0.904296875, 0.9921875], "identification_interval": [0.953125, 0.953125], "CI_unknown_envelope": [0.904296875, 0.9921875], "passed": true, "status": "PASS"}
- TS6b: {"mean": 0.953125, "CI95": [0.904296875, 0.9921875], "identification_interval": [0.953125, 0.953125], "CI_unknown_envelope": [0.904296875, 0.9921875], "passed": true, "status": "PASS"}
- always S500−A3: {"mean": 0.0, "CI95": [0.0, 0.0]}
- 安全层臂碰撞数（包含tuned）：0；unknown：1。

### DEVELOPMENT

| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |
|---|---:|---|---:|---:|---:|
| nominal only | 11.72% | [0.00%, 29.46%] | 11.72% | 35.94% | 52.34% |
| safety baseline | 11.83% | [0.00%, 29.69%] | 11.83% | 0.00% | 88.17% |
| F10 | 27.57% | [19.08%, 37.17%] | 27.68% | 0.00% | 72.32% |
| F13 | 89.06% | [78.24%, 95.87%] | 89.06% | 0.00% | 10.94% |
| S300 | 88.50% | [80.25%, 95.20%] | 88.50% | 0.00% | 11.50% |
| S500 | 94.20% | [90.29%, 97.54%] | 94.20% | 0.00% | 5.80% |
| B-RHR | 11.83% | [0.00%, 29.69%] | 11.83% | 0.00% | 88.17% |
| B-RHR-tuned | 11.83% | [0.00%, 29.69%] | 11.83% | 0.00% | 88.17% |
| A3 zero-shot | 90.74% | [82.59%, 96.99%] | 90.74% | 0.00% | 9.26% |
| always S500 | 94.20% | [90.29%, 97.54%] | 94.20% | 0.00% | 5.80% |
| oracle | 94.42% | [90.62%, 97.66%] | 94.42% | 0.00% | 5.58% |

- TS1: {"mean": 0.8236607142857143, "CI95": [0.6417410714285714, 0.9565011160714282], "identification_interval": [0.8236607142857143, 0.8236607142857143], "CI_unknown_envelope": [0.6417410714285714, 0.9565011160714282], "passed": true, "status": "PASS"}
- TS2: {"mean": 0.05133928571428571, "CI95": [-0.005580357142857143, 0.15069754464285673], "coverage": 0.9285714285714286, "passed": false, "CI_unknown_envelope": [-0.005580357142857143, 0.15069754464285673], "status": "NOT_PASS"}
- TS3a: {"regret": {"mean": 0.036830357142857144, "CI95": [0.0, 0.09821428571428571]}, "gain_over_safety": {"mean": 0.7890625, "CI95": [0.5870256696428571, 0.9453125]}, "passed": false, "comparisons": {"A0_full": {"mean": -0.03459821428571429, "CI95": [-0.09151785714285714, 0.0]}, "G_E": {"mean": -0.03459821428571429, "CI95": [-0.09151785714285714, 0.0]}, "S500": {"mean": -0.03459821428571429, "CI95": [-0.09151785714285714, 0.0]}, "F10": {"mean": 0.6316964285714286, "CI95": [0.546875, 0.7120535714285714]}}, "gain_CI_unknown_envelope": [0.5870256696428571, 0.9453125], "regret_identification_interval": [0.036830357142857144, 0.036830357142857144], "status": "NOT_PASS"}
- TS3b: {"preference_F10_count": 0, "preference_F13_count": 13, "regret_A3_minus_A0": {"mean": -0.18526785714285715, "CI95": [-0.3504743303571428, -0.044642857142857144]}, "status": "INCONCLUSIVE"}
- TS6: {"mean": 0.8236607142857143, "CI95": [0.6417410714285714, 0.9565011160714282], "identification_interval": [0.8236607142857143, 0.8236607142857143], "CI_unknown_envelope": [0.6417410714285714, 0.9565011160714282], "passed": true, "status": "PASS"}
- TS6b: {"mean": 0.8236607142857143, "CI95": [0.6417410714285714, 0.9565011160714282], "identification_interval": [0.8236607142857143, 0.8236607142857143], "CI_unknown_envelope": [0.6417410714285714, 0.9565011160714282], "passed": true, "status": "PASS"}
- always S500−A3: {"mean": 0.03459821428571429, "CI95": [0.0, 0.09151785714285714]}
- 安全层臂碰撞数（包含tuned）：0；unknown：1。

## Failure sources and rescue counts

自身停滞=nominal timeout且无碰撞；安全层导致停滞=nominal碰撞→safety timeout。两者为预注册描述性拆分，其他转移和unknown单列，不强行归类。TRANSFERS.json逐条保留nominal→safety，FAILURE_SOURCES.json逐学生完整结果构成；failure_sources.csv包含各组被F13、S500、原RHR和tuned救回的数量。

| Student | Nominal success/collision/timeout/unknown | Own stagnation | F13/S500/RHR/tuned rescued | Safety induced stagnation | F13/S500/RHR/tuned rescued |
|---|---|---:|---|---:|---|
| base_u_macflow | 0/31/33/0 | 33 | 32/32/0/0 | 31 | 30/30/0/0 |
| base_u_v2_macflow | 0/56/8/0 | 8 | 7/7/0/0 | 56 | 48/48/0/0 |
| base_u_plus_w_v4_wide_macflow | 0/50/14/0 | 14 | 14/14/0/0 | 50 | 50/50/0/0 |
| base_u_v4_macflow | 0/55/9/0 | 9 | 9/9/0/0 | 55 | 49/49/0/0 |
| base_u_plus_w_v6_macflow | 0/11/53/0 | 53 | 19/53/0/0 | 11 | 1/11/0/0 |
| base_u_plus_w_v3_macflow | 0/0/64/0 | 64 | 62/62/0/0 | 0 | 0/0/0/0 |
| base_u_plus_w_v4_train_targeted_macflow | 0/29/35/0 | 35 | 35/35/0/0 | 29 | 29/29/0/0 |
| base_u_v3_macflow | 0/0/64/0 | 64 | 54/54/0/0 | 0 | 0/0/0/0 |
| base_u_plus_w_v5_macflow | 0/39/25/0 | 25 | 24/24/0/0 | 39 | 34/34/0/0 |
| base_u_plus_w_v4_macflow | 0/45/19/0 | 19 | 19/19/0/0 | 45 | 45/45/0/0 |
| base_u_v10_local_macflow | 59/4/1/0 | 1 | 1/1/0/0 | 3 | 3/3/0/0 |
| base_u_v7_local_macflow | 0/0/64/0 | 64 | 55/53/0/0 | 0 | 0/0/0/0 |
| base_u_v8_local_macflow | 0/0/64/0 | 64 | 60/64/0/0 | 0 | 0/0/0/0 |
| base_u_v9_local_macflow | 46/2/16/0 | 16 | 16/16/0/0 | 2 | 2/2/0/0 |

合并分析两类计数：自身停滞469、安全层导致停滞321。自身停滞占两类多数，问题陈述应以分散协调困难为主，安全层是另一部分原因。不能仅从timeout推断真实deadlock。

## Conclusion mapping

TS1、TS6b通过：在本协议专家不可部署、不通信的分散学生上，修正有必要且优于调参后的经典RHR。

TS2未通过：学生的阶段时间表优势未获支持；按任务映射，分散学生使用固定η13已足够，一律S500仍可作默认选择。该结论限于本固定菜单与样本，不把不显著当作普遍等效。

## Inheritance, cost, cache and integrity

TS4在results.json报告学生/教师最佳臂、固定偏好、四臂Spearman。TS5报告原教师D-local/central/shared四臂配对差；TS5_TEACHER_SUPPLEMENT.md/JSON另报教师nominal/safety原有标签与学生的配对差，全部复用。已有准备输入、单核JIT预热后的学生网络/教师10步/四个QP耗时从round30b的10个global及round30的4个local直接复用，出处和hash见REUSED_TIMING.json；它们不等同新tuned完整闭环延迟，未重新采样耗时。round30b等变/act_scale诊断只作原证据引用，不把修正门当作严格旋转等变证明。

copy/core/pipeline/episode/actions与round30b相同，16条逐位回放门直接引用；tuned的原R28参数点动作与85步计数序列逐位通过，12组边界/方向/幅度测试通过。Student.nominal主体AST与round30b相同，唯checkpoint路径定位改变；4名local的旧to_world公式与当前部署逐位一致。没有monkeypatch或修改已有源码、物理、安全层、事件、成功标准、checkpoint，没有git写命令或环境安装。DB_PATH_AUDIT.json检查每个入口/worker模块路径及所有source_file journal路径均在Basin_C1共享库；PoC未写入journal。

每阶段design→preflight封存→missing→default journal→官方单进程merger→postflight，原6名全部七臂标签精确复用。若tuned选择原R28参数，则同样复用已有B-RHR，绝不按改名重复rollout。registration/每次记录所有旧行hash不变与新增行/sha256，未UPDATE/DELETE。各阶段planned/reuse/missing、DB_ROWS、CACHE_POSTFLIGHT、MERGE_RECEIPT保留完整链。unknown不重跑，不算已知失败。

INTEGRITY_START/END对Basin git状态/diff哈希、官方仓库HEAD/status/diff哈希、PoC/round30/round30b全部文件路径/hash核对；INTEGRITY_COMPARISON单列并行任务新增git条目（非本轮所写）。IMPORTED_SOURCE_AUDIT记录每个已有import源，COPY_PROVENANCE/COPY_DIFF.patch记录复制/派生源及最终diff。RESULTS_VALIDATION核门、选择、时序、缓存、冻结文件和资源。RESOURCE_MONITOR保存所有提交时的squeue与运行占用，CPU总上限20，本轮GPU推理申请2shards。没有要求视频。
