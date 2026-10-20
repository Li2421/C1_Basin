源码完整性审计通过，每入口/worker的Basin_C1 shared_rollout_db与默认journals路径断言通过。10个新学生中2个通过蒸馏有效性门。主分析（CONFIRMATORY）：TS1=INCONCLUSIVE，TS2=INCONCLUSIVE，TS3a=INCONCLUSIVE，TS3b=INCONCLUSIVE，TS6=INCONCLUSIVE；合并分析（DEVELOPMENT）：TS1=INCONCLUSIVE，TS2=INCONCLUSIVE，TS3a=INCONCLUSIVE，TS3b=INCONCLUSIVE，TS6=INCONCLUSIVE。按预注册的两类失败拆分，主要来源为自身停滞（自身停滞186条，安全层导致的停滞92条；合并分析集）；timeout只表示超时，不能据此称为deadlock。

# Round30b frame-consistent decentralized students

本轮只修正global教师动作与噪声的坐标系，教师冻结；网络、训练步数、batch、seed、dev选择与round30一致。数据与归一化逐文件hash复用，无重复生成训练rollout。所有代码/输出写入本轮目录，外部写入只有官方DB新身份追加、默认journal和单进程merger。未改动任何已有源码、环境、物理、安全层、成功标准或事件。未运行git写命令，未安装软件。旧10个无效global学生结果**已被取代**，原文件保留；4个有效local学生与冻结选择/标签直接复用，已打开过的旧标签仅列DEVELOPMENT。

## 冻结、门与样本

预注册与收据：PREREGISTRATION.md、PREREGISTRATION.receipt.json。10个学生各只训练一次80,000步、每500步固定dev选择，选取早于闭环。NOISE_GATE.json引用1008/1008历史门，NOISE_IMPLEMENTATION_COMPATIBILITY.json核对迁移路径源码的相同hash。EQUIVARIANCE.receipt.json先于训练；TRAINING_FROZEN.receipt.json先于学生pilot；CHOICES_FROZEN.receipt.json先于任何主评估标签。16条round28回放首动作、位置、终止和步数全部逐位通过。单元测试和组件测试见UNIT_TESTS.json、COMPONENT_TESTS.json。

蒸馏门三条同时满足：非step0、独立2048固定state/noise样本平均cosine≥.90、误差≤.05m/s占比≥80%。未通过不重训、不调参、不替换。逐学生见DISTILLATION_REPORT.md。能力门在固定16个round30 pilot states上两臂seed0执行；合格新学生2个。能力门排除记录见CAPABILITY_GATE.json。遵循round30少于8个能力门合格新学生则整轮INCONCLUSIVE的约束；此时数值仅描述性，不作验证性结论。主分析对象为蒸馏有效且通过预注册能力门者，蒸馏有效完整集合与被能力门排除者单独披露。

## 坐标与等变诊断

目标u与trace第0步ξ逐行经自己的径向/切向R_iᵀ旋转；训练float32，部署numpy float64，数值一致性误差见UNIT_TESTS.json。自身local观测、编号one-hot、自身fold_in(key_for(state,seed,step),i)噪声输入；local输出转world，沿用原限速。转换只依赖自身位置，改变他人目标时自身输出不变，固定输入噪声确定。local教师保留原local噪声和动作定义，与旧local部署一致。

10个global教师的act_scale均不是逐机器人x/y严格各向同性，因此按任务要求，本轮噪声旋转视为近似，而非严格等价的归一化flow变换。ACT_SCALE_AUDIT.json给出全部尺度与比例。等变诊断每教师256个固定训练state、16噪声、随机旋转，不作门；EQUIVARIANCE.json/图给出完整结果。有限16样本的均值差包含Monte Carlo误差，不能单凭该描述性比例断言教师结构性的非等变导致蒸馏失败。

## 论文总表与预注册终点

主分析与合并分析各32个配对state×2seed；controller后state成簇配对bootstrap5000次；每次重抽内重新求best-fixed、best-schedule和oracle。unknown保留，成功率下界0、上界1，不当已知失败、不重跑。表内置信区间为下界估计；上界估计及差值包络见results.json。

### CONFIRMATORY

| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |
|---|---:|---|---:|---:|---:|
| nominal only | 0.00% | [0.00%, 0.00%] | 0.00% | 67.97% | 32.03% |
| safety baseline | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| F10 | 18.75% | [9.38%, 28.12%] | 18.75% | 0.00% | 81.25% |
| F13 | 91.41% | [81.25%, 98.44%] | 91.41% | 0.00% | 8.59% |
| S300 | 86.72% | [77.34%, 94.53%] | 86.72% | 0.00% | 13.28% |
| S500 | 91.41% | [81.25%, 98.44%] | 91.41% | 0.00% | 8.59% |
| B-RHR | 0.00% | [0.00%, 0.00%] | 0.00% | 0.00% | 100.00% |
| A3 zero-shot | 91.41% | [81.25%, 98.44%] | 91.41% | 0.00% | 8.59% |
| always S500 | 91.41% | [81.25%, 98.44%] | 91.41% | 0.00% | 8.59% |
| oracle | 91.41% | [81.25%, 98.44%] | 91.41% | 0.00% | 8.59% |

- TS1: {"mean": 0.9140625, "CI95": [0.8125, 0.984375], "identification_interval": [0.9140625, 0.9140625], "CI_unknown_envelope": [0.8125, 0.984375], "passed": true, "status": "INCONCLUSIVE"}
- TS2: {"mean": 0.0, "CI95": [0.0, 0.0], "coverage": 1.0, "passed": false, "CI_unknown_envelope": [0.0, 0.0], "status": "INCONCLUSIVE"}
- TS3a: {"regret": {"mean": 0.0, "CI95": [0.0, 0.0]}, "gain_over_safety": {"mean": 0.9140625, "CI95": [0.8125, 0.984375]}, "passed": true, "comparisons": {"A0_full": {"mean": 0.0, "CI95": [0.0, 0.0]}, "G_E": {"mean": 0.0, "CI95": [0.0, 0.0]}, "S500": {"mean": 0.0, "CI95": [0.0, 0.0]}, "F10": {"mean": 0.7265625, "CI95": [0.59375, 0.8515625]}}, "gain_CI_unknown_envelope": [0.8125, 0.984375], "regret_identification_interval": [0.0, 0.0], "status": "INCONCLUSIVE"}
- TS3b: {"preference_F10_count": 0, "preference_F13_count": 2, "regret_A3_minus_A0": {"mean": 0.0, "CI95": [0.0, 0.0]}, "status": "INCONCLUSIVE", "numerical_status": "INCONCLUSIVE"}
- TS6: {"mean": 0.9140625, "CI95": [0.8125, 0.984375], "identification_interval": [0.9140625, 0.9140625], "CI_unknown_envelope": [0.8125, 0.984375], "passed": true, "status": "INCONCLUSIVE"}

### DEVELOPMENT

| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |
|---|---:|---|---:|---:|---:|
| nominal only | 27.34% | [0.00%, 60.68%] | 27.34% | 24.22% | 48.44% |
| safety baseline | 27.60% | [0.00%, 61.72%] | 27.60% | 0.00% | 72.40% |
| F10 | 28.65% | [9.90%, 48.96%] | 28.65% | 0.00% | 71.35% |
| F13 | 92.19% | [86.71%, 96.88%] | 92.19% | 0.00% | 7.81% |
| S300 | 83.07% | [67.19%, 95.83%] | 83.07% | 0.00% | 16.93% |
| S500 | 92.71% | [85.94%, 98.44%] | 92.71% | 0.00% | 7.29% |
| B-RHR | 27.60% | [0.00%, 61.72%] | 27.60% | 0.00% | 72.40% |
| A3 zero-shot | 84.64% | [68.23%, 97.40%] | 84.64% | 0.00% | 15.36% |
| always S500 | 92.71% | [85.94%, 98.44%] | 92.71% | 0.00% | 7.29% |
| oracle | 93.23% | [86.98%, 98.44%] | 93.23% | 0.00% | 6.77% |

- TS1: {"mean": 0.6510416666666666, "CI95": [0.3046875, 0.9322916666666666], "identification_interval": [0.6510416666666666, 0.6510416666666666], "CI_unknown_envelope": [0.3046875, 0.9322916666666666], "passed": true, "status": "INCONCLUSIVE"}
- TS2: {"mean": 0.005208333333333333, "CI95": [-0.018229166666666668, 0.036458333333333336], "coverage": 0.8333333333333334, "passed": false, "CI_unknown_envelope": [-0.018229166666666668, 0.036458333333333336], "status": "INCONCLUSIVE"}
- TS3a: {"regret": {"mean": 0.0859375, "CI95": [0.0, 0.21354166666666666]}, "gain_over_safety": {"mean": 0.5703125, "CI95": [0.19791666666666666, 0.8854166666666666]}, "passed": false, "comparisons": {"A0_full": {"mean": -0.08072916666666667, "CI95": [-0.19791666666666666, 0.0]}, "G_E": {"mean": -0.08072916666666667, "CI95": [-0.19791666666666666, 0.0]}, "S500": {"mean": -0.08072916666666667, "CI95": [-0.19791666666666666, 0.0]}, "F10": {"mean": 0.5598958333333334, "CI95": [0.421875, 0.6927083333333334]}}, "gain_CI_unknown_envelope": [0.19791666666666666, 0.8854166666666666], "regret_identification_interval": [0.0859375, 0.0859375], "status": "INCONCLUSIVE"}
- TS3b: {"preference_F10_count": 0, "preference_F13_count": 6, "regret_A3_minus_A0": {"mean": 0.0, "CI95": [0.0, 0.0]}, "status": "INCONCLUSIVE", "numerical_status": "INCONCLUSIVE"}
- TS6: {"mean": 0.6510416666666666, "CI95": [0.3046875, 0.9322916666666666], "identification_interval": [0.6510416666666666, 0.6510416666666666], "CI_unknown_envelope": [0.3046875, 0.9322916666666666], "passed": true, "status": "INCONCLUSIVE"}

## 逐学生失败来源与救回计数

| Student | Nominal success/collision/timeout/unknown | Own stagnation | F13/S500/B-RHR rescued | Safety induced stagnation | F13/S500/B-RHR rescued |
|---|---|---:|---|---:|---|
| base_u_macflow | 0/31/33/0 | 33 | 32/32/0 | 31 | 30/30/0 |
| base_u_v2_macflow | 0/56/8/0 | 8 | 7/7/0 | 56 | 48/48/0 |
| base_u_v10_local_macflow | 59/4/1/0 | 1 | 1/1/0 | 3 | 3/3/0 |
| base_u_v7_local_macflow | 0/0/64/0 | 64 | 55/53/0 | 0 | 0/0/0 |
| base_u_v8_local_macflow | 0/0/64/0 | 64 | 60/64/0 | 0 | 0/0/0 |
| base_u_v9_local_macflow | 46/2/16/0 | 16 | 16/16/0 | 2 | 2/2/0 |

## 结论解释

本轮仅2个新学生满足有效性与能力门，未达到round30预注册的至少8个新学生要求；因此所有验证性终点保持INCONCLUSIVE，下面的数值差异不构成验证性通过。新增2个与旧4个合并只得到6个有效学生，未得到预期的14个有效学生，也不能把此前已打开的4个local学生标签转为新验证证据。

CONFIRMATORY描述性S500−safety=91.41%，95%CI=[0.8125, 0.984375]；S500−B-RHR=91.41%，95%CI=[0.8125, 0.984375]；A3平均oracle遗憾=0.00%。正式结论以各终点status为准。

DEVELOPMENT描述性S500−safety=65.10%，95%CI=[0.3046875, 0.9322916666666666]；S500−B-RHR=65.10%，95%CI=[0.3046875, 0.9322916666666666]；A3平均oracle遗憾=8.59%。正式结论以各终点status为准。

8个学生因余弦阈值未过而无效，说明只修正坐标系仍不足以让所有global教师通过本轮分散化有效性标准。等变诊断及act_scale各向异性提供解释线索，但不足以单独建立失败原因；给学生全局坐标的自身观测是任务建议的后续方向，本轮未追加训练。

## 失败来源与继承/代价

failure_sources.csv及FAILURE_SOURCES.json逐学生报告nominal only的success/collision/timeout/unknown、nominal→safety转移和两类失败被F13/S500/B-RHR救回的数量。TRANSFERS.json保留每个state/seed的逐条转移。自身停滞定义为nominal timeout且无碰撞；安全层导致停滞定义为nominal collision→safety timeout，两组不重叠。其他失败及unknown不硬归入两类；两类计数比较不代表已覆盖所有失败。将自身停滞解释为协调困难的描述性证据，不能从timeout直接推断死锁。

results.json的TS4给出学生/教师最佳臂一致性及四臂Spearman；TS5给出与原D-local教师的配对差。INFERENCE_TIMING.json是单核JIT预热后自身网络、教师10步、四个本地QP的耗时，测量范围不同，不声称完整闭环延迟的同比加速。教师补充臂及四个local学生均复用旧证据，没有重跑。TRAINING_CONFIG.json保留round30参数卡，坐标语义覆盖见预注册及EFFECTIVE_FRAME_CONFIG.json。训练曲线见training_curves.csv/.png/.pdf；全部checkpoint和history位于students/。

## 结论约束

TS1、TS6同时通过时才支持在该分散学生分析集上修正优于安全baseline及经典右手规则。TS2不通过时按任务映射报告阶段结构优势未获支持，并比较固定F13与默认S500；不把置信区间覆盖0当作等效性证明。TS3a不通过且A3系统性选错时才称为负迁移，优先比较稳健默认动作。多个蒸馏门失败结合等变诊断讨论全局自身观测的后续需要，不追加训练或用新输入替换。若自身停滞占两类失败多数，问题陈述应强调分散协调，而不是将所有timeout归因于安全层。

## 缓存、资源、完整性

运行中出现一次协调器中断：Slurm成功提高剩余数组任务的并发，但对两个已完成子任务返回非零。协调器停止，已提交的worker原样继续，未重复提交或重跑；最终使用finalize_existing.py接续后处理。原错误和readback保存在AUTOMATION_INTERRUPTION.json，FINAL_STATUS.json记录最终完成状态。这不改变任何模型、选择、阈值、物理或评估请求。

各阶段*_PLANNED_ROLLOUTS、*_CACHE_PREFLIGHT与封存收据先于提交，只跑缺失；*_CACHE_POSTFLIGHT与MERGE_RECEIPT核官方DB逐条一致，unknown保留。registration/保存所有新增行及sha256和每表原有行前后hash。probe是未标注短探测，只作为官方journal的SUMMARY_ONLY来源，不伪造任务终点。DB_PATH_AUDIT.json逐worker核官方模块与默认journal目录，禁止PoC副本写入。

RESOURCE_MONITOR.json记录各阶段squeue占用及本轮array throttle，CPU总上限20，GPU上限2shards；并行round33既有18CPU时本轮用余下2CPU，空闲释放后才提高本轮并发。源码复制源hash及最终diff见COPY_PROVENANCE.json/COPY_DIFF.patch/ROUND28_COPY_AUDIT.json。受保护PoC及round30全文件路径/hash、两个仓库git状态/diff哈希在INTEGRITY_START/END中，比较在INTEGRITY_COMPARISON.json；并行任务新增status条目单列，不归本轮。IMPORTED_SOURCE_AUDIT.json核已有import源hash。RESULTS_VALIDATION.json为交付一致性检查。无视频交付要求。
