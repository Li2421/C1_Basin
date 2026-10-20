# 历史证据与复用

开始时检索过往配置、checkpoint、数据、预测与报告；首次审计保存在attempt1_blocked/HISTORY_AUDIT.md，后续机器记录为HISTORY_AUDIT.json、TEACHERS.json、TEACHER_SUPPLEMENT_HISTORY_AUDIT.json。

| 证据路径 | 兼容性与本轮处理 |
|---|---|
| diagnostics/decentralized_pipeline_v1/main_protocol.json 与14个best.pkl | 教师逐一按SHA256冻结；全部匹配。训练集按各教师config中的manifest hash定位，而非直接使用adapter manifest。 |
| diagnostics/decentralized_pipeline_v1/EVAL_STATES.json | 32个真实物理初态直接使用，保留原uid别名、RNG namespace、位置/速度/目标；不生成新评估state。官方exact-journal要求physical只有三个数组，因此登记不含冗余timestep=0的新身份表示，original_state_uid保持配对。 |
| diagnostics/decentralized_pipeline_v1/main/raw 与main/traces | 直接复用教师F10/F13/S300/S500的central/shared/local标签，不重复生成。只按本轮明确要求复现16条D-local轨迹作为复制实现门，不作为新DB标签。 |
| diagnostics/decentralized_pipeline_v1/expert 与REPORT.md | 专家32/32、每局规划约0.075s，仅作需要全部目标的DEVELOPMENT特权参照，本轮不运行专家。 |
| 同评估state的共享DB eta0历史记录 | 找到928条：896条D-shared RHR、32条特权专家，不是nominal-only或D-local safety baseline，执行语义不兼容。教师两补充臂仍经精确preflight只跑缺失。 |
| toy_giveway/flowbc.py 与原GiveWayFlowBCAgent | 兼容导出/联合Stage-I，无onestep蒸馏学生；不能替代本轮自身目标/自身noise学生。 |
| double_bottleneck/flowbc_4a_agent.py | 展平联合Stage-I，use_distillation=False；不是本轮分散式一步学生。 |
| new_benchmark_common/macflow.py | 教师联合actor_bc_flow及native采样可只读复用；并未构造actor_onestep_flow。 |
| MACFlow_Official/agents/macflow.py 与utils/networks.py | 只读参考官方ActorVectorField、默认宽度/lr和蒸馏损失；不修改官方仓库，学生实现为独立新文件。 |
| diagnostics/other_controller_classes_v1/probes.py、freeze_choices.py | 原文件含全局替换和其他目录写入，未直接运行；复制参考后在本目录改写，source hash和diff保留。 |
| diagnostics/ring_controller_critic_v1/fits | 原冻结A3/A0、预处理/PCA及选定steps只读使用，不拟合新critic。学生选择在主评估前封存。 |
| diagnostics/controller_flow_blend_v1/stage_2/probes/ | 复用固定16个probe state；学生控制器改变，旧教师probe轨迹不能替代学生R5，需新无标签探测。旧教师R5特征仅用于DEVELOPMENT教师选择参照，不运行新教师probe。 |
| 第三次噪声门与3份完成的数据重建 | 保留原收据及源码快照，核对hash复用；未完成数据重做。数据重建函数逻辑不变，辅助源码审计函数移至独立模块，见DATA_REUSE_CODE_AUDIT.json。 |

新证据缺口：14个学生checkpoint、各自离线误差及自身信息隔离检查；新的pilot能力门与R5；本轮学生执行身份的32state Q2七臂标签；教师两个真正缺失的补充臂。没有因critic选择或输入编码改变而重跑相同execution的闭环标签。

所有新task rollout按design登记→preflight封存→仅缺失→默认journal→官方单进程merger→postflight执行。pilot有两项已披露偏差：runner启动时hash衔接不同步、shadow import将448份journal写入PoC副本。原始证据保持不动，已有rollout未重跑；后者通过官方append_journal重新登记已有raw并合并，不更改结果。

pilot物理去重的已解析/未解析历史state数量见STATE_DEDUP_AUDIT.json。未解析记录不声称完成了逐物理state核验。unknown保留，不作为已知失败、不重跑；timeout不是deadlock。
