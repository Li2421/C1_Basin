# Round30：冻结联合Flow → 分散式一步学生

源码完整性记录未完全通过：既有已审计源码内容hash一致，但Git状态出现本轮目录之外的新增条目，且本轮错误地在PoC副本写入448份journal，写入范围审计未通过（路径清单见INTEGRITY_COMPARISON.json、JOURNAL_PATH_DEVIATION.json）。TS1（安全但卡住）通过；TS2（时间表优势）未通过；TS3a（完整菜单零样本性能）未通过；TS3b（固定菜单controller信息）为INCONCLUSIVE。以上为按预注册计算的结果，必须连同流程偏差和6个step0学生的限制解读。

## 预注册终点

| 终点 | 配对结果 | 判定 |
|---|---|---|
| TS1：S500−safety baseline | +31.70pp，95% CI [+17.97pp, +48.55pp] | 通过 |
| TS2：最佳时间表−最佳固定 | +0.22pp，95% CI [-3.35pp, +2.01pp]；覆盖85.71% | 未通过 |
| TS3a：A3相对oracle平均遗憾 | 4.80pp；A3−safety=+28.24pp，95% CI [+15.07pp, +43.53pp] | 未通过 |
| TS3b：A3遗憾−A0遗憾 | -3.57pp，95% CI [-9.82pp, +2.12pp]；固定偏好F10/F13方向各0/7个达到10pp | INCONCLUSIVE |

CI采用5,000次先controller、再state成簇配对bootstrap；两seed及所有臂保持配对，每次重选oracle/最佳固定/最佳时间表。unknown保留，下界代0、上界代1，绝不称为已知失败或重跑。主评估unknown=0；另报识别区间与unknown包络。教师旧标签为DEVELOPMENT；学生为预注册确认性设计，但不能宣称本轮无流程偏差。

按预注册映射，安全层下的问题仍存在，但时间表优势未成立；失败终止类型、发生时刻和停滞诊断见FAILURE_DIAGNOSTICS.json，不把timeout称为deadlock。

## 成功、安全与转移

14学生×32state×2seed×7臂=6,272条；能力门448条全部入库，14/14通过；教师补充1,792条，只补nominal-only/D-local safety。另有448条无标签probe和16条用户明确要求的历史一致性重放。专家未运行；教师四臂三管线直接复用round28。

加安全层的学生臂碰撞总计0。这只是本批样本证据，不是普遍安全证明。停滞阈值为未达目标、速度<0.01m/s连续40步；正常排队也可能满足，不是正式deadlock判据。

nominal-only下碰撞的601条，加安全层后成功1、timeout 600、unknown 0、碰撞0、其他0。这张配对表描述安全层转移；S500−baseline本身不证明所有停滞都由安全层造成。

完整成功率/CI、碰撞率、超时率见[论文总表](paper_table_complete.md)；[逐条转移表](transition_table.md)、[逐episode指标](rollout_metrics.csv)、[失败与停滞诊断](FAILURE_DIAGNOSTICS.json)。教师B-RHR仅有D-shared历史参照，不冒充D-local；教师A3/A0使用历史R5和原冻结模型，仅作DEVELOPMENT参照。

## 蒸馏、迁移与分散化代价

全部14个学生完成80,000步训练，仅用固定dev MSE选checkpoint。6个dev最优为初始化step0：base_u_macflow, base_u_plus_w_v3_macflow, base_u_plus_w_v4_train_targeted_macflow, base_u_v3_macflow, base_u_v2_macflow, base_u_plus_w_v5_macflow。这些不能当作成功蒸馏的证据，能力门通过也不改变这一点。其余checkpoint亦未根据闭环调参或重训。

所有学生输入统一为自身local_observation行、id和自身2维noise，采用官方ActorVectorField默认宽度512×4与lr=3e-4。14个信息隔离/确定性测试通过；不使用其他机器人目标，不通信。教师输出raw的实际chart随原frame：global教师为world，local教师为径向/切向，执行用原to_world及再限速。仅径向观测与world目标存在信息差，本轮没有把它与去中心化或联合noise边缘化的代价分离。

[蒸馏报告](DISTILLATION_REPORT.md)、[曲线](figures/training_curves.png)、TRAINING_CONFIG.json与TEACHERS.json记录训练、dev误差、seed及输入hash。未使用专家action作为蒸馏标签，不做DAgger。

TS4：教师/学生最佳四臂一致率42.86%；最佳固定类型一致率50.00%，学生多数类型基线71.43%。四臂平均Spearman为0.40834260936075445；逐controller结果和类型列联表见results.json与[继承表](inheritance_table.md)。统一偏好S500会使最佳臂一致率显得较高，因此不能单凭它证明偏好继承。

TS5逐臂学生相对教师D-local/集中TT的配对差与CI见results.json；碰撞、unknown和停滞描述见FAILURE_DIAGNOSTICS.json。性能结论限这14配方、32state、Q2、N=4及本次观测/动作chart，不是N=100或一般多机器人死锁证明。

## 推理耗时

[单核耗时表](inference_timing_table.md)与INFERENCE_TIMING.json给出JIT预热后的每机器人学生单步、教师联合10步Euler及4个本地QP。4份学生在同一CPU顺序执行并不必然快于一个联合教师；研究runner还有重复nominal调用，不能把微基准说成完整管线加速。专家仅引用round28的32/32与约0.075s/episode规划，需全部目标，不能在本轮信息假设下原样部署。

## 流程、写入范围与完整性

1. 复制core.py/pipeline.py与原文件字节一致；episode函数AST与原run.py一致。16条D-local重放首动作、终止类型、长度、完整位置逐位一致。新臂通过单独runtime包装接入，控制器/物理/事件/成功标准保持冻结。
2. design仅用官方connect/transaction追加身份，逐uid冲突检查；每次插入前后已有登记行hash、已有experiment的rollout计数不变。新增行JSON与hash见registration/*。已有rollout不直接SQL插入。
3. pilot启动前后的runner包装重构造成协议记录hash与实际版本不同步，原版本已按hash重建保留。pilot仅运行local baseline/S500，数值执行链相同；这是流程偏差，不以补写收据掩盖，见PILOT_RUNNER_PROVENANCE.json。
4. pilot的shadow import使448份journal误写PoC/source/shared_rollout_db/journals/exp_teacher_student_macflow_v1_pilot/，违反写入范围。既有源码未改动；越界文件原样保留，未擅自删除。通过Basin_C1官方append_journal重新登记已有raw，再经官方merger入库；0条rollout重跑。后续入口先导入并检查官方DB模块路径，未替换任何模块全局变量，见JOURNAL_PATH_DEVIATION.json。
5. 主评估与教师补充均在preflight封存后提交，运行源码保持MAIN_IMPLEMENTATION_FROZEN.receipt.json的hash；辅助shard仅执行原计划尾部缺失，按边界停止，原worker跳过已有raw。ROLLOUT_DEDUP_AUDIT.json核对唯一执行/journal身份。
6. 绘图首次环境缺matplotlib，改用既有.venv-c1-plots；没有安装/升级包、改原Python环境或重跑rollout。初次包装幅度测试仅因±0字节差异失败，修正为严格数值比较后最大误差0，控制代码不变，原测试/log保留。

Git状态中本轮目录之外的新增条目（本轮未写这些条目；工作区另有并行任务）：
- Basin_C1: `?? diagnostics/axis1_state_v1/`
- Basin_C1: `?? docs/generalization_axes_protocol_20261015.md`
- Basin_C1: `?? docs/prompt_round31_gain_attribution_20261015.md`
- Basin_C1: `?? docs/prompt_round32_axis1_state_20261015.md`
- Basin_C1: `?? docs/prompt_round33_axis2_controller_20261015.md`
- Basin_C1: `?? docs/prompt_round34_axis3_cross_scene_20261015.md`
- Basin_C1: `?? docs/prompt_round35_axis4a_geometry_horizon_20261015.md`
- Basin_C1: `?? docs/prompt_round36_axis4b_agent_count_20261015.md`

起止原始记录和完整448份越界路径清单见INTEGRITY_START.json、INTEGRITY_END.json、INTEGRITY_COMPARISON.json；所有import/复制的既有源hash见SOURCE_HASHES.json。不能将既有源码内容hash一致与写入范围合规混为一谈。

DB的逐阶段preflight/postflight、官方合并、旧行保全和去重见*_CACHE_*.json、*_MERGE_RECEIPT.json、FINAL_DB_AUDIT.json。默认新task标签总数8,512；probe不伪造任务outcome，官方DB仅保存其SUMMARY_ONLY来源，R5原始hash索引作为无标签证据。

本轮Slurm分配上限18CPU shard、2GPU shard；训练用2GPU shard，其余闭环CPU。辅助任务在教师补充释放8CPU后提交。后期观测到其他用户任务，本轮10CPU与其合计16CPU，未另行增加资源。无视频要求，未交付视频。

## 交付索引

[历史审计](HISTORY_EVIDENCE.md)、[单元测试](UNIT_TESTS.json)、[能力门](CAPABILITY_GATE.json)、[冻结选择](FROZEN_CHOICES.json)、[总结果](results.json)、[交接](HANDOFF.md)。所有新代码与正式交付位于本目录；上述PoC journal越界是已披露例外，不声称严格源码保护流程全部通过。

## 关键负结果与失败发生时间

本轮不能支持时间表优势和冻结critic零样本迁移在学生上同时成立的主实验结论。safety baseline成功率11.83%，S500为43.53%，F13为42.41%；最佳时间表相对最佳固定只高0.22pp，CI跨0。

A3成功率40.07%，相对四臂oracle平均遗憾4.80pp，超过3pp门槛；虽比safety高28.24pp，仍不满足TS3a。A3相对always-S500/A0/G_E低3.46pp（95% CI [-9.38,0.00]pp，描述性）。固定偏好F10/F13达到10pp者为0/7个，不满足TS3b双向各至少3个的前提，不能确认controller信息关系迁移。

nominal的601条碰撞中，600条加安全层后变为timeout，1条成功。baseline的790个失败全是35s timeout，仅104条有合格连续低速段，其低速起点中位数23.75s。F13的516个失败和S500的506个失败也全为35s timeout；其中合格低速起点中位数分别15.625s、15.95s。其余timeout并非已经证明的低速停滞或deadlock。详见[发生时间表](failure_timing_table.md)及failure_timing.csv。

S500学生相对教师D-local低40.51pp（95% CI [-54.24,-26.00]pp），相对集中TT低49.55pp（95% CI [-65.40,-32.92]pp，下界标签统计）。这些代价包含部分学生蒸馏失败和观测/动作chart信息差，不能单独归因于分散式安全层。
