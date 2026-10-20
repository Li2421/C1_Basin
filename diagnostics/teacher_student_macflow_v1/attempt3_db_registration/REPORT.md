# Round30 第三次运行：噪声门通过，DB 身份登记阻塞

源码完整性审计通过：Basin_C1 与 MACFlow_Official 的 HEAD、git status 和 git diff SHA256 与本次起始记录完全一致，PoC `poc/` 下全部5个 Python 文件 hash 不变。TS1（学生是否安全但卡住）、TS2（时间表优势）、TS3a（零样本迁移性能）、TS3b（是否利用 controller 信息）均为 **NOT_RUN，无法判定**。修订后的噪声门已通过全部1,008个样本，但本轮允许的DB写入路径缺少新精确执行身份的登记接口，依第零节第5条停止；未训练学生或生成闭环标签，交付未全部完成。

## 已完成的续跑

完整读取当前任务文件。`attempt1_blocked/`保持不动；第二次全部产物原样移入`attempt2_gate_tolerance/`，SHA256逐文件校验通过。原PREREGISTRATION.md及收据保留字节不变，并在归档中保留原件。新写PREREGISTRATION_SUPPLEMENT.md及UTC/hash收据，仅更新任务明确修改的噪声门；尚未读取学生结果。重新完成起始完整性审计。

14个教师checkpoint hash与round28 main_protocol一致，14份实际训练manifest均与教师训练配置中的hash精确匹配，见TEACHERS.json。历史证据审计沿用HISTORY_AUDIT.json及前两次归档：Toy/Double-Bottleneck相关实现仅Stage-I，未建立本轮可直接复用的学生证据；round28四臂三管线与专家证据可作后续DEVELOPMENT参照。本次未运行专家。

## 噪声一致性门

按原预注册顺序，每教师实际训练集的前72个train rollout，取第0状态和固定key，共14×72=1,008个样本。全部导入完成后显式恢复JAX x64=False，在单CPU核运行，无环境step。

| 检查 | 结果 |
|---|---|
| trace第0步与独立jax.random.normal逐位一致 | 1,008/1,008通过 |
| native经原to_world与再限速，与非traced nominal逐位一致 | 1,008/1,008通过 |
| 独立解码trace终点，与native动作最大绝对差≤1e-5 m/s | 1,008/1,008通过 |
| trace终点误差最大值 | 7.213342290146585e-8 m/s |
| trace终点误差99.9分位 | 6.82587023567164e-8 m/s |

未调用原traced=True的内置FP64容差断言，未修改它；未切换教师到x64=True。证据见NOISE_GATE.json、NOISE_GATE.receipt.json、noise_gate.log。

另已只读扫描14教师train/dev数据的位置差分速度，所有重建速度均低于reset速度上限；最大约0.460000000000025 m/s，见DATA_VELOCITY_AUDIT.json。数据重建作业21340使用14个单CPU shard、0 GPU。3个教师数据完成，其余在DB阻塞确认后取消；部分输出原样保留。没有训练作业提交，当前无本轮活动作业。训练代码student.py/train.py与sbatch仅为尚未执行验证的草稿，不构成训练完成证据。

## DB 登记接口缺口

任务要求学生checkpoint、新管线和各臂登记为新的执行身份，并要求design→preflight→只跑缺失→journal→官方单进程merge→postflight。第七节把本目录外写入明确限定为两处：默认位置append_journal，以及`python -m shared_rollout_db.src.merger --experiment ...`。没有直接SQL登记的授权，也禁止替换模块全局变量。

静态核查发现：

- 官方merger固定构造Ingestor并调用ingest_jsonl，没有元数据登记参数、回调或插件入口。
- historical_ingest.py的`registered_exact_uid_v1`分支只验证**已经存在**的scenario/state/controller；缺少时返回ambiguous，不创建这些身份。
- 不含success/outcome/termination的纯登记journal会被跳过。
- round28的design.py确实能登记身份，但采用直接INSERT SQL，不属于本轮明确允许的两条写入路径。
- 历史隐式ingest能推断并创建其自身简化profile，但不能作为保留本轮planned exact UID和完整执行语义的通用登记接口。未以伪造outcome、改标签、借用旧身份或损失执行语义的方式引导登记，也没有生成无法合规精确入库的rollout。

具体源码路径、行号、hash和检查结果见DB_INTERFACE_AUDIT.json。这是本次停止原因；前两次接口/容差阻塞均已被新版文档解除。恢复所需的是明确可用的官方精确元数据登记接口，或在任务中明确允许design阶段只追加新身份元数据的写入范围；当前未自行扩大授权。

## 缺项与证据边界

没有pilot生成/去重、训练/checkpoint选择、学生离线测试、能力门、R5、冻结选择、rollout preflight、journal/merge/postflight、成功率/CI总表、转移表、继承表或推理耗时。没有共享DB写入，没有新的闭环标签。不能把本次阻塞归因为学生能力不足，也不能据此判断TS1–TS3或论文主张成立。

INTEGRITY_START.json、INTEGRITY_END.json、INTEGRITY_COMPARISON.json记录起止审计；SOURCE_HASHES.json与各imports文件记录已导入/静态核查源码hash；ARCHIVE_VERIFIED.json记录前两次产物及原预注册不变。结果状态见results.json，资源与已完成数据目录见RESOURCE_AUDIT.json。
