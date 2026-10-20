# Round30 第三次运行交接

根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。任务源为docs/prompt_round30_teacher_student_macflow_20261015.md，版本hash见PREREGISTRATION_SUPPLEMENT.receipt.json。

当前状态：**BLOCKED_EXACT_IDENTITY_REGISTRATION_WRITE_AUTHORIZATION**。本次新版噪声门已经通过1,008/1,008样本，前两次阻塞已解除。当前不是噪声或教师接口失败，而是新执行身份的合规DB登记入口缺失。

先读REPORT.md和DB_INTERFACE_AUDIT.json。官方merger的registered_exact_uid_v1要求state/controller先存在，不能通过纯元数据journal登记；本轮任务只允许默认append_journal和官方merger写入DB，不允许round28 design.py中的直接SQL登记。不要直接SQL写入、替换Ingestor全局、复制DB、伪造outcome或借旧身份来绕过。任务第零节要求缺接口停止，已停止。需要明确可用的官方登记接口或任务对设计阶段元数据追加的明确授权。

归档：attempt1_blocked/原样保留；第二次全部产物原样移入attempt2_gate_tolerance/。ARCHIVE_VERIFIED.json验证两次内容不变。根目录原PREREGISTRATION.md及收据与attempt2归档字节一致；补充预注册仅改噪声门定义，未读取学生结果。不得覆写任何既有收据或训练设置。

已完成：重新起始/结束完整性审计通过，PoC所有5个poc/*.py hash一致；14教师checkpoint与实际训练manifest hash匹配；噪声门1,008样本全部通过（trace误差最大7.21334229e-8、p99.9 6.82587024e-8 m/s）；数据差分速度扫描无reset越界；3个教师train/dev数据重建完成。

作业21340为14单核CPU数据预处理，已scancel；当前无本轮活动作业，0GPU使用，未提交train.sbatch。data/中的部分结果保留，完整者有READY.json，列表见RESOURCE_AUDIT.json。没有训练、checkpoint、学生结果或新rollout；没有共享DB写入/待合并journal。

新代码：core.py为本目录工具；noise_gate.py实施新版门；data.py通过实际RingExchangeEnv.reset和local_observation重建教师观测/学生观测，READY目录内有逐文件DATA_AUDIT.json；student.py/train.py是尚未执行验证的训练草稿；prepare.py为历史路径/hash审计；finalize_attempt3.py是只读结束审计。noise_gate.py防止覆盖门证据。不要重跑prepare.py去覆盖冻结文件。

若后续任务明确解除了DB登记阻塞：保留当前全部产物并重做开始审计；先完成、验证数据与训练实现，再以原冻结80,000步/batch256/验证8批/每500步验证及14个seed训练，不调参、不重训、不替换。原teacher raw的实际坐标依frame而定，global为world、local为径向/切向，执行必须走原to_world及再限速。学生执行仅用自身local_observation行、id和独立噪声。

剩余流程：16个新pilot从预注册202610300000–202610300999区间与DB/round28保留状态去重；能力门；通过者无标签R5与A3/A0/G_E冻结选择；同round28的32state、seed0–1学生七臂及教师缺失两臂；所有rollout preflight必须先于提交，unknown不重跑；默认journal、官方单进程merger/postflight；5,000次controller/state配对bootstrap、TS1–TS5、论文总表/转移/继承/耗时及最终报告。现有训练草稿未经运行测试，不要宣称已完成训练阶段。

最高约束持续适用：所有新内容仅本目录，已有源码/PoC/官方仓库/原环境只读，git只读；无monkey patch；JAX全部导入后恢复x64=False；共享任务合计最多18CPU shard/2GPU shard；专家不运行；无视频要求；timeout不是deadlock，停滞仅诊断。所有终点当前NOT_RUN。
