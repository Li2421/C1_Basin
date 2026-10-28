## 最新用户追加优先级（覆盖下文旧的“不跑论文”说明）

用户要求：四列族跑完784案例后先跑之前指定的五篇论文六版本，检查场景是否太简单，再继续其他修正族。当前正在准备11个唯一固定参数profile，共4312个联合请求、4642个状态0单机诊断请求上界；pilot是联合队列子集，精确复用。没有训练，也不对论文跑新64组调参。原v3只覆盖规则除论文禁令和执行顺序被新指令覆盖，其余冻结、18CPU、90k总硬上限保持。见BASELINE_AFTER_FAMILY4.json、PAPER_HISTORY_CHECK.md、PAPER_FAIRNESS.json、baseline_screen.py。当前尚未实际构造论文controller或运行paper rollout；必须等family_4_complete.json出现。main.py已加hook且已切换到新协调进程31152加载；当前worker原计划不变，已完成数据由官方回收。四列阶段结束后会自动进入论文对照，不会提前启动。

# 当前执行状态与恢复注意事项

当前用户要求严格执行 `docs/prompt_scene_types_oracle_coverage_v3_20261028.md`，已完整阅读。此前 v2 和代表17实例筛查均已停止，旧目录只读。当前只做七个冻结修正族覆盖：全部49档、状态0–7、两个冻结nominal，共784案例；不做重新标定、选族、维度推荐、论文方法或学习训练。

工程已完成并实际运行：49档冻结；新增/缺失单机门2392局全部通过；50组路线引导等价通过；60局跨旧/新时限前缀逐位通过；200组原始/加速逐步认证通过；392物理初态全步随机数链2936088次比较通过；实际生产环境的49份墙体与SCENES逐字节一致，零额外env.step；原harness62文件逐字节等于v1。只用CPU，总配额18，通常1协调+17单核worker。硬上限90000新rollout尝试，budget.py加锁预留；数值unknown保存，不重跑。

当前流水线为Slurm `stb3_pipeline`（最近作业30806），日志 `main.log`。resource_watch.py 与 finish_when_ready.py 为独立的本轮监视/收尾进程。先跑既有四列族，再原三列族，再其余五族（顺序4,0,1,2,3,5,6）。已经完成只加安全投影参照，135/784成功。四列族进行中，最新覆盖应以COVERAGE.json为准；其他族尚未进入正式搜索。

注意：`main.log`里30719、30771的CANCELLED属于已记录的协调程序切换；独立worker继续运行，恢复后官方merger已回收全部记录，并未丢失/重复仿真。新调度每个案例按冻结顺序消化已经核验的缓存直到首个缺失候选，一次只提交这一个缺失候选；rNN是批次号，不是候选序号。真实candidate_index见frontier/labels/SEARCH_STATE。每个case独占文件锁，跨分片领取剩余请求。此前检查实际attempt UID重复数为0。

恢复：`bash diagnostics/scene_types_oracle_coverage_v3/resume.sh`，已有协调程序时不启动第二个。协调程序先等待旧worker全部退出，再单进程merge并读回，然后恢复已保存的原始pending frontier；不要重新生成同名批次计划。SEARCH_STATE保存completed_batches和pending_stage。若实际预算到90000，停止新物理运行；finalize.py补入最后一批已保存结果，并仅靠已有缓存继续依序消化可回答的标签；没有新增rollout或跳过缺失候选。

finalize.py会重建原始候选流、全部总体/逐类曲线、时间与配对统计、49行主表、17类型汇总、4/6倍时限对照、完整性记录。它只在数值任务结束、没有本轮Slurm工作时由finish_when_ready.py调用。完整结果需同时检查COMPLETE.json的complete、delivery_analysis_complete、integrity_passed。到预算上限未完成部分明确列GAPS，不伪装64次失败；有限搜索失败不是数学不可能。

故障检查：先看PIPELINE_ERROR.json、PROTECTION_VIOLATION.jsonl、最新Slurm日志；已保存unknown不要重跑。只处理本轮stb3_，不撤销其他前缀。主线worker基础设施仅补 genuinely_missing，最多三次重试。原数值代码保持冻结；不要随便修改core.py/runner.py/runtime.py/routes.py/guard.py/budget.py/recording_fix.py或共享数值源，它们参与controller_uid，改变会造成不必要的执行身份分叉。调度/分析源可在本目录修复，但要记录。

旧v1全部阶段语义兼容索引33848条，HISTORY_INDEX.json较大、忽略于Git。非精确跨时限复用逐条记REUSE_MANIFEST，不写假新UID精确标签。原始数据、轨迹、SQLite和journal都仍在本地；Git不代表大数据已远程备份。原仓库其他未提交变更已核对保留，PARALLEL_DIRECTORIES登记旧n20监视任务。
