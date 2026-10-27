# 第二阶段安全层修复交接

工作根目录 `/home/zhihan/research/Basin_C1`，本轮目录为本文件所在目录；全部新代码与输出仅在此，v2封版只读。流程环境 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B`。禁止git写操作、运行v2入口、修改已有环境、重定向journal或复制SQLite。

## 当前证据及入口

- 开工 `INTEGRITY_START.json`；历史四场景raw重算 `HISTORY_RAW_COUNTS.json`，来源哈希 `EVIDENCE_INPUTS.jsonl`。
- 新安全层 `stage2_safety.py`，原执行链副本 `dynamics_spatial_mask_20261022.py`，新执行链 `runtime_fix.py`；差异文件记录只换安全层及实际活跃行保存。controller、跟踪层、积分器、物理与事件不变。
- `UNIT_HEAD_ON.json`、`unit_snapshots/`、`STAGE2_FIX.receipt.json`：窄通道及拥挤随机有可行快照本步碰撞反例，单元门未通过；不得提交它们的研究rollout。`STAGE2_FIX.md`说明支持半空间证明所需预测安全前提，不能把任意速度快照说成真实研究轨迹可达。
- `REPLAY_GATE.json`：16条原执行链完整回放逐位通过。原复用的名义算法与原物理可用。
- 圆形初始试跑目录 `lit_circle_five_initial/`，精确360个请求；全量5760身份已注册，但没有通过试跑时不得运行其余5400。
- 唯一回退为预注册离散一步配对形式，gamma=.3；`CIRCLE_FALLBACK_HEAD_ON.json`、`CIRCLE_FALLBACK_UNIT.receipt.json`，严格检查解析区间中心距>=4.0001。回退试跑目录 `lit_circle_five_fallback/`，对应 CASES/PREFLIGHT/PILOT_CASES 在根目录。不得有第二次回退。

## 运行与核查

只使用本轮脚本。提交器 `dispatch_fix.py <stage> pilot|full` 在每次提交前查询全用户squeue，自动执行09:00资源下降规则。worker脚本 `worker_fix.sbatch` 实际入口 `worker_entry_corrected.py`，先核验冻结收据，再执行 `worker_corrected.py`；原worker在队列读取前的括号错误保留于旧文件及迭代日志。两次投影、同controller随机键与原到达停止保持。

worker以controller/state为组原子claim；只运行官方preflight缺失列表中尚无raw的请求。已保存unknown不重跑，raw不可覆盖。确定性类型的seed1只有精确seed0同执行轨迹存在时才保存显式alias，注明零新物理步。journal调用官方默认写入 `shared_rollout_db/journals/exp_baseline_literature_scenes_stage2_collision_fix_v1/`。worker不写SQLite。

`pilot_summary.py <stage>`核对360请求。所有作业结束后，先运行 `merge_fix.py`（本轮锁+全局merger进程检查，官方单进程合并）；成功过门后才能重新官方preflight并提交full。实例与seed不变，回退有独立controller执行身份。门口第二阶段23040与全部第一阶段在原v2身份上复用。

资源最新授权：无人使用时24CPU总份额，23个计算worker+1管理核；新加坡2026-10-23 09:00（UTC01:00）释放6，随后18总份额=17worker+1管理核。`resource_schedule.py`是独立守护，PID在 `CPU_RELEASE_GUARD_PID.json`；`CPU_RELEASE_SCHEDULE.json`为用户指令收据。只能操作本轮自有Slurm作业，不取消其他实验；未完成请求按相同身份恢复，已完成raw/journal保留。

## 结束条件

试跑及唯一回退仍非零碰撞时，写明确缺口，不生成全量假结果。`FINAL_SELECTION.json`记录接受的全量执行或缺口；`results_fix.py`只在全队列齐全时计算5000次配对bootstrap，否则主研究单元格N/A，试跑单独列出。门口表沿用v2原表，不重算同一证据。

所有实际作业结束后生成最终完整性审计、数据库postflight、DYNAMICS_STAGE、REPORT、GAPS、STATUS、ITERATION_LOG及完成标志。完整性有一项已知覆盖限制：`exact_projection_retry.py`间接导入未列入最初1194文件清单；补充哈希在 `SUPPLEMENTAL_IMPORT_AUDIT.json`，不能倒补成首次使用前的覆盖。应在REPORT第一段如实披露，不能声称严格全覆盖。

本轮结果全部为开发证据。timeout不等于deadlock；约束违反与速度违反保留，不伪装unknown。无视频要求。

## 强制停止覆盖上述运行说明

当前STOPPED_SOURCE_PROTECTION，**不要执行本文件前面列出的入口**。编译CBS库内置v2缓存根目录；复制后的库依然写v2。源码保护违规已触发整体停机。dispatch和当前worker入口已加停止哨兵，所有作业与资源guard已停。保留353条raw/trace及未合并journal；不清理、不回滚v2缓存。详情REPORT与READONLY_BOUNDARY_IMPACT。必须先处理边界事件和恢复条件，不能接手续跑。没有COMPLETE.json。

## 第一部分恢复完成（覆盖原停止状态）

用户接受限定缓存例外后已使用resume_backend和native_resume安全库恢复；原危险入口和库仅保留历史，不使用。第一部分结果、DB_POSTFLIGHT_RESUME、新结束审计和COMPLETE是当前入口。第二部分在COMPLETE之后才开始，目录为baseline_literature_scenes_supplement_eta_swap8_v1，独立审计和实验身份。
