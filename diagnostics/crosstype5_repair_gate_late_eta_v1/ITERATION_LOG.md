# Implementation iterations

## Reference intermediate-state restore and import order (before training)
Eight original requery prefix validation episodes were executed. Four requery lookups failed because the reference environment reset rejects tiny legal post-integration speed overshoot. Actual simulator states and source controllers were unchanged. Kept all old records, including numerical unknowns, under their original UID; no same-UID rerun.

The first32 early-validation workers stopped in reference construction before any physical episode because nominal imports altered source path priority and a legacy SCENARIOS relative root pointed to a nonexistent PoC manifest. No raw rollout or journal was produced for that stage.

Added independently versioned snapshot_conditioning.py with explicit authoritative ROOT artifact paths and public restore_augmented_state on a separate observation-only reference environment. Exact positions, velocities and goals are retained; nothing clipped. execution_v2 preloads reference dependencies. Version2 strategy hashes make corrected requests new identities. All original code/receipts remain untouched. Repeat8 prefix validations and32 early-validation cases under the corrected identity before training. Implementation-error physical rerun counter8 of8000; the failed32 workers executed0 physical episodes. The existing16 legacy given replays remain valid because given execution is byte-identical and its path is unchanged.

## Isolated reference process, before training
V2 reference restore succeeded, but one Transformer prefix differed from the historical trajectory at step0 by~2e-7. The frozen generator module sets CPU XLA_FLAGS before JAX backend creation; its eager import changed controller numerical execution, violating the prefix requirement. Seven other prefixes passed; do not relax tolerance. Isolated all conditioning/frozen-generator computation in a synchronous child reference process with its original XLA flags. Main controller process has historical unset XLA_FLAGS and imports no generator module. Child computation serializes with the caller under the same1-CPU Slurm allocation; no actual environment, controller state, physics or RNG is changed. V3 strategy hash includes helper/client and flags, and8 prefix+32 early-reuse checks must pass anew. Preserve all old UIDs and unknowns. V1 prefix8 +V2 prefix8 +V2 early32 account for48 implementation-error engineering executions of8000; the V1 early workers executed0.

## Engineering gate check-path correction
The initial versioned gate scripts retained the same check-metadata filename across versions, so later designs superseded the earlier engineering check-file link. Detected while trying to collect superseded V2 early checks; its32 actual raw records and immutable CASES/protocol/DB identities were intact and officially merged. Old V1/V2 gates remain failed/superseded and are not used as equivalence proof or training labels. Current V3 check files and their frozen hashes are preserved and verified before execution; no later design may overwrite them. New scientific phases will use unique stage-specific filenames. The failed old collection did not execute physics or change any label.


2026-10-19 23:31 SGT：模型菜单检查在无训练/无新物理执行时发现原预注册唯一候选数量算术笔误，原配方8+56×6=344，完整最多415，已以独立补充及收据记录、原封存内容未改。第二次模型检查10项全通过。GPU检查26498未执行即取消（资源排队），转为4GB CPU小检查；26499因上述断言退出、26500通过，二者均无rollout。原CPU每worker6GB的申请使128000MB节点只能同时20worker，待启动20–23号改4GB，23号曾为小检查临时hold，检查结束后已release；运行中的任务未中断、未重跑，helper+parent典型RSS约2.2GB。后续CPU默认4GB、GPU12GB，每次提交全用户24CPU/5GPU admission不变。

2026-10-19 23:43 SGT：新增门控/统计7项检查通过（26503，无训练或rollout）。23号pending分片仅为资源检查临时hold，完成后已release。补齐模型与统计顺序调度、覆盖与新state精确preflight/冻结/汇总/结束审计实现；coordinator只在等待训练阶段重启以载入当前代码，不影响任何物理worker。全仓Python语法检查通过。原34槽标签直接复用，不因编码变化重跑。

2026-10-19T16:00:03.558910+00:00：用户明确指定跨type5完成后进入跨type6。已完整阅读新prompt用于顺序任务确认，未启动跨type6、未创建其目录或提交其作业；前提为跨type5交付且全部Slurm结束。顺序任务记录于NEXT_TASK.json。新prompt内容未用于跨type5实验、候选或模型选择；结束审计仍对新增prompt只stat。

2026-10-19T16:03:34.494303+00:00：用户确认最近一次09:00（2026-10-20新加坡）释放额外6CPU/3GPU。新增纯调度策略RESOURCE_RELEASE_POLICY.json与watchdog；已缩短正在运行的临时CPU18–20号Slurm时限至09:00前，pending21–23号在启动时同样设置，未来GPU临时2–4号同样执行。后续临时worker提前120秒在episode/fit边界drain；09:00后准入18CPU/2GPU。只重启等待中的coordinator，不中断物理任务；已有训练执行与模型检查收据全通过。跨type6继续等跨type5完整交付，资源策略已进入顺序任务记录。

2026-10-20T03:11:25.258585+00:00：夜间26542已完成所有覆盖bootstrap并写入结果/子集/报告，但可选绘图依赖matplotlib缺失导致02:39调度停止，直到11:10状态核查才恢复，约8.5小时空转。此为调度实现缺陷，不归因于资源。26555用NumPy直方图与原生SVG完成同一仿真分布图，逐哈希确认已有bootstrap结果、子集、报告未改，不安装包、不训练、不重跑rollout。COVERAGE_STATISTICS.receipt.json已生成。恢复coordinator继续新state评估。

2026-10-20T03:14:22.124534+00:00：按用户纠正，非关键统计/绘图阶段改用auxiliary_cpu：异常明确记录NONCRITICAL_STAGE_GAPS.json后继续后续rollout；关键执行等价/源码保护不放宽。异常终止时立即更新STATUS/COORDINATOR_STATUS为blocked，避免展示过期running状态。已完成原生SVG XML解析核查；现有统计文件哈希不变。新评估作业26558，18CPU，17090条去重请求，已经运行。

2026-10-20T03:16:28.655183+00:00：按用户要求切换为主agent当前回合持续监视；停止仅在等待的自动coordinator，不中断26558的18个物理worker。后续合并/统计/审计/交付由主agent当场接续。

2026-10-20T03:18:34.427530+00:00：用户再次明确持久规则：不影响实验有效性的缺包必须记缺口后继续，不能阻断训练/rollout/评估。已写USER_EXECUTION_RULES.md、NEXT_TASK.json，并加入最终HANDOFF生成内容；跨轮次继承，无需再确认。

2026-10-20T03:23:34.079132+00:00：用户要求跨type5交付后同一回合正常汇报结果并直接开工跨type6，不再结束回合等待催促。已完整读取2026-10-20新版prompt（明确用户授权读取新增/修订提示词，仅作后续任务指引，不用于本轮科学选择），实际跨type6开工前将再次完整读取。新版数值/绘图环境、未舍入η来源门、隔离重查、训练分配和失败分类已记入NEXT_TASK.json。

2026-10-20T04:04:08.178753+00:00：结束审计26578因4GB内存不足OOM退出；无实验轨迹或数值结果重跑。先读Slurm日志确认为基础设施内存限制，按原输入与审计代码提高至20GB提交26579同一审计，保留原失败日志。
