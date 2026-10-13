# Round 3 重算与恢复入口

仓库根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。先读 `REPORT.md` 的阶段门、`PREREGISTRATION*.md`、`P1_CACHE_PREFLIGHT.json`；不要在现有raw和DB已齐的情况下重新执行rollout。所有结果归属与SHA收据位于本目录。历史资产只读。

P0：`p0.py` 经 `p0.slurm` 的GPU shard拟合；`p0_analyze.py` 仅重算已有预测，零rollout。已完成的 `p0_fits/*/complete.json` 是跳过重训的入口。

P1：`p1_design.py` 只读历史模型/标签并冻结样本、checkpoint适配与profile；`p1_static.py` 用GPU shard做无step一致性审计；`p1_run.py prepare` 写精确请求；然后运行

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan \
  --manifest diagnostics/controller_q_transfer_round3_v1/P1_PLANNED_ROLLOUTS.json \
  --output diagnostics/controller_q_transfer_round3_v1/P1_CACHE_PREFLIGHT.json \
  > diagnostics/controller_q_transfer_round3_v1/P1_PREFLIGHT_STDOUT.json
```

实际提交的P1 array为 `p1_rollout.slurm`，只执行 `P1_MISSING_CASES.json` 中冻结的19,968个genuinely missing cases。因Slurm不允许原job延时，续跑数组 `p1_resume.slurm`（job 19293）依赖原数组 19266 的 `afterany`，已完成worker立即跳过，未完成者只补缺失raw；原因见 `P1_RESOURCE_ADJUSTMENT.md`。worker写 `p1_raw/` 及 `shared_rollout_db/journals/exp_controller_q_transfer_round3_p1_v1/`；实际 `p1_merge.slurm` job 19294 在续跑数组成功后单进程merge/postflight，`p1_analyze.slurm` job 19295 依赖其成功。原先依赖19266的pending merge/analysis job 19290/19292 已取消，避免并行merge。状态不明时先检查worker receipts、raw UID和SQLite，不重新执行unknown或覆盖raw。`p1_features.slurm` job 19291 只做静态查询，无rollout；其完成后 `p1_static_geometry_audit.py` 只读静态特征。完整postflight/主P1分析后可用 `p1_q_decomposition.py` 生成探索性的Q水平/η排序描述，不影响冻结gate。

P2原设计的精确成本/凸包可行性见 `P2_MINIMUM_BUDGET_AUDIT.json` 与 `P2_HULL_FEASIBILITY_AUDIT.json`。不得在不修改明确条件的情况下提交原P2。

若P1 gate失败，先执行 `p2prime_design.py`（冻结352个新state、controller profile、67584个请求与成本）及 `p2prime_run.py setup`，静态 `p2prime_static_audit.py` 通过后，按P1同样方式对 `P2PRIME_PLANNED_ROLLOUTS.json` 做官方preflight，输出 `P2PRIME_CACHE_PREFLIGHT.json`。只有结果无ambiguous、缺失数≤70000且满足预注册才提交 `p2prime_rollout.slurm`；其worker只读取冻结 `P2PRIME_MISSING_CASES.json`。`p2prime_merge.slurm` 单进程merge/postflight/label tensor；`p2prime_features.slurm` 静态查询；两者完成后 `p2prime_train.slurm` 用8个GPU shards拟合30个模型并评价。P3仅在主选择端点失败后运行 `p3_features.slurm` → `p3_train.slurm`，均不新增rollout。

P2′ 运行中为用满 CPU 上限 24，辅助数组 19345 在原22-worker数组 19320 的两个慢区间尾部各处理512个尚未开始的 UID；它调用同一 `p2prime_run.py worker`，参数为 `--index 29,35 --workers 132`，精确区间、理由及文件哈希见 `P2PRIME_TAIL_ASSIST_SUPPLEMENT.md` 和收据。随后三个快分片结束，又用辅助数组 19347 对主分片 3/4/5 的三个未开始尾段各处理256个 UID，参数 `--index 47,57,69 --workers 264`，见 `P2PRIME_SECOND_ASSIST_SUPPLEMENT.md` 和收据。单进程 merger 19342 的依赖已更新为同时等待 19320、19345、19347。若原worker再次读取辅助完成的raw，只验证精确case并跳过rollout；一致重复journal由merger去重。预定唯一UID和统计协议不变。

完整 raw 后，原 merger 19342 因漏登记独立 `experiment` 行而在首个 source_file 外键处停止，未入库任何 P2′ rollout；失败日志已备份为 `P2PRIME_MERGE_FAILED.log`。执行 `p2prime_register_experiment.py` 精确登记冻结协议后，replacement merger 19350 对同一 journal 成功导入67,584个唯一标签、去重1,792条一致重复journal，postflight valid66,747/unknown837/missing0/quarantine0。replacement GPU训练/分析19351完成30fit、`P2PRIME_RESULTS.json`。P2′ unknown 配对界只读修正见 `P2PRIME_PAIRWISE_UNKNOWN_FIX.md`；重算命令是同一 `p2prime_analyze.py`，初版结果完整保留。`p2prime_aliasing_diagnostic.py`、`p2prime_source_count_analysis.py` 均只读已有标签/预测。P3在P2′主门槛失败后由 `p3_features.slurm` job19352 和依赖的 `p3_train.slurm` job19353完成；无新rollout，`p3_analyze.py` 生成 `P3_RESULTS.json`。`finalize_results.py` 从阶段证据重建最终 `results.json`。

新的任务必须复核DB的实际profile、state/eta/seed精确身份及x64，再复用本目录。不要仅靠Git回滚任何未追踪的诊断资产；本任务不做清理、提交或视频验收（AGENTS.md纯Q豁免）。
