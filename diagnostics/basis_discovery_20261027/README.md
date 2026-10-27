# Unified Correction Dictionary Discovery

先读 [report.md](report.md)。本轮15类候选统一实现于
[shared_control/basis_dictionary.py](../../shared_control/basis_dictionary.py)，
由 [runner.py](runner.py) 接入原生产两次安全投影之间；原生产文件和上轮结果未改。

| 文件 | 用途 |
| --- | --- |
| PROTOCOL.json | 历史证据、14困难+8成功对照、冻结nominal、独立拆分 |
| checks.py / checks.json | 真实状态零扩展、原投影、候选机制检查 |
| static_audit.py / static_states.json | 全部15列norm/rank/夹角/投影响应 |
| mechanism_validation.py | C2中心差分、T5阈值和SVD条件数检查 |
| screen.py | 所有单列、所有105对、自动扩展、删列、最小支持集 |
| matched_controls.py | P/P2匹配RMS和R/R2/R3替换对照 |
| compression.py | 自动低一维子族、只补未覆盖案例 |
| holdout.py / confirmatory.py | 开发参数迁移、独立测试及全新确认集 |
| restarts.py / pure_family.py | 真实失败状态恢复、纯低维族全队列检查 |
| final_baselines.py | 历史定制B3加强基线、恢复B3@48 |
| candidates.jsonl / traces/ | 全部逐候选原始记录和NPZ；本地大数据 |
| *_planned.json / *_preflight.json / *_execution.json | 精确预算与原缓存流程 |
| analyze.py / plot.py / write_report.py | 只读结果生成统计、图和报告 |
| verify.py / verification.json | 完整UID、缓存、源码哈希、轨迹存在性核验 |

恢复已完成研究：

```bash
bash diagnostics/basis_discovery_20261027/resume.sh
```

若不在Slurm分配中，脚本会提交CPU作业，默认24核、32GB内存，无GPU；
可用 `C1_CPUS=8 bash .../resume.sh` 改核数。已完成阶段按精确UID和标记复用，
不重复训练或生成轨迹。原数据库、原生产快照及旧结果是依赖，不删除它们。

仿真解释器：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`；
绘图解释器：同项目 `.venv-c1-plots/bin/python`。资源策略从18核按用户要求改为
全部空闲CPU；实际本机24核。所有worker固定单CPU，BLAS/OpenMP为1线程。

仅重算汇总，不需重新提交仿真：

```bash
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" diagnostics/basis_discovery_20261027/analyze.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python diagnostics/basis_discovery_20261027/plot.py
"$PY" diagnostics/basis_discovery_20261027/verify.py
"$PY" diagnostics/basis_discovery_20261027/write_report.py
```

`MINIMAL_FAMILIES.json` 的最小值只针对实际成功witness集合；不是连续参数空间
不可能性的证明。所有Oracle和固定策略均分开报告。原墙体几何/CBF间距不一致
保持原样，坏候选仍计碰撞失败。未训练Generator/Critic，未使用GPU。

Git只提交代码、文档、冻结设计及小型结果；JSONL/NPZ/SQLite等大证据仍在本地。
提交代码不意味着原始数据已异地备份。

追加baseline实际完成536条新仿真，见 [baseline_report.md](baseline_report.md) 与 [论文身份审计](literature_baselines.md)。`baselines.py` 使用独立数据目录和缓存namespace；其源码和主runner均保持各自实验时hash。

完整恢复脚本已实测，冻结设计和两份原始JSONL哈希均不变，新增rollout为0；见 [resume_verification.json](resume_verification.json)。
