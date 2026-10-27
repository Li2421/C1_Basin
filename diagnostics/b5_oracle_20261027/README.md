# B5 实验交付

先读 [report.md](report.md)。新增代码只位于本目录及
[shared_control/correction_b5.py](../../shared_control/correction_b5.py)，不修改旧生产文件。

| 材料 | 路径 |
| --- | --- |
| 真实接口 / 基础族 | runtime.py / ../../shared_control/correction_b5.py |
| A–E与扩展 / 独立测试 | oracle.py, expand.py, supplement.py / independent.py |
| 原四场景 TT/FF 与历史强基线 | legacy.py, legacy_prior.py |
| 机制与32顶点检查 | mechanisms.py, legacy_mechanisms.py / mechanisms/ |
| 原B3回归 | checks.py / regression.json |
| 历史证据与冻结设计 | evidence_audit.json, INDEPENDENT_FROZEN.json, LEGACY_FROZEN.json |
| 逐候选原始结果 | candidates.jsonl, legacy_candidates.jsonl, legacy_prior_candidates.jsonl |
| 规范化全量导出 | candidates_verified.jsonl（保留原数据，补元数据） |
| 轨迹 | traces/, legacy_traces/（NPZ，含原约束/随机流证据） |
| 统计、预算、图 | summary.json, independent_metrics.csv, budget_curves.csv, results.pdf |
| 安全几何发现 | safety_geometry_audit.json |
| 缓存与身份核验 | verification.json, cache_registration_repair.json |
| 可恢复入口 | resume.sh |

恢复已完成研究：

```bash
bash diagnostics/b5_oracle_20261027/resume.sh
```

该入口跳过完成标记，并对旧场景按已记录 UID 去重；不删除证据，不重复训练。
仿真 Python 为 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`，
绘图 Python 为该项目 `.venv-c1-plots/bin/python`。需要当前生产快照和 sibling
`Basin_C1_flow_field_poc_20261004` 原 checkpoint；这些大文件没有复制进本目录。
CPU affinity 为0–17，BLAS/OpenMP为1线程；无需GPU。

只重算汇总：按 resume.sh 末尾顺序运行 analyze.py、plot.py、verify.py、write_report.py。
再次运行静态机制诊断前先保留已有选中状态：mechanisms.py默认从当时完整候选集
重新挑选状态；它不会恢复“初始pilot时”的选择。既有 mechanisms/*.json 已保存精确
source_trace、step、source_candidate。可用下式仅重算最后残余案例（已有结果不必重跑）：

```bash
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OPENBLAS_NUM_THREADS=1 taskset -c 0-17 "$PY" - <<'PY'
import json,sys
from pathlib import Path
p=Path('diagnostics/b5_oracle_20261027')
sys.path.insert(0,str(p.resolve()))
from mechanisms import analyze
old=json.load(open(p/'mechanisms/hs_obst_n08_R140_r045__centralized_sampling__residual_state12.json'))
r=next(r for r in map(json.loads,open(p/'candidates.jsonl'))if r.get('trace')==old['source_trace'])
r['diagnostic_tag']='__residual_state12'
print(analyze(r))
PY
```

`*_planned_rollouts.json` 是每阶段精确物理与控制配置，`*_cache_preflight.json`
和原journals保留历史登记问题的痕迹；最终已由 verify.py 修复元数据并核验入库。
大规模 JSONL、NPZ、SQLite 是本地忽略的研究证据，Git提交不等于数据备份。
参考ZIP缺失，未冒称通过其 run_checks.py。
