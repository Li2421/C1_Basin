# 重算和恢复

仓库根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。
先读 REPORT/HANDOFF、阶段决策门；下列命令不是要求重新生成已经存在的数据。所有标签来源见 DATA_INDEX 和 UID 索引。

## 分析（零 rollout）

```bash
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
$PY diagnostics/ring_state_selection_transfer_v1/verify_sources.py
$PY diagnostics/ring_state_selection_transfer_v1/analyze_learning.py N1
$PY diagnostics/ring_state_selection_transfer_v1/analyze_learning.py N3
$PY diagnostics/ring_state_selection_transfer_v1/summarize_aux.py
$PY diagnostics/ring_state_selection_transfer_v1/mechanism.py
$PY diagnostics/ring_state_selection_transfer_v1/c0_focus.py
$PY diagnostics/ring_state_selection_transfer_v1/censor_energy.py
$PY diagnostics/ring_state_selection_transfer_v1/rank_reversals.py
$PY -m diagnostics.ring_state_selection_transfer_v1.c1.analyze
$PY -m diagnostics.ring_state_selection_transfer_v1.c1.relationship
$PY diagnostics/ring_state_selection_transfer_v1/validate_delivery.py
$PY diagnostics/ring_state_selection_transfer_v1/assemble.py
```

C0_RESULTS/C0_CONTROLLER_CATALOG 是本轮新增C1标签前的冻结普查快照。`controller_audit.py` 和 `legacy_flow_audit.py`可在另建目录/改OUT后重新普查更新DB；不要无意覆盖本轮冻结基准。单独使用已有 `controller_panels/*.npz` 足以重算主要C0统计，不需要依赖未来变化的SQLite。

## 训练与静态查询

已有 fits/complete.json、冻结checkpoint和predictions优先读取。首次拟合命令如下，study.py发现complete receipt会跳过重训；目录已完整时不必再运行。

```bash
srun -p gpu --gres=shard:1 --cpus-per-task=2 --mem=8G --time=02:00:00 \
 env STUDY_PHASE=N1 JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/ring_state_selection_transfer_v1/study.py
# 同样设置 STUDY_PHASE=N3 完成辅助state分布诊断
# 同样GPU环境执行 effective_basis.py、c2_features.py、checkpoint_diagnostic.py
```

C2 fallback由`c2_fallback.slurm`在C1合并成功后按gate自动决定是否执行。模型输入、source-only划分、unknown onehot与mismatch规则在C2_FALLBACK_PREREGISTRATION.md；不得因为低精度结果不理想而更改划分/超参。重新推理时读取各c2_fits/模型与normalization；不要默默重新选择checkpoint。

## C1生成/恢复（通常无需再次运行）

本次已完成的请求全部在DB里。`c1/run.py prepare`只用于首次设计，**不要覆盖完成的protocol/cases**。若确实在中断后恢复，先核对code/protocol hash和已有raw，再重做cache preflight：

```bash
$PY -m shared_rollout_db.plan \
 --manifest diagnostics/ring_state_selection_transfer_v1/c1/planned_rollouts.json \
 --output /tmp/ring_c1_resume_preflight.json > /tmp/ring_c1_resume_preflight.log
# 只允许原missing清单中尚无raw/DB标签的case进入worker；已有unknown保持未知，不能重试覆盖。
# 原作业脚本 c1/rollout.slurm，array0–87、最多22个1CPU worker。
# 单一merger c1/merge.slurm，在所有worker成功后才运行。
```

首次生成实际调用为 `python -m diagnostics.ring_state_selection_transfer_v1.c1.run prepare` → preflight → static audit → sbatch rollout → afterok单merger → analyze。sharedDB已存在的κ端点标签沿用原UID；两个halfκ新增UID，不能把源Flow FF标签错用为同endpoint算子标签。

## 绘图

```bash
OPENBLAS_NUM_THREADS=1 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python \
 diagnostics/ring_state_selection_transfer_v1/figures.py
```

保存PNG/PDF，报告无需额外网络/可视化服务。最终hash见ARTIFACT_MANIFEST，注册时间/hash见各receipt。原资产不可清理或覆盖。
