# Explicit R_risk：VI-augmented R_CERT 独立审计分支

此分支从 `c1-shared-baseline-v1` 恢复最后一次冻结的显式风险实现及其必要复现
依赖。恢复过程逐文件核对 `archive/legacy_pre_git_split/CLASSIFICATION.tsv`
中的 SHA256，没有改写公式、证书、witness、monitor 或控制器。

## 冻结活动实现

- `single_integrator/c1/risk/r_cert.py`：稳定 softplus、normalized soft-min 和
  exact outer max 聚合。
- `single_integrator/c1/risk/vi_r_cert.py`：direct certificate 与八个
  `Pi_U(x)(+-0.5 e_r)` witness 的冻结 VI augmentation；保留 303/2727 和
  42/378 的 certificate counts。
- `single_integrator/c1/rollout_vi_r_cert.py`：当前策略完整可微 rollout、历史
  guard snapshot、原 monitor/latch/termination 语义和无 post-terminal padding。
- `single_integrator/c1/risk/joint_frozen.py`：冻结的联合投影 forward 与 KKT VJP。
- `single_integrator/c1/action_probe.py`、`projection_directional.py`：既有审计辅助。
- `single_integrator/c1/train.py`、`train_deadlock_union.py`、
  `rollout_deadlock_union.py` 及旧 risk 支撑模块只因冻结复现脚本的 import closure
  而恢复；本分支没有运行这些训练入口。

## 独立入口与证据

- 聚合单元审计：`scripts/audit_c1_r_cert_aggregate.py`
- VI 实现检查：`scripts/check_c1_vi_r_cert.py`
- 固定 counterexamples：`scripts/check_c1_vi_r_cert_counterexamples.py`
- 已归档的 prediction/intervention、AD/FD 与 one-sided VJP 诊断脚本保留原定义。
- 小型既有报告 JSON/Markdown 显式纳入 Git；NPZ trace、参数 PKL 和图像仍由
  `.gitignore` 排除，并通过 `docs/assets_manifest.md` 寻址。

## 隔离约束

- 活动代码不 import `single_integrator.c1.direction_a` 或其 score estimator。
- 没有自动融合 score-function training、critic、planner、MPC 或 learned risk。
- 没有默认训练调用；本次只运行 CPU 单元测试和固定合成最小复现。
- `archive/legacy_pre_git_split/` 的其余候选保持非活动状态。

## 回归命令

```bash
JAX_PLATFORMS=cpu .venv-c1/bin/python -m unittest discover -s tests -p 'test_c1_*.py'
JAX_PLATFORMS=cpu .venv-c1/bin/python scripts/audit_c1_r_cert_aggregate.py \
  --out /tmp/c1_r_cert_aggregate_git_split.json
JAX_PLATFORMS=cpu .venv-c1/bin/python scripts/check_c1_vi_r_cert_counterexamples.py \
  --out /tmp/c1_vi_r_cert_counterexamples_git_split.json
JAX_PLATFORMS=cpu bash scripts/test.sh
```

测试记录和源码 SHA256 位于 `docs/test_results_r_risk.txt` 与
`docs/source_manifest_r_risk.sha256`。

## 本次分支恢复验证

- R_risk/VI unittest discovery：28/28 PASS。
- 共享 C1 regression：58/58 PASS。
- generic aggregation 最小复现：稳定区 AD/FD 相符；exact outer-max tie 仍按原实现
  报告为 nondifferentiable，没有平滑。
- 固定 counterexample 最小复现完成，包括 zero-control、jitter、moving outsider、
  retreat、H=850/1000 delay、symmetry 与 `U=[-2,2]x{0}` plateau。
- AST import 审计：34 个活动 R_risk/测试文件中 Direction A import 为 0。
- 测试均为 CPU 单元/固定合成复现；没有执行 GPU robot rollout、旧 intervention、
  训练或参数修改。既有 robot/AD-FD/VJP 报告只作为已完成资产纳入。
