# C1 Toy GiveWay 清理后仓库清单

> 生成于 2026-09-21；描述 `main` 首个基线 commit 前的活动工作树。


## 清理结果

- 活动树不再含 Direction A 或 explicit R_risk / VI-RCERT 模块；两条路线的原文件均在 `archive/legacy_pre_git_split/`，等待各自 branch 恢复。
- 归档保留原相对路径并在 `archive/legacy_pre_git_split/CLASSIFICATION.tsv` 记录原路径、分类和移动前 SHA256。
- 归档条目：legacy 274，Direction A 15，R_risk 42；完整性缺失 0。
- 已删除 509 个 cache 目录、240 个 `.log` 文件和 30 个结果内重复源码快照目录；它们在清理前 inventory 中对应 4,696 个 cache/bytecode 文件、240 个日志和 479 个 snapshot 文件。
- 两个本地虚拟环境保留在原路径供测试使用，但被 `.gitignore` 排除。
- checkpoint、dataset、论文、NPZ/PKL 和其余历史结果均保留原路径；详见 `docs/assets_manifest.md`。

## main 活动文件

| 路径 | 职责 |
|---|---|
| `.gitignore` | 仓库配置或共享文档入口 |
| `C1_DEADLOCK_MONITOR_SEMANTIC_AUDIT.md` | 仓库配置或共享文档入口 |
| `C1_GENERALITY_REQUIREMENTS.md` | 仓库配置或共享文档入口 |
| `README.md` | 仓库配置或共享文档入口 |
| `environment.yml` | 仓库配置或共享文档入口 |
| `flowbc/giveway_dataset.py` | Flow-BC 数据接口、归一化与训练复现 |
| `flowbc/giveway_flowbc_agent.py` | Flow-BC 数据接口、归一化与训练复现 |
| `flowbc/normalization.py` | Flow-BC 数据接口、归一化与训练复现 |
| `flowbc/train.py` | Flow-BC 数据接口、归一化与训练复现 |
| `requirements.txt` | 仓库配置或共享文档入口 |
| `requirements_gpu_working.txt` | 仓库配置或共享文档入口 |
| `scripts/c1_heldout_safety.py` | 共享基线、Safety 或测试入口 |
| `scripts/clean.sh` | 共享基线、Safety 或测试入口 |
| `scripts/evaluate.sh` | 共享基线、Safety 或测试入口 |
| `scripts/evaluate_cbf.sh` | 共享基线、Safety 或测试入口 |
| `scripts/generate_dataset.sh` | 共享基线、Safety 或测试入口 |
| `scripts/giveway_initial_state.py` | 共享基线、Safety 或测试入口 |
| `scripts/plan_baseline_400.py` | 共享基线、Safety 或测试入口 |
| `scripts/requirements-c1-plots.txt` | 共享基线、Safety 或测试入口 |
| `scripts/test.sh` | 共享基线、Safety 或测试入口 |
| `scripts/train.sh` | 共享基线、Safety 或测试入口 |
| `scripts/validate.sh` | 共享基线、Safety 或测试入口 |
| `single_integrator/CBF_PROTOCOL.md` | 共享环境/评估基础设施 |
| `single_integrator/__init__.py` | 共享环境/评估基础设施 |
| `single_integrator/c1/__init__.py` | 共享 C1 rollout/统计基础设施 |
| `single_integrator/c1/differentiable_rollout.py` | 共享 C1 rollout/统计基础设施 |
| `single_integrator/c1/paired_statistics.py` | 共享 C1 rollout/统计基础设施 |
| `single_integrator/c1/requirements.txt` | 共享 C1 rollout/统计基础设施 |
| `single_integrator/c1/scan_socp.py` | 安全投影/求解器基础设施 |
| `single_integrator/c1/socp.py` | 安全投影/求解器基础设施 |
| `single_integrator/c1/termination.py` | 环境、deadlock monitor、终止/结果语义 |
| `single_integrator/c1/training/__init__.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/calibration.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/dataset_starts.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/persistence.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/primal_dual.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/selection.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/step_control.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/c1/training/step_control_restart.py` | 共享原始—对偶/持久化工具，无路线入口 |
| `single_integrator/cbf.py` | 安全投影/求解器基础设施 |
| `single_integrator/compare.py` | 共享环境/评估基础设施 |
| `single_integrator/diagnostics/stalled_outcomes.py` | 共享环境/评估基础设施 |
| `single_integrator/diagnostics/uniform_state_data.py` | 共享环境/评估基础设施 |
| `single_integrator/environment.py` | 环境、deadlock monitor、终止/结果语义 |
| `single_integrator/evaluate.py` | 共享环境/评估基础设施 |
| `single_integrator/evaluate_cbf.py` | 共享环境/评估基础设施 |
| `single_integrator/expert.py` | 共享环境/评估基础设施 |
| `single_integrator/filters.py` | 共享环境/评估基础设施 |
| `single_integrator/generate.py` | 共享环境/评估基础设施 |
| `single_integrator/generate_scene.py` | 共享环境/评估基础设施 |
| `single_integrator/generate_scene_wide.py` | 共享环境/评估基础设施 |
| `single_integrator/outcomes.py` | 环境、deadlock monitor、终止/结果语义 |
| `single_integrator/tests/test_c1_calibration.py` | 共享回归测试 |
| `single_integrator/tests/test_c1_paired_statistics.py` | 共享回归测试 |
| `single_integrator/tests/test_c1_primal_dual.py` | 共享回归测试 |
| `single_integrator/tests/test_c1_socp.py` | 共享回归测试 |
| `single_integrator/tests/test_c1_step_control.py` | 共享回归测试 |
| `single_integrator/tests/test_c1_step_control_restart.py` | 共享回归测试 |
| `single_integrator/tests/test_cbf.py` | 共享回归测试 |
| `single_integrator/tests/test_cbf_solver_status.py` | 共享回归测试 |
| `single_integrator/tests/test_deadlock.py` | 共享回归测试 |
| `single_integrator/tests/test_environment.py` | 共享回归测试 |
| `single_integrator/tests/test_filter_contract.py` | 共享回归测试 |
| `single_integrator/tests/test_outcomes.py` | 共享回归测试 |
| `single_integrator/tests/test_short_scene.py` | 共享回归测试 |
| `single_integrator/tests/test_stalled_outcomes.py` | 共享回归测试 |
| `single_integrator/train.py` | 共享环境/评估基础设施 |
| `single_integrator/validate.py` | 共享环境/评估基础设施 |

## 归档结构

| 分类 | 条目数 | 活动路径状态 | 恢复目标 |
|---|---:|---|---|
| `LEGACY_OR_OBSOLETE` | 274 | 全部退出活动路径 | 只作历史参考 |
| `DIRECTION_A_ONLY` | 15 | main 中无活动文件 | `direction-a-stochastic` |
| `R_RISK_ONLY` | 42 | main 中无活动文件 | `direction-r-risk` |

## 交叉污染静态检查（main）

- 活动 `single_integrator/`、`scripts/` 和 `tests/` 中没有 `direction_a`、`R_CERT`、`vi_r_cert` 或 `single_integrator.c1.risk` import。
- `scripts/` 只保留共享 baseline/Safety、数据生成和测试入口；旧 C1 trainer/driver 均已归档。
- `README.md` 已改为分支职责说明，不再宣告某个旧风险为默认候选。

## 测试状态

- 命令：`JAX_PLATFORMS=cpu bash scripts/test.sh`
- 结果：58 tests，全部通过，耗时 2.230 s。
- 记录：`docs/test_results_main.txt`，SHA256
  `1037e4fd9e5e89388be68100f6bd71bb10641e8b93890316ef3da3ffa00ce519`。
- JAX 在登录节点探测 CUDA 插件时报告 `cuInit` 不可用并回退 CPU；测试未请求 GPU、
  未提交 Slurm 作业，也未运行 rollout 实验或训练。
