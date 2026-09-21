# C1 Toy GiveWay 清理前仓库清单

> 本文件由 2026-09-21 的只读审计生成。在本文件生成前，没有删除、移动或改写仓库源码。


## 审计边界

- 审计时间：2026-09-21（Asia/Singapore）。
- 审计时 Slurm 队列为空；未启动任何实验或训练。
- `.git/` 存在但为空，`git status` 报告“not a git repository”。
- 清理前仓库总量约 11 GiB；其中两个本地虚拟环境约占 6.5 GiB，`results/` 约占 4.0 GiB。
- 下表逐项列出可执行源码、测试、脚本、根级文档和配置。大型结果目录按可审计 bundle 聚合；其中每个 bundle 的文件数和字节数列于后表，SHA256 将写入 `docs/assets_manifest.md`。

## 分类与处置原则

| 分类 | 本轮含义 | 计划 |
|---|---|---|
| `COMMON_CORE` | 两条路线共享的环境、Flow-BC、投影、monitor、通用模型/训练工具和共享测试 | 留在 `main` 的活动路径 |
| `DIRECTION_A_ONLY` | joint Bernoulli + 4D Gaussian residual、score/J_def、相应诊断和测试 | main 中归档；仅在 `direction-a-stochastic` 恢复活动路径 |
| `R_RISK_ONLY` | 最后冻结的 R_CERT/VI、必要复现依赖、诊断和测试 | main 中归档；仅在 `direction-r-risk` 恢复活动路径 |
| `LEGACY_OR_OBSOLETE` | 早期候选、旧训练/评估入口、已结束的实验 driver 和旧说明 | 移入 `archive/legacy_pre_git_split/`，不删除 |
| `GENERATED_OR_DISPOSABLE` | cache、日志、复制源码快照、本地环境/编辑器状态 | 删除明确可再生项；本地环境可保留但不提交 |
| `LARGE_ASSETS` | checkpoint、dataset、论文和历史 rollout/result bundles | 不删除，保留原路径并从 Git 排除；记录 hash |

## 顶层目录概览

| 路径 | 文件数 | 大小 | 分类 | 计划 |
|---|---:|---:|---|---|
| `baseline_309_314/` | 26 | 4.52 MiB | `LARGE_ASSETS` | 原路径保留；二进制/大结果不提交，另入 assets manifest |
| `datasets/` | 954 | 57.51 MiB | `LARGE_ASSETS` | 原路径保留；二进制/大结果不提交，另入 assets manifest |
| `flowbc/` | 4 | 23.49 KiB | `COMMON_CORE` | 保留在 main |
| `papers/` | 2 | 6.88 MiB | `LARGE_ASSETS` | 原路径保留；二进制/大结果不提交，另入 assets manifest |
| `results/` | 73,781 | 4.02 GiB | `LARGE_ASSETS` | 原路径保留；二进制/大结果不提交，另入 assets manifest |
| `scripts/` | 192 | 863.93 KiB | `MIXED` | 见逐文件表；按路线归档/恢复 |
| `single_integrator/` | 143 | 673.20 KiB | `MIXED` | 见逐文件表；按路线归档/恢复 |
| `tests/` | 14 | 68.53 KiB | `MIXED` | 见逐文件表；按路线归档/恢复 |
| `.venv-c1/` | 18,263 | 5.86 GiB | `GENERATED_OR_DISPOSABLE` | 本地环境/工具状态；不纳入 Git，空目录可删除 |
| `.venv-c1-plots/` | 2,690 | 155.34 MiB | `GENERATED_OR_DISPOSABLE` | 本地环境/工具状态；不纳入 Git，空目录可删除 |

## 活动源码、脚本、测试、文档和配置逐项清单

“被引用”基于清理前活动源码的静态 import 和脚本文件名引用；CLI 单独手工运行而没有静态调用的文件会显示“否”，这不表示可以删除。

| 路径 | 分类 | 被其他活动代码引用 | 计划移动/删除 | 理由 |
|---|---|---|---|---|
| `.agents/` | `GENERATED_OR_DISPOSABLE` | 否（独立入口或文档） | 本地环境/工具状态；不纳入 Git，空目录可删除 | 本地状态或可再生内容，不应成为研究源码版本。 |
| `.codex/config.toml` | `GENERATED_OR_DISPOSABLE` | 否（独立入口或文档） | 本地环境/工具状态；不纳入 Git，空目录可删除 | 本地状态或可再生内容，不应成为研究源码版本。 |
| `.git/` | `GENERATED_OR_DISPOSABLE` | 否（独立入口或文档） | 本地环境/工具状态；不纳入 Git，空目录可删除 | 本地状态或可再生内容，不应成为研究源码版本。 |
| `.gitignore` | `COMMON_CORE` | 否（独立入口或文档） | STEP 2 重写后保留在 main | 两条路线共享，或为审计/构建配置。 |
| `.venv-c1-plots/` | `GENERATED_OR_DISPOSABLE` | 否（独立入口或文档） | 本地环境/工具状态；不纳入 Git，空目录可删除 | 本地状态或可再生内容，不应成为研究源码版本。 |
| `.venv-c1/` | `GENERATED_OR_DISPOSABLE` | 否（独立入口或文档） | 本地环境/工具状态；不纳入 Git，空目录可删除 | 本地状态或可再生内容，不应成为研究源码版本。 |
| `ARCHIVE_LOCATION.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive，保留历史说明 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_ARITHMETIC_HARMONIC_CANDIDATE.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_ARITHMETIC_HARMONIC_PROBE_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_BELLMAN_RISK_BOUNDARY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_BOUNDARY_IMPORTANCE_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_COMPLETION_CERTIFICATE_THEORY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_CONDITIONAL_BOUNDARY_CANDIDATE.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_CONDITIONAL_RISK_THEORY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_CROSS_NOISE_AUDIT.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_CURRENT_SPEC.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_DEADLOCK_MONITOR_SEMANTIC_AUDIT.md` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `C1_DEADLOCK_PRIMARY_THEORY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_DEADLOCK_UNION_THEORY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_DIFFICULTY_DISTRIBUTION.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_DOUBLE_PROJECTION_GRADIENT_AUDIT.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_EVENT_HISTORY_CANDIDATE.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_EVENT_HISTORY_PROBE_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_EVENT_PROBABILITY_DIFFERENCE.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_EXACT_MARGIN_THEORY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_GAUSSIAN_TRANSPORT_CANDIDATE.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_GENERALITY_REQUIREMENTS.md` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `C1_GENERATIVE_RISK_RESEARCH.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_GRADIENT_DIAGNOSTIC_STATUS.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_GRADIENT_SEMANTICS_AUDIT.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_HANDOFF_20260918.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_MOMENT_PROBABILITY_CANDIDATE.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_MOMENT_PROBE_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_OPTIMIZER_DIAGNOSTIC_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_ORDERED_GUIDANCE_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_PROBABILITY_GRADIENT_REQUIREMENTS.md` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `C1_PROGRESS_DEBT_THEORY.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_RISK_CANDIDATE_STATUS.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_RISK_TWO_TEST_PROTOCOL.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_R_CERT_PROJECTION_OPTIMALITY_REVIEW.md` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `C1_R_CERT_VALIDATION.md` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `C1_SPEC.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_TERMINAL_PROGRESS_GUARD.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_V3_SPEC.md` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `C1_VI_R_CERT_ADFD_LOCALIZATION.md` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `C1_VI_R_CERT_VALIDATION.md` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `C1_VI_R_CERT_VJP_ONE_SIDED.md` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `README.md` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `codex.sh` | `GENERATED_OR_DISPOSABLE` | 否（独立入口或文档） | 本地环境/工具状态；不纳入 Git，空目录可删除 | 本地状态或可再生内容，不应成为研究源码版本。 |
| `docs/repo_inventory_before_cleanup.md` | `COMMON_CORE` | 否（独立入口或文档） | 审计文档保留在 main | 两条路线共享，或为审计/构建配置。 |
| `environment.yml` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `flowbc/giveway_dataset.py` | `COMMON_CORE` | 是（3） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `flowbc/giveway_flowbc_agent.py` | `COMMON_CORE` | 是（14） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `flowbc/normalization.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `flowbc/train.py` | `COMMON_CORE` | 是（5） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `requirements.txt` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `requirements_gpu_working.txt` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/analyze_c1_arithmetic_harmonic.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_boundary_importance.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_conditional_risk.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_early_gradient.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_event_history.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_g_sensitivity.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_geometry_ranking.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_gradient_closed_loop.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_guard_factorial.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_joint_closed_loop.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_optimizer_diagnostic.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_risk_two_tests.py` | `LEGACY_OR_OBSOLETE` | 是（3） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/analyze_c1_vi_r_cert.py` | `R_RISK_ONLY` | 是（2） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/analyze_giveway_residual_effect.py` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `scripts/audit_c1_arithmetic_harmonic.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_bellman_certificate.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_candidate_boundary.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_candidate_coverage.py` | `LEGACY_OR_OBSOLETE` | 是（3） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_clean_geometry.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_completed_waiting_risk.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_conditional_transversality.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_cross_noise.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_current_gradient_entry.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_deadlock_signal.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_deadlock_union.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_direction_a_contract.py` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `scripts/audit_c1_directional_derivative.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_double_projection_boundary.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_episode_risk.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_equivalent_geometry_solver.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_expanded_safety.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_factorial_parameters.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_final_evidence.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_fixed_horizon_waiting.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_geometry_failures.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_gradient_paths.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_joint_bptt.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_joint_closed_loop.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_joint_witness_checks.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_joint_witness_risk.py` | `R_RISK_ONLY` | 是（12） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/audit_c1_large_local_forecast.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_large_risk_dataset.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_margin_risk.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_matched_baseline_failure.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_multistep_objectives.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_prediction_mismatch.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_pretraining_ablation.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_probability_gradient_limits.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_progress_conditioned_wait.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_progress_debt.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_progress_guard_traces.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_projection_derivative.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_r_cert_aggregate.py` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/audit_c1_risk.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_risk_ordering.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_soft_risk.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_timeout_conversions.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_two_horizon_wait.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_unseen_exceptions.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_v3_support.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_c1_wait_horizon_sweep.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/audit_cbf_status_fix.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/benchmark_c1_episode_pool.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/build_c1_dataset_sets.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/build_c1_v1_sets.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/build_c1_v2_sets.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/c1_heldout_safety.py` | `LEGACY_OR_OBSOLETE` | 是（12） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/calibrate_c1_task_difficulty.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_action_probe.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_cached_directional.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_directional_closed_loop.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_early_forward.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_event_history.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_noise_contract.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_prefix_physics.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_state_vjp.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_v3_deadlock.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_v3_rollout.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/check_c1_vi_r_cert.py` | `R_RISK_ONLY` | 是（2） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/check_c1_vi_r_cert_counterexamples.py` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/clean.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/describe_c1_conditioned_wait.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/diagnose_c1_momentum.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/diagnose_c1_regression.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/diagnose_c1_vi_adfd_layers.py` | `R_RISK_ONLY` | 是（1） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/diagnose_c1_vi_adfd_localize.py` | `R_RISK_ONLY` | 是（1） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/diagnose_c1_vi_vjp_one_sided.py` | `R_RISK_ONLY` | 是（1） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/evaluate.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/evaluate_c1_four_objectives.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/evaluate_c1_frozen_unseen.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/evaluate_c1_geometry.sh` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/evaluate_c1_joint_pilot.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/evaluate_c1_scene_family.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/evaluate_cbf.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/experiment_c1_soft_activity.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/finalize_c1_after_queue.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/focused_c1_p_vs_pg.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/generate_dataset.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/giveway_initial_state.py` | `COMMON_CORE` | 是（3） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/launch_c1_four_objectives.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/launch_c1_frozen_unseen.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/launch_c1_mismatch.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/localize_c1_directional_error.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/localize_c1_nominal_error.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/monitor_c1_exact_pipeline.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/monitor_c1_four_objectives.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/monitor_c1_gradient_job.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/monitor_c1_risk_two_tests.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/monitor_c1_training_job.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/pilot_c1_joint_matched.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/pilot_giveway_residual_effect.py` | `DIRECTION_A_ONLY` | 是（2） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `scripts/plan_baseline_400.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plan_c1_309_314.sh` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plan_c1_400.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_candidate_coverage.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_conditioned_wait.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_diagnosis.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_geometry_ranking.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_gradient_paths.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_guarded_training.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_joint_closed_loop.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/plot_c1_risk_correspondence.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/precompute_c1_seed2.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/prepare_c1_independent_sets.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/prepare_c1_scene_family.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_arithmetic_harmonic.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_boundary_importance.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_certificate_descent.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_conditional_event_sections.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_conditional_risk_cpu.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_deadlock_primary.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_deadlock_union.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_early_gradient.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_event_history.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_event_probability_difference.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_existing_policy_risk.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_gradient_closed_loop.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_guard_factorial.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_matched_baseline.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_moment_probability.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_risk_two_tests.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_v3_descent.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_v3_direction.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/probe_c1_vi_r_cert.py` | `R_RISK_ONLY` | 是（1） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/report_c1_paired_seed.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/report_c1_soft_activity.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/reproduce_c1_cpu_qp_failure.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_arithmetic_harmonic.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_boundary_importance.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_conditional_gpu.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_early_gradient.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_event_history.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_exact_margin.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_experiment_queue.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_gradient_closed_loop.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_guard_factorial.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_moment_probability.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_optimizer_diagnostic.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_ordered_parallel.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_ordered_probe.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_risk_two_tests.sbatch` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_scene_comparison.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_scene_stage_i.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_scene_stage_i_wide.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/run_c1_vi_adfd_layers.sbatch` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/run_c1_vi_adfd_localize.sbatch` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/run_c1_vi_r_cert_check.sbatch` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/run_c1_vi_r_cert_probe.sbatch` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/run_c1_vi_vjp_one_sided.sbatch` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `scripts/run_giveway_residual_effect_pilot.sbatch` | `DIRECTION_A_ONLY` | 是（2） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `scripts/score_c1_risk_outcomes.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/smoke_c1_v0.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/status_c1_goal.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_c1_deadlock_paired.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_c1_four_objectives.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_c1_frozen_unseen.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_c1_joint_pilot.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_c1_mismatch.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_c1_risk_outcomes.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/summarize_focused_c1_p_vs_pg.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/test.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/test_c1_joint_jax.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/test_c1_reliable_geometry.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/train.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/train_c1_309_314.sh` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/train_c1_four_objectives.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/validate.sh` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `scripts/validate_c1_recovered_geometry.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/verify_c1_candidate_coverage.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/verify_c1_clean_geometry.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `scripts/verify_c1_frozen_unseen.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/__init__.py` | `COMMON_CORE` | 是（907） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/__init__.py` | `COMMON_CORE` | 是（564） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/acceptance.py` | `LEGACY_OR_OBSOLETE` | 是（9） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/action_probe.py` | `R_RISK_ONLY` | 是（16） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/differentiable_rollout.py` | `COMMON_CORE` | 是（61） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/direction_a/__init__.py` | `DIRECTION_A_ONLY` | 是（17） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `single_integrator/c1/direction_a/estimators.py` | `DIRECTION_A_ONLY` | 是（3） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `single_integrator/c1/direction_a/policy.py` | `DIRECTION_A_ONLY` | 是（6） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `single_integrator/c1/direction_a/randomness.py` | `DIRECTION_A_ONLY` | 是（6） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `single_integrator/c1/direction_a/rollout.py` | `DIRECTION_A_ONLY` | 是（21） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `single_integrator/c1/direction_a/statistics.py` | `DIRECTION_A_ONLY` | 是（5） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `single_integrator/c1/early_intervention.py` | `LEGACY_OR_OBSOLETE` | 是（9） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/episode_pool.py` | `LEGACY_OR_OBSOLETE` | 是（5） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/episode_rollout.py` | `R_RISK_ONLY` | 是（2） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/evaluate.py` | `LEGACY_OR_OBSOLETE` | 是（3） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/evaluate_certificate.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/evaluate_completion.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/evaluate_deadlock_primary.py` | `LEGACY_OR_OBSOLETE` | 是（3） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/evaluate_deadlock_union.py` | `LEGACY_OR_OBSOLETE` | 是（31） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/evaluate_scene_deadlock_union.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/evaluate_v3.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/frozen_scene_baseline.py` | `LEGACY_OR_OBSOLETE` | 是（5） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/joint_frozen_rollout.py` | `LEGACY_OR_OBSOLETE` | 是（7） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/models/__init__.py` | `COMMON_CORE` | 是（23） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/models/residual.py` | `COMMON_CORE` | 是（9） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/paired_statistics.py` | `COMMON_CORE` | 是（6） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/projection_directional.py` | `R_RISK_ONLY` | 是（8） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/__init__.py` | `R_RISK_ONLY` | 是（129） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/arithmetic_harmonic.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/boundary_importance.py` | `LEGACY_OR_OBSOLETE` | 是（4） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/completion_certificate.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/conditional_gaussian.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/deadlock_geometry.py` | `R_RISK_ONLY` | 是（4） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/deadlock_primary.py` | `R_RISK_ONLY` | 是（13） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/deadlock_union.py` | `R_RISK_ONLY` | 是（29） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/diagnostics.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/evaluation.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/event_history.py` | `LEGACY_OR_OBSOLETE` | 是（9） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/exact_margin.py` | `LEGACY_OR_OBSOLETE` | 是（15） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/gaussian_transport.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/joint_frozen.py` | `R_RISK_ONLY` | 是（32） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/moment_probability.py` | `LEGACY_OR_OBSOLETE` | 是（4） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/ordered_guidance.py` | `LEGACY_OR_OBSOLETE` | 是（20） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/progress_debt.py` | `LEGACY_OR_OBSOLETE` | 是（7） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/r_cert.py` | `R_RISK_ONLY` | 是（9） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/reachability_margin.py` | `LEGACY_OR_OBSOLETE` | 是（2） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/reachability_union.py` | `LEGACY_OR_OBSOLETE` | 是（3） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/risk_function.py` | `R_RISK_ONLY` | 是（12） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/risk_v1.py` | `R_RISK_ONLY` | 是（7） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/risk_v2.py` | `R_RISK_ONLY` | 是（5） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/risk_v3.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/soft_activity.py` | `R_RISK_ONLY` | 是（8） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/risk/temporal_certificate.py` | `LEGACY_OR_OBSOLETE` | 是（8） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/terminal_progress_guard.py` | `LEGACY_OR_OBSOLETE` | 是（5） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/risk/vi_r_cert.py` | `R_RISK_ONLY` | 是（14） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/rollout.py` | `LEGACY_OR_OBSOLETE` | 是（17） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_certificate.py` | `LEGACY_OR_OBSOLETE` | 是（3） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_completion.py` | `LEGACY_OR_OBSOLETE` | 是（5） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_deadlock_primary.py` | `LEGACY_OR_OBSOLETE` | 是（4） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_deadlock_union.py` | `R_RISK_ONLY` | 是（12） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/rollout_early_diagnostic.py` | `LEGACY_OR_OBSOLETE` | 是（13） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_event_history.py` | `LEGACY_OR_OBSOLETE` | 是（8） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_v3.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/rollout_vi_r_cert.py` | `R_RISK_ONLY` | 是（11） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/scan_socp.py` | `COMMON_CORE` | 是（1） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/socp.py` | `COMMON_CORE` | 是（13） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/termination.py` | `COMMON_CORE` | 是（25） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/train.py` | `R_RISK_ONLY` | 是（47） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/train_certificate.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_completion.py` | `LEGACY_OR_OBSOLETE` | 是（4） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_completion_restart.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_deadlock_primary.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_deadlock_union.py` | `R_RISK_ONLY` | 是（36） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `single_integrator/c1/train_exact_margin.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_optimizer_diagnostic.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_ordered_parallel.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_scene_deadlock_union.py` | `LEGACY_OR_OBSOLETE` | 是（1） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/train_v3.py` | `LEGACY_OR_OBSOLETE` | 是（6） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/c1/training/__init__.py` | `COMMON_CORE` | 是（129） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/calibration.py` | `COMMON_CORE` | 是（2） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/dataset_starts.py` | `COMMON_CORE` | 是（5） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/persistence.py` | `COMMON_CORE` | 是（64） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/primal_dual.py` | `COMMON_CORE` | 是（26） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/selection.py` | `COMMON_CORE` | 是（19） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/step_control.py` | `COMMON_CORE` | 是（22） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/c1/training/step_control_restart.py` | `COMMON_CORE` | 是（15） | 保留在 main（通用模型/训练基础设施，无默认路线入口） | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/cbf.py` | `COMMON_CORE` | 是（111） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/compare.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/diagnostics/stalled_outcomes.py` | `COMMON_CORE` | 是（38） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/diagnostics/uniform_state_data.py` | `COMMON_CORE` | 是（2） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/environment.py` | `COMMON_CORE` | 是（175） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/evaluate.py` | `COMMON_CORE` | 是（45） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/evaluate_cbf.py` | `COMMON_CORE` | 是（1） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/expert.py` | `COMMON_CORE` | 是（9） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/filters.py` | `COMMON_CORE` | 是（4） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/generate.py` | `COMMON_CORE` | 是（1） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/generate_scene.py` | `COMMON_CORE` | 是（1） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/generate_scene_wide.py` | `COMMON_CORE` | 是（1） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/outcomes.py` | `COMMON_CORE` | 是（15） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_acceptance.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_audit_regressions.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_calibration.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_completion_certificate.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_deadlock_primary.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_deadlock_union.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_early_analysis.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_early_intervention.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_evaluation.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_exact_margin.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_frozen_sets.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_ordered_guidance.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_paired_statistics.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_primal_dual.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_progress_debt.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_reachability_margin.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_reachability_union.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_risk.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_risk_function.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_risk_v1.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_risk_v2.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_socp.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_soft_activity.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_step_control.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_step_control_restart.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_c1_temporal_certificate.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_terminal_progress_guard.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_trainer_recovery.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_v0.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_v3.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_c1_v3_training.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `single_integrator/tests/test_cbf.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_cbf_solver_status.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_deadlock.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_environment.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_filter_contract.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_outcomes.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_short_scene.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/tests/test_stalled_outcomes.py` | `COMMON_CORE` | 否（独立入口或文档） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/train.py` | `COMMON_CORE` | 是（5） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `single_integrator/validate.py` | `COMMON_CORE` | 是（4） | 保留在 main | 两条路线共享，或为审计/构建配置。 |
| `tests/test_c1_arithmetic_harmonic.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_boundary_importance.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_conditional_gaussian.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_direction_a_frozen_integration.py` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `tests/test_c1_direction_a_monitor_safety.py` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `tests/test_c1_direction_a_mutations.py` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `tests/test_c1_direction_a_policy.py` | `DIRECTION_A_ONLY` | 否（独立入口或文档） | 先归档；在 direction-a-stochastic 恢复 | 仅属于 stochastic residual / true-event score 路线。 |
| `tests/test_c1_event_history.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_gaussian_transport.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_moment_probability.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_projection_directional.py` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `tests/test_c1_r_cert.py` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |
| `tests/test_c1_risk_two_tests.py` | `LEGACY_OR_OBSOLETE` | 否（独立入口或文档） | 移入 archive/legacy_pre_git_split/，保持相对路径 | 不属于两个当前活动定义；保留历史可追溯性但退出活动 import/入口。 |
| `tests/test_c1_vi_r_cert.py` | `R_RISK_ONLY` | 否（独立入口或文档） | 先归档；在 direction-r-risk 恢复 | 属于最后冻结的 explicit R_risk / VI-R_CERT 定义或其最小复现依赖。 |

## 大型资产 bundle（清理前）

这些目录不做表现筛选、不删除。`results/` 中的 `.log`、`source_snapshot/` 和 cache 将在清理时作为可再生副本单独删除；其余原始结果保留。

| 路径 | 文件数 | 大小 | 分类 | 当前用途/归属 | 计划 |
|---|---:|---:|---|---|---|
| `baseline_309_314/CLEANUP.md` | 1 | 1.88 KiB | `LARGE_ASSETS` | checkpoint/基线资产 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `baseline_309_314/checkpoints` | 8 | 3.22 MiB | `LARGE_ASSETS` | checkpoint/基线资产 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `baseline_309_314/planning` | 17 | 1.30 MiB | `LARGE_ASSETS` | checkpoint/基线资产 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `datasets/give_way_si_short_uniform_state_v1` | 451 | 28.54 MiB | `LARGE_ASSETS` | 数据集 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `datasets/give_way_si_short_v1` | 503 | 28.97 MiB | `LARGE_ASSETS` | 数据集 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `papers/tro_deadlock.pdf` | 1 | 6.74 MiB | `LARGE_ASSETS` | 论文参考 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `papers/tro_deadlock.txt` | 1 | 148.88 KiB | `LARGE_ASSETS` | 论文参考 | 原路径保留；不自动删除；写入 SHA256 manifest |
| `results/c1_action_probe_parity_v1/` | 29 | 595.43 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_arithmetic_harmonic_adapter_v1/` | 4 | 33.61 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_arithmetic_harmonic_kernel_v1/` | 4 | 9.61 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_arithmetic_harmonic_probe_v1/` | 258 | 17.82 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_boundary_importance_v1/` | 257 | 14.49 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_candidate_coverage_expansion/` | 651 | 26.39 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_certificate_development/` | 82 | 8.77 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_clean_geometry_audit/` | 115 | 151.02 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_completed_waiting_risk_audit/` | 12 | 578.96 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_completion_development/` | 18 | 7.65 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_conditional_event_sections_v1/` | 78 | 2.56 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_conditional_gaussian_kernel_v1/` | 4 | 8.83 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_conditional_gpu_v2/` | 428 | 31.22 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_conditional_risk_cpu_v1/` | 195 | 15.00 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_critical_cone_kernel_v1/` | 4 | 10.32 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_audit_seed0/` | 7 | 1.63 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_extended_h120_comparison/` | 2 | 24.31 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_extended_h120_seed0_baselines/` | 56 | 12.65 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_extended_h120_seed0_c1/` | 29 | 18.47 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_frozen_sets/` | 3 | 2.71 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_paired_seed0_baselines/` | 406 | 95.21 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_dataset_paired_seed0_c1/` | 144 | 101.94 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_deadlock_primary/` | 338 | 33.64 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_deadlock_union/` | 8,310 | 361.45 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_diagnosis_training/` | 3 | 85.31 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_diagnosis_trajectories/` | 12 | 797.25 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_difficulty_calibration_v1/` | 72 | 3.34 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_direction_a_minimal_v1/` | 10 | 136.69 KiB | `LARGE_ASSETS` | Direction A 诊断资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_closed_loop_5sec_capture_v1/` | 22 | 131.19 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_closed_loop_5sec_v1/` | 21 | 128.00 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_closed_loop_5sec_v2/` | 41 | 769.88 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_closed_loop_v1/` | 41 | 262.04 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_derivative_cpu_v1/` | 4 | 7.03 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_derivative_cpu_v2/` | 30 | 160.76 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_localization_v1/` | 4 | 63.36 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_localization_v2/` | 0 | 0.00 B | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_directional_localization_v3/` | 5 | 86.20 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_early_forward_cpu_v1/` | 1 | 975.00 B | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_early_forward_cpu_v2/` | 2 | 4.02 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_early_gradient_v1/` | 25 | 146.34 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_early_gradient_v2/` | 434 | 30.50 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_event_history_prefix_v1/` | 14 | 1.28 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_event_history_probe_v1/` | 258 | 17.65 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_event_probability_difference_v1/` | 73 | 2.75 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_exact_margin_v1/` | 9 | 3.65 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_fixed_calibration_smoke/` | 3 | 1.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_fixed_horizon_waiting_audit/` | 59 | 7.64 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_focused_p_vs_pg_6seeds/` | 3,973 | 166.54 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_formal_h4_seed0/` | 3 | 1.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_formal_seed0/` | 3 | 559.43 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_four_objectives_multiseed/` | 5,088 | 233.32 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_frozen_unseen_64/` | 20,686 | 675.84 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_gaussian_transport_kernel_v1/` | 5 | 8.31 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_geometry_ranking_analysis/` | 42 | 4.14 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_geometry_solver_safety_audit/` | 20 | 562.21 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_gradient_closed_loop_v1/` | 374 | 30.91 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_gradient_exact_v1/` | 372 | 29.73 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_gradient_ordered_v1/` | 369 | 29.72 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_gradient_paths_guarded/` | 4 | 478.58 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_guard_factorial_v1/` | 453 | 95.68 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_guard_trace_audit_v1/` | 2 | 154.79 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_guarded_h120_comparison/` | 2 | 24.84 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_guarded_h120_seed0/` | 29 | 18.82 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_guarded_seed0/` | 6 | 1.75 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_independent_distribution/` | 533 | 30.73 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_jax_training_audit/` | 24 | 2.65 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_joint_closed_loop_audit/` | 62 | 6.68 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_joint_witness_risk_audit/` | 263 | 20.27 MiB | `LARGE_ASSETS` | 冻结 explicit R_risk / VI-R_CERT 资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_large_risk_association_400/` | 406 | 37.83 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_logic_audit_replay/` | 5 | 1.35 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_logic_smoke_validation/` | 1 | 1.36 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_logic_v11_smoke/` | 4 | 1.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_matched_scene_development/` | 71 | 2.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_matched_scene_stage_i_smoke/` | 56 | 1.98 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_matched_scene_stage_i_v1/` | 7,829 | 617.23 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_matched_scene_stage_i_wide_smoke/` | 56 | 2.26 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_matched_scene_stage_i_wide_v1/` | 7,845 | 686.21 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_moment_probability_probe_v1/` | 257 | 12.53 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_multistep_objective_audit/` | 280 | 10.24 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_nominal_localization_v1/` | 6 | 83.74 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_optimizer_diagnostic_v1/` | 68 | 5.02 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_ordered_parallel_v1/` | 4 | 125.88 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_ordered_parallel_v2/` | 8 | 3.45 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_pause_after_first_large_20260917/` | 7 | 8.79 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_prefix_physics_cpu_v1/` | 10 | 455.61 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_pretraining_audits/` | 2,226 | 6.45 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_primal_dual_seed0/` | 3 | 561.91 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_progress_conditioned_wait_audit/` | 224 | 2.14 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_progress_debt_audit/` | 5 | 298.88 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_risk_deadlock_correspondence/` | 776 | 5.15 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_risk_two_tests_v1/` | 460 | 32.54 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_scene_deadlock_union/` | 6,481 | 266.91 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_soft_activity_edge/` | 12 | 590.43 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_soft_activity_experiment/` | 3 | 262.64 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_soft_v01_h120_seed0/` | 3 | 1.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_two_horizon_wait_audit/` | 222 | 3.00 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v0_smoke/` | 3 | 15.32 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v1_fixed_validation/` | 1 | 8.17 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v1_h120_seed0/` | 4 | 1.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v1_timeout_challenge/` | 7 | 1.80 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v2_episode_eval/` | 8 | 3.39 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v2_episode_seed0_scan/` | 4 | 1.61 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v2_episode_seed0_smoke/` | 2 | 8.55 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v2_episode_validation/` | 1 | 25.76 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v2_frozen_seed0/` | 4 | 1.62 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v2_frozen_sets/` | 3 | 1.55 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v3_audit/` | 8 | 203.02 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v3_risk_outcome_association/` | 18 | 180.89 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_v3_smoke/` | 35 | 6.94 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_vi_adfd_layers_v1/` | 1 | 93.36 KiB | `LARGE_ASSETS` | 冻结 explicit R_risk / VI-R_CERT 资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_vi_adfd_localize_v1/` | 1 | 103.94 KiB | `LARGE_ASSETS` | 冻结 explicit R_risk / VI-R_CERT 资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_vi_r_cert_check_v1/` | 1 | 4.31 KiB | `LARGE_ASSETS` | 冻结 explicit R_risk / VI-R_CERT 资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_vi_r_cert_probe_v1/` | 488 | 23.68 MiB | `LARGE_ASSETS` | 冻结 explicit R_risk / VI-R_CERT 资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_vi_vjp_one_sided_v2/` | 3 | 1.85 MiB | `LARGE_ASSETS` | 冻结 explicit R_risk / VI-R_CERT 资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/c1_wait_horizon_sweep/` | 6 | 666.05 KiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/eval_c1_primal_dual_seed0/` | 203 | 55.53 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/extended_h120_eval_seed0_baselines/` | 56 | 12.65 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/extended_h120_eval_seed0_c1/` | 28 | 9.35 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/giveway_residual_effect_pilot_v1/` | 135 | 19.13 MiB | `LARGE_ASSETS` | Direction A 诊断资产 | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/risk_audit_seed0_baselines/` | 406 | 95.20 MiB | `LARGE_ASSETS` | 历史 C1 实验资产（不作为 main 默认实现） | 原路径保留；大二进制不提交，关键小摘要按分支显式纳入 |
| `results/*.json`（根级聚合） | 22 | 717.11 KiB | `LARGE_ASSETS` / `GENERATED_OR_DISPOSABLE` | 历史摘要或日志；逐扩展名复核 | `.log` 删除；文档/JSON 保留为资产或归档摘要 |
| `results/*.log`（根级聚合） | 100 | 282.79 KiB | `LARGE_ASSETS` / `GENERATED_OR_DISPOSABLE` | 历史摘要或日志；逐扩展名复核 | `.log` 删除；文档/JSON 保留为资产或归档摘要 |
| `results/*.md`（根级聚合） | 12 | 61.57 KiB | `LARGE_ASSETS` / `GENERATED_OR_DISPOSABLE` | 历史摘要或日志；逐扩展名复核 | `.log` 删除；文档/JSON 保留为资产或归档摘要 |
| `results/*.py`（根级聚合） | 1 | 2.97 KiB | `LARGE_ASSETS` / `GENERATED_OR_DISPOSABLE` | 历史摘要或日志；逐扩展名复核 | `.log` 删除；文档/JSON 保留为资产或归档摘要 |
| `results/*.txt`（根级聚合） | 2 | 1.85 KiB | `LARGE_ASSETS` / `GENERATED_OR_DISPOSABLE` | 历史摘要或日志；逐扩展名复核 | `.log` 删除；文档/JSON 保留为资产或归档摘要 |

## 明确可再生项（清理前计数）

| 类型 | 文件数 | 大小 | 处置 |
|---|---:|---:|---|
| Python cache / bytecode | 4,696 | 90.34 MiB | 删除；来源文件/运行配置另行保留 |
| duplicated source snapshots in results | 479 | 2.87 MiB | 删除；来源文件/运行配置另行保留 |
| temporary/Slurm logs | 240 | 4.12 MiB | 删除；来源文件/运行配置另行保留 |

## 清理前结论

- 当前活动目录同时暴露大量旧训练入口、旧风险候选、Direction A 与 VI-R_CERT 文件，存在误 import/误执行风险。
- `COMMON_CORE` 不做算法重构；路线文件先完整归档，再按 branch 恢复。
- 无法确认用途的源码一律归档而非删除；checkpoint、dataset、论文及原始结果一律保留。
- 该清单是处置计划，不代表任何方法或实验结论。

## 非代码配套文件补充核对

下列嵌套 Markdown/requirements 文件不在前述可执行源码扩展名扫描中，已在移动前人工补充：

| 路径 | 分类 | 被其他活动代码引用 | 计划移动/删除 | 理由 |
|---|---|---|---|---|
| `single_integrator/CBF_PROTOCOL.md` | `COMMON_CORE` | 否（协议文档） | 保留在 main | 两条路线共享的安全约束协议。 |
| `single_integrator/c1/IMPLEMENTATION_STATUS.md` | `LEGACY_OR_OBSOLETE` | 否（历史状态文档） | 移入 archive，保持相对路径 | 描述旧 R_risk 迭代与已停训练入口，留在活动目录会造成默认方法混淆。 |
| `single_integrator/c1/requirements.txt` | `COMMON_CORE` | 否（环境配置） | 保留在 main | 两条路线共享的 C1 运行依赖。 |
| `scripts/requirements-c1-plots.txt` | `COMMON_CORE` | 否（环境配置） | 保留在 main | 共享绘图环境依赖，不包含算法入口。 |

清理实施前的依赖复核还做出三项更正：`scripts/plan_baseline_400.py` 是冻结
Flow-BC/Safety 基线的共享复现入口，`scripts/c1_heldout_safety.py` 只实现两条
路线共用的物理安全检查，二者改列 `COMMON_CORE`；
`single_integrator/c1/models/{__init__.py,residual.py}` 是旧确定性 residual 模型，
当前 Direction A 使用自己的三头随机策略，因此这两个文件改列 `R_RISK_ONLY`。
这些更正只改变文件归属，不改变任何算法实现。
