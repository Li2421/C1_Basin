# C1 Toy GiveWay

本仓库保存双机器人 GiveWay 的共享环境、冻结 Flow-BC 基线、联合欧氏安全投影、
deadlock monitor、终止逻辑和两条相互隔离的 C1 研究路线。

分支职责：

- `main`：清理后的共享 C1 基线，不提供任何研究路线的默认算法入口。
- `direction-a-stochastic`：joint Bernoulli gate + 4D Gaussian residual，直接估计真实事件概率的 score-function 路线。
- `direction-r-risk`：最后冻结的 explicit `R_risk` / VI-augmented `R_CERT` 路线，仅供独立审计。

共享基线测试：

```bash
bash scripts/test.sh
```

冻结 Flow-BC/Safety 的 400 回合复现入口保留为共享工具；它会写入新的输出目录，
本轮仓库整理没有运行该评估：

```bash
.venv-c1/bin/python scripts/plan_baseline_400.py --out-dir results/reproduced_400
```

大型 checkpoint、dataset、论文和历史 rollout 仍位于原路径，但不进入 Git。
其路径、大小和 SHA256 见 `docs/assets_manifest.md`。清理和分支恢复规则见
`docs/repo_inventory_before_cleanup.md`、`docs/repo_inventory_after_cleanup.md`
及 `docs/git_split_report.md`。

本地 C1 环境依赖记录在 `single_integrator/c1/requirements.txt`；仓库已有
`.venv-c1` 时，`scripts/test.sh` 会优先使用它。
