# 2026-10-07 目录审查与清理

## 清理前备份

已将当前代码、脚本、文档和选定配置提交并推送至 `origin/main`：
`43b5264`（`chore: snapshot four-scene research workspace before cleanup`）。
新增/修改共 2,154 个文件。SSH 远程读取和 push 均成功，无需变更账户。

初查共有 341,274 个 Git 未跟踪文件，约 13.46 GB，其中 SQLite 主文件约
4.23 GB。普通 Git 不能作为这些大型数据的完整备份。本次快照不包含数据库、
大型原始结果、dataset、checkpoint、视频等；这些资产均保留原路径，未宣称
已远程备份。`diagnostics/` 实际总量约 19 GiB，`shared_rollout_db/` 约 5 GiB。

## 已删除和整理

- 删除 `archive/legacy_pre_git_split/` 的 333 个 Git 跟踪历史副本。
  当前活动 Python/shell 文件未引用该目录；副本可从清理前快照恢复。
  旧审查文档中提到这个目录的内容是历史记录，不再表示当前磁盘状态。
- 删除 184 个 `__pycache__`、`.pytest_cache`、`.ruff_cache` 缓存目录，
  合计约 22,994,859 字节；这些缓存可以重新生成。
- 增补 `.gitignore`：实验生成记录、调度日志、图表、SQLite 主文件/WAL/SHM、
  journals 和 CSV 导出继续留在本地，不再淹没源码变更。
- 重写 README 的四场景目录导航，移除过时的“只有 Give-Way/纯共享基线”描述
  和已经不存在的 `docs/git_split_report.md` 链接。
- 新增仓库级视频规则、全轨迹 MP4 生成/验收工具，以及 batch 包装入口。

恢复旧代码（仅有需要时）：

```bash
git restore --source=43b5264 -- archive/legacy_pre_git_split
```

## 保留与待判明项目

以下项目没有充分证据认定多余，因此未删除：

| 项目 | 保留原因 |
|---|---|
| `diagnostics/` 的旧实验与失败记录 | 多个研究脚本跨实验引用，且用于 provenance 和对照；不能仅凭旧日期删除 |
| `shared_rollout_db/rollout.sqlite*`、journals | 权威缓存索引和写入恢复文件，不能当作普通临时文件清理 |
| `datasets/`、checkpoint、结果 trace | 大型研究资产，未另行备份，可能无法低成本重建 |
| `single_integrator/` | `toy_giveway/` 兼容接口的真实实现，不能因为名字旧而删除 |
| 根目录 `C1_*.md` 和 `docs/` 历史审查 | 保留研究语义、阈值和历史依据；不移动以免破坏已有引用 |
| `environment.yml`、`requirements_gpu_working.txt` | 可能过时/带旧机器路径，但需确认是否仍用于复现 |

本次不启动新的大型仿真，也不补造既有实验的视频成功案例。视频工具的合成
测试只验证渲染和验收行为；正式交付必须使用真实完整 rollout。历史 launcher
尚未逐个嵌入视频命令，需通过新包装入口或在最终汇总步骤显式运行验收。

## 验证

- 视频契约与 MP4 集成测试：11 项通过，包含四场景共 12 个合成视频的全流程
  生成、帧数检查、缺场景/截断/重复/伪死锁/不安全成功拒绝，以及旧完成标记失效。
- 既有 `scripts/test.sh`：58 + 2 + 19 = 79 项通过。当前机器 CUDA 初始化不可用，
  JAX 自动回退 CPU；未运行 GPU 训练。
- Shell 语法与 Git diff 空白检查通过。
