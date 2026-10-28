# 十七类场景的有限候选覆盖

本轮依据 `docs/prompt_scene_types_oracle_coverage_v3_20261028.md`。17 类的全部 49 个难度档，每档固定状态 0–7，接入两个原控制器，共 784 个案例。只运行七个冻结修正族及只加安全投影参照；不重新标定、不选族、不判维度、不训练模型、不跑论文方法。

恢复命令：

```bash
bash diagnostics/scene_types_oracle_coverage_v3/resume.sh
```

运行解释器与输出目录固定；Slurm 名称前缀为 `stb3_`，用户任务总上限 18 CPU、本轮 GPU 0。协调程序持有一个 CPU 份额，最多 17 个单核 worker。先等待遗留 worker 退出，再由唯一协调程序合并 journal，核验官方数据库后继续。已完成数值异常不重跑。

每个案例严格按冻结顺序读取候选，只读已走到的缓存证据；遇到第一个缺失候选才提交。不同案例可处于不同候选序号。第一次成功即停止，最多 64 次。`rNN` 是缺失请求的批次号，真正候选序号写在 `*_frontier.json`、`*_labels.json` 与 `COVERAGE.json`。同一请求独占文件锁，重复候选共享唯一结果。硬上限为实际新 rollout 尝试 90,000 次，不通过删场景、状态或候选凑预算；到上限未完成的族写为缺口。

先登记请求并做官方 preflight；worker 写本实验 journal；批次全部完成后单进程 merger 并读回。旧时限结果不是精确命中：60 局逐位前缀检查通过后，显式迁移旧成功、碰撞和 unknown，逐条记录旧 UID；旧 timeout 只在搜索真正走到时延长。

查看进度：`STATUS.md`、`LIVE_STATUS.json`、`COVERAGE.json`、`REPORT.md`。报告和覆盖表在每次批次推进后更新，未运行完不计为 64 次失败。四倍时限使用实际执行的同一批轨迹，不在六倍时限首次成功后追加搜索。

数值阶段退出后，`finish_when_ready.py` 调用 `finalize.py`：重建逐候选记录、补入最后未完整批次中已保存的真实结局、核验官方索引与保护目录、生成时间统计及曲线。仅在完整性通过且全部规定案例搜索完毕时，`COMPLETE.json` 的 `complete` 才为 true。大数据、轨迹、SQLite 与 journal 保留在本地，Git 提交不代表这些数据已有远程备份。
