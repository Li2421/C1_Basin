# 当前流水线工作区快照（2026-10-27）

当前流水线（异构场景流水线）：3 个种子的 generator 组合，每个场景提出 16 个 η；critic 在菜单标签和留一 generator 提案标签上重训，只在这 16 个提案中选一个。
冻结文件：`hetset/FINAL_FROZEN.json`（8 个验证场景 + 第一批 12 个测试场景），`hetset/FROZEN_T2.json`（第二批 12 个测试场景）。

这份快照原样复制了四个仓库外的任务临时目录。这些目录会随任务删除而消失，所以保存在这里：

| 子目录 | 原位置 | 内容 |
|---|---|---|
| `hetset/` | `/home/zhihan/.claude/jobs/8061f3d8/tmp/hetset` | 流水线全部脚本（generator、critic、场景构建、评估、右手规则、表格）、训练与评估标签（`*.jsonl`、`critic_data*.pkl`）、场景定义（`scenes_*.json`）；`learned/` 是学习型规划器：专家示范、专家重标注数据、全部检查点、单机导航门与评估记录、视频脚本 |
| `scene_search/` | `/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search` | 隔离执行框架 `harness/`（环境、controller、安全投影、运行时） |
| `speed/` | `/home/zhihan/.claude/jobs/7eda8ddd/tmp/speed` | 精确加速用的 numpy threefry 等 |
| `cps_diag/` | `/home/zhihan/.claude/jobs/8061f3d8/tmp/cps_diag` | 一个诊断脚本（`hetset/inv.py`）用到的代码 |

## 注意
- 脚本里仍是原来的绝对路径，例如 `run.py` 里的框架路径、`accel.py` 里的 `speed` 路径、`critic.py` 读仓库里 `diagnostics/critic_pool_supplement_v1/VALIDATION_INPUTS.json`。原临时目录不在时，要把这些路径指到本快照（改路径或建符号链接）。
- 运行环境：rollout 用 `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`（工作目录 `hetset/`，`HS_FAST=1` 开启已核验的精确加速；第二批场景要加 `HS_SCENES=scenes_all2.json`）；表格和视频用 `.venv-c1-plots`。
- 要逐位复现 generator 的提案：`taskset -c 0-17`，并设 `XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12"`，generator 用原始 `critic_data_with_test.pkl` 训练（见 `hetset/step3_t2.py`、`hetset/gen_ens_t2.py`）。
- 运行时还会读取仓库里的：`shared_control/`、`single_integrator/`、`bottleneck_family/`、`shared_rollout_db/src/`；`diagnostics/cross_scene_correction_v1/SELECTED_CONTROLLERS.json`；`diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json`；`diagnostics/critic_pool_supplement_v1/VALIDATION_INPUTS.json`。
