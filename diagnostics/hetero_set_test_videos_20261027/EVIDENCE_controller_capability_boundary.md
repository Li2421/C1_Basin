# 证据：修正修不好本身不合格的 controller（集中式 MPPI）

日期 2026-10-27。数据：12 个测试场景（`diagnostics/hetero_set_critic_selfcheck_v1/scenes_test.json`），状态 8–23，每个 controller 取一个代表。流水线 = generator 16 个提案 + critic，每个场景选一个 η，不看 controller，选择在测试前冻结。最好固定 η = sE22 (0.55, −0.20, 1.42)，即训练场景上平均成功率最高的菜单 η。

## 结论
1. **修正的方向修不好本身有问题的 controller。** 集中式 MPPI 单独一台机器人、没有其他机器人时也到不了目标（0/1440，全部超时；平均速度约 0.07，上限 0.5）。它在只加安全投影和不加安全投影下都是 0%。加修正后的 17–37% 成功是修正项本身把机器人推到目标，不是纠正了死锁：修正越强越"成功"（十字路口 sE22 63–88%，流水线选的较小 η 是 0%）。
2. **因此效果与 controller 有关，但关系特殊：** 关键在 controller 自己能否完成单机导航（能力检查：每台机器人单独、只加安全投影、成功率 ≥ 95%）。通过检查的 7 个 controller 加流水线后都落在 74–96%；没通过的 MPPI 只有 17%。
3. **注意（如实）：** 7 个合格 controller 的提升幅度不一样（因只加安全投影的基线不同），梯度下降 MPC 在 3 个静态障碍场景上用流水线比用最好固定 η 差（56/31/31 对 56/50/88）。所以"差不多"指加修正后的最终水平，不是每个场景都一样。

## 数据（12 个场景合计，每格 192 局，%）
| controller | 单机能力检查 | 流水线 | 最好固定 η | 只加安全投影 | 不加安全投影 |
|---|---:|---:|---:|---:|---:|
| 集中式随机采样 | 100 | 74.0 | 55.7 | 25.0 | 0 |
| 梯度下降 MPC | 100 | 78.1 | 66.7 | 64.6 | 0 |
| 分布式 MPC（打靶） | 100 | 76.0 | 62.5 | 26.6 | 0 |
| 交叉熵 | 100 | 95.8 | 84.9 | 89.6 | 0 |
| 分布式交叉熵 | 100 | 82.3 | 75.0 | 51.0 | 0 |
| 冲突搜索 | 100 | 79.7 | 56.2 | 21.4 | 0 |
| 优先级 A* | 100 | 91.7 | 63.5 | 80.7 | 0 |
| **集中式 MPPI** | **0** | **17.2** | 22.4 | **0** | **0** |

MPPI 用 η13 (1.07, 0.06, 0.27) 为 36.5%。早先跨场景工作（`diagnostics/cross_scene_correction_v1/CAPABILITY_TABLE.md`）中 MPPI 也在几乎所有对穿类场景未通过同一检查。

## 文件
- 视频：`mppi_boundary/`（12 个场景 × 只加安全投影失败 / 修正成功，同一起始状态）；合格 controller 的对照见 `one_pair_per_scene/`。
- 表格：`diagnostics/hetero_set_critic_selfcheck_v1/TEST_PER_SCENE_8_CONTROLLERS.csv`、`TEST_PER_SCENE_7_CONTROLLERS.csv`。
- 能力检查：`diagnostics/hetero_set_critic_selfcheck_v1/code/capgate.py`、`test_capgate.jsonl`；rollout：`code/run8.py`、`test8.jsonl`。
