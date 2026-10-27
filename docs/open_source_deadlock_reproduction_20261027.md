# 开源多机器人死锁方法：文献与证据复算（2026-10-27）

## 范围和历史证据

本轮关注连续空间移动机器人在共享瓶颈中的避碰和到达，并优先核对作者官方代码。实验前检索 `docs/`、`diagnostics/`、`results/`、`datasets/` 的方法名、配置、模型和报告；`docs/drone_3d_deadlock_design_20261013.md` 已引用 Park 等方法，但未找到 LiveNet、Merry-Go-Round 同配置的本地运行。仓库原有的 `bottleneck_family/`、四场景以及 `shared_rollout_db/` 是不同控制器、动力学和场景，不能将其标签或成功率直接当作这些方法的结果。LiveNet 官方仓库公开 CSV、两条轨迹和权重；优先复算这些证据，无需重复生成相同 rollout。

## 候选

| 方法 | 出处与官方来源 | 思路和适用性 | 复现条件 |
|---|---|---|---|
| Merry-Go-Round | IROS 2025，[论文与代码](https://github.com/Aiden-wjLee/merry-go-round) | 去中心化避碰、预防死锁；包含障碍、交换和 20–120 机器人实例。适合研究多机器人扩展，但与 Gap 双向瓶颈不同 | Python 3.11+、CVXOPT、PyQt5；自带地图，仿真有图形界面。作者注明代码在论文后清理过 |
| Decentralized Deadlock-free Trajectory Planning for Quadrotor Swarm | ICRA 2023，[论文主页](https://qwerty35.github.io/research/conference/ICRA2023/)，[代码](https://github.com/qwerty35/lsc_dr_planner) | 栅格多智能体路径规划提供路径点，subgoal 与 linear safe corridor 生成轨迹；针对带障碍物的无人机 | 依赖 ROS 及规划环境；动力学与本仓库不同，完整复现成本较高 |
| LiveNet | L4DC 2025，[论文](https://proceedings.mlr.press/v283/gouru25a.html)，[代码](https://github.com/srikarg89/LiveNet) | 神经网络+可微 CBF，同时约束安全和活性；门口和交叉口与本研究相关 | 作者提供权重、数据和 CSV；论文只测试两机器人。L4DC 是控制领域会议，不能标为 ICRA/IROS/RSS |

## LiveNet 公开数据复算

源码版本 `f2e13cd42a79ecaa7d1d0dfc4a8b760ca93c73e9`；输入为该版本 `experiment_results/*.csv` 和 `experiment_results/histories/*.json`。运行 `python3 scripts/external_baselines/summarize_livenet.py /path/to/LiveNet`；脚本只用 Python 标准库。到达按作者定义：位置误差 <0.05 m 且速度绝对值 <0.05 m/s；碰撞使用 CSV 中的标志。未到达与碰撞可能重叠，不能相加。仅在已到达行统计 makespan。

| 数据 | 行数 | 无碰撞且均到达 | 碰撞行 | 未到达行 | 到达行的平均 makespan |
|---|---:|---:|---:|---:|---:|
| LiveNet Doorway | 50 | 50 | 0 | 0 | 14.2 s |
| LiveNet Intersection | 50 | 50 | 0 | 0 | 11.6 s |
| LiveNet Doorway suite | 28 | 22 | 5 | 1 | 12.956 s |
| SMG-CBF/MPC Doorway suite | 28 | 14 | 13 | 5 | 12.565 s |

作者论文表 2 报告 Doorway LiveNet 为 13.8 s、Intersection 为 11.6 s，论文正文的 suite 成功数为 LiveNet 25/28 和 SMG-CBF 16/28；公开 CSV 复算分别为 14.2 s、11.6 s、22/28 和 14/28。CSV 与论文存在差异，原因尚未确定，不能宣称定量完全复现。作者 50 行默认场景的前 12 列各仅有一个不同值，不能把它们看作 50 个独立随机场景。CSV 没有公布种子及每行完整轨迹。

独立检查作者提供的轨迹：Doorway 首次到达步数 [60,69]，机器人最小中心距离 0.2773 m，按机器人/障碍半径计算到达前最小净间距分别为 0.0345 m、0.0109 m；对应 CSV 行为 [60,71]、最小中心距离 0.2754 m，表明轨迹和 CSV 并非可直接视作同一条运行。Intersection 轨迹到达步数 [50,58] 与 CSV 一致。它们只是离散采样时刻的检查，不构成连续时间安全证明。作者汇总脚本将任一 `-1` 到达索引称为 deadlock；本报告只称未到达，不把 timeout 判为死锁。

论文写仿真时长 18 s；当前 `config.py` 为 20 s，保存轨迹为 90 个采样点；运行设置的历史版本及差异仍需核实。作者当前的 suite 设置仅包含初始速度 0.3 m/s，而论文正文还提到静止初始状态；不能断言两者是同一批配置。当前公开结果不足以证明本仓库 Gap 双向交换上的联合可解、控制器成功或死锁解决。

## 进一步复现条件

当前复算属于复用作者数据的指标与轨迹审计，尚不是独立闭环运行或重新训练。后续若要比较真实相同条件：应先固定任务和动作/动力学接口，分别核对几何有效、单机器人有路与联合可解，再登记外部控制器及完整仿真语义；在 `shared_rollout_db/README.md` 要求的 manifest、preflight、journal、merger 支持该语义后仅补缺失 rollout。不可把外部实验的不同动力学伪装成本仓库已有 state/eta/controller，也不能将 `stalled_groups` 或 timeout 当作已证明死锁。建议先从 MGR 官方交换实例验证环境和终止、碰撞统计，再考虑 Gap 单积分器的明确适配与可行性验证。

## 官方权重的离线动作复现

用作者已保存的两条轨迹作为输入，逐状态运行作者的 `ModelController` 和对应 checkpoint，共各 90 帧 × 2 个 agent；先按作者环境裁剪控制，再和历史控制比对。Doorway 180/180、Intersection 180/180 的两维动作均在 1e-5 内一致；最大绝对误差分别为 3.20e-16、4.17e-17。这验证了当前代码与权重能重现所保存轨迹中的动作，但没有重新模拟状态转移，也不能消除 CSV 与论文的差异。

环境：CPU，Python 3.12，PyTorch 2.13.0+cu130，qpth 0.0.18、cvxopt 1.3.3、cvxpy 1.9.3、numpy 2.5.4、scipy 1.18.1；由于 checkpoint 来自 CUDA，脚本将 `torch.load` 指定 `map_location=cpu`，没有改权重或控制器数学。当前复现使用本机 `/home/zhihan/venvs/gpu/bin/python` 及 `/tmp/basin_livenet_deps` 中依赖，命令：

```bash
PYTHONPATH=/tmp/basin_livenet_deps CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zhihan/venvs/gpu/bin/python scripts/external_baselines/check_livenet_actions.py /tmp/basin_external_LiveNet
```

`/tmp` 克隆与依赖为临时工作副本；永久复现可按上述 commit 检出官方仓库并安装上述版本。作者代码的输入标准差包含零，会产生除零 warning，随后使用 `nan_to_num`；本次保持作者行为。用户需注意：动作复现是已有轨迹上的离线推理，并不是闭环独立再现实验。
