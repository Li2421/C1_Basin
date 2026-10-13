# 三维无人机导航死锁 pilot

低顶通道加侧上方避让舱，N=2，单核 worker、零 GPU。实现三维速度控制、硬安全投影、连续碰撞检查、持续互阻监视器、真实 MACFlow、TT/FF η 修正和固定控制器的 Q 选参。使用理想速度跟踪，不含飞行动力学。

首轮移植已经运行：[pipeline 结果与限制](../docs/drone_3d_pipeline_port_20261013.md)。纯 Flow 27条未成功；显式加入近目标伺服的 TT 组合控制器在6个独立测试初态上由Q选参成功6/6，但固定同一个η也成功6/6，尚未证明自适应选参收益。

设计、文献、历史证据和结果见 [研究记录](../docs/drone_3d_deadlock_design_20261013.md)。当前最终证据为 `diagnostics/drone_3d_pilot_v3/`，三组 deadlock/success 配对都通过；v1 数值中止和 v2 小墙面裕量的记录保留。

## 运行

使用已有 numpy/scipy/clarabel 环境，不需要安装新的仿真平台。

```bash
DRONE_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

# 已有 final 证据：自动 preflight 并复用全部 6 条，不重复仿真。
"$DRONE_PY" -m drone_3d.run_pilot run

# 确有新语义/证据缺口时，在新目录准备；prepare 记录历史审查、注册身份并执行共享 preflight。
# prepare 后不得更改 pilot.py / run_pilot.py，再运行该冻结计划。
"$DRONE_PY" -m drone_3d.run_pilot prepare --output diagnostics/NEW_DRONE_PILOT
"$DRONE_PY" -m drone_3d.run_pilot run --output diagnostics/NEW_DRONE_PILOT
# 读取该目录 protocol.json 的 experiment_uid，用单个进程 merge。
python3 -m shared_rollout_db.src.merger --experiment EXPERIMENT_UID

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$DRONE_PY" -m unittest drone_3d.test_geometry -v

# 只读已保存证据，单线程 ffmpeg，无新 rollout。
.venv-video/bin/python -m drone_3d.render --root diagnostics/drone_3d_pilot_v3 --seed 17
```

新输出目录如果命中已有 DB 轨迹而本地没有对应文件，runner 会要求取回原 trace，而非重新仿真。每个计划上限 6 条，prepare/run 不提交 Slurm、不做训练。路线 controller 的指定让行机和 phase 显式属于其执行语义；`eta=[0,0,0]` 在本 pilot 只是共享缓存的占位，不能当作已经执行 OrthoFlow3。

## 证据与安全语义

NPZ 包含初态、每个物理步、终态、动作、阶段、目标、三维障碍、连续安全余量和 metadata。`swept_clearance` 三列依次是墙面球净距（米）、机间物理球净距（米）、机间排除椭球余量（无量纲）。机间椭球仅是保守几何排除区，不模拟下洗气流。

三个种子仅改变两机共同 y/z 偏移，均保留迎面对称，不代表扰动鲁棒性。视频属于单场景开发交付；正式三维 batch 之前需要扩展统一视频验收入口的二维 trace 契约。二维可扩展母版仍为 `bottleneck_family/`。

## Pipeline 运行

`diagnostics/drone_3d_pipeline_v1/` 保留两次Flow训练、全部失败、组合控制器、Q训练/冻结预测/独立测试及完整视频。新实验必须先读取该目录和共享库的兼容证据；以下已有路径默认复用，不重复训练或生成rollout。

```bash
DRONE_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

# 已有 checkpoint 时保留，不重训。
srun -p cpu -c 1 --mem=2G --time=00:10:00 "$DRONE_PY" -m drone_3d.train_flow

# 已有 Q panel 的72条都精确复用。
srun -p cpu -c 1 --mem=2G --time=00:10:00 "$DRONE_PY" -m drone_3d.experiment run \
  --root diagnostics/drone_3d_pipeline_v1/q_panel

# 只读缓存评价；不训练、不生成 rollout。
"$DRONE_PY" -m drone_3d.critic evaluate

srun -p cpu -c 1 --mem=2G --time=00:05:00 "$DRONE_PY" \
  -m unittest drone_3d.test_geometry drone_3d.test_pipeline -v
```

确需新证据时，在新目录依次 `experiment prepare`、`run --split train`、`run --split dev`、`critic fit`、`run --split test`、`critic evaluate`。`prepare` 可选 `--stage qdata --variant terminal_servo_v1 --placement TT --etas '[[0,0,0],[0,0.5,0],[0.5,0.5,0]]'`，默认checkpoint是上述原Flow。纯Flow使用 `--variant pure_flow`；FF不支持目标伺服组合。新目录遇到完全兼容结果会读取原trace，不应因此重跑。

`train_flow.py --terminal-data .../terminal_teacher --width 128 --steps 8000 --output NEW` 是已做过的监督补充实验，其闭环仍失败；读取旧证据即可回答相同问题，不重复训练。新的控制器语义、checkpoint或物理状态确有缺口时才补跑。

已有视频：[死锁配对](../diagnostics/drone_3d_pipeline_v1/videos/deadlock_show/seed17_full_comparison.mp4)、[独立初态Q选参配对](../diagnostics/drone_3d_pipeline_v1/videos/q_selected/seed2026101318_full_comparison.mp4)。`render.py --baseline BASELINE.npz --intervention SUCCESS.npz --root OUTPUT --seed SEED` 只读真实证据生成完整三维视频与manifest。
