# Bottleneck family：可变数量的死锁场景母版

主配置来自已有死锁研究的 Gap 双向交换设计；串联双瓶颈继承本仓库场景的思路。
完整审查、文献依据和实验边界见 [场景审查](../docs/scalable_scene_audit_20261007.md)。
这是连续空间的新变体，**不是论文复现或已验证的百机器人解锁算法**。

## 生成场景

仅需 numpy；本地 `.venv-video` 可用。下面只生成场景和单机器人路径证据：

```bash
.venv-video/bin/python -m bottleneck_family \
  --agents 100 --preset gap --seed 0 \
  --output diagnostics/my_gap100

# 六种几何 × 五档数量，工程检查，不是控制器大 batch
.venv-video/bin/python -m bottleneck_family.audit \
  --output diagnostics/bottleneck_family_scaffold/audit
```

输出 `instance.json` 和 `preview.svg`。可用配置：

| preset | 变化 |
|---|---|
| `gap` | 两个 staging 区，共享一个窄门 |
| `double` | 串联两个窄门，共享中间房间 |
| `offset` | 串联门上下错位，必须绕行 |
| `wide` | 门宽从 0.5 改为 1.2，测试并行通过 |
| `three_doors` | 同一隔墙上三个并列窄门，测试路线选择 |
| `clutter` | 在两个 staging 区增加矩形障碍 |

`--config path.json` 可传 `Config` 字段，覆盖 preset/agents/seed。例如：

```json
{
  "num_agents": 50,
  "barrier_x": [-1.5, 1.5],
  "openings": [[[-0.7, 0.5]], [[0.7, 0.7]]],
  "extra_rectangles": [[-4.0, 1.0, -3.0, 2.0]],
  "seed": 12,
  "split": "dev"
}
```

`openings` 是每道隔墙的 `(中心 y, 宽度)` 列表；`extra_rectangles` 是
`(xmin,ymin,xmax,ymax)`。更大的 N 若超出 staging 容量会失败，不会偷偷缩小机器人。
`half_length`、`half_height` 可显式调整，但改变面积的实验需与固定地图 N 扫描分开。

## 母版接口

```python
import numpy as np
from bottleneck_family import Config, BottleneckEnv

env = BottleneckEnv(Config(num_agents=20))
obs = env.observation()  # agents [N,6], walls [W,2,2], rectangles [M,4]
snapshot = env.snapshot()  # 兼容 shared_control.HardSafetyFilter
obs, done, info = env.step(np.zeros((20, 2)))
```

机器人是半径固定的二维圆盘，采用 `p[k+1]=p[k]+dt*u[k]`。默认物理 observation
只有位置、速度、相对目标；训练 adapter 另加从公开地图计算的门中心路点向量，
不包含优先级、通行模式或专家路径。实例中的 `paths` 只作为单机器人可达证据；
`SequentialExpert` 的完整零碰撞轨迹另行验证联合可解性。

`step` 接口返回三元组，与旧 Gym 风格场景不同；接入训练必须显式写 adapter。
输入超速或形状错误会报错；碰撞只记录并终止，不修改实际运动。场景支持动态 N，
旧四机器人 checkpoint 不支持直接迁移到这里。

`info['stalled_groups']` 记录持续低位移的邻近机器人簇；正常排队也可能被检测。
它只用于调查，不产生 `deadlock` 标签，也不因此自动终止 episode。
默认终止只有碰撞、全队到达或超时。

## 视频和正式实验

共享视频工具接受 `scenario="bottleneck_family"`，从 `config.num_agents` 读取 N，
逐个绘制全部机器人。素材要求同 [视频协议](../docs/batch_videos.md)。
四场景 batch plan 可额外加入 `bottleneck_family` 的三个角色；原四场景仍是必选。
正式规模扫描需按每个 N/geometry 保存证据，不能拿 N=2 的影片代表 N=100。
没有合格的死锁判定/成功轨迹时应报告缺口，不得将 `stalled_groups` 直接改名充数。

正式 rollout 之前需按 `shared_rollout_db/README.md` preflight。下述开发评估保存
scenario/controller/state 指纹和 preflight 报告，尚未将新 rollout 写入共享数据库。

## Joint Flow Matching 训练与评估

`flow_dataset.py` 用保守的单机器人轮流通行专家生成真实、安全、全队成功的轨迹；
train/dev/test 使用独立 seed，局部门口扰动恢复只从 train/dev 专家状态产生。
`train_flow.py` 复用旧四场景的官方 MACFlow `ActorVectorField`、联合动作 CFM、
Adam 和十步 Euler 采样。每个 N 使用独立 checkpoint，不能把 N=2 模型直接用于 N=50。
训练当前只支持单门 Gap 1；其余障碍变体仍是场景工程检查，尚未训练控制器。

```bash
FLOW_PYTHON=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
ROOT=diagnostics/gap_flow_v1/n10_example
"$FLOW_PYTHON" -m bottleneck_family.flow_dataset --agents 10 \
  --door-width 0.62 --train 48 --dev 12 --test 12 \
  --recovery-anchors 4 --seed 20261007 --output "$ROOT/dataset"
"$FLOW_PYTHON" -m bottleneck_family.train_flow --dataset "$ROOT/dataset" \
  --output "$ROOT/train" --steps 10000 --batch-size 256 --seed 17
"$FLOW_PYTHON" -m bottleneck_family.evaluate_flow --dataset "$ROOT/dataset" \
  --checkpoint "$ROOT/train/best.pkl" --output "$ROOT/raw_dev" \
  --samples-per-step 16 --seed 17
"$FLOW_PYTHON" -m bottleneck_family.evaluate_flow_route_supervisor \
  --dataset "$ROOT/dataset" --checkpoint "$ROOT/train/best.pkl" \
  --output "$ROOT/route_dev" --seed 17
```

`evaluate_flow.py` 是原始闭环策略，可选安全投影。`evaluate_flow_route_supervisor.py`
是**单独标记的组合控制器**：Flow 选择下一位通行者，预测速度在局部安全路径内
可行时直接执行；A* 路线和保守路径跟随器处理不可行的步。其成功率不能归于
纯 Flow，也不能仅凭成功推断死锁解决能力。所有成功率指同一 dev 初态下，
整个仿真 episode **所有机器人**到达且无碰撞；test 归档在训练和调参中不打开。

开发成功 trace 可用 `render_development_video.py` 逐帧转成全程 MP4，
对应 manifest 只标记单场景观察视频，不是原四场景和死锁基线的正式交付。
`evaluate_reciprocal_baseline.py` 提供不设单向路权的安全双向放行基线：
在同初态下检测门口相向机器人接近物理接触、仍相互趋近、连续 5 秒几乎不动。
N=50 可加 `--pair-release`，让队列首先同时放行一对相向机器人，其他机器人等候；
这是明确标记的基线规则，不能把正常排队或单纯 timeout 称作 deadlock。
对应干预视频需要相同完整初态哈希、环境 seed、配置与独立成功轨迹。
`render_development_video.py --role deadlock_show|deadlock_resolution|safety_success`
输出单个全程 MP4；`validate_development_videos.py --root <目录> --sizes 2 10 50`
核验每档三个不同成功轨迹、死锁对照、SHA256 和实际编码帧数。

最终保留集评估必须显式传 `--split test --final-evaluation`，训练和开发评估
默认只能读取 dev nominal 归档。调参之后再一次性打开 test。

## 测试

```bash
.venv-video/bin/python -m unittest discover -s bottleneck_family/tests
# 共享安全层集成测试另需 scipy/clarabel，可使用现有 .venv-c1
C1_PYTHON=/path/to/.venv-c1/bin/python
"$C1_PYTHON" -m unittest discover -s bottleneck_family/tests
```

已覆盖 2/4/10/20/50/100、六种障碍设置、跨 N 的嵌套任务一致性、不可通行几何拒绝、
扫掠碰撞和局部停滞。N=100 零速度输入通过安全层接口检查，尚无其冲突求解性能结论。
