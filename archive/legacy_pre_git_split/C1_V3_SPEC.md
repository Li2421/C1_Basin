# C1 V3：完整时域联合风险实验

2026-09-16。V3 是独立的研究候选版本，未替换 `C1_SPEC.md` 的历史 v2、
固定乘子联合风险实验或冻结基线。入口为 `single_integrator.c1.train_v3`，
不是给旧 `train.py` 增加一个含义不同的隐式默认值。

## 修改依据与范围

旧联合实验预测到 25 秒、执行到 42.5 秒，但 P 和 g 只在 5–15 秒平均。
已见 P+g 三种子执行的 79 条未解决死锁全部在 15 秒后首次触发。
这不证明早期几何没有预测价值，但后期停滞没有直接进入 P+g 评分。
合成轨迹回归进一步验证：仅改变 30 秒后的进展，旧 P+g 不变，V3 有梯度。

保留：冻结 Flow-BC、相同状态和噪声的 Safety reference、物理残差、
双硬投影、原联合 g 的 witness/normalization/undefined domain、5 秒冻结前缀。
未修改任何旧控制器、风险或历史 checkpoint。

V3 改变：P+g 的轨迹支持与分母、训练时域，以及相对近期固定乘子实验恢复
C1 的约束训练日程。故 V3 风险值不能和旧 v2 或旧 5–15 秒分数直接比较。

## 风险与 C1 目标

保持原联合实验的逐步 P 定义：归一化总目标距离的 4 秒平均下降速率，
进展阈值 0.005/s，温度 0.00125/s，逐 agent 未完成 sigmoid 的平滑 OR。
距离为零处使用 1e-30 的平方范数下限，避免精确目标处 NaN 梯度；
旧窗口公式一致性已做回归。g 直接复用 `risk/joint_frozen.py`。

```
R_v3 = mean_{t=100,...,849}(P_t + g_t)
J_live = mean_episodes R_v3
J_def = mean_episodes mean_executed_trainable_actions ||u_C1 - u_safe||²
min J_def subject to J_live <= epsilon
L = J_def + lambda * (J_live - epsilon)
lambda_next = max(0, lambda + dual_lr * (J_live_before_step - epsilon))
```

成功动作本身计入成本，此后为吸收零成本。风险使用固定 750 动作分母，
使持续未完成不会因延长执行而被实际长度重新归一化抵消。
J_def 仍按真正执行的可训练动作数归一化。
风险是任务相关代理，不是概率、成功/失败严格分离代价或活性证明。

训练及对应评估保持近期联合实验的 **deadlock 可恢复协议**：
低速/低进展事件继续记录但不提前停止；成功、碰撞、42.5 秒截止仍权威地
由环境判断。碰撞或不可评分几何显式失败，不能作为低风险或丢弃样本处理。
这不是原 400 回合首次死锁即终止的协议，二者不能混报。

## 数据、优化与记录

- `--prepare` 先冻结 train / validation / test 初态及各自噪声种子。
  分布延续近期联合实验：共享 |x|~U(.55,1.05)，两车 y 独立 U(-.025,.025)。
  检查三组互不重复，并与代码列明的既有 benchmark/近期集合精确去重；
  不声称检查了所有历史数据。默认规模 32/16/64；2/2/4 仅为工程 smoke。
- epsilon=0.8×零残差在全部 train 初态、两组冻结校准噪声上的均值；
  不筛失败或零风险样本。训练从相同 train 初态总体抽样，使用新噪声。
- validation 两组独立固定噪声用于可行优先、最小 J_def 选模。
  test 仅由显式评估入口执行；训练从不使用其回报或轨迹。
- Adam 候选按同批次、同噪声、固定 lambda 的完整 L 复评和减半回溯。
  不增加“每步风险必须下降”的新约束；拒绝则回滚参数与 Adam 状态。
- 最后保存全 train 校准噪声下的重新评分，便于区分训练未收敛和验证差距。
  记录模型、源文件和输入哈希、校准逐回合值、更新前后值、乘子与试探步。
- `--objective risk_only` 只诊断风险优化的经验可达性，优化 J_live；
  不属于最终 C1 约束解，不生成 best_feasible。两步失败不能证明不可达。
- 当前入口拒绝覆盖已有训练目录，保存原子 checkpoint，但尚无恢复入口。
  不应将其当作已具备长期生产恢复能力的训练器。

## 运行

```bash
# 冻结中等规模实验输入；本轮只运行过独立 tiny smoke 集。
.venv-c1/bin/python -m single_integrator.c1.train_v3 --prepare --sets results/c1_v3_sets/sets.json
.venv-c1/bin/python -m single_integrator.c1.train_v3 --sets results/c1_v3_sets/sets.json --out results/c1_v3_seed0

# 只有真实生成可行模型后才使用该文件；没有则明确报告不可行。
.venv-c1/bin/python -m single_integrator.c1.evaluate_v3 --checkpoint results/c1_v3_seed0/best_feasible.pkl --sets results/c1_v3_sets/sets.json --out results/c1_v3_test

# 查看固定预算终点可用 checkpoint.pkl，报告会标记它未由可行选模产生。
.venv-c1/bin/python -m unittest single_integrator.tests.test_c1_v3 single_integrator.tests.test_c1_v3_training
```

需原始相邻 `01_MACFlow_Baseline_Reproduction/MACFlow_Official` 依赖。
本轮该依赖已在本机恢复可用，未重写/替换原始网络。

## 结论门槛与未解决事项

本轮完整链路通过，但微型两步训练未达到 epsilon、未改善实际死锁。
不开展按新 test 结果选学习率、候选风险或 checkpoint 的流程。
后续应先验证死锁分支的方向导数、活动集与 g/P 目标冲突，再冻结更充分
的对照预算；不能因现在有 V3 代码就宣称已解决死锁。
详见 `results/c1_v3_audit/REPORT_ZH.md`。
