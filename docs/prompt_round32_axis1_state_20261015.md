你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`
- 本轮目录：`diagnostics/axis1_state_v1/`

# 零、先读

1. **`docs/generalization_axes_protocol_20261015.md`（共同协议）。** 其中的源码保护、DB 规则、复用规则、标准方法与指标、标准交付，对本轮全部适用，优先级高于本文件以外的一切。
2. `AGENTS.md`、`shared_rollout_db/README.md`。
3. 总结报告 `docs/controller_generalization_ring_summary_20261014.md`，重点是第 0、2、14–17 节。
4. round10（`diagnostics/controller_universal_eta_schedule_v1/`）的预注册与执行代码；round16（`diagnostics/ring_controller_critic_v1/`）的冻结模型；round26（`diagnostics/method_benchmark_v1/`）。

# 一、要回答的问题：泛化轴 1，State

**定义：** 场景与 controller 都已见过，测试训练中没见过的新 state。

| 子项 | 内容 |
|---|---|
| 1a 同分布新 state | 与训练相同的初态生成器，换新的种子 |
| 1b state 分布偏移 | 预先写定的偏移生成器：更不对称的位置、更宽的半径范围、更大的初速度 |
| 1c 按 state 选择 | 逐 state 选动作，比按 controller 选动作多了多少收益 |

**已有证据（DEVELOPMENT，本轮只整理进表，不重跑）：**
- round16 与 round26 的 T2：round10 的 14 个原 controller 在 round15 的训练集中，而 round10 的 64 个 state 不在其中。A3 的平均遗憾为 0.75pp。
- round2：在单个 controller 上，逐 state 选择比最佳固定 η 高 5.21pp [2.65, 7.88]。但在 round4、round8、round16 中，逐 state 选择都没有超出按 controller 选择的效果。

本轮补一个前瞻、预注册、带完整基线的 State 轴检验。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 与 sha256 收据）

**controller：** round10 的 14 个原 Ring Flow（best.pkl）。它们都在 round15 的训练集中，A3 已经见过。

**执行：** 与 round10 相同，FF 位置（PoC `Runtime.action` 的 `chain = "field"`），H700，`key_for` 的未来随机键与 round10 一致，seed 0–1。复现门：先复现 round10 的 16 条记录（共同协议第三节）。

**state：**
- **1a：** 用 round10 的同一个生成器（见 round10 预注册中的 state 生成说明），换一个新的种子区间，生成 64 个 state。区间事先算好，写入预注册，并与 DB 中全部 state 去重。
- **1b：** 把 `ring_exchange/environment.py` 中的 `sample_instance` 复制到本轮目录，只改下列常数，另外起一个新的 SeedSequence 标签（不得与 train、development、test 的标签相同）：

  | 参数 | 原值 | 偏移后 |
  |---|---|---|
  | 起点角度扰动 | ±0.20 rad | ±0.40 rad |
  | 目标角度扰动 | ±0.18 rad | ±0.36 rad |
  | 起点与目标半径 | U(2.30, 2.68) | U(1.40, 2.90) |
  | 初速度大小 | U(0, 0.11) | U(0, 0.30) |

  每个 state 必须通过环境 reset 的有效性检查，不得有初始碰撞。不通过就按预注册规则取下一个种子。生成 32 个 state，并与 DB 去重。

**臂（每个 controller × state × seed）：** nominal only、safety baseline（FF，η = 0）、F10、F13、S300、S500。nominal only 按共同协议第三节写新函数。

**规模：**
- 1a：14 × 64 × 2 × 6 = 10,752 条；
- 1b：14 × 32 × 2 × 6 = 5,376 条。

**冻结选择（在任何评估标签之前完成并留收据）：**
- A3：controller 级选择；另做一份逐 state 的选择（每个 state 取 argmax）。
- A0：controller 级与逐 state 各一份。
- G_E、一律 S500、一律 F10。
- 菜单为 A3 冻结菜单与 {F10, F13, S300, S500} 的交集，写入预注册。
- controller 的 R5 特征：复用 round15 与 round16 已冻结的特征。新 state 的 physical23 特征用冻结的特征代码计算。

# 三、预注册的终点与判定

| 编号 | 子项 | 判定标准 |
|---|---|---|
| ST1（主） | 1a | A3（controller 级）相对四臂 oracle 的平均遗憾 ≤ 3pp（用交叉拟合的 oracle），且 A3 − safety baseline 的 CI 下界 > 0 |
| ST2 | 1b | 在偏移 state 上，同 ST1 的两条标准。另报从 1a 到 1b 遗憾与成功率的变化（描述性） |
| ST3 | 1c | 逐 state 的 A3 − controller 级的 A3：CI 下界 > 0 记为"逐 state 选择有增益"；CI 上界 < 0 记为"有损"；其余记为"未检出差别"。另报交叉拟合的逐 state oracle 相对 controller 级 oracle 的潜在增益 |

**另报（描述性）：** 1a、1b 中从 nominal only 到 safety baseline 的逐条转移表；碰撞率、超时率；伤害数。

**证据等级：** ST1–ST3 为 CONFIRMATORY；round16、round26、round2 的已有结果为 DEVELOPMENT。

# 四、结论映射

| 观察结果 | 结论 |
|---|---|
| ST1 通过 | 对见过的 controller，方法在同分布的新 state 上成立 |
| ST2 通过 | 方法对 state 分布偏移稳健 |
| ST2 不通过 | 报告失效发生在哪类 state（按偏移参数分层），作为方法的局限 |
| ST3 为"未检出差别"，而潜在增益 > 0 | 按 state 选择有空间，但 critic 没能利用；支持"按进度或状态触发切换"这一改进方向 |
| ST3 为"有增益" | 按 state 选择值得进入方法 |

# 五、交付

按共同协议第五节交付，放在本轮目录。AXIS_TABLE 的行至少包括：
- 1a 同分布新 state（本轮，CONFIRMATORY）；
- 1b 分布偏移（本轮，CONFIRMATORY）；
- 1c 按 state 选择（本轮）；
- 已有：T2（round16、round26）与 round2，标 DEVELOPMENT。

REPORT.md 第一段必须直接回答：
- 完整性审计是否通过；
- ST1、ST2、ST3 的结论与关键数字。
