你是接手 Basin_C1 / OrthoFlow3 研究的实现 agent。
- 工作根目录：`/home/zhihan/research/Basin_C1`
- Python：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`

开始前先读 `AGENTS.md`、`shared_rollout_db/README.md`，以及 round27 的报告与代码（`diagnostics/other_controller_classes_v1/`）。严格遵守 AGENTS.md。

# 一、目的：给 round27 补两个基线

round27 只跑了带 OrthoFlow3 修正的 6 个臂：F10、F13、F2、S200、S300、S500。论文需要另外两个基线，才能完整说明“我们的零样本选择”比“不加修正”好多少。

| 基线 | 链路 | 含义 |
|---|---|---|
| **nominal only** | `no_safety` | 只执行 nominal controller 的动作，不加安全投影，也不加修正 |
| **safety baseline** | `hard_safety` | nominal 动作经过硬安全投影，不加修正，即标准的“nominal + 安全过滤器” |

这两种链路的定义沿用项目已有的写法（见 `new_benchmark_common/safety_eta3.py` 中的 `no_safety` 与 `hard_safety`），并在 round27 的 runtime 中以相同方式接入。

# 二、设计（执行前写好 PREREGISTRATION.md，附 UTC 时间与 sha256 收据）

**controller：** round27 的全部 16 个：MPC M0–M5、势场 P0–P5、4 个 Flow TT 参照。直接使用 round27 冻结的实现，先核对代码与参数的 hash，不得重新实现或调参。

**state 与 seed：** round27 主评估的 32 个 state，seed 0–1，seed key 完全相同，以便与 round27 的 6 个臂逐条配对。

**链路：** `no_safety`、`hard_safety`。

**规模：** 16 × 32 × 2 × 2 = **2,048 条完整 rollout**。

**一致性门（必须先通过）：** 用本轮的 runtime，重跑 round27 的 8 条已有记录（orthoflow3 链路，覆盖 MPC、势场、Flow 各类），要求终止类型、步数与首动作字节逐位一致。不通过就停下报告。

**精度：** 全部导入完成之后，显式恢复协议规定的精度（round25 的教训），并记录 x64 状态。

**流程：** 按 shared_rollout_db 流程执行：design → preflight → 只跑缺失 → journal → 单进程 merge → postflight。两条链路登记为新的执行身份。

# 三、指标与报告（描述性，附 CI）

对每个类别（MPC、势场、Flow）与每个 controller，报告以下 8 种做法的成功率：

1. nominal only
2. safety baseline
3. 修正 + 全程 η10（旧做法）
4. 修正 + 全程 η13
5. 修正 + 最佳固定臂（oracle）
6. **修正 + 冻结 A3 的零样本选择**（即 round27 的 A3_full 选择）
7. 修正 + 一律 S500
8. 完整菜单 oracle

第 3–8 项的数字取自 round27 的已有标签，必须与 round27 报告一致。

**附加指标：**
- 碰撞率：nominal only 可能发生碰撞，用连续运动的 swept clearance 检查确认；
- 超时率、数值失败率；
- 停滞诊断（描述性，不得称为 deadlock）。

**关键对比：** 第 6 项相对第 1、第 2 项的成功率差。CI 采用 state 成簇、配对的 bootstrap，共 5000 次，在 controller 层先重抽。

# 四、硬性规则

- **文件写入范围：** 只在新目录 `diagnostics/nominal_safety_baselines_v1/` 中写文件。不修改、不删除历史资产与 round27 的代码或数据，不做 git commit。
- **不得改动核心组件：** 不修改控制器参数、物理、安全投影、事件判定或成功标准。成功标准仍为“全体无碰撞到达”。
- **unknown：** 一律保留，不重跑，不算作失败。timeout 不能直接称为 deadlock。
- **视频：** 与 round27 一致，属于纯 Q 函数的评估数据，按 AGENTS.md 的例外不运行视频入口，并在预注册中写定。
- **资源：** 本 workstation 上用户所有任务合计不超过用户当前授权的上限：目前为 18 个 CPU shard、2 个 GPU shard；如有更新，以用户最新指示为准。

# 五、交付

全部交付物放在 `diagnostics/nominal_safety_baselines_v1/`：
- PREREGISTRATION.md 及收据；
- 一致性门报告；
- DB 的 preflight 与 postflight 记录；
- results.json；
- 一张论文用的总表：类别 × 8 种做法的成功率与 CI，另附碰撞率、超时率；
- REPORT.md；
- HANDOFF.md（不依赖本对话也能读懂）。

REPORT.md 第一段必须直接给出：在 MPC、势场与 Flow 上，我们的零样本选择相对 nominal only 与 safety baseline 的成功率提升（附 CI）。
