# Round20 预注册：跨场景无标签探测选择

权威任务：`docs/prompt_round20_cross_scene_probe_selection_20261014.md`。已先读 `AGENTS.md`、`shared_rollout_db/README.md`、`docs/batch_videos.md`、总结报告第 6、11 节，并查阅 round13 的代码、协议、报告、逐臂汇总及旧视频证据。此文与 SHA256 收据写入时，不读取或生成任何 Round20 评估成功标签。

## 历史证据和缺口

`diagnostics/cross_scene_eta_schedule_v1/`：E8 四场景 FF runtime、Flow-reference 三算子、七臂与原几何 40 state×4 seed 的真实 rollout；可复用代码、原几何等价目标、开发集成功率和视频候选。`diagnostics/controller_ff_factorial_repair_v1/`：controller、checkpoint、环境配置。`diagnostics/controller_state_transfer_v1/`：Toy 宽度变体的先例，需确认其状态、参数、checkpoint 与本设计的执行语义才可复用。`diagnostics/four_controller_axis_v2/`：Four controller 轴，几何及 runtime 不同，不作目标标签复用。新八几何变体、独立 pilot/probe/evaluation state、所需 probe 特征和新标签尚无兼容证据；逐项查询共享数据库，只有 exact semantics 可复用，不因输入编码变化丢弃兼容标签。原几何复现门选 round13 既有的 8 条记录，逐位核对结局、步数、首动作及可用完整精度数值；不通过则停止新标签运行并报告。

## 固定设计

runtime 为 E8 的原生四场景 FF，加 round13 绝对步数 η 替换。三个 Flow-reference 算子为 flow_isotropic、flow_anisotropic、flow_coupled，各用本场景 checkpoint、原归一化和安全投影。仅改变几何参数与每步 η；不修改物理、控制器、安全投影、事件或成功标准。Toy corridor_width 原 0.40，变体 0.44、0.50、0.56；DB bottleneck_width 原 0.50，变体 0.60、0.75、0.90；Ring obstacle_radius 原 0.90，变体 0.75、1.05。Four 原几何仅参与开发参照 probe。H：Toy/DB 850，Ring 700，Four 1100。t1=round(3H/7)：Toy/DB 364，Ring 300，Four 471；t2=round(5H/7)：Toy/DB 607，Ring 500，Four 786。η 取 E8 完整 float64 值。

每变体独立 16 pilot state，flow_isotropic、F10/F15 各 seed 0 一次；两臂成功率都严格 <20% 才排除。先验证实体放置、通道连通和容量，不调整半径或替换变体。pilot 只用于此有效性门，不参加 D1–D3。所有新 state 按固定 namespace 与索引生成，pilot/probe/evaluation 各自独立；新 physical state 与旧数据库及本轮其他 split 去重，失败时报告，不按结果改抽样。

每有效变体及四原场景各 16 个新 probe state，flow_isotropic 分别 η13/η15，运行到 t1，屏蔽成功、碰撞、死锁终止与标签读取，保持状态推进和动作计算。每个 state/η 的 stall = 曾连续至少 40 步速度范数 <0.01 的机器人比例；场景 stall 为 16 state 的均值。progress ratio = state 内起点到 t1 的目标距离净下降严格 >0 的机器人比例，再对 state 取平均。minimum pair distance = t=0..t1 所有机器人中心间最小距离，再对 state 取平均。若数值异常，记录为 probe unknown；不得以终止标签代替特征。规则 R：stall(η15)>stall(η13)+0.05 时选 S13_t1，否则选 S15_t2。特征、各变体与原场景的选择、代码 hash、UTC 和 SHA256 收据必须在读取任何 Round20 评估标签前冻结。

评估为每个有效变体×3 算子×40 全新 state×七臂 `F10,F13,F15,S13_t1,S13_t2,S15_t1,S15_t2`×seed 0,1；全有效 13440 条。初态、随机种子、同一算子的七臂配对。先设计请求和标准 preflight，仅执行真正缺失 seed；probe 与 pilot 也分别先存请求清单、运行官方 preflight 并留收据。worker 只写本目录原始证据及 journal，单进程官方 merge、postflight。unknown 原样保留，不重跑，不当作观测失败。Toy/DB 仅原事件触发才记 deadlock；timeout 单列。

## 终点及统计

round13 四原场景、三算子等权平均的七臂开发集源均值最高是 **S15_t2**，下界成功率 0.884375；作为 D2 固定基线。此项由旧 `ARM_RATES.csv` 只作开发集计算，已在新标签前写定。变体成功率 = 3 算子各 40 state×2 seed 的平均。每变体 oracle = 七臂最高成功率，遗憾 = oracle 减已选臂；并列按上述臂顺序。D1：全部有效变体平均遗憾≤3pp 且最大≤10pp。D2：R 相对固定 S15_t2 的各变体遗憾差，等价于基线成功率减 R 成功率；要求平均遗憾更低且按 state 成簇配对 95% bootstrap CI 上界<0。D3：逐变体 `max(S13_t1,S15_t2)` 与七臂 oracle 差≤3pp 的变体比例，描述性报告。D4：原四场景选择与 round13 事后最优的比较，仅 DEVELOPMENT，不参与 D1–D3。Bootstrap 5000 次，固定种子 202610142020，按变体内 state 重采样，保留同 state 的算子、seed、臂配对；D2 再对变体等权平均。记录点估计、区间及 unknown 下/上界；若 unknown 边界跨过判定门，则 INCONCLUSIVE。有效变体<4 时整体证据 INCONCLUSIVE。D3 报每项是否达标及比例，不把未指定的比例阈值伪装成总体显著性门。

报告每变体两种 stall、progress、minimum pair distance、规则选择、七臂成功率、失败事件与步数。结论仅限 Toy/DB/Ring 现有拓扑族内的几何变体。数值报告第一段直接给 D1/D2/D3 状态，事实与解释分开。

## 视频、资源与交付

Toy、DB、Ring 各至少三个从真实初态到终止的完整视频：两个不同真实死锁解决案例（`deadlock_show`、`deadlock_resolution`，同初态同 seed 的基线与成功干预）及独立 `safety_success`，用 `docs/batch_videos.md` 统一入口校验。Ring 缺原生死锁判据则如实留下缺口；缺失时 launcher 非零退出，不宣布 batch 视频完成。连续运动安全证据须来自 simulator。

用户 2026-10-14 最新授权：确认无人使用时，本 workstation 所有任务合计最多 22 CPU shard；GPU 仍最多 2 shard。提交前和运行中检查队列占用，并扣除别人占用。所有文件只写本目录；不修改、删除旧资产，不做 git commit。
