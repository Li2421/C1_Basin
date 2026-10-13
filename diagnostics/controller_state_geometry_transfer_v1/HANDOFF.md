# Independent Research Handoff — Controller-conditioned Generalization

**先读状态：原任务最重要的强state-dependent η选择封口目标未达成。** 已完成有限预注册pilot、失败诊断、等预算旧面板迁移、few-shot校准、独立因素与机制坐标实验；不能将结果表述为方法主要缺陷已被排除。下面不依赖原对话。

**复现设置必须一同交接：** Toy critic的JAX x64=False，factor/normalization为True。匹配设置后全部138个checkpoint的η选择重现，59个集成评估组成功率变化0。初次审计漏掉该配置曾产生19.8pp差异，已修复；那是精度不匹配的敏感性，不是相同设置下的不稳定。Q数值不是逐位一致，详见C7及`REPLAY_SENSITIVITY_recorded.json`。

## 1. 问题与背景

项目在每步执行冻结参考动作、OrthoFlow3三维η修正与硬安全FF投影；一个episode内η固定。Critic预测无碰撞且按时全员到达的Q(s,C,η)。旧四场景LOSO把geometry、Flow、N、horizon等混在一起；旧DB连续宽度族可以迁移(C,η)关系，但state信号很弱。现在需要区分预测Q、给定C选η、校准场景η偏好及success basin泛化。

## 2. 实际完成及关键实现

结果根目录：`/home/zhihan/research/Basin_C1/diagnostics/controller_state_geometry_transfer_v1`。复用旧Toy width13,312条；新增7,172次真实尝试。旧Toy拟合39个、独立因素90个、固定单位归一化诊断9个，共138个，均3训练seed；四场景校准复用旧LOSO预测。完整脚本、参数、每seed checkpoint/预测、raw、factor完整轨迹、数据索引、预注册及哈希均在目录内。

Pilot依次测试自然初态、加深bay、广初态、固定H400/H200；宽度、C、η独立控制，安全/成功标准未改。广初态Q有35.2%方差涉及state，但各pilot交叉拟合自适应增益均未达到5pp且CI>0的门槛。广初态main没有运行。

独立因素面板固定地图/半径/速度，交叉width(.50/.65/.80)、单/双并行开口拓扑、N2/N4、H400/H850，保持跨几何同物理state、跨N嵌套任务。使用同一N通用解析导航参考和两种FF矩阵，不能称为Flow跨N迁移。

## 3. 最重要定量结果

| 面板 | state 方差占比 | S×η 无偏能量 | 交叉拟合 η 自适应增益 | 95% CI | 通过门槛 |
| --- | --- | --- | --- | --- | --- |
| 旧自然初态 / H850 | 0.138 | 0.01224 | -0.008 | [-0.031, 0.000] | 否 |
| 加深 bay / H850 | 0.158 | 0.01441 | -0.027 | [-0.043, 0.000] | 否 |
| 广初态 / H850 | 0.352 | 0.03075 | -0.016 | [-0.070, 0.016] | 否 |
| 广初态 / H400 | 0.516 | 0.04721 | -0.031 | [-0.062, 0.008] | 否 |
| 广初态 / H200 | 0.744 | 0.00635 | -0.008 | [-0.023, 0.000] | 否 |


旧Toy内插：ENT MANY的S×η相关0.391、无偏skill−0.084；ALL提高到0.662、0.217。MANY−FEW Q MSE=-0.032,CI[-0.057, -0.006]；两者选择成功均1.0，几乎无state选择空间。高Q相关或100%选择均不能封口。

| 目标 | zero-shot模型 | k1 prior | k1 prior+条件 | k4 prior | k4 prior+条件 | k4 条件−prior / CI |
| --- | --- | --- | --- | --- | --- | --- |
| toy_width | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.000 / [0.000, 0.000] |
| toy_give_way | 0.288 | 0.359 | 0.429 | 0.506 | 0.503 | -0.003 / [-0.047, 0.027] |
| double_bottleneck | 0.125 | 0.988 | 0.990 | 0.997 | 0.999 | 0.001 / [0.000, 0.004] |
| four_way_intersection | 0.938 | 0.936 | 0.963 | 0.976 | 0.985 | 0.009 / [0.000, 0.026] |
| ring_exchange | 0.300 | 0.576 | 0.607 | 0.760 | 0.742 | -0.018 / [-0.029, -0.007] |


| 留出因素 | ENT | +raw | +机制 | 仅机制 | 固定per-C η | 交叉拟合oracle | 机制−ENT / CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WIDTH | 0.758 | 0.766 | 0.766 | 0.711 | 0.711 | 0.789 | 0.008 / [-0.016, 0.031] |
| EXTRAP | 0.883 | 0.875 | 0.867 | 0.875 | 0.859 | 0.891 | -0.016 / [-0.039, 0.000] |
| COMBINATION | 0.719 | 0.719 | 0.688 | 0.594 | 0.750 | 0.719 | -0.031 / [-0.094, 0.000] |
| FAMILY | 0.797 | 0.776 | 0.766 | 0.266 | 0.766 | 0.859 | -0.031 / [-0.083, 0.021] |
| FAMILY_REVERSE | 0.688 | 0.297 | 0.375 | 0.057 | 0.625 | 0.745 | -0.312 / [-0.365, -0.260] |
| COUNT | 0.599 | 0.099 | 0.490 | 0.438 | 0.401 | 0.651 | -0.109 / [-0.156, -0.047] |
| TIME | 0.891 | 0.193 | 0.870 | 0.536 | 0.865 | 0.953 | -0.021 / [-0.115, 0.062] |


Factor outcome={'timeout': 1406, 'success': 1665, 'numerical_failure': 1}；唯一未知在训练集，最终测试0未知、0碰撞。四场景旧Flow数据实际7,680条、51未知，不是12,800条；后者只是张量容量。见`history_audit.json`。

Factor交叉拟合state oracle增益也仅约±0.5pp，仍不满足5pp的强选择门槛。显式机制坐标在EXTRAP/COUNT改善Q，却未提高选择；反向拓扑成功损失31.25pp。固定单位缩放在FAMILY改善原机制模型Q MSE约0.119，但成功仍比ENT低3.65pp；COUNT仍较差，TIME差异不明确。该诊断排除了“只要修正缩放就能恢复选择”的简单解释，不排除表示缺陷。

以上选择率来自原保存预测，匹配精度GPU重放均保持。不匹配精度时近并列η排序可明显变化；独立重放/复拟合入口已显式恢复原精度。不能把首轮审计遗漏归因成方法缺陷或理论机制证据。

## 4. 假设、已排除与未排除

- 支持：指定旧Toy预算下几何覆盖改善Q估计；少量目标标签可恢复DB/Ring场景级η regime。
- 不支持：MANY在饱和面板中有选择优势；加深bay足以产生有意义state选择空间。
- 未定：真正依赖state的η排序能否跨几何泛化；高精度pilot是否揭示Q2/Q4 oracle估计遗漏的headroom；主要方法缺陷能否被排除。
- 显式坐标的效果依留出类型不同。上表给出逐项配对结果；跨路由拓扑仍是同一母版/解析程序，不是原四场景跨机制证明。
- 精确controller身份与物理state本身已知，不是本面板的信息缺口。但没有测试未知controller函数、未知Flow或大N。
- 尚未排除：sample support、表示/归纳偏置、训练目标、OOD特征缩放、有限future-seed精度、场景×Flow嵌套。不要复述旧报告中无范围限制的“均已排除”。
- 已定位的复现问题：factor训练导入的x64副作用必须显式保留。修复独立入口后全部η选择重现；跨精度/CPU后端仍需再验证，未用事后改规则替换原结果。

## 5. 对Q、η选择及success basin的影响

Q预测可改善而argmax没有额外价值；场景prior可恢复成功而不迁移state条件结构。理论应分别形式化这三种问题。当前标签仅K4/K8离散候选、Q2/Q4，不能推断连续success basin体积/拓扑、B15鲁棒性或普遍enlargement。交叉拟合oracle是有限标签选择参考，不是真实oracle上界；负regret不能解释为超oracle。

## 6. 最重要未解决问题

构造并独立确认有可靠state×η选择headroom的面板，再冻结全新state/geometry测试。优先复用已有raw预测确认是候选饱和还是Q估计精度不足；本次不继续无限搜索。其次是把路由拓扑与真实Flow程序变化解耦；归一化诊断表明缩放只解释部分Q退化，选择失败仍待解释。本报告没有完成强选择或跨Flow机制泛化封口。

## 7. 可直接使用的证据与入口

- `REPORT.md`：完整A–G报告、每阶段解释和限制；`results.json`：完整机器结果。
- `PREREGISTRATION.md`及补充文件、`*_receipt.json`：原始门槛、时间、hash、执行调整。
- `data/legacy_diagnostic.npz`、`data/factors.npz`：标签/unknown/实体/显式特征；`data/*index.json`：原始来源和SHA。
- `models/legacy_diagnostic/`、`models/factors/`：每seed `.msgpack`、`.npz`预测和训练索引、`.json`超参/loss/时间。
- `models/normalization/`、`normalization_results.json`：9个固定物理单位归一化探索性拟合及配对结果。
- `CHECKPOINT_REPLAY_gpu_recorded.json`、`REPLAY_SENSITIVITY_recorded.json`、`replay_predictions/`：全部138个checkpoint的正确精度重放；不带recorded文件是首次精度不匹配诊断；`checkpoint_tools.py`提供修复后的入口。
- `calibration_results.json`：每seed、k、子集、paired CI和未知上下界；原预测在`data/calibration_prediction_index.json`列出。
- `factors/{protocol,cases,geometry_validation}.json`、`factors/raw/`、`factors/traces/`：独立因素真实证据。
- `pilot_revision/`、`broad_pilot/`：失败pilot也保留；`broad_main_study.py`仅恢复入口，**未运行且门槛未通过**。
- `FINAL_AUDIT.json`、`MANIFEST.json`、`source_implementation_audit.json`、`NUMERICAL_HANDLING.md`、`EXECUTION_NOTES.md`：审计、成本、数值失败、代码与数据版本。
- `RUNBOOK.md`：复算、恢复、cache/merge命令；`figures/`：可共享PNG/PDF。

工作根 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。主Git基线`a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e`，但历史diagnostics/PoC不少未纳入Git；以SHA快照和数据索引为准。未修改或删除历史证据。新增实验是纯Q研究，依AGENTS豁免视频交付。
