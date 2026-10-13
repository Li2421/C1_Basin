# Controller-conditioned Generalization：state、几何与机制验证

最终生成 UTC：2026-10-11T06:54:58.099929+00:00。代码基线：`a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e`。

**交付状态：完成有限实验、诊断和下游验证，但最高优先级的强 state-dependent η selection 封口目标尚未达成。** 五个预先限定的面板/时间预算均未通过“state 信号 + 可靠选择收益”的联合门槛；不能把本报告写成方法主要缺陷已被排除。条件成立才启动的 `broad_main` 没有运行。所有未通过、数值未知和执行调整均保留。

**复现核查：** 匹配原始精度设置后，138个checkpoint的全部η选择均重现，59个集成评估组的成功率变化为0。初次重放漏设因素模型的64位模式，曾导致最多19.8pp差异；已纠正，并保留为精度不匹配的敏感性记录。C7说明这一执行修正，不能将其误写成相同设置下的选择不稳定。

## A. Task Overview

研究问题是 Q(s,C,η) 的 state 条件结构能否同时迁移到新初态、新几何，以及这种预测是否能提高给定 controller 时的 η 选择。本研究另检验少量目标标签的校准、独立物理因素和显式机制坐标。

核查了 `docs/controller_generalization_audit_20261011.md` §10–§11、DB width 实验及其 entity critic、原始 rollout/配置、Flow/FF/OrthoFlow3/安全投影实现。旧 DB 实验 entity 外推 B2≈0.626 属于低 state 信号的单宽度族；不能一般性排除表示或训练目标缺陷。其 ENT EXTRAP_B 的给定-C η 成功为0.175、oracle0.575；联合选(C,η)为0.5、oracle1.0。历史“外推选择等于oracle”不能泛化到该对照。

新发现的兼容资产 `../controller_state_transfer_v1/` 在接手时仍在运行。复用了其13,312条 Toy width 标签，未重新生成。它的独立历史分析使用24个训练state；本任务使用不同的16-state训练划分及等预算分配，因此已有模型不能替代本次拟合。历史新报告随该作业结束出现；本目录保存接手快照和后续核查版本，不改历史文件。

`history_audit.json` 另纠正计数：四场景 Flow 原始张量有12,800槽位，`observed` 确认实际7,680条，含51条数值未知、0碰撞。槽位数不是 rollout 数。

## B. Implementation Summary

- `pilot.py`、`pilot_revision_analysis.py`、`broad_pilot_analysis.py`、`horizon_pilot.py`：隔离 pilot，split-half 无偏析因与选择门槛。state、state×η、geometry×η、controller×η 及高阶项分别保存。
- `revision_study.py` / `broad_study.py` / `supplement.py` / `prefix_study.py`：新请求注册、精确缓存 preflight、真实 rollout、append-only journals 与单进程 merge。只补证据缺口。
- `toy_data.py`、`learn.py`、`main_analysis.py`：项目 `Critic('state_field_compact')`、原 entity encoder、MLP、state-only、原始参数、机制特征和一个固定 ranking-loss 诊断。每个 seed 的 checkpoint、训练索引、loss、全单元格预测均保存。
- `calibration.py`：0/1/2/4目标state，20组固定嵌套子集；旧四场景直接复用三个seed的 source-only LOSO预测。
- `factor_runtime.py` / `factor_study.py`：未改的 `bottleneck_family` 圆盘动力学、连续运动碰撞检查、硬安全投影、OrthoFlow3和十步FF；新增可变N的个体几何路径导航参考。它是解析控制器，不是固定N Flow checkpoint迁移。
- `factor_data.py` / `factor_analysis.py`：全交叉物理面板，七种留出方式，机制坐标与移除 time-margin 的消融。
- `metrics.py`、`test_contracts.py`：同物理state配对bootstrap、独立未来seed评估、oracle/置换/析因代码验证、N任务前缀、endpoint匹配、安全证书、四条原生H400与H850前缀复现。

Toy 几何变量 `corridor_width` 同时改变走廊和 bay 开口；其余动力学、碰撞/成功定义及controller固定。Bay深度修正只改`.68→.84`。广初态是安全的相向近/远遭遇与初始bay占用混合，保留固定目标、零速度与真实t0空监测器；不是伪造中间轨迹。修正依据仅来自pilot，独立最终广初态没有生成outcome。

先记录两个失败的几何pilot，再以单独预注册的探索性补充测试初态覆盖；最后固定H400/H200作前缀标签检验。每一步的旧失败结论保持不变。所有门槛失败后关闭环境搜索，未降低标准；依据 `ADDENDUM_DIAGNOSTICS_AND_FACTORS.md` 转为旧面板有限诊断及独立因素研究。明确不能用这些替代有效强选择面板。

## C. Experimental Results

### C1. Pilot：Q变化不等于选择空间

门槛：state相关信号≥20%；within-(G,C) S×η能量≥.005且95%CI下界>0；交叉拟合state oracle相对(G,C)固定η收益≥.05且95%CI下界>0。物理state bootstrap2000次。

| 面板 | state 方差占比 | S×η 无偏能量 | 交叉拟合 η 自适应增益 | 95% CI | 通过门槛 |
| --- | --- | --- | --- | --- | --- |
| 旧自然初态 / H850 | 0.138 | 0.01224 | -0.008 | [-0.031, 0.000] | 否 |
| 加深 bay / H850 | 0.158 | 0.01441 | -0.027 | [-0.043, 0.000] | 否 |
| 广初态 / H850 | 0.352 | 0.03075 | -0.016 | [-0.070, 0.016] | 否 |
| 广初态 / H400 | 0.516 | 0.04721 | -0.031 | [-0.062, 0.008] | 否 |
| 广初态 / H200 | 0.744 | 0.00635 | -0.008 | [-0.023, 0.000] | 否 |


旧初态pilot复用1,024条；加深bay新增2,048条(Q4)；广初态新增1,024条(Q2)，H400/H200复用其成功时间。另4条原生H400执行用于前缀复现，均匹配。广初态H850的state方差已达35.2%，说明确有state-dependent Q；其选择门槛仍未通过。H200平均成功仅1.95%，没有作为正式研究环境。

这些结果**不证明总体真实oracle没有headroom**。Q2/Q4的交叉拟合oracle是有噪声的选择器，不是数学上界；argmax并列与有限未来seed会压低收益。这里能说的是预注册的可评估性门槛未满足，不能排除更高精度下的排序信息。完整15项析因、CI与seed记录见各`*_pilot_results.json`、`pilot_results.json`及`data/*pilot*.npz`。

H200中部分bootstrap样本的总信号为0，state方差比例CI未定义；原派生文件保留NaN，汇总`results.json`规范化为null，不把它写成有限CI。

![Pilot门槛与区间](figures/pilot_gate.png)

### C2. 旧 Toy 面板：严格分离state和几何的有限迁移诊断

Pilot state0–7；训练8–23；最终测试24–31。训练宽度.41/.47/.53/.59，留出内插.44/.50/.56。MANY=4宽度×8state，FEW=2端点宽度×16state；两者同一16个不同state、各1,024标签。FEW×2加seed2/3，共2,048；ALL=4宽度×16state×Q2，共2,048。测试Q4，用seed0/1选交叉拟合oracle，seed2/3评估模型实际闭环结果。

以下是三训练seed集成；每个seed与所有CI在`legacy_diagnostic_results.json`。成功率是**给定C只选η**，不是联合选择。

Q MSE使用所有最终future seed的经验均值；另存跨seed残差乘积估计，避免把Bernoulli方差误认模型误差。S×η在每个(G,C)内同时去除S和η均值，因此包含与G/C交互的局部排序变化。C×η先平均state，G×η先平均state/controller。Pair-order只评估经验Q差≥.25的η对，属于有限标签指标。固定η/prior仅由源训练标签选择；oracle与目标场景固定η均由独立的第一半future seed选择，第二半评估。统计单位为完整物理state块，2000次配对bootstrap；所有几何/controller与随机seed随该state一同重采样。

| 模型/预算 | Q MSE | S×η r | S×η skill | 给定 C 选η成功 | matched−mismatch | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| ENT_MANY | 0.033 | 0.391 | -0.084 | 1.000 | 0.001 | [0.000, 0.004] |
| ENT_FEW | 0.065 | 0.421 | -0.603 | 1.000 | 0.004 | [0.000, 0.013] |
| ENT_FEW_x2 | 0.057 | 0.435 | -0.315 | 0.990 | -0.004 | [-0.013, 0.000] |
| ENT_ALL | 0.029 | 0.662 | 0.217 | 1.000 | 0.000 | [0.000, 0.000] |
| STATE_MANY | 0.055 | 0.531 | 0.176 | 0.990 | 0.003 | [-0.004, 0.013] |
| MLP_MANY | 0.045 | 0.368 | -0.483 | 1.000 | 0.003 | [0.000, 0.009] |
| ENT_raw_MANY | 0.022 | 0.549 | 0.208 | 1.000 | 0.004 | [0.000, 0.013] |
| ENT_mech_MANY | 0.024 | 0.479 | 0.141 | 0.990 | -0.010 | [-0.031, 0.000] |
| RANK_MANY | 0.030 | 0.404 | -0.008 | 1.000 | 0.000 | [0.000, 0.000] |


| 旧Toy内插模型 | seed17 success / S×η r | seed23 success / S×η r | seed41 success / S×η r |
| --- | --- | --- | --- |
| ENT_MANY | 1.000 / 0.293 | 1.000 / 0.415 | 1.000 / 0.311 |
| ENT_FEW | 1.000 / 0.401 | 0.990 / 0.404 | 1.000 / 0.377 |
| ENT_FEW_x2 | 0.990 / 0.482 | 0.990 / 0.372 | 1.000 / 0.362 |
| ENT_ALL | 1.000 / 0.598 | 1.000 / 0.778 | 1.000 / 0.478 |
| STATE_MANY | 1.000 / 0.532 | 0.990 / 0.495 | 1.000 / 0.460 |


| 旧Toy内插模型 | Q correlation | C×η r | G×η r | 联合(C,η)成功 |
| --- | --- | --- | --- | --- |
| ENT_MANY | 0.907 | 0.935 | 0.588 | 1.000 |
| ENT_FEW | 0.844 | 0.918 | 0.356 | 1.000 |
| ENT_ALL | 0.922 | 0.952 | 0.395 | 1.000 |
| STATE_MANY | 0.826 | — | 0.848 | 0.979 |


上述全Q关系与联合选择不是给定-C选择的替代指标；STATE对照去除controller身份但保留state和几何。

MANY−FEW 的Q MSE差为 -0.032，配对CI [-0.057, -0.006]；选择成功差为0，CI[0,0]。因此几何覆盖改善了该面板的Q估计，不能据此宣称选择泛化受益。ALL使S×η相关提高到0.662，但其matched−mismatch为0。显式参数、机制和ranking对照都未给出稳定的state选择增益。

EXTRAP仅训练.41/.47/.53，按预注册在.59/.62的新state测试：S×η r=0.817，skill=0.748，给定-C选择成功=1.000。这一结果仍处于几乎普遍成功的η区域。

原始数据总计14,336条（13,312复用+1,024补seed），{'timeout': 2920, 'success': 10824, 'deadlock': 592}，无数值未知、无碰撞。训练seed17/23/41，ENT6000步，MLP3000步，固定最终参数，无test选checkpoint。`models/legacy_diagnostic/`有39个拟合。

### C3. Few-shot target calibration

目标校准与最终测试state完全分离；校准只读目标seed0/1。prior+条件为 `clip(mean_cal Q + P(test) - mean_cal P(cal),0,1)`，与场景级加性offset写法代数相同，不能当作两个独立方法。k0是source prior / zero-shot模型。Toy width每个目标state成本32条，旧四场景每个40条；本阶段均从兼容缓存复用，新增rollout为0。

| 目标 | zero-shot模型 | k1 prior | k1 prior+条件 | k4 prior | k4 prior+条件 | k4 条件−prior / CI |
| --- | --- | --- | --- | --- | --- | --- |
| toy_width | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.000 / [0.000, 0.000] |
| toy_give_way | 0.288 | 0.359 | 0.429 | 0.506 | 0.503 | -0.003 / [-0.047, 0.027] |
| double_bottleneck | 0.125 | 0.988 | 0.990 | 0.997 | 0.999 | 0.001 / [0.000, 0.004] |
| four_way_intersection | 0.938 | 0.936 | 0.963 | 0.976 | 0.985 | 0.009 / [0.000, 0.026] |
| ring_exchange | 0.300 | 0.576 | 0.607 | 0.760 | 0.742 | -0.018 / [-0.029, -0.007] |


所有关键差值具有配对state CI，三个训练seed和20个校准子集变异另存。上表Four/Ring是成功下界；同一决策的上界也保留。历史51条未知不被删除；校准用已知结果，空单元回退source prior。涉及未知的oracle/regret按下界补全只是敏感性估计，不能声称其真实regret已完全识别。

DB与Ring的大幅提升主要来自目标场景η regime恢复。k4时Ring加入条件结构比prior差约1.81pp，配对CI排除0；Toy/DB未见可靠正的额外state选择价值。Toy width本来接近100%，不适合判断校准价值。不能把校准后的高成功率当作条件结构已迁移。

完整样本效率（k0的条件模型就是zero-shot；标签成本是部署时需购买的名义成本，本次新增成本0）：

| 目标 | k | 标签成本/目标几何 | prior success | 条件 success | 条件95% CI | 交叉拟合oracle | 条件regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toy_width | 0 | 0 | 0.992 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| toy_width | 1 | 32 | 1.000 | 0.999 | [0.997, 1.000] | 1.000 | 0.001 |
| toy_width | 2 | 64 | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| toy_width | 4 | 128 | 1.000 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| toy_give_way | 0 | 0 | 0.238 | 0.288 | [0.087, 0.538] | 0.600 | 0.312 |
| toy_give_way | 1 | 40 | 0.359 | 0.429 | [0.329, 0.542] | 0.600 | 0.171 |
| toy_give_way | 2 | 80 | 0.456 | 0.489 | [0.389, 0.603] | 0.600 | 0.111 |
| toy_give_way | 4 | 160 | 0.506 | 0.503 | [0.386, 0.626] | 0.600 | 0.098 |
| double_bottleneck | 0 | 0 | 0.375 | 0.125 | [0.063, 0.175] | 1.000 | 0.875 |
| double_bottleneck | 1 | 40 | 0.988 | 0.990 | [0.976, 1.000] | 1.000 | 0.010 |
| double_bottleneck | 2 | 80 | 0.992 | 0.995 | [0.987, 1.000] | 1.000 | 0.005 |
| double_bottleneck | 4 | 160 | 0.997 | 0.999 | [0.996, 1.000] | 1.000 | 0.001 |
| four_way_intersection | 0 | 0 | 0.975 | 0.938 | [0.875, 0.988] | 0.938 | 0.000 |
| four_way_intersection | 1 | 40 | 0.936 | 0.963 | [0.916, 0.995] | 0.938 | -0.025 |
| four_way_intersection | 2 | 80 | 0.966 | 0.978 | [0.940, 0.999] | 0.938 | -0.040 |
| four_way_intersection | 4 | 160 | 0.976 | 0.985 | [0.955, 1.000] | 0.938 | -0.048 |
| ring_exchange | 0 | 0 | 0.200 | 0.300 | [0.188, 0.400] | 0.788 | 0.487 |
| ring_exchange | 1 | 40 | 0.576 | 0.607 | [0.529, 0.684] | 0.788 | 0.181 |
| ring_exchange | 2 | 80 | 0.708 | 0.688 | [0.640, 0.734] | 0.788 | 0.100 |
| ring_exchange | 4 | 160 | 0.760 | 0.742 | [0.706, 0.773] | 0.788 | 0.046 |


![Few-shot样本效率；k0的prior为源域prior](figures/calibration.png)

### C4. Independent factor variation

地图固定16×4m，r=.16、vmax=.5、dt=.05。width=.50/.65/.80；单开口与双并行开口两种路由拓扑；N2/N4；H400/H850。物理几何×N共12配置，各16个state、C2、K4、Q2，共3,072条真实轨迹；6,144个时间预算标签是相关前缀，不能计为额外独立rollout。共享state/目标跨几何不变；N4保留N2前两个任务，新增机器人也穿越共享隔墙。

C固定为同一几何导航程序上的isotropic/anisotropic endpoint-matched FF矩阵。控制器与η在每个配置独立交叉。导航只使用单机器人路径，没有多机器人优先级；数值失败不重试。两拓扑均几何有效且有单机器人路径证据；实际成功的config-state数为 191/192，其余未被宣称几何不可解。timeout没有冒充deadlock。

实际outcome：{'timeout': 1406, 'success': 1665, 'numerical_failure': 1}。唯一数值未知在训练state6；所有最终测试state8–15完整。见`NUMERICAL_HANDLING.md`、原JSON和全轨迹NPZ。训练只对已观测结果写Bernoulli似然；原数据仍保留unknown。总体因素统计给出三种单调时间补全敏感性范围。

| 独立因素 | 末水平−首水平成功差 | state bootstrap 95% CI |
| --- | --- | --- |
| topology | 0.223 | [0.141, 0.299] |
| N | -0.403 | [-0.449, -0.359] |
| width | 0.141 | [0.106, 0.178] |
| horizon | 0.172 | [0.137, 0.205] |


WIDTH=.50/.80→.65；EXTRAP=.50/.65→.80；COMBINATION留出(双开口,N4,.65)；FAMILY=单开口→双开口，REVERSE相反；COUNT=N2→N4；TIME=H400→H850。全部最终评估使用新state。其他因素在训练中独立覆盖，不能用“配置数量大于因素数量”解释结果。

### C5. Explicit mechanism coordinates 与跨拓扑验证

原始参数输入width/N/H/开口数/构造内孔数。机制输入：净开口/直径比、忽略互动的欧氏时间余量上界、单机器人可行路径绕行比、内部障碍分量数、开口数、N/开口数、初始到门距离离散度、最小墙间隙。全部是预测时可计算量，无最终标签或未来轨迹特征。

净开口比不是实测吞吐capacity；时间余量不证明联合可解；visibility图路径是可行路径证据，不是严格最短路。内部障碍数只在本构造中对应0/1孔；不是任意自由空间的通用拓扑算法。

| 留出因素 | ENT | +raw | +机制 | 仅机制 | 固定per-C η | 交叉拟合oracle | 机制−ENT / CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WIDTH | 0.758 | 0.766 | 0.766 | 0.711 | 0.711 | 0.789 | 0.008 / [-0.016, 0.031] |
| EXTRAP | 0.883 | 0.875 | 0.867 | 0.875 | 0.859 | 0.891 | -0.016 / [-0.039, 0.000] |
| COMBINATION | 0.719 | 0.719 | 0.688 | 0.594 | 0.750 | 0.719 | -0.031 / [-0.094, 0.000] |
| FAMILY | 0.797 | 0.776 | 0.766 | 0.266 | 0.766 | 0.859 | -0.031 / [-0.083, 0.021] |
| FAMILY_REVERSE | 0.688 | 0.297 | 0.375 | 0.057 | 0.625 | 0.745 | -0.312 / [-0.365, -0.260] |
| COUNT | 0.599 | 0.099 | 0.490 | 0.438 | 0.401 | 0.651 | -0.109 / [-0.156, -0.047] |
| TIME | 0.891 | 0.193 | 0.870 | 0.536 | 0.865 | 0.953 | -0.021 / [-0.115, 0.062] |


| 留出因素 | ENT Q MSE | 机制 Q MSE | ΔMSE / CI | ENT S×η r | 机制 S×η r |
| --- | --- | --- | --- | --- | --- |
| WIDTH | 0.098 | 0.104 | 0.006 / [-0.013, 0.026] | 0.197 | 0.055 |
| EXTRAP | 0.155 | 0.126 | -0.029 / [-0.054, -0.003] | 0.154 | 0.078 |
| COMBINATION | 0.119 | 0.109 | -0.010 / [-0.096, 0.067] | 0.258 | 0.161 |
| FAMILY | 0.195 | 0.362 | 0.168 / [0.039, 0.277] | 0.017 | 0.080 |
| FAMILY_REVERSE | 0.210 | 0.315 | 0.105 / [0.084, 0.127] | -0.020 | -0.049 |
| COUNT | 0.442 | 0.301 | -0.141 / [-0.205, -0.074] | -0.036 | 0.032 |
| TIME | 0.197 | 0.196 | -0.001 / [-0.053, 0.046] | 0.043 | 0.012 |


这些表使用原训练进程保存的预测；C7以原精度设置重放全部checkpoint，选择成功率相同。精度模式必须显式保留。

机制−ENT选择差CI严格为正的预设留出：无；严格为负：FAMILY_REVERSE, COUNT。逐seed、H400/H850分层、C×η/G×η/S×η、交叉拟合regret、全局固定η、state错配及time-margin消融见`factor_results.json`。共90个固定拟合，不做超参数搜索。显式坐标是否改善Q与是否改善选择分别判断。

独立因素面板的S×η能量虽非零，其交叉拟合state oracle相对目标(G,C)固定η增益仍只在−0.0052到+0.0052之间，远低于原5pp门槛。因而它也不能替代最高优先级的强选择验证。机制坐标在EXTRAP改善Q MSE约0.029、COUNT约0.141，却没有相应成功提升；反向拓扑迁移成功反而降低31.25pp。这是预测与选择应分别评价的直接证据。

#### 固定单位归一化诊断（探索性，9次拟合，0新增rollout）

标签无关的`feature_shift_audit.json`发现：源域常量坐标在目标域变化，原始参数标准化后可达到100/200/2250个源域标准差。按单独冻结的`NORMALIZATION_DIAGNOSTIC.md`，对FAMILY/COUNT/TIME使用固定物理单位缩放（仍只从源域估计均值），保留原模型与全部结果。

| 留出 | 物理单位机制模型成功 | 相对ENT差 / CI | 相对原机制模型ΔQ MSE / CI |
| --- | --- | --- | --- |
| FAMILY | 0.760 | -0.036 / [-0.063, -0.005] | -0.119 / [-0.150, -0.079] |
| COUNT | 0.479 | -0.120 / [-0.193, -0.042] | 0.034 / [-0.039, 0.093] |
| TIME | 0.885 | -0.005 / [-0.026, 0.021] | -0.017 / [-0.061, 0.030] |


FAMILY相对原机制模型的Q MSE改善0.119，说明输入缩放确实解释部分预测退化；但它相对原ENT成功仍降低3.65pp，CI严格低于0。COUNT仍低于ENT，TIME差异不明确。归一化修正不足以挽救选择泛化，不能把原失败全部归因于缩放，也不能由此排除更合适的表示。详见`normalization_results.json`。

预设移除time-margin消融：WIDTH成功从0.766变为0.773；FAMILY从0.766降至0.266，Q MSE却从0.362降至0.294。这不是time-margin的因果充分性证明：大幅变化还受到预测饱和/缩放/域外排序影响，原预测和重放结果均保留。

这些是**同一瓶颈母版、同一解析导航程序内的跨路由拓扑/任务因素迁移**。比连续宽度扫描更强，但仍不能写成原四场景跨机制或未知Flow泛化已解决。

### C6. 成本、原始证据与执行审计

新物理rollout总数7,172（2,048+1,024+1,024+4+3,072）；没有生成条件门槛失败的广初态main数据。累计逐rollout时间约4.76 CPU小时；138次拟合累计约13.20分钟（39旧面板+90独立因素+9归一化）。完整成本见`FINAL_AUDIT.json.cost`。rollout每worker1CPU、最多16并行；拟合通过Slurm、最多4GPU shards、禁用JAX预分配。GPU一度排队，资源释放后按原4-shard请求启动；降低请求的尝试未应用。Slurm accounting被集群关闭，累计时间不含全部初始化、排队和分析，也不是精确CPU/GPU利用率积分。

交付目录约200MiB磁盘空间（含原始记录、checkpoint、3,072条轨迹及推理重放；压缩备份另计）。精确文件字节数与SHA由`MANIFEST.json`记录。额外GPU重放使用1shard、2CPU，未计入138次拟合的13.20分钟；它没有新增训练或rollout。资源扩展始终受预注册上限约束，未使用扩展的16,384-label额度。

原始新记录在`pilot_revision/raw/`、`broad_pilot/raw/`、`legacy_supplement/raw/`、`prefix_audit/raw/`、`factors/raw/`；factor完整状态轨迹在`factors/traces/`。现有标签路径和SHA在`data/*index.json`。全部7,172次尝试已入库且逐条核对record hash；其中7,171条是精确可复用标签、1条是保留的数值未知，按缓存规则仍显示missing而非有效标签，没有重试。3,072条factor轨迹hash/帧数全部匹配，抽取每配置成功/timeout和未知前缀独立重算连续碰撞间隙，均与simulator记录一致。原始历史数据、报告和checkpoint未覆盖。纯Q实验按AGENTS不运行视频验收入口。

### C7. Checkpoint重放：精度设置与近并列排序

在GPU上对全部138个保存参数进行新进程推理；另对七种代表结构做CPU重放。不重训、不改变标签或原预测。原Toy拟合是`jax_enable_x64=False`；factor/normalization导入`poc.field_controller`后设置为True。其encoder中的默认dtype identity matrix使pooling升为64位，尽管参数是float32。独立重放入口最初漏设这个全局配置；修复后按模型族显式恢复，复拟合入口也设置相同模式。

匹配精度的GPU重放最大Q绝对差为0.001048；全部138个checkpoint的argmax均未改变，59个集成评估组的选择成功率差为0。以下并列给出原结果和正确重放；Q浮点值不承诺逐位一致。

| 留出 | 原ENT | 重放ENT | 原+机制 | 重放+机制 | 重放机制−ENT / CI |
| --- | --- | --- | --- | --- | --- |
| WIDTH | 0.758 | 0.758 | 0.766 | 0.766 | 0.008 / [-0.016, 0.031] |
| EXTRAP | 0.883 | 0.883 | 0.867 | 0.867 | -0.016 / [-0.039, 0.000] |
| COMBINATION | 0.719 | 0.719 | 0.688 | 0.688 | -0.031 / [-0.094, 0.000] |
| FAMILY | 0.797 | 0.797 | 0.766 | 0.766 | -0.031 / [-0.083, 0.021] |
| FAMILY_REVERSE | 0.688 | 0.688 | 0.375 | 0.375 | -0.312 / [-0.365, -0.260] |
| COUNT | 0.599 | 0.599 | 0.490 | 0.490 | -0.109 / [-0.156, -0.047] |
| TIME | 0.891 | 0.891 | 0.870 | 0.870 | -0.021 / [-0.115, 0.062] |


MANY−FEW的Q MSE差和配对CI仍分别约−0.0322、[−0.0572,−0.0063]。主要因素机制对照的结论保持，未发现正确精度GPU重放改变选择成功率。

严格Q近等容差检查（rtol2e−5、atol2e−6）并非每个模型都通过，原概率文件仍是精确复算参考。初次**不匹配精度**的重放使454个集成选择变化（原top-two间隔均<1e−4），成功率最多变化19.79pp；它说明部署不能随意改精度，**不证明原设置下存在19.79pp的编译不稳定**。这是审计代码的修正，未改变模型、标签或原选择规则。CPU跨后端的数值差异另存，不等价于原GPU设置重现。

正确记录：`CHECKPOINT_REPLAY_gpu_recorded.json`、`CHECKPOINT_REPLAY_cpu_recorded.json`、`REPLAY_SENSITIVITY_recorded.json`。不带`_recorded`的文件保留首次精度遗漏证据，解释已纠正；全部预测在`replay_predictions/`。复现精度/导入副作用是已定位并修复的技术问题，不能把它当作支持某个理论机制的证据。

## D. Hypothesis Evaluation

| 假设 | 判断 | 解释 |
| --- | --- | --- |
| 强state且有可用state×η选择空间的Toy面板 | REJECTED | 本次有限pilot均未满足联合门槛，不等于总体不存在这种环境。 |
| 强state选择结构的跨几何泛化 | INCONCLUSIVE | 缺少合格面板；不能用旧宽度高成功替代。 |
| 几何覆盖改善Q估计 | SUPPORTED | 旧Toy等预算MANY−FEW的MSE配对CI<0，局部结论。 |
| MANY比FEW提高选择成功 | REJECTED | 该面板success差0且headroom极低。 |
| 少量目标标签修复η regime | SUPPORTED | DB/Ring明显改善，校准/test严格分离。 |
| 校准迁移了真正state条件选择结构 | INCONCLUSIVE | prior常已解释改善；Ring k4条件项反而损害。 |
| 显式机制坐标改善目标迁移 | INCONCLUSIVE | 依留出方式分别见上表；探索性、未多重比较校正。 |
| 方法主要缺陷已经被排除 | REJECTED | 证据不足以支持排除，不等于已证明存在某个唯一硬伤。 |


初始pilot门槛和等预算比较来自原预注册；广初态、时间预算及后续独立因素的补充按时间冻结，保留先前失败。所有机制解释与特征比较属于预设的探索性分析，不能与最初的强封口假设混写。CI未作全实验家族多重校正。

## E. Generalization Diagnosis

**数据事实：** 原entity critic能预测部分state×η变化，扩展源几何改善Q估计；但给定controller时普遍成功η造成严重选择饱和。场景级prior校准能恢复很大一部分LOSO性能，额外条件结构不保证有益。独立交叉设计给出可直接检验的width、拓扑、N、H效应与交互，而非场景身份的一张混杂表。

此外，不匹配精度可能放大近并列η分数的排序差异；匹配精度后，全部checkpoint的选择重现。它是部署与复现约束，与原始pilot缺少可确认headroom是不同问题。

**机制解释（探索性）：** 可通行宽度、路径绕行与时间预算可能组织一部分η偏好；正面结果只支持这些可计算量在指定任务下有预测用途，不证明它们是充分或因果不变的机制坐标。负面结果也可能含有限state、表示、训练目标、OOD归一化与Q2精度问题，不能单一归因于“几何数量不够”。

**理论猜想：** 未来理论需要分别定义全Q可迁移性、argmax排序可迁移性与目标prior可校准性。Q相关高而η选择无headroom时，这三者无法由同一个指标判定。离散K4/K8标签不支持success basin体积、拓扑变化或普遍扩大结论。

## F. Limitations

1. 最高优先级的有效强选择面板仍缺失；本任务没有封口排除方法缺陷。
2. 每项关键最终比较只有8个物理test state，Q2/Q4而非Q16；置信区间反映固定配置/模型下的有限state变异，不是普遍域保证，也不是B15认证。
3. Controller两类均在训练见过；没有未知controller函数或未知Flow checkpoint泛化。Factor阶段使用新的解析几何导航，不能移植结论到原Flow四场景。
4. 只有N2/N4、两个路由拓扑、少量离散宽度和η；不能声称支持N100或任意地图。Toy width同时改变bay开口。
5. 交叉拟合oracle会因有限seed与并列而表现低于固定η；负regret不是超过真实oracle。naive oracle另存，具有乐观偏差。
6. 单机器人路径、实际联合成功和无碰撞是不同证据层。未成功配置不能自动标为几何不可达或死锁。
7. 校准20个子集来自同一个有限pool，并非20套独立目标环境；训练seed仅3个。历史Four/Ring未知导致部分regret未识别。
8. 显式特征source-only标准化放大源域常量的目标变化；固定单位诊断改善部分Q误差但没有解决选择。须同时检查`feature_shift_audit.json`和`normalization_results.json`，不能直接断言物理坐标无用。
9. Q浮点预测不是逐位可复现；匹配精度GPU重放的所有η选择已验证相同。更改精度/后端须重验排序；未重放历史四场景源checkpoint。

## G. Reproducibility

工作目录：`/home/zhihan/research/Basin_C1`。解释器：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。`environment_requirements.txt`固定包版本；`source_implementation_audit.json`包含实际代码/冻结Flow hash；初始Git工作树在`git_status_baseline.txt`。代码/配置均有快照；Git未覆盖的原始资产仍保留，Git commit本身不构成数据备份。

绘图使用已有`/home/zhihan/venvs/gpu/bin/python`（matplotlib3.10.9），包版本见`plot_environment_requirements.txt`，未修改共享环境。Critic复拟合/推理必须显式匹配x64模式，见C7与`RUNBOOK.md`。

预注册SHA256：`d62ba45f534fc3950b365d6f4fd00c916c1a80988ad7497fc72400a0c233c880`，UTC `2026-10-11T05:30:08.625959+00:00`。所有补充的哈希/时间收据在`*_receipt.json`；没有覆盖原预注册。执行偏差/技术修正见`EXECUTION_NOTES.md`。

```bash
cd /home/zhihan/research/Basin_C1
C1PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
# 仅重算已完成数据；不会重训或新增rollout
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.pilot
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.pilot_revision_analysis
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.broad_pilot_analysis
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.horizon_pilot
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.main_analysis evaluate --legacy
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.calibration
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.factor_analysis evaluate
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.test_contracts
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.final_audit
/home/zhihan/venvs/gpu/bin/python -m diagnostics.controller_state_geometry_transfer_v1.figures
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.report
```

新rollout必须先相应`study design`、`shared_rollout_db.plan`，再提交本目录Slurm脚本；工人只执行preflight缺口且保留已完成文件，最后merge。`RUNBOOK.md`给出全部入口与如何在新输出目录重新拟合。不要绕过 `broad_main_study.py` 的失败门槛守卫。

机器主结果：`results.json`；精细指标：`legacy_diagnostic_results.json`、`calibration_results.json`、`factor_results.json`；实际case/协议/未来seed在各stage目录。完整性与哈希索引见`FINAL_AUDIT.json`、`MANIFEST.json`。静态可导出图在`figures/`（PNG/PDF）。
