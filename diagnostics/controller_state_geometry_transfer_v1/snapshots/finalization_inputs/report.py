"""Build standalone report/handoff/results from audited, completed artifacts."""
from pathlib import Path
import json,datetime
import numpy as np
from .main_analysis import write
HERE=Path(__file__).resolve().parent
read=lambda n:json.loads((HERE/n).read_text())
def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def ci(a,scale=1):return '['+', '.join(fmt(x*scale) for x in a)+']' if a else '—'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def main():
 legacy=read('legacy_diagnostic_results.json');cal=read('calibration_results.json');factor=read('factor_results.json');audit=read('FINAL_AUDIT.json');receipt=read('preregistration_receipt.json');hist=read('history_audit.json');hp=read('horizon_pilot_results.json');pilots=[read('pilot_results.json'),read('pilot_revision_results.json'),read('broad_pilot_results.json'),hp['results']['400'],hp['results']['200']]
 assert audit['passed'] and read('phase2_complete.json')['completed']
 pilot_names=['旧自然初态 / H850','加深 bay / H850','广初态 / H850','广初态 / H400','广初态 / H200'];pt=table(['面板','state 方差占比','S×η 无偏能量','交叉拟合 η 自适应增益','95% CI','通过门槛'],[[n,fmt(p['state_fraction']),fmt(p['state_eta_energy'],5),fmt(p['oracle_adaptive_gain']),ci(p['oracle_adaptive_gain_ci95']),'否'] for n,p in zip(pilot_names,pilots)])
 names=['ENT_MANY','ENT_FEW','ENT_FEW_x2','ENT_ALL','STATE_MANY','MLP_MANY','ENT_raw_MANY','ENT_mech_MANY','RANK_MANY'];lr=[]
 for n in names:
  r=legacy['results'][n]['interpolation']['ensemble'];lr.append([n,fmt(r['q_mse']),fmt(r['state_eta']['corr']),fmt(r['state_eta']['unbiased_skill']),fmt(r['success']),fmt(r['matched_minus_mismatched']),ci(r['matched_minus_mismatched_ci95'])])
 lt=table(['模型/预算','Q MSE','S×η r','S×η skill','给定 C 选η成功','matched−mismatch','95% CI'],lr)
 ct=table(['目标','zero-shot模型','k1 prior','k1 prior+条件','k4 prior','k4 prior+条件','k4 条件−prior / CI'],[[n,fmt(v['results']['ensemble']['0']['methods']['zero_shot']['success']['mean']),fmt(v['results']['ensemble']['1']['methods']['prior_only']['success']['mean']),fmt(v['results']['ensemble']['1']['methods']['prior_conditional']['success']['mean']),fmt(v['results']['ensemble']['4']['methods']['prior_only']['success']['mean']),fmt(v['results']['ensemble']['4']['methods']['prior_conditional']['success']['mean']),fmt(v['results']['ensemble']['4']['conditional_minus_prior']['mean'])+' / '+ci(v['results']['ensemble']['4']['conditional_minus_prior']['state_ci95'])] for n,v in cal.items()])
 regs=['WIDTH','EXTRAP','COMBINATION','FAMILY','FAMILY_REVERSE','COUNT','TIME'];fr=[]
 for reg in regs:
  vals=[factor['results'][f'{arm}_{reg}']['all']['ensemble'] for arm in ['ENT','ENT_raw','ENT_mech','MECH']];base=vals[0];cmp=factor['paired_feature_comparisons'][f'ENT_mech-ENT:{reg}:all'];fr.append([reg,*[fmt(v['success']) for v in vals],fmt(base['per_controller_fixed']),fmt(base['crossfit_oracle']),fmt(cmp['mean'])+' / '+ci(cmp['ci95'])])
 ft=table(['留出因素','ENT','+raw','+机制','仅机制','固定per-C η','交叉拟合oracle','机制−ENT / CI'],fr)
 fq=table(['留出因素','ENT Q MSE','机制 Q MSE','ΔMSE / CI','ENT S×η r','机制 S×η r'],[[reg,fmt(factor['results'][f'ENT_{reg}']['all']['ensemble']['q_mse']),fmt(factor['results'][f'ENT_mech_{reg}']['all']['ensemble']['q_mse']),fmt(factor['paired_feature_comparisons'][f'ENT_mech-ENT:{reg}:all']['delta_q_mse'])+' / '+ci(factor['paired_feature_comparisons'][f'ENT_mech-ENT:{reg}:all']['q_mse_ci95']),fmt(factor['results'][f'ENT_{reg}']['all']['ensemble']['state_eta']['corr']),fmt(factor['results'][f'ENT_mech_{reg}']['all']['ensemble']['state_eta']['corr'])] for reg in regs])
 ef=table(['独立因素','末水平−首水平成功差','state bootstrap 95% CI'],[[k,fmt(v['last_minus_first_mean']),ci(v['state_ci95'])] for k,v in factor['factor_effects'].items()])
 manyfew=legacy['paired_comparisons']['ENT_MANY-ENT_FEW:interpolation'];extra=legacy['results']['ENT_EXTRAP']['extrapolation']['ensemble']
 positive=[r for r in regs if factor['paired_feature_comparisons'][f'ENT_mech-ENT:{r}:all']['ci95'][0]>0];negative=[r for r in regs if factor['paired_feature_comparisons'][f'ENT_mech-ENT:{r}:all']['ci95'][1]<0]
 mechanisms='PARTIALLY SUPPORTED' if positive else 'INCONCLUSIVE'
 h={'H0_strong_state_selection_panel':'REJECTED','H1_strong_state_selection_transfer':'INCONCLUSIVE','H2_MANY_selection_benefit_in_legacy_panel':'REJECTED','H2_Q_geometry_coverage_benefit':'SUPPORTED' if manyfew['q_mse_ci95'][1]<0 else 'INCONCLUSIVE','H3_target_regime_calibration':'SUPPORTED','H3_conditional_selection_gain':'INCONCLUSIVE','H4_explicit_coordinates_transfer':mechanisms,'all_major_method_defects_excluded':'REJECTED'}
 overall={'schema':'controller_state_geometry_transfer_v1_results','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'delivery_status':'bounded_experiments_completed_with_primary_scientific_gap','original_strong_state_selection_goal_met':False,'strong_panel_gate_passed':False,'hypothesis_evaluation':h,'preregistration':receipt,'pilot_results':dict(zip(pilot_names,pilots)),'legacy_transfer':legacy,'few_shot_calibration':cal,'independent_factors_and_mechanisms':factor,'audit':audit,'history_audit':hist,'cost':audit['cost'],'phase_status':{'1':'bounded pilots completed; strong premise FAILED; low-headroom diagnostic transfer completed','2':'completed on compatible legacy data; no qualifying strong panel','3':'completed crossed width/topology/N/horizon panel','4':'completed feature comparisons and time-margin ablation','5':'completed audit/report/handoff'}}
 write(HERE/'results.json',overall)
 text=f'''# Controller-conditioned Generalization：state、几何与机制验证

最终生成 UTC：{overall['created_utc']}。代码基线：`{receipt['git_head']}`。

**交付状态：完成有限实验、诊断和下游验证，但最高优先级的强 state-dependent η selection 封口目标尚未达成。** 五个预先限定的面板/时间预算均未通过“state 信号 + 可靠选择收益”的联合门槛；不能把本报告写成方法主要缺陷已被排除。条件成立才启动的 `broad_main` 没有运行。所有未通过、数值未知和执行调整均保留。

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

{pt}

旧初态pilot复用1,024条；加深bay新增2,048条(Q4)；广初态新增1,024条(Q2)，H400/H200复用其成功时间。另4条原生H400执行用于前缀复现，均匹配。广初态H850的state方差已达35.2%，说明确有state-dependent Q；其选择门槛仍未通过。H200平均成功仅1.95%，没有作为正式研究环境。

这些结果**不证明总体真实oracle没有headroom**。Q2/Q4的交叉拟合oracle是有噪声的选择器，不是数学上界；argmax并列与有限未来seed会压低收益。这里能说的是预注册的可评估性门槛未满足，不能排除更高精度下的排序信息。完整15项析因、CI与seed记录见各`*_pilot_results.json`、`pilot_results.json`及`data/*pilot*.npz`。

### C2. 旧 Toy 面板：严格分离state和几何的有限迁移诊断

Pilot state0–7；训练8–23；最终测试24–31。训练宽度.41/.47/.53/.59，留出内插.44/.50/.56。MANY=4宽度×8state，FEW=2端点宽度×16state；两者同一16个不同state、各1,024标签。FEW×2加seed2/3，共2,048；ALL=4宽度×16state×Q2，共2,048。测试Q4，用seed0/1选交叉拟合oracle，seed2/3评估模型实际闭环结果。

以下是三训练seed集成；每个seed与所有CI在`legacy_diagnostic_results.json`。成功率是**给定C只选η**，不是联合选择。

{lt}

MANY−FEW 的Q MSE差为 {fmt(manyfew['delta_q_mse'])}，配对CI {ci(manyfew['q_mse_ci95'])}；选择成功差为0，CI[0,0]。因此几何覆盖改善了该面板的Q估计，不能据此宣称选择泛化受益。ALL使S×η相关提高到0.662，但其matched−mismatch为0。显式参数、机制和ranking对照都未给出稳定的state选择增益。

EXTRAP仅训练.41/.47/.53，按预注册在.59/.62的新state测试：S×η r={fmt(extra['state_eta']['corr'])}，skill={fmt(extra['state_eta']['unbiased_skill'])}，给定-C选择成功={fmt(extra['success'])}。这一结果仍处于几乎普遍成功的η区域。

原始数据总计14,336条（13,312复用+1,024补seed），{legacy['counts']}，无数值未知、无碰撞。训练seed17/23/41，ENT6000步，MLP3000步，固定最终参数，无test选checkpoint。`models/legacy_diagnostic/`有39个拟合。

### C3. Few-shot target calibration

目标校准与最终测试state完全分离；校准只读目标seed0/1。prior+条件为 `clip(mean_cal Q + P(test) - mean_cal P(cal),0,1)`，与场景级加性offset写法代数相同，不能当作两个独立方法。k0是source prior / zero-shot模型。Toy width每个目标state成本32条，旧四场景每个40条；本阶段均从兼容缓存复用，新增rollout为0。

{ct}

所有关键差值具有配对state CI，三个训练seed和20个校准子集变异另存。上表Four/Ring是成功下界；同一决策的上界也保留。历史51条未知不被删除；校准用已知结果，空单元回退source prior。涉及未知的oracle/regret按下界补全只是敏感性估计，不能声称其真实regret已完全识别。

DB与Ring的大幅提升主要来自目标场景η regime恢复。k4时Ring加入条件结构比prior差约1.81pp，配对CI排除0；Toy/DB未见可靠正的额外state选择价值。Toy width本来接近100%，不适合判断校准价值。不能把校准后的高成功率当作条件结构已迁移。

### C4. Independent factor variation

地图固定16×4m，r=.16、vmax=.5、dt=.05。width=.50/.65/.80；单开口与双并行开口两种路由拓扑；N2/N4；H400/H850。物理几何×N共12配置，各16个state、C2、K4、Q2，共3,072条真实轨迹；6,144个时间预算标签是相关前缀，不能计为额外独立rollout。共享state/目标跨几何不变；N4保留N2前两个任务，新增机器人也穿越共享隔墙。

C固定为同一几何导航程序上的isotropic/anisotropic endpoint-matched FF矩阵。控制器与η在每个配置独立交叉。导航只使用单机器人路径，没有多机器人优先级；数值失败不重试。两拓扑均几何有效且有单机器人路径证据；实际成功的config-state数为 {factor['joint_witnesses']['config_state_success_any_eta_controller_seed']}/{factor['joint_witnesses']['total_config_states']}，其余未被宣称几何不可解。timeout没有冒充deadlock。

实际outcome：{factor['outcomes']}。唯一数值未知在训练state6；所有最终测试state8–15完整。见`NUMERICAL_HANDLING.md`、原JSON和全轨迹NPZ。训练只对已观测结果写Bernoulli似然；原数据仍保留unknown。总体因素统计给出三种单调时间补全敏感性范围。

{ef}

WIDTH=.50/.80→.65；EXTRAP=.50/.65→.80；COMBINATION留出(双开口,N4,.65)；FAMILY=单开口→双开口，REVERSE相反；COUNT=N2→N4；TIME=H400→H850。全部最终评估使用新state。其他因素在训练中独立覆盖，不能用“配置数量大于因素数量”解释结果。

### C5. Explicit mechanism coordinates 与跨拓扑验证

原始参数输入width/N/H/开口数/构造内孔数。机制输入：净开口/直径比、忽略互动的欧氏时间余量上界、单机器人可行路径绕行比、内部障碍分量数、开口数、N/开口数、初始到门距离离散度、最小墙间隙。全部是预测时可计算量，无最终标签或未来轨迹特征。

净开口比不是实测吞吐capacity；时间余量不证明联合可解；visibility图路径是可行路径证据，不是严格最短路。内部障碍数只在本构造中对应0/1孔；不是任意自由空间的通用拓扑算法。

{ft}

{fq}

机制−ENT选择差CI严格为正的预设留出：{', '.join(positive) or '无'}；严格为负：{', '.join(negative) or '无'}。逐seed、H400/H850分层、C×η/G×η/S×η、交叉拟合regret、全局固定η、state错配及time-margin消融见`factor_results.json`。共90个固定拟合，不做超参数搜索。显式坐标是否改善Q与是否改善选择分别判断。

这些是**同一瓶颈母版、同一解析导航程序内的跨路由拓扑/任务因素迁移**。比连续宽度扫描更强，但仍不能写成原四场景跨机制或未知Flow泛化已解决。

### C6. 成本、原始证据与执行审计

新物理rollout总数7,172（2,048+1,024+1,024+4+3,072）；没有生成条件门槛失败的广初态main数据。完整成本见`FINAL_AUDIT.json.cost`：{audit['cost']}。rollout每worker1CPU、最多16并行；拟合通过Slurm、最多4GPU shards、禁用JAX预分配。GPU一度排队，资源释放后按原4-shard请求启动；降低请求的尝试未应用。Slurm accounting被集群关闭，累计时间来自逐rollout/逐fit记录，不能当作精确GPU利用率积分。

原始新记录在`pilot_revision/raw/`、`broad_pilot/raw/`、`legacy_supplement/raw/`、`prefix_audit/raw/`、`factors/raw/`；factor完整状态轨迹在`factors/traces/`。现有标签路径和SHA在`data/*index.json`。postflight确认所有7,172条已精确入库，无缺失/歧义；原始历史数据、报告和checkpoint未覆盖。纯Q实验按AGENTS不运行视频验收入口。

## D. Hypothesis Evaluation

{table(['假设','判断','解释'],[['强state且有可用state×η选择空间的Toy面板','REJECTED','本次有限pilot均未满足联合门槛，不等于总体不存在这种环境。'],['强state选择结构的跨几何泛化','INCONCLUSIVE','缺少合格面板；不能用旧宽度高成功替代。'],['几何覆盖改善Q估计','SUPPORTED','旧Toy等预算MANY−FEW的MSE配对CI<0，局部结论。'],['MANY比FEW提高选择成功','REJECTED','该面板success差0且headroom极低。'],['少量目标标签修复η regime','SUPPORTED','DB/Ring明显改善，校准/test严格分离。'],['校准迁移了真正state条件选择结构','INCONCLUSIVE','prior常已解释改善；Ring k4条件项反而损害。'],['显式机制坐标改善目标迁移',mechanisms,'依留出方式分别见上表；探索性、未多重比较校正。'],['方法主要缺陷已经被排除','REJECTED','证据不足以支持排除，不等于已证明存在某个唯一硬伤。']])}

初始pilot门槛和等预算比较来自原预注册；广初态、时间预算及后续独立因素的补充按时间冻结，保留先前失败。所有机制解释与特征比较属于预设的探索性分析，不能与最初的强封口假设混写。CI未作全实验家族多重校正。

## E. Generalization Diagnosis

**数据事实：** 原entity critic能预测部分state×η变化，扩展源几何改善Q估计；但给定controller时普遍成功η造成严重选择饱和。场景级prior校准能恢复很大一部分LOSO性能，额外条件结构不保证有益。独立交叉设计给出可直接检验的width、拓扑、N、H效应与交互，而非场景身份的一张混杂表。

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
8. 显式特征source-only标准化对源域常量在目标域变化时可能放大输入；在解释跨拓扑/COUNT/TIME失败时须同时检查`feature_shift_audit.json`，不能直接断言物理坐标无用。

## G. Reproducibility

工作目录：`/home/zhihan/research/Basin_C1`。解释器：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。`environment_requirements.txt`固定包版本；`source_implementation_audit.json`包含实际代码/冻结Flow hash；初始Git工作树在`git_status_baseline.txt`。代码/配置均有快照；Git未覆盖的原始资产仍保留，Git commit本身不构成数据备份。

预注册SHA256：`{receipt['sha256']}`，UTC `{receipt['utc']}`。所有补充的哈希/时间收据在`*_receipt.json`；没有覆盖原预注册。执行偏差/技术修正见`EXECUTION_NOTES.md`。

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
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.figures
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.report
```

新rollout必须先相应`study design`、`shared_rollout_db.plan`，再提交本目录Slurm脚本；工人只执行preflight缺口且保留已完成文件，最后merge。`RUNBOOK.md`给出全部入口与如何在新输出目录重新拟合。不要绕过 `broad_main_study.py` 的失败门槛守卫。

机器主结果：`results.json`；精细指标：`legacy_diagnostic_results.json`、`calibration_results.json`、`factor_results.json`；实际case/协议/未来seed在各stage目录。完整性与哈希索引见`FINAL_AUDIT.json`、`MANIFEST.json`。静态可导出图在`figures/`（PNG/PDF）。
'''
 (HERE/'REPORT.md').write_text(text)
 handoff=f'''# Independent Research Handoff — Controller-conditioned Generalization

**先读状态：原任务最重要的强state-dependent η选择封口目标未达成。** 已完成有限预注册pilot、失败诊断、等预算旧面板迁移、few-shot校准、独立因素与机制坐标实验；不能将结果表述为方法主要缺陷已被排除。下面不依赖原对话。

## 1. 问题与背景

项目在每步执行冻结参考动作、OrthoFlow3三维η修正与硬安全FF投影；一个episode内η固定。Critic预测无碰撞且按时全员到达的Q(s,C,η)。旧四场景LOSO把geometry、Flow、N、horizon等混在一起；旧DB连续宽度族可以迁移(C,η)关系，但state信号很弱。现在需要区分预测Q、给定C选η、校准场景η偏好及success basin泛化。

## 2. 实際完成及关键实现

结果根目录：`{HERE}`。复用旧Toy width13,312条；新增7,172次真实尝试。旧Toy新拟合39个、独立因素新拟合90个，均3训练seed；四场景校准复用旧LOSO预测。完整脚本、参数、每seed checkpoint/预测、raw、factor完整轨迹、数据索引、预注册及哈希均在目录内。

Pilot依次测试自然初态、加深bay、广初态、固定H400/H200；宽度、C、η独立控制，安全/成功标准未改。广初态Q有35.2%方差涉及state，但各pilot交叉拟合自适应增益均未达到5pp且CI>0的门槛。广初态main没有运行。

独立因素面板固定地图/半径/速度，交叉width(.50/.65/.80)、单/双并行开口拓扑、N2/N4、H400/H850，保持跨几何同物理state、跨N嵌套任务。使用同一N通用解析导航参考和两种FF矩阵，不能称为Flow跨N迁移。

## 3. 最重要定量结果

{pt}

旧Toy内插：ENT MANY的S×η相关0.391、无偏skill−0.084；ALL提高到0.662、0.217。MANY−FEW Q MSE={fmt(manyfew['delta_q_mse'])},CI{ci(manyfew['q_mse_ci95'])}；两者选择成功均1.0，几乎无state选择空间。高Q相关或100%选择均不能封口。

{ct}

{ft}

Factor outcome={factor['outcomes']}；唯一未知在训练集，最终测试0未知、0碰撞。四场景旧Flow数据实际7,680条、51未知，不是12,800条；后者只是张量容量。见`history_audit.json`。

## 4. 假设、已排除与未排除

- 支持：指定旧Toy预算下几何覆盖改善Q估计；少量目标标签可恢复DB/Ring场景级η regime。
- 不支持：MANY在饱和面板中有选择优势；加深bay足以产生有意义state选择空间。
- 未定：真正依赖state的η排序能否跨几何泛化；高精度pilot是否揭示Q2/Q4 oracle估计遗漏的headroom；主要方法缺陷能否被排除。
- 显式坐标的效果依留出类型不同。上表给出逐项配对结果；跨路由拓扑仍是同一母版/解析程序，不是原四场景跨机制证明。
- 精确controller身份与物理state本身已知，不是本面板的信息缺口。但没有测试未知controller函数、未知Flow或大N。
- 尚未排除：sample support、表示/归纳偏置、训练目标、OOD特征缩放、有限future-seed精度、场景×Flow嵌套。不要复述旧报告中无范围限制的“均已排除”。

## 5. 对Q、η选择及success basin的影响

Q预测可改善而argmax没有额外价值；场景prior可恢复成功而不迁移state条件结构。理论应分别形式化这三种问题。当前标签仅K4/K8离散候选、Q2/Q4，不能推断连续success basin体积/拓扑、B15鲁棒性或普遍enlargement。交叉拟合oracle是有限标签选择参考，不是真实oracle上界；负regret不能解释为超oracle。

## 6. 最重要未解决问题

构造并独立确认有可靠state×η选择headroom的面板，再冻结全新state/geometry测试。优先复用已有raw预测确认是候选饱和还是Q估计精度不足；本次不继续无限搜索。其次是把通过的跨拓扑坐标与真实Flow程序变化解耦，并确认归一化/常量维度的OOD放大是否解释失败。本报告没有完成这两个封口。

## 7. 可直接使用的证据与入口

- `REPORT.md`：完整A–G报告、每阶段解释和限制；`results.json`：完整机器结果。
- `PREREGISTRATION.md`及四类addendum、`*_receipt.json`：原始门槛、时间、hash、执行调整。
- `data/legacy_diagnostic.npz`、`data/factors.npz`：标签/unknown/实体/显式特征；`data/*index.json`：原始来源和SHA。
- `models/legacy_diagnostic/`、`models/factors/`：每seed `.msgpack`、`.npz`预测和训练索引、`.json`超参/loss/时间。
- `calibration_results.json`：每seed、k、子集、paired CI和未知上下界；原预测在`data/calibration_prediction_index.json`列出。
- `factors/{{protocol,cases,geometry_validation}}.json`、`factors/raw/`、`factors/traces/`：独立因素真实证据。
- `pilot_revision/`、`broad_pilot/`：失败pilot也保留；`broad_main_study.py`仅恢复入口，**未运行且门槛未通过**。
- `FINAL_AUDIT.json`、`MANIFEST.json`、`source_implementation_audit.json`、`NUMERICAL_HANDLING.md`、`EXECUTION_NOTES.md`：审计、成本、数值失败、代码与数据版本。
- `RUNBOOK.md`：复算、恢复、cache/merge命令；`figures/`：可共享PNG/PDF。

工作根 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。主Git基线`{receipt['git_head']}`，但历史diagnostics/PoC不少未纳入Git；以SHA快照和数据索引为准。未修改或删除历史证据。新增实验是纯Q研究，依AGENTS豁免视频交付。
'''
 (HERE/'HANDOFF.md').write_text(handoff);print('REPORT.md, HANDOFF.md, results.json written')
if __name__=='__main__':main()
