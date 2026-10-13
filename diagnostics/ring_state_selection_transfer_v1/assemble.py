"""Assemble final independent Chinese reports only after applicable stages finish."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
def write(f,v):(P/f).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}pp'
def interval(x):return f'[{100*x[0]:.2f}, {100*x[1]:.2f}]pp'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)
def main():
 n=read('N1_RESULTS.json');rep=read('REPRODUCTION.json');sim=read('HISTORICAL_SIMULTANEOUS_REPRODUCTION.json');c0=read('C0_RESULTS.json');focus=read('C0_FF_TT_FOCUS.json');energy=read('C0_CENSOR_ENERGY.json');legacy=read('LEGACY_FLOW_FOCUS.json');c1=read('c1/RESULTS.json');rel=read('c1/RELATIONSHIP.json');decision=read('C2_DECISION.json');aux=read('N2_RESULTS.json');shift=read('N3_SUMMARY.json');basis=read('EFFECTIVE_BASIS_AUDIT.json');src=read('SOURCE_VALIDATION.json');assert src['status']=='PASS';valid=read('FINAL_VALIDATION.json');assert valid['status']=='PASS'
 fallback=read('C2_FALLBACK_RESULTS.json') if (P/'C2_FALLBACK_RESULTS.json').exists() else None
 newfits=sum(3 for f in (P/'fits').glob('*/complete.json') if not read(str(f.relative_to(P)))['reused'])+(21 if fallback else 0)
 results=dict(schema='ring_controller_state_selection_transfer_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Controller-conditioned generalization; N1 and N2/N3 are supporting evidence',status='COMPLETED APPLICABLE STAGES; formal stages follow explicit gates',N1=n,C0=dict(inventory='C0_RESULTS.json',focus=focus,censor_energy=energy,legacy_real_Flow=legacy),C1=c1,C1_all_fixed_sensitivity=read('c1/ALL_FIXED_SENSITIVITY.json'),C2=decision,C2_fallback=fallback,C3=read('C3_ASSET_AUDIT.json'),N2=aux,N3=shift,effective_basis='EFFECTIVE_BASIS_AUDIT.json',reproduction=rep,cost=dict(existing_unique_rollouts_reused=src['all_reused_unique'],new_rollouts=c1['new_rollouts'],new_network_fits=newfits,reused_N1_checkpoints=30,new_rollout_episode_cpu_hours=c1['cpu_episode_hours']),data_index='DATA_INDEX.json',reuse_index='REUSED_ROLLOUT_INDEX.json',validation='FINAL_VALIDATION.json',preregistration_receipts=['REGISTRATION.json','AMENDMENT_RECEIPT.json','C1_TIE_RECEIPT.json','CHECKPOINT_DIAGNOSTIC_RECEIPT.json','C2_FALLBACK_RECEIPT.json','C1_FAMILY_GATE_RECEIPT.json'])
 write('results.json',results)
 curve=table(['训练 state','原标量 Q：成功率 / gain CI95','16-head MLP：成功率 / gain CI95','kNN：成功率 / gain CI95'],[[sz]+[pct(next(r for r in n['summary'] if r['n']==sz and r['arm']==a)['success_lower'])+' / '+interval(next(r for r in n['summary'] if r['n']==sz and r['arm']==a)['gain_ci95']) for a in ['scalar','heads','knn']] for sz in [32,64,128,256,512]])
 c1table=table(['Controller','固定 η (两半)','固定成功率','state oracle 成功率','gain / CI95','未知数'],[[r['controller'],str(r['discovery_fixed_eta']),pct(r['fixed_crossfit_success']),pct(r['oracle_crossfit_success_lower']),pp(r['gain'])+' / '+interval(r['gain_ci95']),r['unknown']] for r in c1['primary']])
 c2text='正式 C2 状态：'+decision['status']+'。'+decision['reason']+'\n'
 if fallback:
  rr=[]
  for m in fallback['records']:
   for r in m['rows']:
    if r['split']=='joint_unseen_states':rr.append([m['arm'],r['heldout_kind'],f"{r['matched']['q_mse']:.4f}",pct(r['matched']['success_lower']),pp(r['matched']['gain']),pp(r['matched_minus_mismatched_success'])])
  c2text+='\nQ-only fallback 已实际运行：每个源 Controller 16 个 TRAIN state、8 个 VAL state、8 个完全未用于拟合/选择的 TEST state；源 κ=-1,0,0.5，留出 -0.5（内插）及1（外推）。同一组32历史 state 全部属于 DEVELOPMENT。训练/验证种子0–3，测试种子4–7；另外单独保存16个已见 state 上的 held-controller 结果。每种表示3个优化种子，source-VAL NLL选择 checkpoint。\n\n'+table(['表示','留出类型','Q MSE','成功率','相对目标 discovery 固定 η','matched−mismatched'],rr)+'\n\nR0无 Controller；R1仅三个源身份 one-hot，未知身份为全零；R2真实κ；R3项目compact64/agent；R4实际投影FF动作对η的有限差分基底（η0及η13）；R4_action追加η0动作；RB为未投影解析基底消融。全部输入补到相同维度、同一64×64网络和训练预算，归一化仅用源训练数据。错配时把-0.5的descriptor换为+0.5，把+1换为-1，physical state与η不变。R0预测必须完全不变；R1的未知/已知身份对照不能解释成部署收益。\n\n另有配对Q风险分析：R4仅基底的平均MSE=0.1754，R0=0.1793，差值CI跨0；R4_action为0.1448，相对R0改善0.0345（未多重修正CI[0.0073,0.0679]），但matched相对mismatched的Q改善CI仍跨0，不能确认收益来自正确Controller信息。两留出Controller的这8个test state均有固定η达到100%观测成功率；选择端点因此没有可测headroom。证据C2_PAIRED_Q_DIAGNOSTIC.json。\n\n这些选择指标仅为描述性诊断：8个独立物理 test state、Q4精度、16个训练 state均不足以代替正式C2。目标 discovery 固定η本身使用了目标的0–3 seed标签，只是参照，绝不是zero-shot输入。没有使用目标标签训练、归一化或选checkpoint。低Q MSE或正相关不等于selection泛化成立。'
 shiftrows=[[r['source_region'],r['arm'],r.get('mode'),pct(r['success']),pp(r['gain']),interval(r['gain_ci95'])] for r in shift if 'mode' in r]
 basisrows=[r for r in basis['rows'] if r['controller']!='flow'];errmax=max(r['unprojected_affine_error'] for r in basisrows);zero=read('ZERO_ETA_PROJECTED_ACTION_AUDIT.json');zrange=[min(r['zero_eta_projected_action_max_pair_difference'] for r in zero['rows']),max(r['zero_eta_projected_action_max_pair_difference'] for r in zero['rows'])]
 reg=read('REGISTRATION.json');c1gate=c1['gate'];report=f'''# Ring State-conditioned η Selection and Controller Transfer v1

最终研究报告；生成于 {results['utc']}。研究主线以用户目标对齐补丁及 `docs/next_experiments_ring_state_selection_20261011.md` §3 为准。**本研究已经执行数据审计、学习曲线、Controller intervention pilot及适用的后续诊断；没有把未执行的正式泛化阶段写成成功。**

## A. Research Overview

目标是识别 Controller 如何改变闭环 Q(s,C,η)，区分 Q 整体水平、η偏好和state×Controller×η，并检验当前Controller表示对未见Controller的迁移边界。N1只为后续Controller实验建立合理样本尺度；N2/N3为辅助，不替代Controller主线。几何N4/N5按新优先级后置。

历史Ring固定native Flow v10、FF、H700的Q512结论被原始预测和标签复现：成功率{pct(n['historical_q512_recalculation']['success_lower'])}，固定η13 {pct(n['fixed13_success'])}，gain {pp(sim['adaptive_minus_hindsight_best_fixed_point'])}，原协议同时对照16个固定候选的CI {interval(sim['simultaneous_max_over_16_fixed_state_bootstrap_95'])}。这个历史确认结论保留历史身份；本轮对旧256个确认state的所有新模型与探索均为 **DEVELOPMENT**。不把旧确认重新包装成新确认。

## B. Implementation Summary

1. 先读取AGENTS、共享DB规范、Ring下一阶段文档及generalization audit，追溯 `diagnostics/state_nuisance_rootcause_v1/real_state_expansion/` 与 `heldout_adaptive_selection_v1/` 的protocol、代码、冻结checkpoint、预测和逐seed标签。发现历史Q512本身是physical23+η的64×64 SiLU标量MLP，并非entity encoder。对照采用既存physical23→16-head MLP，避免重复模型命名。
2. N1复用30个canonical-prefix checkpoint（两网络×五个规模×三seed），仅对两个新嵌套state排列补48次拟合。三个排列中的512集合相同，只计原三fit，不伪造九次独立重复。kNN及η-prior重算。checkpoint诊断只推理历史checkpoint，零新增训练。
3. C0从SQLite只读扫描Ring controller配置及标签，核对state/η/seed精确identity、horizon、dt、安全层和evaluator，发现{c0['ring_controllers']}个有对应配置的Ring profile；当前native geometry的{c0['exact_native_geometry_controllers']}个exact profile产生{len(c0['pair_panels'])}个兼容配对面板，仅2个达到所用Q16矩形规则。低seed、analytic goal endpoint和nativeFlow endpoint分别保留边界。进一步审计真实Flow旧协议面板，不能与当前FF自由初态数据混并。
4. C1只新增缺失标签，既有κ=-1/0/+1程序沿用原controller UID；κ=±0.5新增精确profile。`c1/runtime.py`复用FlowEndpointMatchedField和原FF积分/硬投影/模拟器，只扩展M(κ)。32个固定原始source state、K16、Q8、五κ加native FF anchor，共24576个请求；preflight复用3840条（含38条数值未知），新增{c1['new_rollouts']}次。worker写journal，一个merger验证后追加DB，未覆盖历史记录。
5. 静态basis/descriptor直接调用controller函数；代码禁止environment.step。8state×6controller×2η锚点完成96项局部导数/有限扰动核查；另为C2缓存32state×5controller的表示。所有特征仅用部署可得state、geometry、controller和固定查询key。
6. N2物理特征ridge及N3双向chirality切分都复用标签。优先级补丁到达时已有18次N3 scalar拟合，立即停止旧顺序并保存；后来在C1 rollout占22CPU时用剩余资源完成辅助分析，不抢占主线rollout。共72次N3拟合。
7. 全部代码写入本独立目录。历史未跟踪资产没有被修改，因此没有依赖Git恢复原始证据；DB只按规范追加新profile/rollout。纯Q任务适用AGENTS的视频豁免。

异常与修复：C0结果构造曾出现重复字典key，修复后完整重跑派生分析；N3分析曾在预测文件尚未完成时启动，文件就绪后完整重跑。CPU进程输出JAX CUDA插件初始化警告，但明确指定CPU backend，未中断rollout；真正执行异常另计unknown，绝不混为普通失败。C1标签已合并后，模块方式运行分析出现本地import路径错误；修复package导入后只重跑分析，恢复被failed依赖阻止的C2作业，没有重做任何rollout。有限差分审计未发生查询失败。所有未完成日志均保留，不作为有效结果。

## C. Experimental Results

### N1：样本效率与复现

训练512state×K16×Q4，独立VAL32×K16×Q16，DEVELOPMENT256×K16×Q16。新比较评价seed8–15，0–7用于固定η/历史discovery参照。模型fit seed17/23/41，state排列canonical及2026101101/02；标量网络5953参数，16-head为6736参数（多约13%，非严格等参数）；训练步数/优化器/候选集/划分一致。NN同AdamW lr0.0008、wd0.0001、clip5、batch128、12000步上限，62个checkpoint按三fit均值VAL NLL选择。kNN k∈1/3/5/9/17/33仅按VAL NLL选，统一physical23输入。固定η13保留在完整K16内。原32训练state的标准化沿用历史协议，新子集因此共享少量无标签source统计；不含test outcome。

{curve}

固定prior在这些配置均选择η13。原标量网络的协议内开发n*为512；16-head为128；kNN为64但增益仅约0.85pp，不能视为达到Q512水平。kNN64在保守五规模99%区间下不显著，需到128；heads128与scalar512在99%区间下仍为正。五规模99%区间和每个subset/fit结果见`N1_RESULTS.json`。CI以256个物理state为单位，先在state内平均fit/subset，不把模型seed或rollout当作独立环境证据；它不穷尽重新抽取整个训练总体的误差。

配对比较：128state下heads比scalar高4.74pp，CI[2.65,7.02]pp；但heads128比scalar512低1.77pp，CI[-3.12,-0.50]pp。heads256比scalar512低0.71pp，CI[-1.78,0.29]pp：未证实同等性能，也不能将“不显著”当作等价。`N1_PAIRED_COMPARISONS.json`保留所有对照。

有限失败诊断显示checkpoint目标也是原因之一。canonical scalar128只改用VAL selected-success选checkpoint，从89.27%变为91.72%；同规则对scalar512却降至固定prior水平88.33%。这项仅一个state子集的事后探索不替换主要n*，也不证明换目标必然修复。网络readout、优化/模型选择和有限state覆盖尚未完全分离。完整Q MSE、NLL、排序准确率、tie-aware top1、经验oracle regret与unknown上下界见机器结果；高Q相关没有被当作selection证据。

证据：`REPRODUCTION.json`（Q32/Q512预测最大差0、argmax一致）；`HISTORICAL_SIMULTANEOUS_REPRODUCTION.json`；`N1_RESULTS.json`；`CHECKPOINT_DIAGNOSTIC_RESULTS.json`；`learning_curve.png/pdf`。

### C0：Controller-dependent η selection

高精度nativeFF/TT匹配24个state×16η×16seed，共12288条，74条unknown。它们保持Flow checkpoint、geometry、state、η、seed、H700及评价一致，改变修正placement/执行程序。TT−FF平均Q的censor bounds为{interval(focus['q_TT_minus_FF_bounds'])}，配对CI包络{interval(focus['q_level_censor_ci95'])}。不能把这个候选池平均Q差与最佳η成功率差混淆。

两者在两个discovery半分的最佳固定η均为13，**fixed cross-controller regret=0**。但controller×η及state×controller×η的独立seed-half能量在保守unknown传播后仍为正：{energy['energy_censor_bounds']}，state-bootstrap包络{energy['paired_state_bootstrap_censor95']}（单位为概率平方，负能量估计没有截断）。这否定“Controller只改变常数Q level”。强discovery间隔≥0.25的严格排序反转，两个方向分别有15/18个state在独立半分仍确认，不是单纯argmax并列。

跨Controller state-oracle结果依赖tie规则：任意bank-order tie使TT matched−mismatched达71.35pp；改为discovery全局prior tie后约7.81pp、CI触及0。FF方向为8.59pp、CI[1.04,17.71]pp。这个敏感性必须保留：oracle使用discovery标签，不是可部署selector；小样本、多重比较、过去开发使用限制新确认结论。证据：`C0_FF_TT_FOCUS.json`、`C0_CENSOR_ENERGY.json`、`C0_RANK_REVERSALS.json`。

真实Flow缓存提供另一层事实：在180个相同state、两个相同η、Q16、固定committed第一步的旧协议下，v9和v7偏好不同候选。v9采用v7最佳η损失22.92pp（CI[18.30,27.57]）；反向9.24pp（CI[3.40,14.69]）。较强方向在保守多重比较敏感性下仍正。**这里η池不是本轮K16，execution不是本轮自由t0 FF；只能支持该条件Q中的真实Controller依赖，不能冒充C3。** 证据`LEGACY_FLOW_FOCUS.json`、`legacy_flow_panels/`、`FLOW_CHECKPOINT_INDEX.json`。共审计38个替代profile、68个旧协议配对面板，复用19712个唯一标签。

### C1：连续Controller family pilot

M(κ)=2.5(I+κJ)，κ∈[-1,-0.5,0,0.5,1]；nativeFF另作anchor，**不等于κ=0**。固定Flowv10和每步同key的未投影nominal endpoint，保留原pseudo-time10步、FF安全/速度投影、H700、dt0.05、K16、机器人几何和成功判据。κ同时改变有效响应角度与增益，不能仅归因于旋转角。原始32state按历史顺序固定，没有删除困难state、最强η或数值未知。

{c1table}

以上是4/4与反向4/4的平均；primary tie优先discovery全局prior。独立seed选择/评价，physicalstate bootstrap5000，未知按区间保留。Q8的7/8 coverage仅是探索性指标，**不冒充B15**；最小η覆盖数只针对至少有候选达到阈值的state，不能把未覆盖者叫几何不可行。所有η失败也只能说明候选/seed中的失败。

预注册门：极端κ之间固定η交叉遗憾≥5pp且对两个方向校正后至少一个方向CI下界>0，并至少3/5κ的state-oracle gain≥5pp、CI>0及半分均值为正。实际极端对A={c1gate['A_extreme_fixed_regret_pass']}，全20方向同时A={c1gate['A_family_simultaneous_fixed_regret_pass']}，B={c1gate['B_headroom_at_least3_pass']}，合格headroom controller={c1gate['headroom_controllers']}；整体selection benchmark合格={c1gate['selection_transfer_qualified']}。全20方向遗憾和同时区间保留在`c1/RESULTS.json`。在读取完整C1结果前，另作了晚期family-wide门澄清，保留原极端对主对照及更严格20方向同时对照；见C1_FAMILY_GATE_CLARIFICATION与时间/hash receipt，此修订没有改变最终B失败的决策。

**更强固定基线敏感性**：κ=+1相对crossfit discovery固定基线的+7.42pp，不等于已经可靠超过最强通用固定候选。用全部K16的hindsight fixed同时bootstrap参照，点估计仅+5.86pp，CI[-0.39,8.98]pp，最坏unknown CI[-0.78,8.59]pp；见c1/ALL_FIXED_SENSITIVITY.json。因此本轮没有确认新的selection benchmark。极端κ固定regret名义门为+5.08pp，但两半分别+10.16pp与0，反向regret−1.17pp且CI跨0；它不足以证明双方最优η已稳定反转。\n\n**门槛未过不等于Controller只影响Q level，也不等于不存在state交互。** `c1/RELATIONSHIP.json`保存完整Q(κ,η)曲线、独立half的controller/state/η分解；这些是辅助prediction证据，不能替代selection门。总unknown={c1['unknown_total']}，没有选择性重试或剔除。该32state面板的nativeFF固定η成功率95.31%，高于历史256state的88.33%；因此pilot资格结论只适用于这批固定source state，不能宣称整个controller family在原始state总体饱和。新增episode累计{c1['cpu_episode_hours']:.2f} CPU小时（episode计时和，不包括进程初始化），均值{c1['mean_episode_seconds']:.3f}秒。Slurm并行22CPU，剩余资源用于GPU静态查询/训练和分析；配置与job ID均保留。

### C2：Held-out Controller generalization

{c2text}

证据：`C2_DECISION.json`；已执行fallback，`C2_SPLITS.json`、`C2_STATIC_FEATURES_AUDIT.json`、`C2_FALLBACK_RESULTS.json`、`c2_fits/`保存可重放的训练/归一化/预测。历史controller模型并不满足本轮holdout（部分见过+1算子且候选仅K4），因此不能当作本轮source-only checkpoint直接复用。

### C3与辅助N2/N3、几何阶段

C3正式新Flow transfer未伪称完成：{decision['C3_status']}。数据库存在v7/v8/v9等checkpoint及大量legacy标签，但当前geometry、自由t0、FF、K16、统一seed协议下没有替代真实Flow的兼容完整面板。最小后续heldout64state×K16×Q16仅测试部分就需16384条；源Controller标签规模应以N1及C2证明的可靠尺度确定。旧checkpoint确实存在且已保存hash，不能因为索引困难重复生成legacy标签。

N2：16个部署可算物理特征（半径、角度gap、相反agent配对相位、goal antipodal偏角、径向/切向初速度、最小间距、直线距离），不使用未来轨迹。source512训练、32VAL只选8个ridge候选；按state均值中心化Q以隔离η交互。最优为一次线性ridge，heldout η-contrast相对固定prior R²={aux['heldout_contrast_R2_vs_prior']:.4f}，selection gain {pp(aux['gain_vs_eta13'])}，CI {interval(aux['gain_ci95'])}。物理关联弱，只支持可证伪的routing-chirality等候选解释，未得到足以支撑跨Controller不变性的简洁机制。

N3：以**初态goal相对起点的平均反向角偏差是否>0**固定划分，未按标签选boundary；每方向128训练state，3子集×3fit，与同规模随机训练切分配对；旧256测试按区域划分127/129。结果：

{table(['源区域','模型','训练支持','成功率','相对η13 gain','CI95'],shiftrows)}

主要shift−random对照CI均跨0；不能声称这个切分证明普遍state外推失败或成功。它只在一个较温和的chirality半空间上未观察到稳定额外损失。N2/N3均固定Controller，不能独立证明跨Controller机制。N4/N5几何实验 **NOT EXECUTED—DEFERRED BY USER PRIORITY**，没有抢占C0–C2预算。

## D. Hypothesis Evaluation

| 假设 | 判断 | 证据边界 |
|---|---|---|
| nativeRing存在可部署state-conditioned selection收益 | SUPPORTED | 历史Q512独立确认被精确复现；本轮新比较为development |
| 原协议标量Critic需较多state才取得可靠gain | SUPPORTED（协议内） | 三嵌套子集、三fit；n*=512不是总体理论下界 |
| Critic存在无法由优化/目标解释的固有结构性缺陷 | INCONCLUSIVE | heads更省样本，但checkpoint目标也显著改变表现 |
| Controller只改变Q整体水平、η排序不变 | REJECTED（所审计协议） | FF/TT独立half严格排序反转、交互能量；legacy真实Flow固定η反转 |
| 连续κ family满足预注册selection-transfer benchmark门 | {'SUPPORTED' if c1gate['selection_transfer_qualified'] else 'REJECTED（本pilot资格门）'} | 不推断连续η basin或所有state分布 |
| 未投影RB就是实际FF执行/闭环的精确表示 | REJECTED | 逐步投影改变局部导数和η=0动作 |
| R4已证明优于现有descriptor并实现真实heldFlow迁移 | INCONCLUSIVE | 静态可算/局部有效不等于终端Q充分性；正式C3未完成 |
| 一组简洁物理state量已解释可迁移选择机制 | INCONCLUSIVE | 线性contrast增益有限、selection无可靠提升 |

预注册与探索性文件的时间/hash分别保留；checkpoint目标、tie敏感性和legacy覆盖最大面板分析明确为探索性，不冒充预注册独立确认。

## E. Controller–Q Relationship and Transferability

1. **Controller对Q的影响**：在固定物理state、geometry和η下改变FF/TT执行程序，会改变候选池平均Q；真实Flow在固定committed-first-action的旧协议也改变Q。不能只用环境身份解释全部差异。
2. **Controller对η排序和success basin**：FF/TT共享最佳固定η13却存在state层严格偏好反转；因此“全局最佳η没变”不意味着Controller对选择无用。旧Flow两个候选中最佳固定η可改变。当前证据只覆盖离散候选/固定预算，未识别连续basin形状、连通性或边界。
3. **表示的充分性**：endpoint nominal14在五算子间完全相同，会发生输入alias；现有field compact64确实区分算子（静态最大差5），所以不能把所有现有descriptor一概称为不可辨识。区分controller身份也不保证能预测闭环Q。
4. **Effective Correction Basis**：未投影仿射公式 R=M⁻¹[I−(I−M/L)^L] 数值误差最大{errmax:.2g}。完整投影动作的η Jacobian与RB相对差约23%–94%；即使η=0，五算子执行动作的pair差约{zrange[0]:.3f}–{zrange[1]:.3f}（4agent联合速度L2）。局部ε=0.001/0.01导数多数稳定，但有限扰动和活跃约束可破坏线性。R4只编码当前state/key附近响应；不含未来遇到的状态、约束切换和时间预算，不能宣称闭环精确等价。
5. **Held-out Controller**：正式selection-transfer以C1门和独立state规模为准，结果见C2状态；本轮Q-only fallback的五种表示和两个机制消融均已训练与评估，但低规模负结果不能排除可学性，正结果也不构成真实Flow确认。
6. **已确认可迁移的关系**：未投影仿射算子的解析执行响应可计算；历史Q512同controller同几何的新state选择收益真实。尚未确认完整闭环Controller–Q关系从仿射family迁移到真实未见Flow/跨geometry，不能把这两层结论混在一起。
7. **后续关键条件**：先为正式Controller选择实验保留包含强通用η的可靠headroom，保持free-t0/FF/seed/evaluator一致，充分覆盖state；将η0投影动作与投影Jacobian、未投影RB分开消融，并使用真正独立heldFlow与state。不要用大量重复rollout弥补语义或候选饱和问题。

## F. Limitations and Next Steps

所有新模型分析均为DEVELOPMENT；C1为32个旧source state的pilot，Q8无法给B15判定；新seed不使旧state池变成独立state确认。原32source state对nativeFF较容易（η13=95.31%），不代表历史256state（88.33%）总体；不得把pilot未达headroom推广到所有state。本轮“原Critic”指历史Q512标量网络，未测试所有entity critic架构，不能对它们给出样本量下界。有限η/K16模型不能外推连续η空间，16head比较也不具备天然off-gridη预测能力。状态bootstrap不等于跨场景总体置信区间；多面板共享state/标签，不能把253或68个面板当作独立重复。N1 n*依赖architecture、损失和checkpoint规则；“heads用更少数据可靠为正”不同于“用更少数据达到Q512同等性能”。

优先后续问题是：哪种Controller表示能在**有效决策headroom**及可靠state预算下预测Controller引起的η排序变化？其次是局部投影响应加基准动作是否足够，还是需要描述将要遇到的状态/约束。真实Flow checkpoint外推必须恢复自由t0 FF一致性，并先复用可兼容标签；geometry阶段后置。对门失败不扩大/删减候选池来制造收益。

## G. Reproducibility and Evidence Index

Git HEAD `{reg['git']}`（仓库本来有未提交研究资产；没有将Git clean误当完整版本）。初始预注册UTC `{reg['utc']}`，SHA256 `{reg['sha256']}`；后续用户priority/tie/checkpoint/fallback修订均有独立receipt。`DATA_INDEX.json`记录数据与直接代码hash，`ARTIFACT_MANIFEST.json`记录最终产物，`SOURCE_VALIDATION.json`逐条验证原导出标签与SQLite一致。实际复用**{src['all_reused_unique']:,}**个唯一已有rollout，新增**{c1['new_rollouts']:,}**次；不把重叠面板重复计数。原导出N1标签106496条，其中unknown468；C0及legacy复用见独立UID索引。

主要环境：`/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`，NumPy/SciPy/JAX/Flax/Optax；GPU拟合经Slurm shard，CPU rollout每worker1核；绘图用`.venv-c1-plots/bin/python`。详细版本见`ENVIRONMENT.json`。所有命令在仓库根目录执行：

```bash
# 无需重新生成任何rollout；先重算已有数据分析
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OPENBLAS_NUM_THREADS=1 $PY diagnostics/ring_state_selection_transfer_v1/analyze_learning.py N1
OPENBLAS_NUM_THREADS=1 $PY diagnostics/ring_state_selection_transfer_v1/c0_focus.py
OPENBLAS_NUM_THREADS=1 $PY diagnostics/ring_state_selection_transfer_v1/censor_energy.py
OPENBLAS_NUM_THREADS=1 $PY -m diagnostics.ring_state_selection_transfer_v1.c1.analyze
OPENBLAS_NUM_THREADS=1 $PY -m diagnostics.ring_state_selection_transfer_v1.c1.relationship
# 复核后生成报告（不会触发rollout）
OPENBLAS_NUM_THREADS=1 $PY diagnostics/ring_state_selection_transfer_v1/validate_delivery.py
OPENBLAS_NUM_THREADS=1 $PY diagnostics/ring_state_selection_transfer_v1/assemble.py
```

训练/静态查询的完整Slurm命令见`RUN_COMMANDS.md`；已有模型/预测优先直接读取，避免重复训练。C1的`cases.json`、`planned_rollouts.json`、`cache_preflight.json`、`cached_rollout_uids.json`、`raw/`、worker receipt、merge audit及`labels.npz`形成完整provenance链。不要重跑prepare覆盖完成protocol；恢复只能读取已冻结missing清单和已有raw，先重新preflight。新unknown不得以重试覆盖。
'''
 (P/'REPORT.md').write_text(report)
 handoff=f'''# 独立交接：Ring Controller–Q 与η Selection

## 研究问题和位置

本项目是Basin_C1 / OrthoFlow3。核心目标是学习可迁移Q(s,C,η)中的Controller作用，并在未见Controller下有效选η。本轮按用户修订执行N1+C0→C1→条件性C2/C3，N2/N3仅辅助。几何N4/N5已后置。本目录为`diagnostics/ring_state_selection_transfer_v1/`；本文件可脱离对话阅读，完整方法/统计在`REPORT.md`，结构化结果在`results.json`。

## 实际完成的工作

核对历史Ring Q512模型/协议/标签并重新推理冻结checkpoint；复用canonical学习曲线checkpoint，补两个嵌套state排列的公平拟合，比较标量Q、16-head、kNN、prior。原Q512本来就是物理23维+η的MLP，不是entity encoder。只读普查Ring数据库并构造精确state/η/seed匹配的Controller面板，保留数值未知。新增连续FF算子κ=−1,−0.5,0,0.5,1的缺失标签，原生FF另作anchor，固定nativeFlowv10端点/geometry/K16/H700/安全层。执行静态EffectiveBasis可计算性、投影非线性核查，及适用的Controller表示泛化诊断。N2/N3、checkpoint规则诊断均零新增rollout。

累计复用{src['all_reused_unique']:,}个唯一已有rollout，新增{c1['new_rollouts']:,}次，新增网络拟合{newfits}次、复用N1冻结checkpoint30个。新增rollout episode累计{c1['cpu_episode_hours']:.2f}CPU小时。共享数据库只按preflight/journal/单merger流程追加，未覆盖历史数据。完整原始UID索引`REUSED_ROLLOUT_INDEX.json`和`c1/cases.json`；数值未知作为未知保留。

## 最重要的定量结果

- **历史确认复现成功**：Q512=93.54%，固定η13=88.33%，+5.21pp，原max-over16 CI[2.64,7.94]pp；冻结logits最大误差0、选择完全一致。历史256state的确认身份属于旧实验；本轮对其新比较都是DEVELOPMENT。
- **样本效率**：历史VAL-NLL协议、三子集/三fit下，原标量Q的开发n*=512；16-head n*=128；kNN n*=64但只有约+0.85pp。heads128=91.77%、heads256=92.83%，Q512=93.54%。heads128相对scalar128 +4.74pp CI[2.65,7.02]；但仍显著低于Q512。不能声称更少state已经达到同等性能。
- **不只有数据因素**：仅改canonical scalar128的验证checkpoint目标，成功率89.27%→91.72%；同规则却使scalar512降到88.33%。模型选择/目标确实影响selection，不能把当前问题完全归为数据不足或完全归为网络结构。
- **C0直接Controller关系**：固定geometry/Flow/state/seed，FF vs TT的平均Q下降约13.5–14.7pp；全局最佳固定η均13、fixed错配遗憾0，但强state×Controller排序反转在独立半分上成立。保守censor交互能量CI为正。任意argmax ties可把TT的matched−mismatched夸大到71.35pp，prior-aware tie后只有约7.81pp且CI触0；FF方向8.59pp CI[1.04,17.71]。不要把交叉拟合oracle当成可部署selector。
- **真实Flow旧证据**：v9/v7在180相同state、两个η、Q16下偏好不同候选，fixed跨Controller损失22.92pp / 9.24pp。条件是committed原生第一步和旧execution，η池亦不同；不能当成本轮自由t0 FF C3的完成证据。
- **C1门结果**：A（极端κ固定η遗憾门）={c1gate['A_extreme_fixed_regret_pass']}；B（至少3/5κ的可靠state headroom）={c1gate['B_headroom_at_least3_pass']}；selection-transfer合格={c1gate['selection_transfer_qualified']}。各κ具体成功率和CI见下表。

{c1table}

- **C1更强检查**：κ=+1相对discovery固定η的+7.42pp虽通过单Controller主规则，但同时对照完整K16的最强固定η后仅+5.86pp，CI[-0.39,8.98]pp；不能视为可靠新benchmark。极端κ错配固定regret +5.08pp也不稳定于两半（10.16/0），反向CI跨0。
- **C2正式状态**：{decision['status']}。{decision['reason']} {'Q-only fallback已实际跑七个表示/消融×三fit；细表见REPORT和C2_FALLBACK_RESULTS.json。只有16训练state/源Controller、8完全heldstate，不代替正式selection确认。' if fallback else ''}
- **C2表示诊断的实际限制**：R4_action的Q MSE约0.145，比无Controller约0.179低，但正确/错配描述的Q差CI跨0；8个完全held state上固定η观测成功100%，没有selection提升空间。不能据此宣称R4支持Controller泛化，也不能据此排除其潜力。
- **C3正式状态**：{decision['C3_status']}。已完成数据库/真实checkpoint盘点和可兼容旧协议分析，未声称新的真实heldFlow迁移成功。
- **有效修正基底**：解析RB在未投影affine积分误差<3.4e−8，但完整投影FF导数与RB可差23%–94%；η=0时五算子动作仍相差0.333–0.361联合L2。nominal14发生alias，compact64能够区分；局部响应的可计算性不等于终端Q充分性。
- **辅助state解释/外推**：物理特征linear ridge的heldout η-contrast R²约0.056，selection无可靠收益；预先定义chirality半空间N3未发现稳定额外外推损失，但只检验一个较温和切分，不证明普遍state鲁棒。

## 已支持、否定和尚未解决

SUPPORTED：历史nativeRing state-selection收益；Controller影响Q且不止整体level；特定legacy真实Flow会改变最佳固定η；原协议的样本规模与模型选择影响selection。

REJECTED：从“最佳固定η不变”推断Controller只改Q水平；将未投影RB视为完整安全FF/闭环精确响应；把低Q误差自动当作selection泛化。

INCONCLUSIVE：Controller表示是否足以迁移复杂终端Q；R4是否可靠优于compact64；真实自由t0 FF heldFlow迁移；连续η success basin；跨geometry机制不变性；Critic是否存在不可由训练目标/覆盖解释的固有表达缺陷。

C1资格门失败只限制本family作为预注册selection benchmark的资格，**不等于证明不存在Controller×state×η交互**。旧开发数据的新分析和小规模Q-only fallback不能升级为确认。

## 对后续研究的实际影响

把Controller–Q作为主线，避免再次用每Controller/geometry只有16state的失败排除可学性。N1说明模型readout和checkpoint目标需要作为控制变量。Mechanism特征至少应区分未投影映射、完整投影Jacobian与η0执行动作；局部state/key响应不包含未来轨迹/约束切换。C0显示即使同一通用固定η最优，Controller仍可能改变state层排序，不能仅看全局bestη。

最有价值的下一问：在含强固定η、真实headroom和足够state覆盖的面板上，source-only模型能否通过有效执行响应预测未见Controller引起的η排序变化，并在独立state上超过固定与错配参照？之后再固定geometry换真实Flow，最后跨geometry。若补标签，先按精确execution身份查DB；不要重复当前已追加的20736条，不重试覆盖unknown，不删除η13制造收益。

## 可直接使用的资产

- `REPORT.md` / `results.json`：完整实施、指标、证据边界及各阶段状态。
- `PREREGISTRATION.md`、`PRIORITY_AMENDMENT.md`及各`*_RECEIPT.json`：事前设计、目标修订及hash/时间证据。
- `DATA_INDEX.json`、`SOURCE_VALIDATION.json`、`REUSED_ROLLOUT_INDEX.json`、`FINAL_VALIDATION.json`：逐条provenance和一致性。
- `N1_RESULTS.json`、`N1_predictions.npz`、`fits/`、`N1_PAIRED_COMPARISONS.json`、`CHECKPOINT_DIAGNOSTIC_RESULTS.json`：学习曲线与失败诊断。
- `C0_CONTROLLER_CATALOG.json`、`C0_RESULTS.json`、`controller_panels/`、`C0_FF_TT_FOCUS.json`、`C0_CENSOR_ENERGY.json`、`C0_RANK_REVERSALS.json`：直接Controller机制证据。
- `LEGACY_FLOW_FOCUS.json`、`C3_ASSET_AUDIT.json`、`legacy_flow_panels/`、`FLOW_CHECKPOINT_INDEX.json`：真实Flow旧证据及与当前任务不兼容的原因。
- `c1/`：冻结protocol、exact preflight、cases、逐rollout raw、Slurm receipts、merge审计、labels.npz、RESULTS.json、RELATIONSHIP.json。这是本轮新增可共享数据库资产。
- `EFFECTIVE_BASIS_AUDIT.json`、`EFFECTIVE_BASIS_QUERIES.npz`、`ZERO_ETA_PROJECTED_ACTION_AUDIT.json`、`C2_STATIC_FEATURES.npz`：无rollout的Controller表示与局部有效性证据。
- `C2_DECISION.json`、`C2_FALLBACK_RESULTS.json`、`c2_fits/`：正式门和低成本诊断须分开理解。
- `N2_RESULTS.json`、`N3_SUMMARY.json`及预测：辅助state机制/分布证据。
- `RUN_COMMANDS.md`、实验脚本、`ENVIRONMENT.json`、`ARTIFACT_MANIFEST.json`：重算与恢复入口。

工作目录 `/home/zhihan/research/Basin_C1`。先读REPORT与机器结果，再读所需原始面板；不要从日志摘要或单个Q相关系数推断泛化成功。
'''
 (P/'HANDOFF.md').write_text(handoff)
 print('REPORT/HANDOFF/results assembled',results['cost'])
if __name__=='__main__':main()
