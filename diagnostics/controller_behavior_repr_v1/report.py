"""Render the preregistered conclusion, complete result index and handoff."""
from core import *

def short(n):return 'base_u' if n=='base_u_macflow' else n.replace('base_u_plus_w_','plus_w_').replace('base_u_','').replace('_macflow','').replace('blend_','')
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}pp'
def ci(x):return '['+', '.join(f'{100*v:+.2f}' for v in x)+']pp'
def num(x):return 'NA' if x is None else f'{x:.3f}'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|',' / ') for x in row)+' |' for row in rows)+'\n'
def primary_table(res):
    return table(['覆盖','表示','规则','成功率下界','E1 vs R0 同时CI','E2 vs prior 同时CI','E2 vs R0 同时CI','E3 possible/certain','SPT'],[[s['coverage'],s['representation'],s['policy'],pct(s['success_lower']),ci(s['E1']['ci95_simultaneous']),pp(s['E2']['vs_src']['lower'])+' '+ci(s['E2']['vs_src']['ci95_family_simultaneous']),pp(s['E2']['vs_R0']['lower'])+' '+ci(s['E2']['vs_R0']['ci95_family_simultaneous']),f"{s['E3']['possible_count']}/{s['controller_count']} ; {s['E3']['certain_count']}/{s['controller_count']}",s['SPT']] for s in res['summaries'] if s['primary']])

def main():
    gate();real=read(HERE/'real_results.json');blend=read(HERE/'blend_results.json');sensitivity=read(HERE/'blend_imputed_results.json');info=read(HERE/'STAGE1.json');maininfo=info['results']['main'];audit=read(HERE/'FINAL_AUDIT.json');r5audit=read(HERE/'R5_AUDIT.json');replay=read(HERE/'STAGE0_REPRODUCTION.json');receipt=read(HERE/'PREREGISTRATION.receipt.json');coverage=read(HERE/'COVERAGE.json');truth=read(HERE/'TRAIN_TRUTH.json');now=utc()
    cells={(c['representation_family'],c['coverage']):c['passed'] for c in real['cells']};rp=cells['R5','REAL6'];cp=cells['R3','REAL6+BLEND'];both=cells['R5','REAL6+BLEND'];i1=next(r for r in maininfo['I1'] if r['population']=='real' and r['statistic']=='progress');p_rule=any(s['SPT'] for s in real['summaries'] if s['representation']=='P_rule')
    if rp and cp:verdict='两种补救各自充分：表示替换与锚点邻域覆盖增广均可迁移';branch='both_remedies_sufficient'
    elif rp:verdict='表示是瓶颈：现有覆盖下闭环行为表示即可迁移';branch='representation'
    elif cp:verdict='覆盖是瓶颈：静态表示在锚点邻域增广下即可迁移';branch='coverage'
    elif both:verdict='两者都需要：闭环行为表示加上锚点邻域覆盖';branch='both_required'
    elif i1['ci95'][0]>0 or p_rule:verdict='信息存在，但学习通路没有用上';branch='learning_pathway'
    else:verdict='表示（探测设计）仍不足：当前200步、两个η的R5没有建立真实Flow regime的可迁移编码';branch='probe_design_insufficient'
    hypotheses={
        'H_R5_REAL6_sufficient':'SUPPORTED' if rp else 'REJECTED',
        'H_anchor_neighborhood_coverage_sufficient_with_R3':'SUPPORTED' if cp else 'REJECTED',
        'H_R5_plus_anchor_coverage_sufficient':'SUPPORTED' if both else 'REJECTED',
        'H_positive_monotone_delta_progress_information_real':'SUPPORTED' if i1['ci95'][0]>0 else 'INCONCLUSIVE',
        'H_P_rule_safe_positive_transfer':'SUPPORTED' if p_rule else 'REJECTED',
        'H_static_recipe':maininfo['I4']['H_static_recipe'],
        'H_behavior_regime':maininfo['I4']['H_behavior_regime'],
        'H_independent_eta13_real_controller_coverage':'INCONCLUSIVE',
        'H_learning_pathway_is_the_remaining_bottleneck':'SUPPORTED' if branch=='learning_pathway' else 'INCONCLUSIVE',
        'H_any_R5_statistic_or_nonlinear_model_has_no_information':'INCONCLUSIVE',
        'H_real_safe_positive_transfer_under_prescribed_family':'SUPPORTED' if any(c['passed'] for c in real['cells']) else 'REJECTED',
        'H_real_complete_transfer':'SUPPORTED' if any(s['CT'] for s in real['summaries']) else 'REJECTED',
        'H_prospective_confirmation':'INCONCLUSIVE'}
    old5=read(R5/'stage_2/results.json');calbaselines=[]
    for cov in ('SPARSE','DENSE'):
        rows=[r for r in old5['rows'] if r['split']=='LOPO' and r['coverage']==cov and r['arm']=='R0' and r['necessary']];calbaselines.append(dict(coverage=cov,controllers=[r['held_controller'] for r in rows],**{k:float(np.mean([r['baselines_success_lower'][k] for r in rows])) for k in ('src','near','R0','target_fixed')}))
    result=dict(status='COMPLETE',utc=now,evidence_level='DEVELOPMENT',verdict=verdict,conclusion_mapping_branch=branch,hypotheses=hypotheses,real_cells=real['cells'],real_SPT_arms=[s['key'] for s in real['summaries'] if s['SPT']],real_CT_arms=[s['key'] for s in real['summaries'] if s['CT']],blend_SPT_arms=[s['key'] for s in blend['summaries'] if s['SPT']],imputed_blend_SPT_arms=[s['key'] for s in sensitivity['summaries'] if s['SPT']],I1_real_progress=i1,I4=dict(H_static_recipe=maininfo['I4']['H_static_recipe'],H_behavior_regime=maininfo['I4']['H_behavior_regime']),new_rollouts=0,new_probes=0,shared_db_writes=0,database_connections=0,independent_new_F2_seed_fits=165,primary_seed_fits=135,sensitivity_seed_fits=30,per_task_prediction_artifacts=82,fit_seeds=[17,23,41],source_equivalent_sparse_reuse=True,preregistration=receipt,model_freeze_sha256=sha(HERE/'MODEL_FREEZE.json'),audit_status=audit['status'],round5_calibration_comparison_baselines=calbaselines,artifacts={k:str(HERE/v) for k,v in [('stage0','STAGE0.json'),('R5_audit','R5_AUDIT.json'),('stage1','STAGE1.json'),('real','real_results.json'),('blend','blend_results.json'),('imputed_blend','blend_imputed_results.json'),('checkpoints','CHECKPOINT_INDEX.json'),('historical_inputs','HISTORICAL_ASSET_INDEX.json'),('final_audit','FINAL_AUDIT.json')]})
    write(HERE/'results.json',result)
    intro=f'''# Round7：闭环行为表示 × 锚点邻域覆盖

状态 **COMPLETE**；UTC {now}。全部证据 **DEVELOPMENT**。新增rollout **0**，新增probe **0**，共享DB写入 **0**，DB连接 **0**。纯Q实验，无视频验收要求。

## 结论与证据边界

**{verdict}。** 按预注册第四节映射：真实LOCO四格均未达到SPT；真实14Flow的I1主检验ρ={i1['rho']:.3f}，95%CI [{i1['ci95'][0]:.3f}, {i1['ci95'][1]:.3f}]，下界不大于0；P_rule也未达到SPT。因此没有走“信息存在但学习通路没用上”的分支。CT亦未成立。

这里的“表示不足”是本次固定probe、表示和学习规则下的操作性结论。ρ的CI跨零不证明真实相关严格为零，也不证明160维原始轨迹没有任何非线性信息；不能把“不支持正单调信息”写成信息论意义的无信息。覆盖臂失败只否定本次锚点邻域融合增广的充分性；增加独立η13型真实controller仍为 **INCONCLUSIVE**。

事实：R5与覆盖增广在许多规则下恢复了安全回退，却没有提高两个必要真实Flow的选择收益。解释：200步进展差可能偏好短期快速运动，但未反映H700是否能完成交换。猜想：后段停滞、绕障进度或投影持续性可能更有用；这些没有在本轮作为新特征试验。

{table(['表示判定族','覆盖','SPT通过','通过臂','CT通过臂'],[[c['representation_family'],c['coverage'],c['passed'],', '.join(c['SPT_arms']) or '无',', '.join(c['CT_arms']) or '无'] for c in real['cells']])}

R3判定族为3条主要规则；R5判定族为PCA2/Δprog×3规则。R5-Dprog是文件名中对R5-Δprog的ASCII拼写。不得把argmax或κ=.5描述性结果升级成主结论。

## 预注册、历史复用与实现

先读取AGENTS.md及round4/5/6协议、报告、代码与冻结资产，随后在 {receipt['utc']} 留下PREREGISTRATION.md及sha256收据：`{receipt['sha256']}`。新统计和重放均晚于收据。完整用户设计原文已嵌入预注册。

复用路径：`../controller_pathway_fix_v1/` 的R0/F2/F4配置、checkpoint与预测；`../controller_flow_blend_v1/` 的TRAIN/终点标签、Stage0曲线、96state静态特征及R5 raw；`../controller_loco_real_flow_v1/` 的真实7折标签、R0与128state静态特征；`../controller_q_transfer_round3_v1/` 的P1 24state验证集。execution语义未变，旧标签保持有效。缺口只有新表示/源集合的小型模型，不需要新rollout或Flow查询。3068项历史资产hash见HISTORICAL_ASSET_INDEX.json。

独立核对发现任务说明的一处历史事实差异：round6真实LOCO的R3均值实际为128 TRAIN state，融合LOPO为96；源码及round6预注册一致。本轮R3×REAL6精确复用128state旧fit；REAL6+BLEND保留真实Flow128state均值，融合取其已有96state均值；阶段1描述均取共同96state。这保留了真实Flow的表示输入，但融合特征的平均state组成不同，是覆盖比较的额外局限。

F2冻结同折、同fit seed的R0，加16/16/16 SiLU残差，末层零初始化，tanh界1；AdamW lr0.0008、weight decay0.01、clip5、batch128头，12000step，seed17/23/41，3fit平均P1源验证NLL选共同step，JAX x64=false。REAL6+BLEND的R0、prior和P1验证只由原6源真实Flow决定。新增controller均满足两锚点在源中；训练与验证无目标controller，所有变换source-only。所有新预测及模型先冻结到MODEL_FREEZE.json，再执行新终点评价。

105个主真实seed fit、30个主融合seed fit、30个插补敏感性seed fit，合计 **165**。另复用R3旧fit，输出 **82** 个臂×折预测包（每包3seed）及全部checkpoint。相同SPARSE源模型每表示只优化一次，其余折共享SHA并各存预测；不重复训练。新F4及R0、prior、P_rule参数/选择见classifiers/、CLASSIFIERS_INDEX.json。首个正式3seed GPU fit耗时46.37s，作为正式结果保留。

资源按用户实时更新为所有任务共同18CPU/3GPU shard、本轮优先。实际真实LOCO105seed fit使用Slurm GPU；融合主/敏感性60seed fit使用Slurm CPU，每job1核、最多14CPU并发，另有3个GPU job各1CPU、预留1CPU编排。CPU/GPU仍保持x64=false，backend不同不声称bitwise等价，未根据终点挑选后端。原队列20个尚未启动的融合任务改为CPU调度，未删臂。CPU日志中的CUDA插件无设备信息是插件探测警告；实际backend=cpu，全部拟合完成，非rollout失败。资源修订先于新终点评价，见RESOURCE_EXECUTION_ADDENDUM及收据、SLURM_SUBMISSIONS.json。

阶段0后先执行了一个正式F2成本测量任务，阶段1的信息量分析与其独立运行；阶段1脚本本身只读TRAIN_ONLY.npz、无标签表示与旧Stage0曲线，不读任何终点seed，不拟合Q。这项执行顺序并行不参与方法选择，完整源验证预算固定。

## 阶段0：硬门复现与R5审计

15个task，R0、F2-PCA2悲观κ1/保守、F4的所有逐目标选择和成功率/收益与round6精确一致。R0最大logit误差 {max(r['R0_logit_max_abs_error'] for r in replay['rows']):.3g}；F2最大logit误差 {max(r['F2_logit_max_abs_error'] for r in replay['rows']):.3g}；logit误差不是另加的选择门。所有主要选择一致才解锁后续阶段。STAGE0_REPRODUCTION.json保留逐项收据；FINAL_AUDIT另检查155条复用R3终点行及η直方图与round6一致。

R5形状(26,160) float32；排列为controller→state0..15→η10/13→[progress,projection_fraction,minimum_pair_distance,mean_speed,stalled_steps]。831条完整raw trace独立重算与raw统计最大误差 **{r5audit['max_abs_error_vs_raw_statistics']:.1f}**；与float32 NPZ最大误差 **{r5audit['max_abs_error_vs_float32_features']:.9g}**，转float32后完全一致。25个controller各32个完整probe，base_u_v3为31个；唯一NaN为该行105..109，即state10/η13。失败CBFSolverError发生于132步，partial只作描述，不生成任何统计特征。无缺失零填充；stalled_steps中的829个真实零来自完整trace，不是插补。

{table(['统计量','有效格数','min','median','mean','max'],[[k,v['n'],num(v['min']),num(v['quantiles'][2]),num(v['mean']),num(v['max'])] for k,v in r5audit['distribution'].items()])}

16probe state与R4全部192state、R5全部128state无交集；融合TRAIN96全部属于真实TRAIN128，和真实终点64state无交集。主方案所有controller删除state10/η13五维，R5为155维；所有差值统一使用另外15state。敏感性只在阶段1/3，用折内源且不含base_u_v3的均值插补该格，PCA160维、差值16state；不使用132步partial。

## 阶段1：controller级信息与识别

真值：14真实Flow采用历史Stage0 best_eta与observed_valid_curve；融合采用96 TRAIN state的valid curve，regime遵循round6四候选顺序[10,13,2,4]。各controller曲线、K16 best与regime见TRAIN_TRUTH.json。融合K16 best落在四候选外的数量为 {sum(t['best_K16'] not in (2,4,10,13) for t in truth[14:])}。**稀疏掩码使η10与η13在不同state子集上观测，ΔS包含state组成差异。**

### I1：零参数信息检验

{table(['人群','统计量差值','ρ','95%CI','退化bootstrap次数','性质'],[[r['population'],r['statistic'],num(r['rho']),'['+', '.join(num(v) for v in r['ci95'])+']',r['bootstrap_degenerate'],'主检验' if r['primary'] else '探索性'] for r in maininfo['I1']])}

每项5000次bootstrap：真实Flow按controller，融合按pair，合并按14个单Flow簇+4个pair簇。常量重抽样记退化，剔除并报告数量；stalled_steps几乎恒为0，该探索性相关尤其不能按831个独立观测解读。主ρ未证明正关系，也不是等价性检验。

![Δprog vs ΔS](figures/I1_delta_progress_vs_delta_success.png)

### I2：固定ridge α=1的留出ΔS预测

{table(['表示','划分','留出Spearman','符号准确率'],[[r['representation'],r['bed'],num(r['spearman']),pct(r['sign_accuracy'])] for r in maininfo['I2']])}

真实LOO仅13源；融合LOPO为14真实锚点+9个非留出pair融合，锚点保留。14真实Flow中ΔS正号仅2个，预测全负的符号准确率本就为12/14=85.71%；高符号准确率不能替代排序或少数类识别证据。

### I3：14个Flow的LOO regime分类

{table(['表示','分类器','准确率','相对64.29%多数类差'],[[r['representation'],r['classifier'],pct(r['accuracy']),pp(r['delta_vs_fixed_majority'])] for r in maininfo['I3'] if r['classifier']!='source_majority'])}

没有预定表示/分类器超过9/14多数类；完整混淆矩阵、逐目标预测及source-majority结果在STAGE1.json和tables/I3_main.csv。

![Confusion matrices](figures/I3_confusion_main.png)

### I4：配方与regime的描述性预测

{table(['表示','d(base,v10)','d(base,v9)','d(v10,v9)','该表示预定关系成立'],[[rep]+[num(maininfo['I4'][rep]['triplet_distances'][k]) for k in ('base_v10','base_v9','v10_v9')]+[maininfo['I4'][rep]['prediction_satisfied']] for rep in ('R3','R3-PCA2','R5-PCA2','R5-Dprog')])}

H_static_recipe = **{maininfo['I4']['H_static_recipe']}**；H_behavior_regime = **{maininfo['I4']['H_behavior_regime']}**。这是距离关系预测，不证明训练配方是静态描述的唯一因果因素。7×7距离表与每controller完整近邻排序在tables/I4_*；只用14真实Flow拟合描述性z-score/PCA。

{table(['面板Flow','raw Δprog','符号'],[[short(r['controller']),f"{r['delta_progress']:.6f}",r['sign']] for r in maininfo['I4']['panel_delta']])}

7个面板Flow全部Δprog>0，P_rule因此对每个目标都选择η13；短期进展的正号无法区分η10/η13最终成功regime。

![Distances](figures/I4_panel_distances.png)

## 阶段2：真实Flow LOCO主检验

终点仅64个loco_test state × seed4–7。全部主要臂联合E1 max-T，5000次物理state成簇bootstrap，各controller与3fit先等权平均；不把同state的controller/seed当独立样本。真实主要族18臂+唯一P_rule，两个coverage中完全相同的P_rule仍逐行展示，重复相同列不改变max-T。E1下界须严格大于−2pp。E2每个R3三臂/R5六臂判定族上，联合vs prior及R0两比较做max-T；SPT要求两个下界都>0且possible E3=0。argmax/κ.5只在JSON与CSV描述，不参与判定。

### 覆盖与防泄漏

{table(['留出Flow','总源C数','新增融合C数','新增regime数量'],[[short(r['held']),r['source_count'],len(r['added']),str({eta:sum(v['regime']==eta for v in r['added']) for eta in (10,13,2,4)})] for r in coverage if r['coverage']=='REAL6+BLEND'])}

完整新增controller名称/所属pair/regime见COVERAGE.json。base_u折仅pair2/3，v10折仅pair0/1，v4_wide折仅pair1/3，v9折仅pair0/2，其余全4pair。该增广引入6或12个附近exemplar，并增加标签量；不是独立新η13型真实Flow。

### 给定controller时的η选择（主终点）

{table(['覆盖','基线','成功率下界','成功率上界','必要C相对prior保守收益'],[[s['coverage'],s['representation'],pct(s['success_lower']),pct(s['success_upper']),pp(s['E2']['vs_src']['lower'])] for s in real['summaries'] if s['representation'] in ('B0','source_prior')])}

{primary_table(real)}

R5保守/F4多数回退η10，在base_u与v10上相对prior收益为0；safe fallback没有证明Q学会controller作用。R3×REAL6保守规则在v10上+17.58pp、base_u为0，但v9上−10.16pp，因此仍因灾难失败。κ1的possible E3是2/7，不能只列v9一个受损目标而把总灾难率写1/7。

P_rule必要controller合并收益较大，却有5/7可能灾难，整体E1相对R0的同时CI低于门；未达到安全正迁移。不能用base_u/v10收益掩盖其余目标伤害。

{table(['覆盖','表示','规则','目标','vs prior 保守收益','vs R0 保守收益','η10占比','η13占比'],[[r['coverage'],r['representation'],r['policy'],short(r['controller']),pp(r['gain_src_lower']),pp(r['gain_R0_lower']),pct(r['choice_histogram'][10]),pct(r['choice_histogram'][13])] for r in real['rows'] if r['primary'] and r['controller'] in ('base_u_macflow','base_u_v10_local_macflow','base_u_v9_local_macflow')])}

所有7目标、全部16个η直方图及逐目标描述CI在real_results.json、tables/real_per_target.csv；下图覆盖全部目标。

![Factorial endpoints](figures/real_factorial_E1_E2_E3.png)

![All target choices](figures/real_eta_histograms_all_targets.png)

### Q预测（独立于选择与regime识别）

{table(['覆盖','模型','valid-head MSE','valid-seed NLL'],[[s['coverage'],s['representation'],f"{s['Q_prediction']['mean_valid_head_MSE']:.5f}",f"{s['Q_prediction']['mean_valid_seed_NLL']:.5f}"] for s in real['summaries'] if s['policy']=='argmax' and s['Q_prediction'] is not None])}

MSE针对终点有效seed比例的Q估计，保留有限Q4标签噪声；NLL仅valid seed计入。unknown未当作失败。逐目标chosen-Q高估区间见Q_prediction字段。F4的regime概率和P_rule不是成功概率Q，不伪造Q误差。这里的Q误差、上节η选择收益、阶段1 controller级regime准确率是三个不同问题，不能相互替代。

## 阶段3：融合族LOPO次检验

4fold，SPARSE14真实Flow，DENSE加9个非留出pair融合。终点仅32state × seed2–3；必要3个controller沿用旧discovery gate，不用终点重选。R3旧模型全部复用、纳入本轮同时推断族。

{table(['覆盖','基线','成功率下界','成功率上界'],[[s['coverage'],s['representation'],pct(s['success_lower']),pct(s['success_upper'])] for s in blend['summaries'] if s['representation'] in ('B0','source_prior')])}

{primary_table(blend)}

融合SPARSE的R3-PCA2悲观/保守仍达到SPT；R5两种表示在两个coverage均未达到SPT。这保留了round6“融合族有静态表示正迁移，真实Flow没有”的差异；不能据此证明R5优于R3。融合controller由已知锚点插值、相关性更强，且TRAIN regime可能随λ变为η4，不是纯二类连续η10↔η13过渡；它不等价于新真实checkpoint。

![Blend factorial](figures/blend_factorial_E1_E2_E3.png)

### 插补敏感性（不进入主判定）

阶段1进展差主检验的插补结果如下（同一预定5000次成簇bootstrap）：

{table(['人群','插补ρ','插补95%CI'],[[r['population'],num(r['rho']),'['+', '.join(num(v) for v in r['ci95'])+']'] for r in info['results']['imputed']['I1'] if r['statistic']=='progress'])}

{table(['表示','分类器','插补LOO准确率'],[[r['representation'],r['classifier'],pct(r['accuracy'])] for r in info['results']['imputed']['I3'] if r['representation'] in ('R5','R5-PCA2','R5-Dprog','R3-PCA2+R5-PCA2') and r['classifier']!='source_majority'])}

{primary_table(sensitivity)}

插补后R5仍无SPT；DENSE Δprog的保守/悲观收益还可能更差，说明缺失格处理不能笼统称完全无影响。该方案同时恢复所有完整controller的第16个state差值，绝非只改变base_u_v3一行。阶段1敏感性I1–I4完整保存在STAGE1.json的imputed分支，CSV与混淆图一并交付；不按敏感性终点改主方案。

## 假设判定

{table(['假设','判定'],list(hypotheses.items()))}

REJECTED表示“预定方案在当前DEVELOPMENT面板不充分”；不等于证明该研究方向不可能。H_positive_monotone_delta_progress_information_real为INCONCLUSIVE而不是证明零相关，主映射中的“无信息”只指未满足预注册ρCI下界>0的信息门。学习通路仍可能有限，但本轮未识别为剩余唯一瓶颈。

## 成本对照与下一步最小设计

R5原始部署成本每controller **32×200=6,400模拟步**，无成功标签，但需要闭环模拟器；本轮全部复用，不新增。round5 k=1校准每controller **32个完整H700 episode**（8个已掩码η头×Q4），需要成功/unknown标签，最大预算32×700=22,400模拟步，真实episode可能早停；不能只按同为32次就称计算成本相等。本轮主缺失方案只用31格/15个成对state，但既有数据采集成本仍按原32次计。

以下直接引用round5 REPORT.md的“目标少样本校准参照”表。该表是 **LOPO的3个必要融合controller等权、32个test state、终点seed2–3** 的绝对成功率下界，不是7个真实Flow均值，也不是相对R0收益。fixed列为用预定前k TRAIN state标签选出的固定η；critic+offset列为同一原critic加目标标签logit校准。目标校准标签不含终点seed。

{table(['覆盖','同表源prior基线','同表R0基线','同表最近源基线','discovery固定目标η参照'],[[r['coverage'],pct(r['src']),pct(r['R0']),pct(r['near']),pct(r['target_fixed'])] for r in calbaselines])}

'''
    oldreport=(R5/'REPORT.md').read_text();cal=oldreport.split('### 目标少样本校准参照（使用目标标签）',1)[1].split('### 每fold/每表示结果',1)[0].strip()
    next_design='''
下一步先做 **0 rollout** 的源数据设计：从已保存831条完整raw轨迹定义后段goal进展、投影持续性、相对运动或绕障进度等候选，先预注册并仅在源TRAIN内选择；不把这里的200步低速统计升级为正式deadlock标签。Ring旧评价器没有runtime deadlock事件，timeout须单列。若完整200步轨迹仍不足，再单独授权并预注册更长horizon的probe。

最小独立前瞻计划（当前尚无通过SPT的R5候选，须先在源内冻结候选后才执行）：4个新controller，优先不同训练seed的真实Ring Flow，覆盖至少2个候选η13型和2个η10型；新锚点组合融合可作工程先导，但不能替代独立真实controller结论。为每个新controller采集32次短probe，共 **128次200步无标签probe=25,600步**。用独立24state×η{10,13}×Q2作必要性发现，共 **384个完整episode**；终点另用全新64state×K16×Q4，共 **16,384个完整episode**。总计 **16,768个有标签完整H700 episode +128个短probe=16,896次请求**，完整episode上限11,737,600步。发现state、终点state、probe state三者与所有历史state去重；同一冻结规则评价全部controller，不按终点筛选。

若测试更长probe（例如现有26controller×16state×2η×H700），另需 **832次无标签H700请求，582,400步**；这是未来新协议成本，不在本轮运行，不把旧200步标签/partial假作完整700步轨迹。先复用现有raw的0-rollout方案可避免不必要采集。

独立覆盖假设的后续成本也保留：每新增真实controller，128 TRAIN state×8掩码头×Q4=4,096完整episode，P1兼容验证24state×K16×Q4=1,536，另32短probe；若新增2个独立η13真实controller，源标签预算为 **11,264完整episode+64短probe**，训练Flow本身的计算成本另计。这是独立覆盖的试验计划，当前结果不支持把它写成已解决方案。若之后建立了probe信息但选择仍失败，可另预注册以probe预测regime先验、再接源固定η曲线的结构化模型；对当前已有标签/轨迹的开发成本为0新rollout。

边界：所有数据早已打开，均DEVELOPMENT；固定Ring、FF、H700、K16和几何，真实必要controller仅base_u与v10_local两个。source-only bootstrap没有把controller总体不确定性估进去；14Flow/4pair信息检验样本少。融合增广还混杂额外标签量、源采样分布与静态均值state组成。不能推广到四场景、N=100、任意真实Flow或连续η basin；安全正迁移不等价于严格死锁解决。

## 审计、文件与复算

FINAL_AUDIT状态为PASS_WITH_DOCUMENTED_RESOURCE_UPDATE：除用户任务文档资源段的外部更新外，其余3067项历史资产hash不变。原失败审计保留FINAL_AUDIT_INITIAL.json，PROMPT_EXTERNAL_CHANGE.diff记录唯一差异。原预注册hash、全部模型冻结hash均未变化；155条R3复用结果/直方图与round6逐值一致；unknown相同η精确抵消、不同η保守区间通过独立检查。没有连接共享DB，因此只声明本任务DB写入0，不宣称共享DB全局字节不变。所有新增文件只在本目录，没有git commit。

核心文件：PREREGISTRATION.md与receipt；STAGE0.json / STAGE0_REPRODUCTION.json / R5_AUDIT.json；STAGE1.json / TRAIN_TRUTH.json；TASKS.json / FIT_TASKS.json / COVERAGE.json；fits/、shared_sparse_fits/、classifiers/及CHECKPOINT_INDEX.json；MODEL_FREEZE.json；real_results.json、blend_results.json、blend_imputed_results.json及state_vectors；results.json；figures/的PNG/PDF、tables/的CSV；REPORT.md、HANDOFF.md。

复算入口（不重训、不读写DB、不生成rollout）：

```bash
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/controller_behavior_repr_v1/evaluate.py real
$PY -B diagnostics/controller_behavior_repr_v1/evaluate.py blend
$PY -B diagnostics/controller_behavior_repr_v1/evaluate.py blend imputed
$PY -B diagnostics/controller_behavior_repr_v1/verify.py
$PY -B diagnostics/controller_behavior_repr_v1/figures.py info
$PY -B diagnostics/controller_behavior_repr_v1/figures.py real
$PY -B diagnostics/controller_behavior_repr_v1/figures.py blend
$PY -B diagnostics/controller_behavior_repr_v1/figures.py blend imputed
$PY -B diagnostics/controller_behavior_repr_v1/report.py
```

只有确实丢失拟合产物时才按FIT_TASKS.json补对应critic，不盲目重训；既有rollout/probe永远不重跑。不得调用历史probe.py merge/worker或rollout merger。资源上限与其他任务共享，重新提交前须检查占用。
'''
    (HERE/'REPORT.md').write_text(intro+'\n'+cal+'\n'+next_design)
    handoff=f'''# Round7 独立Handoff

状态COMPLETE，UTC {now}，全部 **DEVELOPMENT**。工作根目录 `/home/zhihan/research/Basin_C1`；输出 `diagnostics/controller_behavior_repr_v1/`；Python `{PY}`。不依赖本对话。

目标：区分真实未见Flow的η选择迁移瓶颈是表示、锚点邻域覆盖还是学习通路。遵循AGENTS.md与冻结PREREGISTRATION。新增rollout/probe=0，DB连接/写入=0，只写新目录，无commit，纯Q无需视频。

结论：**{verdict}**。真实四格全无SPT/CT；ρ(Δprog,ΔS)真实14Flow={i1['rho']:.3f}，CI[{i1['ci95'][0]:.3f},{i1['ci95'][1]:.3f}]，未建立正单调信息；所有7面板Δprog>0，零参数P_rule全选η13，possible E3=5/7。安全R5规则多回退η10，在base_u/v10上相对prior收益0。学习通路瓶颈未被本轮单独识别，独立η13真实controller覆盖仍INCONCLUSIVE。I4静态配方距离预测SUPPORTED、行为regime距离预测REJECTED，均描述性。

融合LOPO次检验：R3-PCA2 SPARSE悲观/保守保留SPT，R5主/插补均无SPT。融合家族结果不推广真实Flow。所有数值和family CI见REPORT及三个*_results.json。不能用Q误差、regime分类准确率代替η选择收益；不能把Ring timeout称deadlock。

已完成：阶段0从冻结checkpoint重放15task，R0/F2κ1/保守/F4全部选择/成功与round6一致；831完整probe重算与raw统计误差0、float32特征逐值一致，唯一失败base_u_v3/state10/η13/132步保留。主删除5维得到155、delta删整个state10用15state；敏感性仅阶段1/3源均值插补。阶段1I1–I4全部主/敏感性、散点/混淆/距离CSV。105真实+30融合主+30插补seed fit=165，82个task×3seed预测及checkpoint齐全，SPARSE共享模型不重训。F4及P_rule参数冻结，终点只真实seed4–7、融合seed2–3，unknown配对保守且同η对消。

历史事实差异：round6真实R3均值实际128TRAINstate、融合96，非提示概述中的全部96。真实R3 baseline精确复用128；增广保留真实128、融合96，state组成差异为局限。预注册先已记录。用户随后将资源更新为共同18CPU/3GPU、本轮优先；资源补充收据先于新终点评价。真实GPU、融合CPU，x64false和科学超参数保持。历史prompt仅资源段在外部更新，唯一hash差异已解释；3067个其余历史资产不变。原审计保留，最终PASS_WITH_DOCUMENTED_RESOURCE_UPDATE。

预注册UTC {receipt['utc']}，sha256 `{receipt['sha256']}`。MODEL_FREEZE在新终点评价之前；FINAL_AUDIT验证hash、无泄漏、165fit、155复用行与η直方图、unknown算术。全共享DB未作字节不变声明；无DB连接。

下一步：先从已有raw源TRAIN内预注册新probe统计，0新rollout。若候选冻结，4个新controller（至少2个候选η13真实Flow）/全新state前瞻：384发现完整episode+16,384终点完整episode+128短probe=16,896请求，未执行。原R5每C6,400步；round5 k1每C32完整H700且需要目标成功标签。报告直接保留round5少样本校准表，其比较对象是3个必要融合controller，source prior78.65%，SPARSE R0 78.65%、DENSE R0 78.82%，不是7真实Flow。

继续工作先读REPORT.md、PREREGISTRATION.md及资源补充、FINAL_AUDIT.json。结果入口results.json；阶段文件、checkpoints索引、figures/PNG-PDF、tables/CSV、运行/代码均在本目录。REPORT末尾有只从冻结缓存复算命令；不得运行历史rollout/probe/merge，也不得修改旧资产或用终点调参。边界Ring/FF/H700/K16固定几何、真实必要C仅2；所有结论DEVELOPMENT。
'''
    (HERE/'HANDOFF.md').write_text(handoff)
    write(HERE/'DELIVERY_INDEX.json',dict(utc=utc(),files=[dict(path=str(f.relative_to(HERE)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(HERE.rglob('*')) if f.is_file() and (f.suffix in ('.py','.md','.csv','.png','.pdf') or f.name in ('results.json','FINAL_AUDIT.json','CHECKPOINT_INDEX.json')) and not any(p in f.parts for p in ('tmp','cache','mplconfig'))]))
    print(json.dumps(dict(status='COMPLETE',verdict=verdict,branch=branch,hypotheses=hypotheses)))
if __name__=='__main__':main()
