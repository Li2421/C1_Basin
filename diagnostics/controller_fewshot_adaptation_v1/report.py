from core import *

def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}pp'
def ci(x):return '['+', '.join(f'{100*v:+.2f}' for v in x)+']pp'
def short(n):return 'base_u' if n=='base_u_macflow' else n.replace('base_u_plus_w_','plus_w_').replace('base_u_','').replace('_macflow','')
def table(h,rows):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|',' / ') for v in row)+' |' for row in rows)+'\n'

def main():
    gate();audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='PASS';res=[read(HERE/'endpoints'/f'main_B{B}_with_E3_CI.json') for B in BUDGETS];desc=[read(HERE/'endpoints'/f'stageB_B{B}.json') for B in BUDGETS];eff=read(HERE/'SAMPLE_EFFICIENCY.json');pool=read(HERE/'POOL_AUDIT.json');pr=read(HERE/'PREREGISTRATION.receipt.json');cost=read(HERE/'PROBE_COST_REFERENCE.json');now=utc();last=res[-1];names=pool['controller_order']
    es=next(s for s in last['summaries'] if s['key']=='R|E-safe');cs=next(s for s in last['summaries'] if s['key']=='R|C-F2-safe');source_supported=[c for c in eff['source_knowledge_contrasts'] if any(v['supported_sample_efficiency'] for v in c['budget_savings'].values())];positive_state=[x for x in eff['statewise_vs_controller'] if x['budget']>0 and x['ci95_simultaneous'][0]>0]
    hypotheses=dict(H_fewshot_safe_positive_transfer='SUPPORTED DEVELOPMENT',H_complete_transfer_on_both_necessary_controllers='SUPPORTED DEVELOPMENT at B128 R E-safe and C-F2-safe',H_safe_transfer_with_B64_or_less='REJECTED under registered methods and 200 repetitions',H_H_prior_saves_labels='SUPPORTED' if any(c['source_method'].startswith('H') for c in source_supported) else 'REJECTED under registered mixture-Beta implementation',H_critic_saves_labels='SUPPORTED' if any(c['source_method'].startswith('C') for c in source_supported) else 'REJECTED under registered frozen-offset implementation',H_statewise_additional_success_value='SUPPORTED' if positive_state else 'INCONCLUSIVE: no positive simultaneous CI at B>0',H_R_shortlist_acquisition_helpful='SUPPORTED DESCRIPTIVE: E-safe R reaches128; U/A >128; no preregistered cross-strategy causal CI',H_within_2pp_of_oracle_for_both_necessary_by128='REJECTED for E-safe R',H_source_knowledge_has_no_possible_value='INCONCLUSIVE',H_independent_new_eta13_source_controller_value='INCONCLUSIVE',H_prospective_confirmation='INCONCLUSIVE')
    verdict='少样本适应可行：预设regime候选集R下，E-safe在128条目标标签首次达到SPT/CT；H和冻结critic没有建立额外预算节省'
    result=dict(status='COMPLETE',utc=now,evidence_level='DEVELOPMENT',verdict=verdict,conclusion_mapping='Partial second-row pattern: few-shot works but no incremental H/C sample-efficiency; H is harmful rather than equivalent to E, so do not claim statistical equivalence',hypotheses=hypotheses,primary_budget_grid=BUDGETS,repeats=200,minimum_primary_SPT=[b for b in eff['budgets'] if isinstance(b['B_SPT'],int)],minimum_primary_CT=[b for b in eff['budgets'] if isinstance(b['B_CT'],int)],sample_efficiency=eff,main_budget_results=[str(HERE/'endpoints'/f'main_B{B}_with_E3_CI.json') for B in BUDGETS],stageB_descriptive_results=[str(HERE/'endpoints'/f'stageB_B{B}.json') for B in BUDGETS],R_Esafe_B128=es,R_CF2safe_B128=cs,new_rollouts=0,new_probes=0,critic_training=0,shared_db_writes=0,database_connections=0,preregistration=pr,adaptation_freeze_sha256=sha(HERE/'ADAPTATION_FREEZE.json'),final_audit_status=audit['status'],cost_reference=cost)
    write(HERE/'results.json',result)
    text=f'''# Round8：真实新controller的少样本η适应

状态 **COMPLETE**，UTC {now}。全部证据 **DEVELOPMENT**。新增rollout/probe **0**，critic训练 **0**，共享DB连接/写入 **0**。所有采集是从既有标签池无放回抽取的模拟；纯Q不需要视频。

## 回答与结论映射

**{verdict}。** 在预注册预算网格0/4/8/16/32/64/128中，首次通过SPT/CT的是 **R + E-safe，B=128**。R + C-F2-safe也在128通过，额外预算节省为 **0条**。H及其他组合128以内未通过；不是“所有方法都失败”的分支。64和128之间未测试，因此128是网格中的首次通过值，不是任意整数预算的精确最小值。

必要目标base_u/v10_local在R + E-safe、B128分别相对η10提高 **+26.47pp / +22.38pp**，合并 **{pp(es['E2']['lower'])}**，同时95%CI **{ci(es['E2']['ci95_simultaneous'])}**。7目标平均提高 **{pp(es['E1']['lower'])}**，同时CI **{ci(es['E1']['ci95_simultaneous'])}**；200次抽样的possible/certain E3以及any-sample target E3均为0，因此CT成立。

“安全正迁移”与“接近最佳固定η”是两个门：E-safe R在B128仍距base_u/v10的discovery固定最佳η约3.80pp/3.79pp保守上界，两个必要目标的 **B_2pp均>128**。五个非必要目标在η10基线已经距对应hindsight固定η≤2pp，故 **B_2pp=0**；整体安全正迁移仍需防止适应误伤它们。

按用户第五节：没有“H/C首次SPT预算更小且配对同时CI支持”的实例，故不支持“源知识可迁移为额外样本效率”。结果部分符合“少样本可行但H/C未带来额外价值”一行；**H明显有害，不能写H≈E或统计等价**。源知识的预设四头候选菜单R本身有工程价值：E-safe在R为128，U/A为>128；这是跨策略阈值的描述性比较，未做预注册跨策略因果CI，也不能当作critic学会了未见controller的证据。A的采集由H驱动，E-A也含源采集知识。

事实：H/C没有节省SPT/CT预算，H的部分高预算选择造成灾难；C-F2-safe在B128的平均收益低于E-safe。解释：强源混合曲线与目标分布不匹配可能限制适应。猜想：更弱的形状先验、目标总体成功水平与regime分离可能更合适，本轮未调整或追加这些方法。

## 预注册、复用与标签池

预注册UTC **{pr['utc']}**，SHA256 `{pr['sha256']}`；先读AGENTS.md、round8、早期草稿及round4–7证据，再冻结PREREGISTRATION。早期草稿未执行，其256预算/S/C-FT不纳入本轮。

可复用路径：`../controller_behavior_repr_v1/` 的REPORT/HANDOFF/results及零样本终点；`../controller_pathway_fix_v1/` 的REAL6 F2-R3-PCA2与R0配置；`../controller_loco_real_flow_v1/` 的StageA/B标签、协议、静态特征、冻结R0及controller_prior；`../controller_q_transfer_round3_v1/` 的P1；`../controller_flow_blend_v1/` 的Stage0曲线、融合标签及粗略目标校准表。旧标签execution语义保持有效。新证据缺口只有pool预测与重复适应统计，没有任何训练或rollout必要性。

主面板7折，每折6源；必要base_u/v10_local。目标池：TRAIN128state×8掩码head×Q4=4096格，加P1 24state×K16×Q4=1536格，合计5632；P1按controller名称重排，不沿用其原数组顺序。主评价64loco_teststate seed4–7；hindsight η仅seed0–3的observed-valid curve选出。TRAIN、P1、test的UID及physical内容两两无交集，controller checkpoint/horizon/dt/FF/success/conditioning核对通过，原完整execution payload保存于POOL_AUDIT。

{table(['目标','可采集格','unknown','TRAIN格','P1格'],[[short(r['controller']),r['available'],r['unknown'],r['TRAIN_count'],r['P1_count']] for r in pool['rows']])}

StageB新增7Flow只用TRAIN4096格做目标池，其他13Flow作源，P1 24state作描述评价；source验证也用过P1，故不进入主判断。没有对应13源LOO的冻结R0/F2，StageB C明确INAPPLICABLE，只执行E/H及safe/P0。每η可用格/有效/unknown/state覆盖在POOL_AUDIT与pools/完整保留，全部方法/策略无“无法采集”事件。

200重复无放回，预算为同一长度128路径的前缀；每个目标/策略/重复的RNG固定为202610130800+target×10000+strategy×1000+repeat。所有适应方法共享标签，预算/策略/方法的配对关系保留。unknown消耗预算，后验和偏置均不按成功/失败更新；P1的Y=0只有在U=0时才是有效失败。

## 固定适应规则与冻结证据

P0固定源η10。E用每η Beta(1,1)，未采valid标签的head不参与，平局先η10再编号最小。E-safe需P(candidate>η10)≥.9。H为预注册的共轭mixture-Beta：每个源component的a=1+TRAIN成功seed数、b=1+失败数，controller先验等权；目标s/f既更新component权重的Beta-binomial边际似然，也与a/b相加。H-safe概率按同一controller混合分量中的两head联合后验计算。源每head约256条标签决定其浓度，**H并非任意强度可调的regime先验**；浓度/权重更新方式的局限保留，未根据终点弱化先验。

R取固定四候选{{2,4,10,13}}；这个菜单由任务/历史总体预声明，不声称每折6源都有四种最佳regime（StageA没有η4型）。A先从H posterior抽controller分量、再抽各head Beta值并选最大，在该head剩余格中均匀抽state/seed，完全没有未来标签搜索。

C-R0/C-F2用冻结3fit的成功概率q，offset=sum(y−q)/(n+8)，q'=clip(q+offset,0,1)。n只计valid目标标签，probability-space residual与round5的未收缩logit-offset不同；本轮n0=8固定。C每fit逐state选argmax并平均真实成功率，C-ctl每fit在评估state的无标签预测均值上选一个固定η。safe需3fit top1一致且平均优势≥.10，否则η10。评估初态可用于无标签平均，终点seed标签不进入这些规则。

冻结模型仅补152个poolstate预测，旧64state终点预测直接复用。R0前4state预测与旧cache max误差0；JAX x64=false，GPU通过Slurm单shard/1CPU推理；采集和统计用CPU最多14job×1CPU，资源提交前检查用户所有任务共同18CPU/3GPU上限。推理、14目标采集路径与选择先冻结ADAPTATION_FREEZE，再打开新终点统计。

## 统计与安全门

每预算统一max-T同时CI，覆盖12方法×3策略、单份P0、E1、必要合并/逐目标E2、预定H−E/C−E/C−Cctl配对以及每目标oracle差。主预算族491个效应（均含unknown上下界），没有按表现删臂；每个B各自一族，不宣称跨预算全局FWER。

5000次**精确交叉重抽**：同一64physicalstate权重跨7目标共享，每target对200次采集重复独立重抽200次，所有方法/策略/预算共享该target重抽权重；fit先平均、不作为独立样本。主种子202610130883、StageB描述202610130884。CI估计200重复平均性能，同时另报单次采集效果分位数与any-catastrophe，不是单次部署保证。

unknown按配对上下界：同η完全对消；不同η的lo=Ya−Yb−Ub、hi=Ya+Ua−Yb。E3 possible以每target/rep平均fit与state保守收益≤−10pp判断，certain用上界；平均E3和“任一次抽样”target比例都报告。SPT严格要求E1 CI下界>−2pp、possible E3平均=0、E2合并CI下界>0；CT再要求两个必要目标点收益>0。

### B_SPT / B_CT与预算不确定性

{table(['方法/策略','B_SPT','B_CT','SPT bootstrap 2.5/50/97.5%','SPT右删失率'],[[b['key'],b['B_SPT'],b['B_CT'],str(b['B_SPT_bootstrap']['quantiles']),pct(b['B_SPT_bootstrap']['right_censored_rate'])] for b in eff['budgets']])}

预算bootstrap保留>128右删失，内部129仅编码而非实测预算；预算节省不把删失值当128。决策bootstrap按外层交叉重抽中心和重抽E3，保留原max-T CI宽度，并非嵌套再估CI。R + E-safe区间[64,>128]，R + C-F2-safe的SPT区间[0,>128]、CT区间[64,>128]反映阈值不稳定；原零样本F2在v9的损失接近−10pp硬阈值，state重抽可使E3门翻转。主点判定仍B128，不能从bootstrap偶发B0改写为零样本通过。

### 预算学习曲线（安全方法主表）

{table(['B','策略/方法','7目标success下界','E1 vs prior同时CI','必要E2保守收益/同时CI','E3平均possible/certain','any-sample目标possible比例','SPT/CT'],[[r['budget'],s['key'],pct(s['success_lower']),ci(s['E1']['ci95_simultaneous']),pp(s['E2']['lower'])+' '+ci(s['E2']['ci95_simultaneous']),pct(s['E3']['possible_rate'])+' / '+pct(s['E3']['certain_rate']),pct(s['E3']['any_sample_target_possible_rate']),str(s['SPT'])+'/'+str(s['CT'])] for r in res for s in r['summaries'] if s['key'] in ['R|E-safe','R|H-safe','R|C-R0-safe','R|C-F2-safe','R|C-ctl-F2-safe']])}

![Safe curves](figures/learning_curves_safe.png)

![Raw curves](figures/learning_curves_raw.png)

图按U/R/A与方法展示，success、E3平均和any-sample均将必要/非必要目标分开；全部数值及同时CI见endpoints/和CSV。

![Simultaneous effect bands](figures/R_safe_simultaneous_effects.png)

### 每目标B_2pp与最佳固定η差距

{table(['方法/策略','目标','必要','B_2pp点指标','同时CI认证预算'],[[r['key'],short(r['controller']),r['necessary'],r['B_2pp'],r['B_2pp_certified']] for r in eff['target_B2pp'] if r['key'] in ['P0','R|E-safe','R|H-safe','R|C-F2-safe','R|C-ctl-F2-safe']])}

全37组合×7目标的B_2pp、bootstrap区间、删失率在SAMPLE_EFFICIENCY.json及tables/target_B2pp.csv。point指标用200重复平均paired保守oracle差≥−2pp，认证版本再要求预算族同时CI下界≥−2pp；两者不混用。oracle由seed0–3选后在seed4–7评价，本身也可能不优于η10；oracle−prior保守分母≤0的gap-closed记NA，其他比例不clip0..1。

{table(['B128组合','必要目标','vs prior下界','oracle gap保守上界','固定η差距闭合比例'],[[r['key'],short(r['controller']),pp(r['gain_lower']),pp(r['G']['gap_upper']),pct(r['G']['gap_closed_lower_point']) if r['G']['gap_closed_lower_point'] is not None else 'NA'] for r in last['rows'] if r['target'] in [1,4] and r['key'] in ['R|E-safe','R|H-safe','R|C-F2-safe','R|C-ctl-F2-safe']])}

## 源知识节省了多少标签

{table(['策略','源方法','vs E的SPT预算节省','vs E的CT预算节省','预算节省SUPPORTED'],[[r['strategy'],r['source_method'],r['budget_savings']['SPT']['point_budget_saved'],r['budget_savings']['CT']['point_budget_saved'],r['budget_savings']['SPT']['supported_sample_efficiency']] for r in eff['source_knowledge_contrasts']])}

只有首次SPT/CT更早、且该预算相对E的7目标配对成功差同时CI下界>0才判样本效率SUPPORTED，本轮无实例。R C-F2-safe与E-safe都是128，节省0；H>128、E-safe为128，H的节省为<0且右删失，不能给精确负预算。两个方法都>128时节省INCONCLUSIVE。完整删失率与条件两者未删失的预算差区间保存在JSON，条件区间不能当全体预算差区间。

{table(['R策略 B128配对比较','7目标差保守下界','同时95%CI'],[[x['key'],pp(x['lower']),ci(x['ci95_simultaneous'])] for x in last['inferences'] if x['metric']=='paired_all' and x['key'] in ['R|H-safe-E-safe','R|C-R0-safe-E-safe','R|C-F2-safe-E-safe','R|C-ctl-F2-safe-E-safe']])}

R H-safe较E-safe约−14.64pp，CI[-20.89,−7.87]pp；R C-F2-safe约−3.91pp，CI[-6.71,−0.97]pp。它们在大预算下不等价于E，更没有建立额外选择优势。

## 逐state选择是否有价值

{table(['B','策略/同骨干比较','逐state−controller保守收益','同时95%CI','判定'],[[r['budget'],r['key'],pp(r['lower']),ci(r['ci95_simultaneous']),r['verdict']] for r in eff['statewise_vs_controller'] if r['budget'] in [0,16,64,128]])}

B>0没有C−C-ctl成功率配对差的同时CI下界>0，故额外逐state成功率价值 **INCONCLUSIVE**。R B128 C-F2-safe−C-ctl-F2-safe点差−1.61pp，CI[-3.91,+0.77]pp；符合controller固定选择点均值更高的描述，但没有证明普遍无价值。另一方面C-F2-safe在R B128的E3=0，C-ctl-F2-safe仍有2/1400个target×repeat灾难（两目标各一次），因此后者未过SPT；均值更高与rare catastrophe并存，不能用一个指标覆盖另一个。

## StageB描述扩展

其他7Flow作目标、13源混合，目标pool只TRAIN，评价P1。历史源模型验证使用同一P1开发分布，本轮H源分量本身仍只由TRAIN构造。StageB的oracle在同一P1数据选出和评价，含hindsight选择偏差；仅作描述参照。所有预算的E/H及safe结果见stageB_B*.json和CSV，以下仅列128，均不进SPT/CT主结论。

{table(['方法/策略','success下界','相对源prior保守差/同时CI','E3平均possible','any-sample目标比例'],[[s['key'],pct(s['success_lower']),pp(s['E1']['lower'])+' '+ci(s['E1']['ci95_simultaneous']),pct(s['E3']['possible_rate']),pct(s['E3']['any_sample_target_possible_rate'])] for s in desc[-1]['summaries']])}

![StageB descriptive](figures/StageB_descriptive_curves.png)

## 成本与研究路线

R5每controller32×200=6400步且无成功标签。按H700上限换算 **9.14条** 完整rollout步数；按StageA所有TRAIN请求的实际记录episode_length均值，约 **9.25–10.65条**。这是模拟步数描述，未计模型/安全求解开销，标签信息不等价；这里只引用真实TRAIN raw，不混入P1或终点来选择策略。

{table(['目标','真实TRAIN记录数','平均recorded steps','R5步数等价完整请求数'],[[short(r['controller']),r['n'],f"{r['mean_recorded_steps']:.2f}",f"{r['R5_6400_step_equivalent_rollouts']:.2f}"] for r in cost['rows']])}

round5的粗略固定前kstate校准表对应3个必要融合controller，k1=32完整H700且需要目标标签。本轮7真实目标、200次重复、多种随机采集、unknown与严格安全门，取代其样本效率估计；不能拿round5的83.33%融合绝对成功率与本轮真实Flow直接比较。

对部署研究最小路线是 **预设少数η候选 + 少量目标标签 + 明确安全回退**。当前可审查候选为R + E-safe B128：对新真实controller随机采128条来自预设四头的标签，冻结Beta规则；未来需要新controller/新state确认。本轮支持开发面板上的互补路线，没有支持加入冻结critic或强源mixture能进一步减预算。对非必要目标，η10零标签已经够好；实际部署缺乏目标必要性真值，统一适应仍需避免误伤。

新的独立η13真实controller仍有研究价值，可用于检验源覆盖/先验misspecification，但当前证据不足以优先宣称它必然比目标适应更省：按旧128state×8head×Q4加24state×K16×Q4源标签设计，每个新controller5632完整episode；两个为11264完整episode+64短probe，Flow训练成本另计。仅标签数量已相当于88个目标各128条适应，尚未计新Flow训练。应先用已有源内资料研究弱混合/目标成功水平与regime分离等结构，再单独预注册新覆盖确认。

前瞻最小确认建议：4个全新真实Flow（至少2个候选η13型、2个其他型），冻结R E-safe与P0/R C-F2-safe对照；每target采128条来自全新采集state的四头标签，共512；另用全新64state×K16×Q4终点评价，共16384；独立24state×{{η10,η13}}×Q2作必要性发现共384。总计 **17280有标签H700 episode**，本轮未运行；是否额外采32短probe/controller取决于新预注册对照，不把R5作为当前必要成本。

## 假设、限制与交付审计

{table(['假设','判定'],list(hypotheses.items()))}

边界为Ring、FF、H700、K16、固定几何、仅2个必要真实Flow；已打开数据全部DEVELOPMENT。采集pool为两类state分布混合且head稀疏，跨η的state组成不同；模拟无放回与真实新采集环境的差异保留。0次样本灾难只描述200重复经验结果，不证明新controller风险恒为0；预算CI估均值且硬E3阈值有不稳定性。H结论仅对应本次源count浓度的共轭mixture，C仅概率offset与冻结骨干，不能否定所有源知识或controller–Q结构可学性。StageB共享P1开发分布不作确认。

FINAL_AUDIT **PASS**：原预注册/采集冻结hash不变，1588历史输入hash不变；主/描述pool无放回、unknown计数正确，source排除目标；H后验权重独立重算max误差{audit['H_conjugate_weight_max_error']:.3g}，offset独立重算max误差{audit['offset_probability_space_max_error']:.3g}；R0/F2零预算与round7逐目标success/gain精确一致；交叉bootstrap与独立标量计算一致。没有DB连接，不作共享DB全局不变声明。所有文件只在本目录，无commit。

文件入口：PREREGISTRATION及收据；POOL_AUDIT、PROTOCOL、HISTORICAL_ASSET_INDEX；pools/、predictions/及PREDICTION_RECEIPT；simulations/的paths、计数、posterior、offsets、choices；ADAPTATION_FREEZE；endpoints/的原/带E3CI结果和bootstrap；SAMPLE_EFFICIENCY/BUDGET_BOOTSTRAP_DRAWS；figures/ PNG-PDF、tables/ CSV；results.json、REPORT、HANDOFF、FINAL_AUDIT。不依赖本对话即可复算。

冻结结果的统计复算命令（无训练、无rollout、无DB）：

```bash
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ROUND8_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
# bi=0..6对应冻结预算；通过Slurm或当前允许CPU余量执行。
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/evaluate.py 6 main
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/evaluate.py 6 stageB
# 完整预算统计已在endpoints/；汇总不会重做采集。
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/efficiency.py finalize
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/audit.py
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/figures.py
$ROUND8_PY -B diagnostics/controller_fewshot_adaptation_v1/report.py
```

完整重算全部预算时逐bi运行evaluate.py与efficiency.py bi，再finalize；不要重新调用setup覆盖原freeze，不要运行历史rollout/probe/merger，不要把已打开终点调参后的新策略称确认。
'''
    (HERE/'REPORT.md').write_text(text)
    handoff=f'''# Round8 独立交接

任务COMPLETE，UTC {now}，全部DEVELOPMENT；工作根目录/home/zhihan/research/Basin_C1；Python {PY}。不依赖本对话。

结论：{verdict}。7真实LOCO、必要base_u/v10_local，R E-safe B128首次SPT/CT：E1 +7.25pp CI[+2.24,+12.61]；E2 +24.42pp CI[+6.91,+42.78]；E3 possible/certain与any-sample均0。必要目标分别+26.47/+22.38pp，仍距固定oracle约3.80/3.79pp，因此必要B_2pp均>128；五个非必要B_2pp=0。C-F2-safe同样B128，预算节省0，其收益较E-safe低3.91pp，同时CI[-6.71,-.97]。H在128内无SPT且部分设置灾难多，不能写H≈E。U/A E-safe亦>128；R预设候选菜单有描述性工程价值，与H/C的额外先验价值分开。

逐state价值：B>0没有C-Cctl成功率正同时CI；R B128 CF2safe-ctl点差-1.61pp、CI[-3.91,+.77]，INCONCLUSIVE。CF2safe过SPT而ctl仍有2次target×repeat灾难，是均值/rare risk区别，不作逐state普遍优越或无价值结论。

执行：先读AGENTS/round8及旧证据，在{pr['utc']}冻结PREREGISTRATION，sha256 {pr['sha256']}；新分析在收据后。7主目标pool各5632格(TRAIN4096+P11536)，unknown70/72/74/85/26/69/15。P1按名称对齐；TRAIN/P1/test UID和physical两两无交集；endpoint只testseed4..7，oracle仅0..3。StageB7目标pool只TRAIN、13源、P1描述评价，C无合适冻结模型不适用。所有采集200重复，无放回，预算0/4/8/16/32/64/128是路径前缀；U/R/A、E/H/C/ctl/safe全部按预注册固定。A由H posterior驱动并给全部方法相同标签，E-A也带源采集帮助。

H实现为原source成功/失败count+Beta1的mixture共轭后验，没有自行选prior强度；source每head约256样本浓度是方法局限。C概率residual offset=sum(y-q)/(n+8)，unknown只占预算，safe3fits一致且优势>=.10。R0/F2 REAL6冻结，只补pool预测、复用旧endpoint logits，JAXx64false；无模型训练。ADAPTATION_FREEZE早于新终点读入。5000精确state×采集重复交叉bootstrap，每B统一max-T(主491效应)，预算decision bootstrap保留原CI宽度且右删失，不作跨B全局确认。

全部阶段完成，新增rollout/probe0，DB连接/写入0，只写本目录，不commit，纯Q无视频。共同18CPU/3GPU，GPU推理1shard，CPU采集/统计最多14job。FINAL_AUDIT PASS，1588历史输入不变；unknown/无放回/源排除目标/H公式/offset/零预算旧数值/交叉bootstrap独立检查均过。

下一步优先预设少数候选+目标标签安全适应的前瞻确认。4新controller：目标采集512episode+独立必要性384+终点16384=17280完整H700episode，本轮未执行。两个新η13源按旧TRAIN/P1补数据需要11264完整+64短probe，Flow训练另计，额外先验预算收益尚未验证。R5每C6400步约9.14H700上限请求或9.25–10.65条已记录TRAINepisode步数，缺成功标签，信息不等价。

主要入口results.json、REPORT.md、SAMPLE_EFFICIENCY.json、POOL_AUDIT.json、FINAL_AUDIT.json；逐采集/后验/offset/choices在simulations/；模型pool预测在predictions/；endpoints/保存所有B主/StageB结果与bootstrap；figures/和tables/完整曲线。REPORT末尾有复算命令。继续前读预注册，绝不生成新rollout、连接DB写入或用testseed4..7改策略。边界Ring/FF/H700/K16固定几何、真实必要目标仅2、DEVELOPMENT。
'''
    (HERE/'HANDOFF.md').write_text(handoff)
    write(HERE/'DELIVERY_INDEX.json',dict(utc=utc(),files=[dict(path=str(f.relative_to(HERE)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(HERE.rglob('*')) if f.is_file() and (f.suffix in ('.py','.md','.csv','.png','.pdf') or f.name in ('results.json','FINAL_AUDIT.json','SAMPLE_EFFICIENCY.json')) and not any(v in f.parts for v in ('cache','tmp','mplconfig'))]))
    print(json.dumps(dict(status='COMPLETE',verdict=verdict,source_budget_savings_supported=len(source_supported)),ensure_ascii=False))
if __name__=='__main__':main()
