"""Self-contained Chinese report; execution states are never fabricated."""
from common import *
import shutil

def percent(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}pp'
def ci(x):return '['+', '.join(pp(v) for v in x)+']'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def save(name,text):
    path=HERE/name
    if path.exists() and path.read_text()!=text:
        d=HERE/'report_versions';d.mkdir(exist_ok=True);dest=d/(path.stem+'_'+sha(path)+path.suffix)
        if not dest.exists():shutil.copy2(path,dest)
    path.write_text(text)
def main():
    s0=read(HERE/'STAGE_0_RESULTS.json');folder=HERE/'stage_1'
    s1=read(folder/'results.json') if (folder/'results.json').exists() else None
    s2=read(HERE/'stage_2/results.json') if (HERE/'stage_2/results.json').exists() else None
    raw_count=len(list((folder/'raw').glob('*.json')));timing=read(folder/'PILOT_TIMING.json') if (folder/'PILOT_TIMING.json').exists() else None
    stage1state='完成；门通过' if s1 and s1['gate']['passed'] else '完成；内部λ门未通过' if s1 else 'RUNNING — 已提交CPU采集，尚未分析完整面板'
    stage2state='完成' if s2 else 'NOT EXECUTED — GATE FAILED' if s1 and not s1['gate']['passed'] else 'RUNNING — 正式协议/采集/特征和下游训练已提交' if (HERE/'stage_2/DISPATCH.json').exists() else 'NOT EXECUTED — 等待Stage1门/正式阶段提交'
    done=bool(s2 or (s1 and not s1['gate']['passed']))
    result=dict(schema='controller_flow_blend_round5_results_v1',updated_utc=utc(),status='COMPLETE' if done else 'IN PROGRESS',
        task='Distinguish controller coverage, representation and method explanations of unseen-controller eta selection',
        stages=dict(stage0='COMPLETE EXPLORATORY; zero rollout',stage1=stage1state,stage2=stage2state),stage0=s0,stage1=s1,stage2=s2,
        observed_stage1_raw_count=raw_count,new_rollouts_stage0=0,resource_authorization=read(HERE/'RESOURCE_AUTHORIZATION.json'),
        execution_boundary='All old round4 states DEVELOPMENT; fresh Stage2 test only if gate passes; no continuous basin inference',
        hypotheses=dict(real_flow_eta_regime_variation='SUPPORTED — existing sparse TRAIN evidence; not new confirmation',
            mean_static_features_LOO_outperform_majority='REJECTED IN THIS DESCRIPTIVE TEST',
            blend_internal_regime_coverage='SUPPORTED' if s1 and s1['gate']['passed'] else 'REJECTED FOR THIS GRID/PILOT' if s1 else 'INCONCLUSIVE — RUNNING',
            coverage_vs_representation_vs_method='INCONCLUSIVE — main factorial study not executed' if not s2 else s2.get('rootcause_verdict','INCONCLUSIVE')))
    write(HERE/'results.json',result)
    rt=[]
    for r in s0['rows']:
        rt.append([r['controller'],r['best_eta'],percent(r['success_lower'][r['best_eta']]),percent(r['success_lower'][10]),percent(r['success_lower'][13]),ci(r['comparisons']['10']['ci95_simultaneous']),ci(r['comparisons']['13']['ci95_simultaneous'])])
    s0table=table(['Flow','经验bestη','best成功下界','η10下界','η13下界','best−10同时CI','best−13同时CI'],rt)
    classtable=table(['表示','NN LOO','Logistic LOO','多数类LOO'],[[a,*[percent(v[m]['accuracy']) for m in ('NN','logistic','majority')]] for a,v in s0['representations'].items()])
    asym=table(['表示','base_u最近邻','v10_local最近邻'],[[a,v['nearest_controller'][1],v['nearest_controller'][4]] for a,v in s0['representations'].items()])
    priortab=table(['阶段/折','R3 prior','R4_action prior','R3+R4 prior'],[[key,*[str(v[a]) for a in ('R3','R4_action','R3_R4_action')]] for key,v in s0['frozen_controller_priors'].items()])
    decomp=table(['阶段','折','表示','target固定−model固定','model固定−逐state','总regret'],[[r['stage'],r['fold'],r['arm'],pp(r['regime_component_lower_success_difference']),pp(r['state_component_lower_success_difference']),pp(r['total_lower_success_regret'])] for r in s0['regret_decomposition']])
    pilot='尚未完成，不能依据部分outcome宣称regime翻转或gate结果。'
    if s1:
        pilot=table(['pair','λ','η2下界','η4下界','η10下界','η13下界','13−10配对CI','明确赢家','unknown'],[[r['pair'],r['lambda_value'],*[percent(q) for q in r['success_lower']],ci(r['ci95_conservative_delta']),r['unambiguous_winner_13_vs_10'],r['unknown']] for r in s1['rows']])
        pilot+='\n\n'+table(['pair','η13胜任λ','η10胜任λ','通过内部λ门'],[[r['pair'],str(r['eta13_competent_lambdas']),str(r['eta10_competent_lambdas']),r['passes']] for r in s1['pairs']])
        pilot+=f"\n\n12个内部controller共4608请求；valid={s1['valid']}、unknown={s1['unknown']}。内部λ门{sum(r['passes'] for r in s1['pairs'])}/4，要求≥3/4。锚点复用{s1['reused_endpoint_seed_slots']}个K4 seed格；每η仅12/24state，详见stage_1/results.json.original_endpoint_partial_curves。"
    prereg=[]
    for f in sorted(HERE.glob('PREREGISTRATION*.receipt.json')):
        p=read(f);prereg.append([Path(p['file']).name,p['utc'],p['sha256']])
    timingtext='计时未完成。' if not timing else f"98次pilot平均{timing['mean_seconds']:.2f}s、median {timing['median_seconds']:.2f}s、p90 {timing['p90_seconds']:.2f}s；全Stage1初估{timing['stage1_cpu_hours']:.2f}CPU小时/{timing['stage1_wall_hours_24']:.2f}h@24CPU，Stage2完整标签初估{timing['stage2_full_cpu_hours']:.2f}CPU小时/{timing['stage2_wall_hours_24']:.2f}h@24CPU。实际全量尾部、特征、训练、队列与合并另计，不能视作任务完成时间保证。"
    s2details='尚无完整Stage2结果，不能写选择收益或根因结论。'
    finalcost=None
    if (HERE/'stage_2/COST_GATE.json').exists():
        finalcost=read(HERE/'stage_2/COST_GATE.json')
        timingtext+=f"\n\nStage2正式成本门：复用2,304个pilot seed格（2,299 valid、5 unknown不重跑），新增{finalcost['new_closed_loop_rollouts']}个完整H700 episode及832个短probe，共{finalcost['total_new']}、cap65,000。按全pilot平均{finalcost['mean_seconds_full_pilot']:.2f}s估计{finalcost['predicted_cpu_hours']:.2f}CPU小时，@24CPU={finalcost['predicted_wall_hours_24']:.2f}h、@18CPU={finalcost['predicted_wall_hours_18']:.2f}h；K16包含更多可能慢的η，仍非完成时间保证。"
    adjustments='执行调整：历史协议扫描遇到states为计数整数而非列表，首次Stage2设计在新outcome前停止，修正解析后继续；未修改设计/预注册。GPU静态作业首次以JAX_PLATFORMS=gpu请求后端，当前JAX展开到不存在的ROCm而初始化失败，未生成任何特征；改为显式cuda并重提，等待训练的作业也在未启动时替换/重接依赖。见STAGE2_PREPARATION_RESUME.log、stage_2/GPU_FEATURE_REPAIR.json、GPU_TRAINING_REPAIR.json、code_versions/。CPU新标签未重跑，模拟执行identity没有改变。CPU标签收尾时，提前释放原有probe数组的调度依赖，利用空闲CPU槽位；probe不读取闭环标签，其单进程merge仍等待标签merge结束。见stage_2/PROBE_EARLY_START.json。'
    if s2:
        post=read(HERE/'stage_2/POSTFLIGHT.json');probe=read(HERE/'stage_2/probes/POSTFLIGHT.json')
        result['rollout_accounting']=dict(new_closed_loop=4608+post['raw'],new_short_probes=probe['requests'],new_total=4608+post['raw']+probe['requests'],historical_training_seed_slots=57344,historical_validation_seed_slots=21504,historical_stage0_test_seed_slots=16384,unique_historical_seed_slots_used=95232,pilot_seed_slots_reused_in_stage2=2304,unknown_stage1=s1['unknown'],unknown_stage2_panel=post['unknown'],unknown_probes=len(probe['unknown_probe_uids']),unknown_never_retried=True)
        timingtext+=f"\n\n实际全研究新增{result['rollout_accounting']['new_closed_loop']:,}次完整闭环请求及{probe['requests']}次短probe，总计{result['rollout_accounting']['new_total']:,}次。Stage2已合并61,440个格子标签（含2,304个pilot缓存）：有效{post['valid']:,}、unknown {post['unknown']}、缺失/隔离均0。probe有效统计{probe['known_statistics']}、unknown {len(probe['unknown_probe_uids'])}；R5整体未训练，保留失败与partial trace。历史TRAIN/验证/Stage0旧test分别57,344/21,504/16,384 seed格，共95,232；2,304个本轮pilot复用不计入历史独立标签数。"
        relationship='本轮主析因已执行；两臂均仅3个必要controller，低于事前要求4个，主结论INCONCLUSIVE。R4_action的必要controller描述性success为DENSE79.69%、SPARSE78.30%，没有取得正式联合迁移确认。R3与特权坐标的DENSE−SPARSE分别−9.90pp、−26.91pp（同时CI均小于0），因此本设定没有支持增加融合源即可修复；这仍不能唯一归因表示或方法。R5有一个unknown探测，完整保留而未填补，整臂未执行。'
        stage2_note='正式预注册与receipt已冻结，采集、特征、拟合、评价均完成。R5未训练属于事前输入完整性分支，并非零性能。'
        handoff_end='本轮已完成Stage0–2及所有适用分支，新增标签、模型和预测实际存在。必要controller仅3个，低于4个；禁止把描述性失败改写成正式方法缺陷结论。优先复用现有预测检查源验证NLL为何选中R3/oracle的很早训练步（50/100等），以及held-pair输入是否在源特征支持范围内。这些是后续待验证假设，不是本轮已确认根因；不得在已打开终点上调参后冒充新确认。R5失败的一个probe原样保存，不重跑、不填0；因此闭环行为表示优于静态表示的假设仍未检验。未来独立确认另需新state、重新预注册及预算。'
    else:
        relationship='主析因尚未完成，不能写成跨controller选择成功。R3旧冻结输出带有部分controller信息，但选择成功与准确regimeindex不同；t0局部投影响应是否不足，需要真实析因证据。'
        stage2_note='尚无完整Stage2结果，不能借Stage0或pilot替代它们。'
        handoff_end='Stage2尚未执行完成，不能定位coverage/representation/method根因。恢复须遵循正式阶段gate、协议与缓存规则。'
    if s2:
        result['hypotheses']['held_blend_selection_transfer']=('SUPPORTED IN DENSE' if any(v['passed'] for v in s2['primary']['DENSE']['representations'].values()) else 'INCONCLUSIVE' if s2['primary']['DENSE']['necessary_count']<4 else 'REJECTED: JOINT CRITERION NOT MET; NOT ZERO-EFFECT PROOF')
        result['hypotheses']['coverage_vs_representation_vs_method']=s2['rootcause_verdict']
        write(HERE/'results.json',result)
        gate_rows=[r for r in s2['gates']['rows'] if r['split']=='LOPO']
        gt=table(['覆盖','pair/fold','目标C','η_src','targetCFη','L保守下界','95%CI','必要'],[[r['coverage'],r['fold'],r['controller'],r['source_prior_eta'],str(r['target_cf_fixed_eta']),pp(r['L_lower']),ci(r['ci95_conservative']),r['necessary']] for r in gate_rows])
        pt=[]
        for coverage,group in s2['primary'].items():
            for arm,v in group['representations'].items():
                vals=[ci(v['comparisons'][c]['ci95_simultaneous']) if 'comparisons' in v else '未检验' for c in ('src','near','R0','mismatch')]
                pt.append([coverage,arm,group['necessary_count'],*vals,v['status']])
        primary_table=table(['覆盖','表示','必要C数','vs源prior同时CI','vs最近源同时CI','vsR0同时CI','matched−mismatch同时CI','联合门'],pt)
        cover=[]
        for scope,g in s2['coverage_effect'].items():
            for arm,v in g.get('effects',{}).items():cover.append([scope,len(g['targets']),arm,pp(v['mean']),ci(v['ci95_simultaneous'])])
        allrows=table(['划分','覆盖','fold','目标C','表示','必要','success下界','vs源prior','vs最近源','vsR0','Q2 MSE'],[[r['split'],r['coverage'],r['fold'],r['held_controller'],r['arm'],r['necessary'],percent(r['success_lower']),pp(r['gain_lower']['src']),pp(r['gain_lower']['near']),pp(r['gain_lower']['R0']),f"{r['q_mse_observed_Q2']:.4f}"] for r in s2['rows']])
        residual=[];cal=[]
        for coverage in ('DENSE','SPARSE'):
            for arm in ['R0','R3','R4_action','R_oracle','R5_behavior']:
                rr=[r for r in s2['rows'] if r['split']=='LOPO' and r['coverage']==coverage and r['arm']==arm and r['necessary']]
                if not rr:continue
                avg=lambda key:sum(r[key] for r in rr)/len(rr)
                residual.append([coverage,arm,percent(avg('success_lower')),percent(avg('controller_prior_success_lower')),pp(sum(r['gain_lower']['state_residual'] for r in rr)/len(rr)),str([r['learned_controller_prior_eta'] for r in rr])])
                for k in (1,2,4):
                    cr=[r['calibration'][str(k)] for r in rr]
                    cal.append([coverage,arm,k,32*k,percent(sum(r['fixed_success_lower'] for r in cr)/len(cr)),percent(sum(r['offset_success_lower'] for r in cr)/len(cr))])
        s2details=f'''### 实际训练与部署设置

新增32自然test、16probe初态均做全DB及全部历史protocol物理hash去重；旧96train为DEVELOPMENT。每原Flow复用128state8-headQ4，共57,344TRAIN seed格；P1 14controller24stateK16Q4共21,504验证seed格复用。14原Flow始终为源，融合源按LOPO/LOAO排除目标。模型physical23+pad513→64/64/16 MLP，3fitseeds17/23/41，AdamW/12,000steps/NLL选步沿用round4；源controller均匀抽样，source-only归一化。源验证只含原Flow，没有融合源验证标签，须注意源验证目标与融合任务的偏移。

实际独立优化{s2['independent_optimization_fits']}次fit、{s2['checkpoint_references']}个per-task checkpoint引用；SPARSE相同源训练重复折直接复用SHA一致冻结模型。R0/R3/R4_action/R_oracle/R5输入网络参数预算相同；R_oracle已知[λ,R3_A,R3_B]是特权坐标，原Flow自融合[0,R3_C,R3_C]。R5每C用16probe×2η×200step，160个无终局标签统计特征，需要模拟器，可部署假设不同。

probe已执行832次，其中831次提供完整统计、1次数值失败。按事前分支，全部12个R5预测task明确NOT EXECUTED；另外48个task完成，实际84次独立fit。partial trace/unknown保留，不删controller/probe、不填0、不重跑。无标签probe在DB仅登记UNLABELED_SHORT_PROBE source_file与独立UID/manifest，不制造success=0，不修改schema。见probes/POSTFLIGHT.json、R5_FEATURES.npz和RAW_TRACE_INDEX.json。

### 必要性门（仅seed0–1；不读终点）

{gt}

### 覆盖×表示主结果（仅新state seed2–3）

{primary_table}

每表示内部四项配对max-T同时CI、5000次shared-state cluster bootstrap；3fit先平均。主检验必要C≥4；缺输入不伪造结果。跨表示不宣称整个研究familywise error已经控制。Q2有标签噪声，不能代替success选择证据。

### DENSE−SPARSE直接覆盖效应

{table(['目标集合','C数','表示','配对差','跨表示同时CI'],cover)}

主union由两臂discovery必要门定义，两个臂在同一目标/物理state上比较；intersection/all为敏感性。额外源同时增加label量，不把这个实验当纯无混杂的controller覆盖因果估计。

### Controller prior与逐state增益

{table(['覆盖','表示','state选择success','模型controller固定prior success','state残差保守差','模型priorη/3fits'],residual)}

prior只用held controller在96旧train初态的静态/无标签行为输入预测均值，不用目标标签。以上分解描述性，与主门分开；精确argmax index与闭环goodη不等价。

### 目标少样本校准参照（使用目标标签）

{table(['覆盖','表示','k个目标state','fullrollout数','校准固定η success','critic+offset success'],cal)}

取预定前k个S_train，未观测head回退源prior；不读seed2–3。k1=32个full H700 episode，R5=32个200step短probe，episode数量相同而CPU/步数不同，不能称成本精确相等。

### 每fold/每表示结果（含LOAO次要划分）

{allrows}

LOAO只留出包含某A的融合pair，原A checkpoint仍在14源中；不是未见原锚点/真实checkpoint的验证。

### 根因分类

**{s2['rootcause_verdict']}**。按预注册映射分类，必要controller少于4优先INCONCLUSIVE。若oracle没过联合门，必须检查是否只有最近源baseline已正确选regime这一项使门失败，不得据此声称映射无法学。统计未显著不证明效应为零；小样本、源验证、优化与近邻天花板边界同时保留。
'''
    report=f'''# Controller Flow Blending：覆盖、表示和方法的辨识研究

更新时间UTC：{result['updated_utc']}。整体状态：**{result['status']}**。

## A. Research Overview

核心目标是学到可迁移的Q(s,C,η)中controller作用，在未见controller上选择闭环成功率更高的η。上一轮真实Flow LOCO未获独立选择确认，但必要折均为η13经验最优、源先验为η10；增加七个源没有增加η13经验最优样例。本轮先复核controller η regime，再用连续world-space Flow融合跨越regime，满足pilot门后才做覆盖×表示析因。Q误差、regime分类与逐state选择不是同一终点。

## B. Implementation Summary

读取AGENTS.md、完整round5 prompt、round4 HANDOFF/REPORT/results/SOURCE_REGIME_COVERAGE及shared DB规范。新代码只在本目录；历史数据、checkpoint和报告只读、不commit。纯Q不要求视频。

Stage0由stage0.py执行，复用14个controller共享128state的随机平衡8-head Q4标签、静态特征以及冻结的controller_prior/predictions；共{s0['reused_training_seed_slots']}个TRAIN seed格（有效{s0['valid_seed_slots']}、unknown{s0['unknown_seed_slots']}），零rollout、零重训critic。每η64state，按128个共享初态bootstrap5000次，保留head请求mask；经验bestη同数据选择，仅探索性，报告unknown上下界和14×2 max-T区间。NN/logistic只训练regime分类器，LOO只由13源做normalization、L2=1固定无调参。

blend_runtime.py复用旧RingFlowAdapter、native v10 runtime、FF积分和安全投影。在每个共同world h处调用两个Flow自己的chart/normalizer，凸组合世界速度；h0用共同ξ凸组合两种反归一化。basis参考nominal为融合场无修正积分后终端chart clipping的凸组合；这个原prompt未具体定义的接口选择已事前明确。边界λ0/1保留原float32运行顺序；ENDPOINT_EQUIVALENCE_AUDIT.json的160次静态η/state/seed检查，端点最大误差为0，绕过分支的world公式也为0，邻近λ误差≤7.1e−7；禁止env.step，不以端点分支替代公式检查。global/local环境schema名称不同，其余物理配置逐项相同；初次严格schema检查停止，排除仅chart名称后继续，尚未产生outcome。

Stage1设计为4个固定锚点pair×3内部λ×24旧S_train×K4[2,4,10,13]×Q4=4608，98timing计入其中。新UID包括两个锚点完整execution identity、λ、runtimeSHA、nominal/chart规则。design.py先登记experiment/UID，再官方DBpreflight（4608确切缺失、0reuse、0ambiguous），仅执行missing；rollout.py写raw+journal，单进程merger、postflight、stage1_analyze.py。固定物理Ring/H700/FF/安全/评价器/JAX x64=false与seed语义。

Stage1翻转门事前明确检查三个内部λ：它们才是Stage2的12个目标/新增源；仅端点已知差异不证明内部新controller提供覆盖。原锚点标签前24state只有随机8头，K4每η12state，没有补跑缺失锚点。此操作化比把端点计入的解释更严格，报告实际定义，不伪称两种解释完全相同。{stage2_note}

{adjustments}

## C. Experimental Results

{table(['阶段','状态','新增rollout/模型'],[['Stage0','COMPLETE EXPLORATORY','0 rollout；无critic重训'],['Stage1',stage1state,f'当前raw={raw_count}/4608'],['Stage2',stage2state,'见正式阶段证据；未执行不填0性能']])}

### Stage0：14真实Flow regime

{s0table}

regime计数η2/4/10/13分别2/1/9/2。表中率为unknown保守下界，不是将unknown标成失败；best定义来自valid均值。比较区间是保守配对/按请求mask的探索性max-T，best同标签选出，不能称独立最优确认。sparse随机head意味着不同η覆盖不同state，随机分配不等于每state配对全K16。

{classtable}

分类混淆矩阵和14×13源归一化距离在STAGE_0_RESULTS.json。没有任何表示的这个LOO分类器超过多数类；14小样本、严重不平衡、η4类只有1例，不能推出表示不包含任何regime信息。

### fold1/fold4：现象、解释及regret分解

{asym}

{priortab}

**核查纠正**：上一轮摘要称fold4 R3“选η13”并不等于冻结代码的输出。Stage A fold4的模型自身prior为η4（3fit均如此），fold1为[η4,η10,η10]；Stage B也没有直接学到η13 prior。η4在v10旧测试state表现很好，故“经验η13-type”是标签regime的描述，不能代替模型实际所选η。成功决策不要求精确复现同一数据上经验argmax的index。

{decomp}

上述point项从lower-success算出，相加恒等；各component的配对unknown保守上下界和bootstrap见JSON，不把分别最坏的component CI相加当精确总CI。fold1大量regret来自controller固定prior错误，fold4 R3在开发面板固定prior已很好而逐state偏离反而造成损失。静态均值NN并未找到另一个η13经验最优controller，能描述少数regime邻域稀疏，但不能将模型不对称因果归咎距离或源数量。

### Stage1：融合regime地图

{pilot}

### Stage2：覆盖×表示

{stage2state}。预定SPARSE14原Flow、DENSE另加9非留出pair融合Flow，4 LOPO；R0/R3/R4_action/R_oracle/R5。主终点必须为新testseed2–3选择success，必要门只用0–1，source normalization与原Flow P1验证NLL选步，3fits。若未执行，不存在coverage effect、R5数据、checkpoint或选择性能结果；不能借Stage0或pilot替代它们。

{s2details}

## D. Hypothesis Evaluation

{table(['命题','判定'],list(result['hypotheses'].items()))}

“LOO均值静态分类超过多数类”的否定只对应既定分类器和14点面板，不是对controller条件Q普遍可学习性的否定。门未通过不能用于把coverage/representation/method三种根因逐一排除。

## E. Controller–Q Relationship and Transferability

已知真实Flow改变η regime而非仅Q水平；Stage0不同best/η10/13曲线支持这一现象。{relationship}

融合族内插、留pair和特权已知融合坐标不能自动推广到全新真实Flow checkpoint或跨几何。R5若执行具有200step模拟行为查询部署假设，不是静态descriptor；其成功只能支持该proxy更有用，不能证明因果机制已识别。coverage对比同时改变总标签数量，明确其有限可辨识性；DENSE显著/SPARSE不显著也须检查直接DENSE−SPARSE差。

## F. Limitations and Next Steps

所有旧state为DEVELOPMENT；Stage0经验best存在同数据选择偏差、head覆盖差异、controller少数类稀疏，独立seed≠独立state。Pilot只有24state/Q4/K4，专门找regime边界，不能推断完整K16选择headroom或连续三维success basin形状。融合同时改变速度场与h0初始噪声分布，不能把结果只归因于速度场。R_oracle仍含压缩的锚点R3，而非完整程序的无损真坐标。必要controller门不足、源验证仅原Flow、NLL与选择目标可能偏移、未控制额外标签数量，均阻止唯一根因判断。统计相关、机制解释和理论猜想分别陈述。{handoff_end}

## G. Reproducibility and Cost

工作根目录{ROOT}，Python {PYTHON}，代码版本a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e（无commit）。DATA_REUSE_INDEX.json记录历史资产hash；PREREGISTRATION*.receipt.json记录事前UTC/hash；stage_1/protocol.json含state内容、精确UID/η、checkpoint及两chart。runtime和model JAX x64必须false。

{table(['预注册','UTC','SHA256'],prereg)}

{timingtext}

复现入口（既有raw永不重新运行；先audit再missing）：

```bash
cd {ROOT}
{PYTHON} diagnostics/controller_flow_blend_v1/stage0.py
{PYTHON} diagnostics/controller_flow_blend_v1/blend_runtime.py
{PYTHON} diagnostics/controller_flow_blend_v1/design.py create
{PYTHON} -m shared_rollout_db.plan --manifest diagnostics/controller_flow_blend_v1/stage_1/PLANNED_ROLLOUTS.json --output diagnostics/controller_flow_blend_v1/stage_1/CACHE_PREFLIGHT_RECHECK.json
# 已有MISSING_CASES是原冻结计划；重提作业先核查DB/raw，不重跑unknown。
{PYTHON} diagnostics/controller_flow_blend_v1/rollout.py merge
{PYTHON} diagnostics/controller_flow_blend_v1/rollout.py postflight
{PYTHON} diagnostics/controller_flow_blend_v1/stage1_analyze.py
# Stage2从已保存标签与冻结逐seed预测重算；不会重训或生成rollout。
{PYTHON} diagnostics/controller_flow_blend_v1/analyze.py
{PYTHON} diagnostics/controller_flow_blend_v1/render_report.py
{PYTHON} diagnostics/controller_flow_blend_v1/final_audit.py
```

SLURM_SUBMISSIONS.json为实际作业/commands；CPU24/GPU8按需，用户额外6CPU/4GPU授权到2026-10-14 08:00+08，之后18CPU/4GPU。RESOURCE_AUTHORIZATION.json及独立systemd cutoff定时器只撤本任务r5x_额外作业，不动其他人的作业。未用GPU时不预占GPU。raw命名为精确rollout UID，不覆盖原数据。
'''
    save('REPORT.md',report)
    handoff=f'''# 独立Handoff：Controller Flow Blend Round5

当前UTC：{result['updated_utc']}；任务状态 **{result['status']}**。本文件可脱离对话读取，未完成阶段明确标注。

## 背景和研究目标

目标：从物理state和controller查询学会Q(s,C,η)，在未见controller下获得η选择闭环成功收益。上一轮round4真实Flow LOCO必要折仅2/7，其留出controller经验η13、源priorη10；增加7源没有增加η13样例。独立Stage C的R3 gain+5.21pp CI[-.66,11.08]，错配差+7.51pp CI[2.53,12.49]；未过联合选择门。因此无法唯一定位覆盖不足、表示不足或方法缺陷。

## 实际实施和已完成证据

Stage0已执行，读14个原Flow共享128state平衡8-headQ4标签、静态特征和冻结模型；复用57,344训练seed格，零新增rollout/critic拟合。Stage1实现world-field Flow convex blend，A=base_u/v10、B=wide/v9，4pairs×λ.25/.5/.75。初值同ξ，两chart速度在共同h处转换世界坐标融合，保留FF/OrthoFlow3/投影/H700。端点静态公式及原runtime检查误差0，禁止env.step；x64=false。stage_1/protocol.json冻结4,608K4Q4请求；preflight全部确切缺失、登记experiment先于merge。98timing包含在pilot标签内。

{table(['阶段','当前状态'],[['Stage0','COMPLETE EXPLORATORY'],['Stage1',stage1state],['Stage2',stage2state]])}

14Flow经验regimeη10=9、η13=2、η2=2、η4=1。R3 NN/logistic LOO准确率64.3%=多数类；R4_action NN35.7%，logistic64.3%；组合也是64.3%。分类只对14个controller均值描述，无调参，不支持源外独立确认。

重要事实纠正：fold4冻结R3 controller prior实际选η4，3fit都如此，而不是摘要的η13。它在旧测试面板很好；fold1 prior[4,10,10]失败。精确index分类与选择success不可混同。旧报告中也存在Stage A组合逐state−modelprior point+2.21pp CI[-.01,4.44]，因此“所有point都不高于prior”过强；没有可靠正的state残差选择证据。

## 本轮最重要结果

{pilot}

{timingtext}

{s2details}

## 成立、否定和未解决

{table(['假设','判定'],list(result['hypotheses'].items()))}

Stage0未超过多数类不能证明表示没有regime信息；pilot不检验selection泛化。{relationship} 合成family的结果不能直接覆盖真实未见checkpoint、跨geometry或连续success basin。

## 可直接复用资产及下一步入口

目录：{HERE}。先读REPORT.md、results.json和PREREGISTRATION_STAGE_1.md；STAGE_0_RESULTS.json给14条曲线CI/距离/混淆/冻结prior/regret分解；DATA_REUSE_INDEX.json给历史hash。ENDPOINT_EQUIVALENCE_AUDIT.json给语义检查。stage_1/含协议、CASES/MISSING_CASES、CACHE_PREFLIGHT、PILOT_TIMING、raw和labels/results（只有真正产生后存在）；shared DB的experiment为exp_controller_flow_blend_v1_stage_1。

重放分析命令：{PYTHON} {HERE}/stage0.py；{PYTHON} {HERE}/stage1_analyze.py（仅postflight成功后）；{PYTHON} {HERE}/render_report.py。历史round4可复用{PREVIOUS}/LABELS.npz、stage_b/LABELS.npz、static_features/、controller_prior/，勿覆盖。

{handoff_end} Stage2证据：stage_2/GATE_RESULTS.json、results.json、TRAIN_LABELS.npz、DISCOVERY_LABELS.npz、ENDPOINT_LABELS.npz、static_features/、fits/、shared_sparse_fits/、probes/RAW_TRACE_INDEX.json；复算使用analyze.py，再render_report.py和final_audit.py。完整证据审计在FINAL_CONSISTENCY_AUDIT.json。

资源：明天2026-10-14 08:00+08撤6CPU/4GPU额外容量，剩18CPU/4GPU可继续任务。资源授权/调度在RESOURCE_AUTHORIZATION.json、SLURM_SUBMISSIONS.json；只撤本人本任务r5x_作业。所有旧round4 state结果DEVELOPMENT，未来确认另需全新state。
'''
    save('HANDOFF.md',handoff)
    print(json.dumps(dict(status=result['status'],stage1=stage1state,stage2=stage2state,report=str(HERE/'REPORT.md'))))

if __name__=='__main__':main()
