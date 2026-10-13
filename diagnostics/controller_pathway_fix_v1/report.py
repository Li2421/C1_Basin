"""Chinese self-contained report and handoff from actually executed arms."""
from util import *
import shutil,csv
def pp(x):return f'{100*x:+.2f}pp'
def pct(x):return f'{100*x:.2f}%'
def ci(x):return '['+', '.join(pp(z) for z in x)+']'
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def save(name,content):
    p=HERE/name
    if p.exists():
        q=HERE/'report_versions'/(p.stem+'_'+sha(p)+p.suffix);q.parent.mkdir(exist_ok=True)
        if not q.exists():shutil.copy2(p,q)
    p.write_text(content)
def main():
    blend=read(HERE/'blend_results.json');real=read(HERE/'real_results.json');elig=read(HERE/'STAGE2_ELIGIBILITY.json');rep=read(HERE/'STAGE0_REPRODUCTION.json')
    fits=[read(f) for f in (HERE/'fits').glob('*/complete.json')];count=3*sum(f['optimization_performed'] for f in fits)
    positive=blend['positive_transfer_candidates'];realpositive=real['positive_transfer_candidates']
    primary_positive=[p for p in positive if not p['method'].endswith('pessimistic0.5')];primary_real=[p for p in realpositive if not p['method'].endswith('pessimistic0.5')]
    hypotheses=dict(H_method_harm=blend['H_method'],H_positive_blend_transfer='SUPPORTED DEVELOPMENT' if primary_positive else 'INCONCLUSIVE: no primary safe arm established positive gain over both baselines',H_positive_real_Flow_transfer='SUPPORTED DEVELOPMENT' if primary_real else 'INCONCLUSIVE: no primary eligible arm established positive real transfer',H_high_dimension_is_unique_rootcause='INCONCLUSIVE: repair interventions combine representation, shrinkage, uncertainty and fallback; no unique causal identification',H_statewise_generalization='INCONCLUSIVE: not a prospective state-selection confirmation')
    result=dict(status='COMPLETE',updated_utc=utc(),evidence_level='DEVELOPMENT',task='Controller pathway inductive-bias repair with cached labels',stages=dict(stage0=rep['status'],stage1='COMPLETE: all prescribed arms and frozen postprocessing',stage2='COMPLETE: unconditional F3 and source-defined eligible retraining; oracle-low INAPPLICABLE for zero-shot real Flow'),new_rollouts=0,shared_db_writes=0,model_training=dict(independent_fit_seeds=count,prediction_tasks=len(fits),fit_seeds=[17,23,41],logistic_tasks=len(list((HERE/'logistic').glob('*/complete.json'))),source_equivalent_sparse_reuse=True),reused_seed_label_slots=dict(training=94208,validation=21504,blend_test_discovery_and_endpoint=24576,real_test_discovery_and_endpoint=57344,unique_total=197632),resource_policy=read(HERE/'RESOURCE_POLICY.json'),hypotheses=hypotheses,stage0=rep,stage1=blend,stage2=real,conditional_eligibility=elig,figures=read(HERE/'FIGURE_INDEX.json'))
    result['primary_positive_candidates']=dict(blend=primary_positive,real=primary_real)
    result['sensitivity_positive_candidates']=dict(blend=[p for p in positive if p not in primary_positive],real=[p for p in realpositive if p not in primary_real])
    result['database_provenance_audit']=read(HERE/'DB_CONCURRENT_ACTIVITY.json')
    write(HERE/'results.json',result)
    sections=[];summary_rows=[]
    for bed,r in [('融合族',blend),('真实Flow LOCO',real)]:
        sr=[];dr=[]
        for s in r['summaries']:
            e1=s['E1'];e2=s['E2'];e3=s['E3'];ratio=e2['gap_closed_ratio_lower_point']
            planned=s.get('planned_real_primary',True)
            evidence='敏感性 κ=.5' if s['policy']=='pessimistic0.5' else ('源门预定' if planned else '辅助；源门未过')
            sr.append([s['coverage'],s['method'],pct(s['success_lower']),ci(e1['ci95_simultaneous']),e1['passed'],ci(e2['vs_src']['ci95_descriptive']),ci(e2['vs_R0']['ci95_descriptive']),f'{ratio:.3f}' if ratio is not None else 'NA',f"{e3['certain_count']}–{e3['possible_count']}/{s['controller_count']}",s['eligible'],evidence])
            rows=[d for d in r['rows'] if d['coverage']==s['coverage'] and d['method']==s['method']]
            q=[d['D2'] for d in rows if 'mean_predicted' in d['D2']];heads=[d for d in r['D3'] if d['coverage']==s['coverage'] and d['method']==s['method']]
            optimism=sum(d['overestimate_upper'] for d in q)/len(q) if q else None
            freq1=np.mean([d['heads'][1]['selection_probability'] for d in heads]);freq8=np.mean([d['heads'][8]['selection_probability'] for d in heads]);coverage1=np.mean([d['heads'][1]['requested'] for d in heads]);coverage8=np.mean([d['heads'][8]['requested'] for d in heads])
            cor=next(d for d in r['D1_correlations'] if d['coverage']==s['coverage'] and d['method']==s['method'] and d['metric']=='convex_hull')
            dr.append([s['coverage'],s['method'],f"{cor['pearson']:.3f}" if cor['pearson'] is not None else 'NA',pp(optimism) if optimism is not None else '非Q模型',pct(freq1),pct(freq8),f'{coverage1:.0f}/{coverage8:.0f}'])
            summary_rows.append(dict(bed=bed,coverage=s['coverage'],method=s['method'],success_lower=s['success_lower'],E1_lower=e1['ci95_simultaneous'][0],E1_upper=e1['ci95_simultaneous'][1],E1_pass=e1['passed'],E2_src_lower=e2['vs_src']['ci95_descriptive'][0],E2_src_upper=e2['vs_src']['ci95_descriptive'][1],E2_R0_lower=e2['vs_R0']['ci95_descriptive'][0],E2_R0_upper=e2['vs_R0']['ci95_descriptive'][1],E3_possible=e3['possible_count'],eligible=s['eligible']))
        sections.append(f'''### {bed}

H_method 操作判定：**{r['H_method']}**。安全修复数{len(r['safe_repairs'])}，正迁移候选{len(r['positive_transfer_candidates'])}。每个controller/fold/fit的选择、unknown边界、source prior、target fixed及η频率保存在对应results.json的rows。

{table(['覆盖','arm/规则','成功下界','E1 vs R0同时CI','E1过门','E2 vs源prior描述CI','E2 vsR0描述CI','gap闭合比例','E3确定–可能','安全合格','规则证据类别'],sr)}

E1同时CI family包含本测试床的全部实际评价方法及coverage臂；5000次同物理state共享成组bootstrap，3fit平均。E2仅既定必要controller（融合3、真实2），CI描述性；gap闭合比例不clip，bootstrap分母不正时CI为NA，次数见机器结果。E3范围保留unknown，合格要求possible E3=0。真实床不符合源门但同架构生成的其他固定policy只作辅助，最后列明确标记；不能以真实终点选择policy。

{table(['覆盖','arm/规则','D1凸包距离/损失Pearson','D2 selected Q高估上界','D3选η1','D3选η8','源η1/η8请求数'],dr)}

D1全部最近邻/凸包距离、solver状态、Pearson/Spearman在D1/D1_correlations。不同折的source normalization不同，controller与state非独立，相关性不等于因果。D2有unknown保守区间与valid-Q估计，F4概率是regime概率，不能冒充Q，故D2不适用。D3每个η头的请求/valid/unknown/源success/controller覆盖都保留。选择稀有好head不能自动解释为缺标签：请求分配平衡，而源成功率可能不平衡。
''')
    safe_list=table(['测试床','安全修复规则数（含敏感性）','安全修复涉及架构'],[[name,len(r['safe_repairs']),', '.join(sorted({s['method'].split('__')[0] for s in r['safe_repairs']}))] for name,r in [('融合',blend),('真实',real)]])
    candidate_list=table(['测试床','主要正迁移候选（DEVELOPMENT）','κ=.5敏感性候选'],[['融合',str(primary_positive),str([p for p in positive if p not in primary_positive])],['真实',str(primary_real),str([p for p in realpositive if p not in primary_real])]])
    preregs=table(['文件','UTC','sha256'],[[Path(read(f)['file']).name,read(f)['utc'],read(f)['sha256']] for f in sorted(HERE.glob('PREREGISTRATION*.receipt.json'))])
    headline='修复能够消除伤害，按预注册操作映射支持H_method。融合族SPARSE的有界残差+悲观/保守规则有正收益；真实Flow尚无同时满足安全与正收益的规则。高维过参数化是否为唯一根因仍INCONCLUSIVE。' if blend['safe_repairs'] else '预定修复未同时证明非劣与零灾难，H_method作为充分解释未得到支持。'
    key_results='''融合族SPARSE：F2-PCA2+κ=1悲观选择，在3个必要controller上相对源prior及R0的保守收益为+7.81pp（上界+8.33pp），描述性95% CI [1.56,14.06]pp；E1同时CI下界−1.48pp超过−2pp非劣界，E3=0/12。F2-PCA2+保守偏离收益+4.69pp，CI [1.04,8.33]pp，E1下界−1.04pp，E3=0/12。分别闭合target固定η差距55.6%和33.3%；因2次bootstrap分母非正，融合比例CI记NA。

真实Flow：F2-PCA2+保守偏离在2个必要controller上相对源prior提高+8.79pp，描述性CI [4.30,13.48]pp；相对R0提高+7.49pp，CI [3.06,12.11]pp。但在非必要的v9_local上相对源prior损失−10.16pp，E3=1/7，故不能判为安全正迁移。该收益来自v10_local（+17.58pp），base_u为0，两个η13型controller的不对称仍在。B1+保守偏离、F1-PCA1/2/3+保守偏离、F4可满足安全门，但在必要controller上与源prior完全相同（E2=0）。因此尚没有已验证“既安全又有效”的真实Flow修复，不应只看平均收益。

机制诊断：融合DENSE的旧R3/oracle被选η的平均预测高估上界分别+5.17/+12.99pp，凸包距离与相对R0损失的描述性Pearson为.647/.975；有界残差PCA2 argmax的对应高估为−6.57pp。降维/收缩大幅减少误选η1/η8；请求头数本身平衡，不能将错误简单归为这些head没标签。相关性不是因果，各目标明细及solver状态见D1–D3机器结果。'''
    db_note='最终全库hash审计最初FAIL，原记录保留为FINAL_AUDIT_INITIAL_FAIL.json。只读调查发现本轮期间Drone3D独立任务导入66条source行标签，并登记其独立experiment；本目录experiment/source行均0。DB_CONCURRENT_ACTIVITY.json保留基线/当前hash及外部来源。无法证明共享DB每个字节变化的完整归属，故全库不变性标NOT VERIFIED；本轮实际使用的737个历史NPZ/checkpoint/代码文件哈希全一致，新增脚本不写DB，最后调查仅以mode=ro、immutable、query_only执行SELECT。最终审计以限定范围PASS_WITH_NOTES交付，不能伪称共享DB全局未变化。'
    report=f'''# Controller pathway fix：方法偏置、伤害消除与零样本迁移

状态 **COMPLETE**；更新时间UTC {result['updated_utc']}；全部结果 **DEVELOPMENT**。新增rollout **0**，共享DB写入 **0**。

## A. Research Overview

目标是解释并修复Q(s,C,η)的controller通路在未见controller上过度外推、argmax选择被高估η的风险。本轮使用已经打开的Ring/FF/H700/K16标签，检验降维、向R0收缩、悲观/保守选择和两段式regime模型，主终点是实际缓存闭环success，而非Q误差。{headline}

先核查round5：3个必要controller，原主门要求4，故当时INCONCLUSIVE；原R3最大损失−57.81pp、oracle−52.08pp；DENSE−SPARSE分别−9.90pp CI[−17.71,−2.08]、−26.91pp CI[−34.23,−19.59]。R0并非所有目标Q2 MSE都约.06，其最大.165；不能把少数例子推广到全部。核查详见BACKGROUND_FACT_AUDIT.json。round4七折中1/4两必要折，Stage C只有旧确认背景，不在本轮重新确认。14原Flow经验regime分布9/2/2/1沿用源标签核查。

## B. Implementation Summary

读取AGENTS.md、round6完整prompt、round4/5 HANDOFF/REPORT/results/protocol/fits/静态特征和P1验证标签，HISTORICAL_ASSET_INDEX.json逐文件hash。prepare.py登记38个冻结重放任务和48个融合新模型预测任务；replay.py直接加载原checkpoint、原输入与normalization，以JAX x64=false GPU推理。无Flow静态查询、无env.step调用，训练/终点脚本无DB连接与merger。只在本目录写文件，不修改历史数据，不commit，纯Q无视频。

阶段0：所有38任务argmax和原报告success均逐值一致。14个任务辅助logit误差超过自主追加的2e-5门，最大0.0016608；原log保持FAIL，不伪称bitwise重放。归一化重算逐值一致，numpy float32与历史CUDA前向也有小差异（Q最大0.000429），支持浮点/后端差异解释，但没有证明具体kernel是唯一原因。PREREGISTRATION_REPLAY_NUMERICS_ADDENDUM.md明确主用户复现门PASS、辅助逐值门未过；F3用原冻结预测缓存，F2用逐值一致的R0。此调整发生在修复训练前，未按新终点调参。

阶段1：util.Data只把源TRAIN标签用于训练与经验regime；controller级R3在共同96初态上求均值、source-only zscore/SVD取1/2/3维，目标仅投影、不参与PCA。F1-oracle-low输入9维已知融合坐标及源锚点regime。每个输入physical23+pad513，F1沿用64/64/16-head SiLU、AdamW lr.0008/wd.0001/clip5/12k步骤、17/23/41、3fit平均源验证NLL选step。F2先冻结同折R0，再训练16/16/16零输出初始化残差，tanh界±1、wd.01；只对valid heads计loss。F1/F2共48任务×3seed引用，但相同SPARSE源模型只优化一次并SHA复用。F4是source-only固定L2=1分类器，置信度.6，否则src prior。F3a κ1为主、.5为敏感性，全部固定；F3b三fit top1一致且平均预测优势≥.10才偏离，否则prior。F3相当于ensemble部署，原argmax每fit部署后平均，两者区分。统计首次只因NumPy int64无法JSON序列化而停止，转成Python int后从冻结预测重算，没有改统计值、调参或重训，见ANALYSIS_SERIALIZATION_REPAIR.json。

阶段2：资格由融合E1 simultaneous CI下界>−2pp、possible E3=0冻结到STAGE2_ELIGIBILITY.json。所有真实七折B1-F3无条件评价；合格架构只训一次再报告预定合格policies。oracle-low的真实Flow输入需目标自身regime，零样本不可得，故明确INAPPLICABLE，绝不读取目标标签补onehot。实际新MLP独立fit **{count}**，模型预测task **{len(fits)}**，小logistic tasks **{result['model_training']['logistic_tasks']}**。来源规模：训练94,208、P1验证21,504、融合test/discovery24,576、真实test/discovery57,344；去重复用 **197,632 seed标签槽位**（包含unknown，不等同全valid），无新增标签。

资源：与另一任务共享18CPU，本任务优先，GPU最多3shards且每job1CPU；实测3fits37.43s，成本门保留4倍余量低于8h，COST_GATE.json。Slurm执行commands在SLURM_SUBMISSIONS.json与分支dispatch。所有脚本/输入/参数/输出索引独立保存。

## C. Experimental Results

{key_results}

{table(['阶段','状态'],[['0','完成：argmax/success复现；logits非bitwise且公开记录'],['1','完成：F1/F2/F3/F4、E1–E3、D1–D3'],['2',f"完成：{elig['real_fit_tasks']}个真实拟合任务；oracle-low不适用"]])}

{''.join(sections)}

{safe_list}

{candidate_list}

全部图表在figures/：blend/real_E1,E2,E3以及D1,D2,D3各PNG/PDF，共24个文件；FIGURE_INDEX.json。图表使用固定旧matplotlib依赖只读加载，font cache写入本目录。

## D. Hypothesis Evaluation

{table(['假设','判定'],list(hypotheses.items()))}

上述H_method按用户给定“安全修复+基线灾难”映射判定，是方法偏置解释伤害的操作性证据。单纯回退固定η成功不等于critic已学会controller关系。F1同时降维和移除descriptor逐state变化；F2同时缩小网络、强weight decay及限制残差；F3三seed分歧不是校准的Bayesian不确定性。不能把这些干预当成严格证明高维参数化是唯一因果机制。

## E. Controller–Q Relationship and Transferability

明确分开：Q拟合/选择高估(D2)、给定controller的η选择(E1/E2)、controller级regime/prior校准(F4/回退)。恢复安全可支持修复controller通路的归纳偏置，只有E2同时优于src prior与R0才算正迁移候选。全部修复性能由原终点seed评价，没有目标标签offset/调参。源regime信息与特权融合组成不是新真实Flow的oracle输入。必要controller只有3不再被包装成round5旧主门通过；本轮E2按预定规则仅作DEVELOPMENT描述。

## F. Limitations and Next Steps

范围为Ring、固定geometry、FF/H700/K16、给定真实Flow与合成速度融合族；离散η不能刻画完整连续success basin。本轮全部数据早已打开，multiple arms固定并用simultaneous E1保护安全检验，仍非独立确认。真实LOCO仅在该折优化中留出controller，方法资格来自含这些原Flow的旧融合源面板，不能称完全独立新controller确认。未发现正迁移不证明其不存在；恢复安全也不排除覆盖/表示不足。D1相关不因果；D2末端标签噪声/unknown；D3计数不等于有效controller覆盖。源验证只含原Flow，其NLL最优step未必优化融合selection；本轮不打开终点调参。oracle-low是特权压缩regime表示，并非完整真坐标。三fit置信区间按物理state抽样，不包括对新controller总体或大量训练seed的随机抽样不确定性。

最小前瞻方案（未执行）：先从本轮候选冻结一个模型/规则；至少两个新真实Flow controller，部署只用无标签静态表示，不读目标TRAIN regime；每controller64个全新自然state×K16×Q8=16,384闭环请求（先DB精确复用审计、另行成本/预注册），seed0–3仅选独立fixed参照与门，4–7仅评价；source checkpoint冻结，不重新选policy。若只存在安全无收益修复，则下一步是源数据内CV检查controller级regime可辨识性与NLL/selection目标偏移，使用现有标签，不直接扩rollout。该建议不在本轮自动生成数据。

## G. Reproducibility

{db_note}

Python {PY}；工作目录{ROOT}；git a83a76dc8f4bc1bc7526aa59b006f8ae2dd4909e；JAX x64=false。随机拟合17/23/41，bootstrap5000、种子20261013061/62，PCA/logistic deterministic。数据源hash和DB只读hash、checkpointSHA、source normalization/PCA参数、每seed logits均在本目录。

{preregs}

```bash
cd {ROOT}
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/verify_statistics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/evaluate.py blend
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/evaluate.py real
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/plots.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/report.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/audit.py
```

重训通过Slurm，参见submit.py与各dispatch commands；完成checkpoint缓存不重复优化，禁止改终点seed或读取目标标签构造输入。最终审计FINAL_AUDIT.json独立验证收据、历史输入未修改、全部已执行模型/图表/结果；全库不变性与本任务零写入分开登记。失败记录保留，未解决的审计FAIL不可标交付完成。
'''
    save('REPORT.md',report)
    handoff=f'''# 独立Handoff — Controller pathway fix round6

任务COMPLETE，全部证据 **DEVELOPMENT**；{result['updated_utc']} UTC。本文件不依赖对话。

研究问题：高维controller输入与controller数稀少是否导致critic对未见controller外推、选择被高估η，以及降维、收缩、保守选择或两段式regime是否能消除伤害并获得迁移收益。

实际实施：复用round4真实Flow LOCO和round5连续Flow融合LOPO全部兼容标签/静态特征/冻结checkpoint，重放38任务；物理state23维+controller填充513、16-head MLP，源-onlyPCA/normalization/regime，源P1 NLL选step，fit seeds17/23/41，unknown保留、sameη对消，5000次配对物理statebootstrap。六个F1/F2重训arm和F4/F3均执行；真实LOCO只重训源E1/E3门合格架构。oracle-low原始留出Flow需未知regime，标不适用，禁止目标标签泄漏。

复现：argmax/success全一致；辅助logit门14/38未过，max.0016608，原FAIL记录与补充协议保留，F3用原cache避免后端微小数值改变规则。新增rollout **0**，DB写 **0**；复用197,632标签槽位（含unknown）；新独立MLP fit {count}、预测task {len(fits)}。

核心结论：{headline} 融合操作判定 **{blend['H_method']}**；真实 **{real['H_method']}**。

{key_results}

{safe_list}

{candidate_list}

{table(['假设','证据判定'],list(hypotheses.items()))}

结论边界：安全回退并不证明controller→Q已学会，正迁移候选要求必要controller同时优于源prior与R0；候选仍需前瞻确认。低维表示改变有效输入维数与descriptor的state变化，残差同时改变容量/收缩/惩罚，3fit分歧不是校准不确定性；不能唯一归因高维过参数化。coverage/representation与NLL目标偏移仍可能限制正迁移。固定Ring/FF/H700/K16，不能外推跨geometry、连续basin或任意真实Flow。

证据：{HERE}/REPORT.md完整表格；results.json全结果；blend_results.json/real_results.json含每controller E1/E2/E3、D1–D3、bootstrap与unknown边界；BACKGROUND_FACT_AUDIT.json历史数字；HISTORICAL_ASSET_INDEX.json数据hash；PREREGISTRATION*.md及UTC/hash收据；STAGE0_REPRODUCTION.json；STAGE2_ELIGIBILITY.json条件分支；fits/与shared_sparse_fits/逐seedcheckpoint/logits；logistic/分类参数；figures/24张PNG/PDF；FINAL_AUDIT.json最终审计。

{db_note}

可复算：使用REPORT G节命令，evaluate.py从缓存重算、不生成rollout。未经冻结的新policy不能在已打开终点上选出后称确认。后续最有价值的是对候选做新controller/newstate冻结验证；如无正候选，先源数据内审计regime识别与selection校准，避免仅扩大controller标签。前瞻最小设计见REPORT F节，未在本轮运行。
'''
    save('HANDOFF.md',handoff)
    f=HERE/'METRICS.csv'
    with f.open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)
    print(json.dumps(dict(status='COMPLETE',new_fit_seeds=count,positive_blend=positive,positive_real=realpositive)))
if __name__=='__main__':main()
