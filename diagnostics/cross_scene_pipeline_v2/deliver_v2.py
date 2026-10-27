"""Chinese development report from sealed evidence, then final integrity delivery."""
from common import *
from audit_v2 import database,training,integrity,cpu_handoff,cutoff_accounting
from names_v2 import METHOD_NAMES,SCENE_NAMES,CONTROLLER_NAMES
import numpy as np,subprocess

def pp(x):return f'{100*x:+.2f}'
def pct(x):return f'{100*x:.2f}%'
def difference(row):return f"{pp(row['estimate'])}个百分点，95%区间[{pp(row['ci'][0])}, {pp(row['ci'][1])}]"
def triple(eta):return '（'+', '.join(f'{v:.8g}'for v in eta)+'）'

def candidate_label(key,eta):
    name={'sE29':'η-A，扩展范围候选','sE05':'η-B，扩展范围候选','eta13':'η13','safety':'无修正候选'}.get(key,'场景级生成器提案'if key.startswith('proposal_')else'学习选 η v1的生成器提案'if key.startswith('v1_proposal_')else'固定修正候选')
    return f'{name}（{key}，{triple(eta)}）'

def method_label(method):
    if method in('eta_A','eta13'):
        data=read(HERE/'INPUTS_V2.json');key='sE29'if method=='eta_A'else'eta13'
        return METHOD_NAMES[method]+' '+triple(data['etas'][data['eta_keys'].index(key)])
    return METHOD_NAMES[method]

def selected_etas(fold,name,ch,method):
    choice=ch['methods'][method]
    if not choice['applicable']:return '不适用：该场景超出训练范围，按规定不执行'
    if method=='v1_pipeline':
        old=read_evidence(ROOT/'diagnostics/cross_scene_pipeline_v1/folds'/fold['name']/'CHOICES.json')[name];pairs={}
        for c,indices in enumerate(old['pipeline']):
            for s,a in enumerate(indices):
                eta=old['candidate_etas'][s][a];key=old['candidate_keys'][a]if a<49 else f"v1_proposal_state{old['state_indices'][s]}_index{a}"
                pairs.setdefault((key,etahex(eta)),eta)
        return '；'.join(candidate_label(key,eta)for(key,h),eta in pairs.items())
    if method=='scene_xstate':
        weights=np.asarray(choice['weights_by_state']);indices=np.flatnonzero(weights.sum(0)>0)
        return '各状态的并列候选按权重平均；候选并集：'+'；'.join(candidate_label(ch['candidate_keys'][i],ch['candidate_etas'][i])for i in indices)
    return '；'.join(candidate_label(r['key'],r['eta'])for r in choice['selected_candidates'])

def comparisons_document(group,heading):
    tables=read(HERE/'EVALUATION_LABEL_TABLES.json');stat=read(HERE/'STATISTICS.json');out=[f'# {heading}\n\n','方法名称遵循说明中的名称表。候选代号仅定位身份，括号中的三个数依次为朝目标、侧移、互斥；精确float64与逐格映射保存在冻结CHOICES.json。unknown计失败，并列参照按冻结权重平均。\n\n']
    for name,row in tables[group].items():
        fold=next(f for f in read(HERE/'FOLDS.json')if f['name']==row['fold']);ch=read(HERE/'folds'/fold['name']/'CHOICES.json')[name]
        out.append(f"## {SCENE_NAMES[name]}（{name}）\n\n| 方法 | 成功率 | unknown计成功上界 | 完成格/适用格 | 选中的η（朝目标, 侧移, 互斥） |\n|---|---:|---:|---:|---|\n")
        y=np.asarray(row['y']);u=np.asarray(row['unknown']);done=np.asarray(row['completed']);app=np.asarray(row['applicable'])
        for i,m in enumerate(row['methods']):
            if app[:,:,i].any():rate=pct(y[:,:,i].mean());upper=pct(np.minimum(1,y[:,:,i]+u[:,:,i]).mean());counts=f'{done[:,:,i].sum()}/{app[:,:,i].sum()}'
            else:rate=upper=counts='不适用'
            out.append(f"| {METHOD_NAMES[m]} | {rate} | {upper} | {counts} | {selected_etas(fold,name,ch,m)} |\n")
        out.append('\n')
    return ''.join(out)

def primary_report():
    s=read(HERE/'STATISTICS.json');p=s['primary'];g=s['development_goals'];c=s['loso']['comparisons'];eta=read(HERE/'INPUTS_V2.json');aa=eta['etas'][eta['eta_keys'].index('sE29')]
    flags='、'.join(f"{k}{'满足'if g[k]['passed']else'不满足'}"for k in('G1','G2','G3'))
    first=(f"{METHOD_NAMES['v2']}减{METHOD_NAMES['global_fixed']}为{difference(p)}；{flags}，三条开发目标{'全部满足'if g['all_passed']else'未全部满足'}。"
        f"减{METHOD_NAMES['eta_A']}（η-A，扩展范围候选sE29={triple(aa)}）为{difference(c['v2_minus_eta_A'])}；减{METHOD_NAMES['knn']}为{difference(c['v2_minus_knn'])}；"
        f"留一族测瓶颈族为{difference(s['lofo']['lofo__bottleneck']['comparisons']['v2_minus_global_fixed'])}，测文献族为{difference(s['lofo']['lofo__literature']['comparisons']['v2_minus_global_fixed'])}。\n\n")
    lines=[first,'本轮仅是开发证据：在本修正族、本控制器集与本安全层下，对这11个去重场景。去重规则、超范围规则与25%边距均在看过学习选 η v1（按状态和控制器选）的结果之后确定；未使用保留给最终检验的新场景定义或结果，不外推未测拓扑。该保留目录只检查os.stat元数据。\n\n',
        '学习选 η v2的生成器为整个场景提出5组η；critic在训练场景候选平均成功率的先验对数几率之上加λ×3×tanh(raw/3)，λ只用外层训练场景内部留一结果选择。将预测成功率对状态、合格控制器类型平均后，整个场景选一组η。场景超范围时执行所有场景一组 η（训练场景里最好）；状态超范围时仅在打分时退回先验。所有31折模型、先验、带宽、λ、超范围判定、提案及全部选择均在任何本轮新rollout之前冻结。\n\n',
        '状态输入为沿用的6维初始物理编码；场景描述为说明规定的12维尺度无关量。标准化只使用该折训练场景的训练状态，超范围边界用该折训练场景的全部初始状态；先验按场景、类型、状态等权，用该折训练场景全部状态，unknown计失败。合格控制器与种子0、数值执行语义、成功标准均沿用前轮。\n\n',
        '控制器：'+'、'.join(f'{n}（{code}）'for code,n in CONTROLLER_NAMES.items())+'；分布式模型预测控制的既定实例为DM0。每场景沿用既有能力门通过的类型，不补齐未通过类型。\n\n',
        '764次未共享排程中有263次内部拟合与其他折完全相同，本轮共享这些源集合、先验、标准化、损失、采样和种子均相同的检查点，实际完成501次拟合。优化器、验证与早停沿用指定实现；固定初始化key=0，三个种子控制独立采样流。训练、每种子曲线及选择见TRAINING_SUMMARY.md和TRAINING_AUDIT.json。\n\n',
        '| 开发目标 | 未四舍五入数值判定 | 是否满足 |\n|---|---|---|\n',
        f"| G1 | 差值{pp(p['estimate'])}个百分点，最低逐场景差值{pp(g['G1']['worst_scene_difference'])}个百分点；要求分别≥+3、≥−10 | {'是'if g['G1']['passed']else'否'} |\n",
        f"| G2 | 两个留一族方向差值分别{pp(g['G2']['directions']['lofo__bottleneck'])}、{pp(g['G2']['directions']['lofo__literature'])}个百分点；均要求≥−5 | {'是'if g['G2']['passed']else'否'} |\n",
        f"| G3 | 18场景留一的v2·只在 49 个旧候选里选减所有场景一组 η（训练场景里最好）为{pp(g['G3']['full18_estimate'])}个百分点；要求≥0 | {'是'if g['G3']['passed']else'否'} |\n",
        '\n主比较的95%区间仅描述不确定性，开发目标严格按G1–G3的原始JSON数值判断。5000次成对自助法依次重抽场景→控制器类型→状态，各方法共用抽样索引；条件于已冻结拟合与选择，不包含重新拟合的不确定性。unknown和未完成格计失败；另报unknown成功上界和仅完成格结果。timeout不等于deadlock，未认证正式死锁解决。\n\n',
        '| 次要比较 | 差值与95%区间 |\n|---|---|\n']
    for key,a,b in [('v2_minus_eta_A','v2','eta_A'),('v2_minus_knn','v2','knn'),('v2_minus_v1_pipeline','v2','v1_pipeline'),('v2_minus_v2_49','v2','v2_49')]:lines.append(f'| {method_label(a)}减{method_label(b)} | {difference(c[key])} |\n')
    robust=p['robust'];lines.append(f"\n逐场景差值中位数{pp(robust['median'])}个百分点；两端各去1场景截尾均值{pp(robust['trimmed_mean_drop_one_each_tail'])}个百分点；Wilcoxon双侧p={robust['wilcoxon_two_sided_p']:.8g}，符号检验双侧p={robust['sign_test_two_sided_p']:.8g}；胜/负/平={robust['wins']}/{robust['losses']}/{robust['ties']}。\n\n")
    principal=('v2','global_fixed','eta_A','knn','v1_pipeline')
    lines.append('| 留出场景 | '+' | '.join(method_label(m)for m in principal)+' | 学习选 η v2减所有场景一组 η（训练场景里最好）（百分点） |\n|---|'+'---:|'*6+'\n')
    for name,row in s['loso']['per_scene'].items():lines.append(f"| {SCENE_NAMES[name]}（{name}） | "+' | '.join(pct(row['rates'][m])for m in principal)+f" | {pp(p['per_scene'][name])} |\n")
    lines.append(f"\n18场景留一场景，v2·只在 49 个旧候选里选减所有场景一组 η（训练场景里最好）：{difference(s['full18']['comparisons']['v2_49_minus_global_fixed'])}，全部直接使用旧标签。\n\n")
    generated=[]
    for fold in read(HERE/'FOLDS.json'):
        if not fold['generator']:continue
        for name,ch in read(HERE/'folds'/fold['name']/'CHOICES.json').items():
            selected=ch['methods']['v2']['selected_index']
            if selected>=49:generated.append(f"{fold['name']}的{SCENE_NAMES[name]}（{name}）：{selected_etas(fold,name,ch,'v2')}")
    lines.append('学习选 η v2选中生成器提案的场景：'+('；'.join(generated)if generated else'无')+'。\n\n')
    lines.append('| 方法 | 11场景等权成功率 | unknown计成功上界 | 仅完成格成功率 |\n|---|---:|---:|---:|\n')
    for method,row in s['loso']['rates'].items():
        completed=s['completed_only']['loso']['rates'][method];lines.append(f"| {method_label(method)} | {pct(row['rate'])} | {pct(row['unknown_success_upper'])} | {pct(completed)if completed is not None else '无完成格'} |\n")
    gen=s['generator_only'];lines.append(f"\n只用生成器仅在{gen['applicable_scene_count']}个不超范围场景描述：场景等权成功率{pct(gen['rate'])}，unknown计成功上界{pct(gen['unknown_success_upper'])}；适用场景集合不同，不参与开发目标或与11场景宏平均的直接比较。未完成格共{s['uncompleted_cells']}个，逐格清单与完成格覆盖数见UNCOMPLETED_CELLS.json及STATISTICS.json。\n\n")
    if 'completed_only'in gen:lines.append(f"只用生成器另报仅完成格：场景等权成功率{pct(gen['completed_only']['rate'])}，每场景内仍按控制器类型等权；覆盖数见STATISTICS.json。\n\n")
    diagnosis=read(HERE/'V1_DIAGNOSIS.json');observed=diagnosis['macro']['nearest3_other_states'];lines.append(f"v1诊断复算：每个场景一组 η（本场景其他状态选，参照）为{pct(diagnosis['macro']['scene_cross_state'])}。最近3个其他状态的选择复算为{pct(observed)}，与说明提供的67.90%有差异；距离并列按状态顺序、候选并列按规范顺序，本项按说明只报告，不作为调整模型或选择的依据。其余诊断和差异见V1_DIAGNOSIS.json。\n\n")
    lines.append('逐场景全部方法、成功率与所选η三个数见METHOD_COMPARISON.md；两个留一族方向见LOFO_METHOD_COMPARISON.md；18场景零rollout对照见FULL18_METHOD_COMPARISON.md。三处改进的成对归因见ATTRIBUTION.md，排序诊断见RANKING_DIAGNOSTICS.md；各折λ、带宽与触发维度见LAMBDA_AND_GUARD.md。\n\n')
    db=read(HERE/'DB_POSTFLIGHT.json');cost=read(HERE/'ACTUAL_COST.json');lines.append(f"本轮按UID去重计划请求{db.get('planned_unique_requests',db['unique_requests'])}条，其中{db['unique_requests']}条具有已完成结果；数值unknown {db['new_unknowns']}条，保留且不重跑；完成结果、身份、元数据、η精确字节与轨迹哈希均与数据库相符，无未合并journal。登记仅INSERT新身份，登记前后已有行逐表哈希相同。\n\n")
    lines.append(f"试跑和正式新执行的单CPU单局耗时合计{cost['single_cpu_episode_wall_hours_sum']:.2f}小时；最长单局{cost['maximum_episode_seconds']/60:.2f}分钟；正式阶段墙钟{cost['stage_wall']['main']['seconds']/60:.2f}分钟。用户临时授权后无他人运行时最多23个worker加1协调CPU；遇实际CPU/内存需求让出，GPU使用0。没有按无新结果时长撤销重交。逐UID独占从一开始避免长任务阻塞后续请求；仅因时限或实际资源需要用已验证检查点交接。\n")
    if(HERE/'CPU_PLACEMENT_HANDOFF_CHECK.json').exists():
        handoff=read(HERE/'CPU_PLACEMENT_HANDOFF_CHECK.json')
        steps='、'.join(str(r['paused_next_step'])for r in handoff['handoffs'])
        lines.append(f"\n用户要求继续加速后，发现两局长请求仍绑定能效核心，而性能核心空闲。通过原有操作检查点分别在第{steps}步保存，旧进程退出后由Slurm正常重新分配到CPU 0/1续跑，未绕过资源分配或修改亲和性限制。两类核心指令集标志一致，源代码、模型、候选、选择和请求不变；最终轨迹与保存进度中的位置、速度前缀逐字节相同。核对见CPU_PLACEMENT_HANDOFF_CHECK.json。\n")
    if(HERE/'USER_CUTOFF_EVIDENCE.json').exists():
        e=read(HERE/'USER_CUTOFF_EVIDENCE.json');lines.append(f"\n按用户最后指令，最后一局只用生成器的描述性请求按4000步截断结算。执行进程在安全检查点第{e['safe_pause_step']}步暂停；仅保存前4000步作为结算轨迹，多出的{e['extra_simulated_steps_not_used_for_cutoff_report']}步不用于截断结算。原4800步请求与完整操作检查点保留，这1格标记为未完成、主值计失败，另报仅完成格结果；不生成原时限timeout或deadlock标签，不写入部分rollout行或journal。学习选 η v2及全部G1–G3所需格子均已完成，判定不受该截断影响。部分执行实际耗时{e['actual_partial_episode_wall_seconds']/60:.2f}分钟，另列于ACTUAL_COST.json；与已完成执行合计{cost['total_completed_and_partial_episode_wall_hours']:.2f}单CPU小时。\n")
    (HERE/'REPORT.md').write_text(''.join(lines));return first

def auxiliary_reports():
    s=read(HERE/'STATISTICS.json');comp=s['loso']['comparisons'];failures=[]
    writers={
      'METHOD_COMPARISON.md':lambda:comparisons_document('loso','11个去重场景留一场景：全部方法与η'),
      'LOFO_METHOD_COMPARISON.md':lambda:comparisons_document('lofo__bottleneck','留一族：文献与对穿训练，测试瓶颈')+'\n'+comparisons_document('lofo__literature','留一族：对穿与瓶颈训练，测试文献'),
      'FULL18_METHOD_COMPARISON.md':lambda:comparisons_document('full18','18场景留一场景：49候选零rollout对照')}
    for name,writer in writers.items():
        try:(HERE/name).write_text(writer())
        except Exception as exc:failures.append(dict(file=name,type=type(exc).__name__,message=str(exc)))
    attrs=['# 三处改进的成对归因\n\n','全部限定在49个旧候选内，直接复用既有执行标签；这些是看过学习选 η v1（按状态和控制器选）结果后预设的开发比较，不外推到新拓扑。\n\n| 比较 | 差值与95%区间 |\n|---|---|\n']
    for method in('v2_49_raw_description','v2_49_no_fallback','v2_49_clip','v2_49_no_prior','v2_49_lambda1','v2_49_per_cell','v2_49_per_controller'):
        attrs.append(f"| {METHOD_NAMES['v2_49']}减{METHOD_NAMES[method]} | {difference(comp['v2_49_minus_'+method])} |\n")
    (HERE/'ATTRIBUTION.md').write_text(''.join(attrs))
    diag=read(HERE/'RANKING_DIAGNOSTICS.json');lines=['# 排序诊断\n\n','同格排序准确率：在同一状态和控制器下，对全部成功/失败候选对计算成功者分数较高的比例，并列计一半；仅计成败都有的格子，场景内格子等权，再按场景等权。场景Spearman比较候选场景分与真实类型等权成功率。应用状态级打分退回；场景级退回仅决定执行选择。\n\n']
    names={'full_selected':'学习选 η v2的critic（所选λ）','full_lambda1':'v2·修正不收缩的critic（λ=1）','no_prior':'v2·无先验的critic','raw_selected':'v2·原始场景描述的critic','prior':'各候选训练场景平均成功率先验'}
    for group,rows in diag['groups'].items():
        lines.append(f'## {group}\n\n| 场景 | 对象 | 同格排序准确率 | 场景Spearman |\n|---|---|---:|---:|\n')
        for n,objects in rows.items():
            for mode,r in objects.items():
                accuracy=r['cell_pair_ranking_accuracy'];rho=r['scene_score_spearman'];lines.append(f"| {SCENE_NAMES[n]}（{n}） | {names[mode]} | {accuracy:.6f} | {f'{rho:.6f}'if rho is not None else '无定义（常量）'} |\n")
        for mode,r in diag['macro'][group].items():lines.append(f"| 场景等权平均 | {names[mode]} | {r['cell_pair_ranking_accuracy']:.6f} | {r['scene_score_spearman']:.6f} |\n")
    (HERE/'RANKING_DIAGNOSTICS.md').write_text(''.join(lines))
    guard=read(HERE/'GUARD.json')['folds'];lines=['# 各折λ、先验带宽、超范围与提案选择\n\n| 折与场景 | λ | 原始描述λ | 带宽 | 场景触发维度 | 状态触发比例 | 学习选 η v2所选η |\n|---|---:|---:|---:|---|---:|---|\n']
    for fold in read(HERE/'FOLDS.json'):
        folder=HERE/'folds'/fold['name'];lam=read(folder/'LAMBDA_SELECTION.json');prior=read(folder/'PRIORS.json');choice=read(folder/'CHOICES.json')
        for n,ch in choice.items():
            g=guard[fold['name']][n]['scale_independent'];method='v2'if fold['generator']else'v2_49';selection=selected_etas(fold,n,ch,method)
            raw=lam.get('raw',{}).get('selected_lambda','不适用');lines.append(f"| {fold['name']}，{SCENE_NAMES[n]}（{n}） | {lam['full']['selected_lambda']} | {raw} | {prior['bandwidth']} | {g['scene_trigger_dimensions']} | {pct(np.mean(g['states_out_of_range']))} | {selection} |\n")
    (HERE/'LAMBDA_AND_GUARD.md').write_text(''.join(lines))
    axis=['| 泛化框架 | 比较与范围 | 差值与95%区间 |\n|---|---|---|\n',f"| 跨场景 | 11个去重场景留一场景，{METHOD_NAMES['v2']}减{METHOD_NAMES['global_fixed']} | {difference(s['primary'])} |\n"]
    for key,label in [('lofo__bottleneck','文献与对穿→瓶颈'),('lofo__literature','对穿与瓶颈→文献')]:axis.append(f"| 跨场景 | 留一族{label} | {difference(s['lofo'][key]['comparisons']['v2_minus_global_fixed'])} |\n")
    axis.append(f"| 状态 | {METHOD_NAMES['v2_49']}减{METHOD_NAMES['v2_49_per_cell']}；同时细化到状态与类型 | {difference(comp['v2_49_minus_v2_49_per_cell'])} |\n")
    axis.append(f"| 控制器 | {METHOD_NAMES['v2_49']}减{METHOD_NAMES['v2_49_per_controller']} | {difference(comp['v2_49_minus_v2_49_per_controller'])} |\n")
    (HERE/'AXIS_TABLE.md').write_text(''.join(axis));return failures

def finalize(first,failures):
    stat=read(HERE/'STATISTICS.json');db=read(HERE/'DB_POSTFLIGHT.json');audit=read(HERE/'INTEGRITY_COMPARE.json');assert audit['passed']and db['passed']
    gaps=['# 证据缺口与范围\n\n','本轮为开发证据；规则在看过学习选 η v1（按状态和控制器选）结果后预设，只限本修正族、控制器集、安全层与11个去重场景。保留的新场景没有读取定义或结果，本轮不构成最终检验。\n\n',
        f"未完成格{stat['uncompleted_cells']}个；数值unknown {db['new_unknowns']}条。unknown保留不重跑，未完成格计失败并另报完成格结果。timeout不等于deadlock。\n\n",
        '最近3个其他状态的诊断参考值未完全复现，详见REPORT.md和V1_DIAGNOSIS.json；按说明只报告，不影响模型、选择或判定。\n\n',
        '只用生成器的适用场景集合不同，按条件集合描述，不参与开发目标。置信区间条件于冻结模型与选择，不包含重新拟合训练过程。\n']
    if failures:gaps.append('\n附表排版失败（主线JSON与统计已封存）：\n'+''.join(f"- {r['file']}：{r['type']}，{r['message']}\n"for r in failures))
    if(HERE/'USER_CUTOFF_EVIDENCE.json').exists():gaps.append('\n用户授权最后一局按4000步结算，仅涉及只用生成器的缺口6机状态7、优先级时空A*规划描述性格子。保留原4800步请求与操作检查点，未写伪timeout标签；学习选 η v2的主比较和G1–G3全部格子完成，不受此缺口影响。见USER_CUTOFF_AMENDMENT.json、USER_CUTOFF_EVIDENCE.json。\n')
    (HERE/'GAPS.md').write_text(''.join(gaps))
    reserved_status='元数据未变'if audit['prospective_metadata_unchanged']else'已登记并行任务的元数据变化已单列，按说明豁免'
    with(HERE/'REPORT.md').open('a')as f:f.write(f"\n结束审计通过：{audit['source_files']}个复用输入哈希与{audit['frozen_files']}个冻结文件未变；两个仓库git status/diff符合开工基线；保留最终检验目录{reserved_status}，未打开或哈希其任何文件。数据库与训练审计通过。\n")
    with(HERE/'HISTORY_CHECK.md').open('a')as f:f.write('\n## 完成核对\n\n501次唯一拟合、31折全部模型与选择冻结；旧49候选不重跑，仅按官方预检补新提案缺失请求。原执行语义、数据库、轨迹和两仓库完整性审计通过；保留最终检验的新场景仅os.stat元数据。诊断参考数字差异已单列，没有据此调整模型或选择。\n')
    (HERE/'STATUS.md').write_text(f"# 当前状态\n\n{utc()}：COMPLETE。501次唯一拟合、31折冻结、rollout与统计已交付，最终审计通过。\n\n"+first)
    write(HERE/'DELIVERY_CODE_SHA256.json',dict(files={str(p):sha(p)for p in HERE.glob('*.py')}))
    copied=read(HERE/'AUDIT_SOURCE_COPY.receipt.json');copied['final_target_sha256']=sha(HERE/'audit_v2.py');write(HERE/'AUDIT_SOURCE_COPY.receipt.json',copied)
    files=['REPORT.md','METHOD_COMPARISON.md','ATTRIBUTION.md','RANKING_DIAGNOSTICS.md','LOFO_METHOD_COMPARISON.md','FULL18_METHOD_COMPARISON.md','AXIS_TABLE.md','LAMBDA_AND_GUARD.md','HISTORY_CHECK.md','V1_DIAGNOSIS.json','INPUTS_V2.json','GUARD.json','FOLDS.json','ALL_FOLDS_FROZEN.receipt.json','TRAINING_SUMMARY.md','TRAINING_AUDIT.json','TOOLING_COPY.receipt.json','PILOT_COST.md','STATUS.md','GAPS.md','ITERATION_LOG.md','DB_POSTFLIGHT.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','STATISTICS.json','ACTUAL_COST.json','DELIVERY_CODE_SHA256.json','AUDIT_SOURCE_COPY.receipt.json','CPU_PLACEMENT_HANDOFF.receipt.json','CPU_PLACEMENT_RUNNING_CHECK.json','CPU_PLACEMENT_HANDOFF_CHECK.json']
    reports={n:sha(HERE/n)for n in files if(HERE/n).exists()}
    for n in('USER_CUTOFF_AMENDMENT.json','USER_CUTOFF_EVIDENCE.json','USER_CUTOFF_STATISTICS.receipt.json'):
        if(HERE/n).exists():reports[n]=sha(HERE/n)
    complete=dict(utc=utc(),status='COMPLETE',development_only=True,primary=stat['primary'],development_goals=stat['development_goals'],reports=reports,
        unique_requested_rollouts=db.get('planned_unique_requests',db['unique_requests']),unique_completed_rollouts=db['unique_requests'],new_executions=read(HERE/'ACTUAL_COST.json')['new_execution_count'],uncompleted_cells=stat['uncompleted_cells'],optional_table_failures=failures,
        reserved_prospective_data_not_read=True,all_final_audits_passed=True)
    write(HERE/'COMPLETE.json',complete);write(HERE/'HANDOFF.json',complete)
    write(HERE/'PROGRESS.json',dict(utc=utc(),stage='complete',remaining=0,running=0,pending=0,complete=True,report='REPORT.md'))
    print(first,flush=True)

def main():
    assert(HERE/'main_COMPLETE.json').exists()
    if(HERE/'RESULTS_SEALED.receipt.json').exists():
        for p,h in read(HERE/'RESULTS_SEALED.receipt.json')['files'].items():assert sha(p)==h,p
    else:
        database();cutoff_accounting();training();cpu_handoff()
        seal('RESULTS_SEALED.receipt.json',[HERE/n for n in('STATISTICS.json','EVALUATION_LABEL_TABLES.json','RANKING_DIAGNOSTICS.json','DB_POSTFLIGHT.json','TRAINING_AUDIT.json')],non_main_formatting_uses_sealed_evidence=True)
    first=primary_report()
    try:failures=auxiliary_reports()
    except Exception as exc:failures=[dict(file='auxiliary_reports',type=type(exc).__name__,message=str(exc))]
    integrity();finalize(first,failures)

if __name__=='__main__':main()
