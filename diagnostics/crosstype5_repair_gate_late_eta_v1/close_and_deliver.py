"""Closing explicit-input audit, stat-only parallel boundaries, self-contained delivery."""
from common import *
from audit import snapshot
from candidate_menu import METHODS

def diff(a,b):return {k:dict(before=a.get(k),after=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}

def audit_end():
    start=read(HERE/'INTEGRITY_START.json');expanded=dict(start);expected=dict(start['input_sha256'])
    for addition in sorted(HERE.glob('INPUT_ADDITIONS_*.json')):
        for row in read(addition)['files']:
            if row['path'] in expected:assert expected[row['path']]==row['sha256']
            expected[row['path']]=row['sha256']
    expanded['input_sha256']=expected;end=snapshot(expanded);write(HERE/'INTEGRITY_END.json',end)
    inputs=diff(expected,end['input_sha256']);docs=diff(start['existing_document_sha256'],end['existing_document_sha256']);repos={}
    def filtered(status,new):
        added={str(Path(p).relative_to(ROOT)) for p in new}
        return '\n'.join(line for line in status.splitlines() if line[3:] not in added and not line[3:].startswith('diagnostics/'))
    for repo,old in start['repositories'].items():
        now=end['repositories'][repo];changes={k:dict(before=old[k],after=now[k]) for k in ('head','diff_sha256') if old[k]!=now[k]}
        a=filtered(old['status'],{}) if repo==str(ROOT) else old['status'];b=filtered(now['status'],end['user_new_prompts_stat_only']) if repo==str(ROOT) else now['status']
        if a!=b:changes['status']=dict(before=a,after=b)
        if changes:repos[repo]=changes
    parallel={}
    for folder in start['diagnostic_metadata_stat_only'].keys()|end['diagnostic_metadata_stat_only'].keys():
        changes=diff(start['diagnostic_metadata_stat_only'].get(folder,{}),end['diagnostic_metadata_stat_only'].get(folder,{}))
        # Never read or hash these paths to explain a concurrent task's change.
        changes={k:v for k,v in changes.items() if k not in expected}
        if changes:parallel[folder]=changes
    result=dict(utc=utc(),passed=not inputs and not docs and not repos,input_count=len(expected),historical_input_changes=inputs,unexpected_document_changes=docs,repository_changes=repos,parallel_changes_stat_only=parallel,user_added_prompts_stat_only=end['user_new_prompts_stat_only'],forbidden_trees_stat_only=True,no_git_writes=True)
    write(HERE/'INTEGRITY_COMPARISON.json',result)
    seal('INTEGRITY_COMPARISON.receipt.json',[HERE/'INTEGRITY_START.json',HERE/'INTEGRITY_END.json',HERE/'INTEGRITY_COMPARISON.json',HERE/'close_and_deliver.py',*sorted(HERE.glob('INPUT_ADDITIONS_*.json'))])
    return result

def format_gain(r):
    if r is None:return 'N/A（缺少可评估样本）'
    return f"{r['point']:+.2f}pp [{r['CI95'][0]:+.2f}, {r['CI95'][1]:+.2f}]"

def main():
    assert (HERE/'FINAL_STATISTICS.receipt.json').exists()
    audit=read(HERE/'INTEGRITY_COMPARISON.json') if (HERE/'INTEGRITY_COMPARISON.receipt.json').exists() else audit_end()
    final=read(HERE/'FINAL_RESULTS.json');coverage=read(HERE/'COVERAGE_RESULTS.json');g=final['goals'];budget={stage:read(HERE/f'{stage}_BUDGET.json') for stage in ('training','coverage','evaluation')};completeness={stage:read(HERE/f'{stage}_COMPLETENESS.json') for stage in budget};development=read(HERE/'DEVELOPMENT_VALIDATION.json')
    g1=g['G1'];g2=g['G2'];c=coverage['groups']['类型等权合并']['comparisons'];g3=g['G3'];g4=g['G4'];parts=[]
    parts.append(f"完整性审计{'通过' if audit['passed'] else '未通过'}：核查 {audit['input_count']} 个显式受保护输入；禁止前缀与其他非输入diagnostics仅stat，并行变化单列，不据此判失败。旧源码/旧证据/PoC/官方仓库只读；无Git写操作。具体差异见 INTEGRITY_COMPARISON.json。全部19个类型均为开发证据，不作未见controller确认性主张。")
    parts.append(f"pipeline是否学会了不卡就不修：G1 {'达到' if g1['achieved'] else '未达到'}，高安全投影组 {g1['controllers']} 个controller，相对{METHODS[6]} {format_gain(g1['gain'])}，伤害数 {g1.get('harm_count')}，判为不修正比例 {g1.get('no_repair_fraction')}；G2 {'达到' if g2['achieved'] else '未达到'}，低安全投影组 {g2['controllers']} 个controller，相对{METHODS[1]} {format_gain(g2['gain'])}，误判为不修正的可观察比例 {g2.get('mistaken_no_repair_fraction')}。仅G1和G2同时达到才按prompt称本轮解决了不会不修正的问题。")
    parts.append('新的后期η来源是否提出了更好的η、pipeline是否用上：G6，时间表生成器提议对相对同数量冻结提议对 '+format_gain(c['时间表生成器提议对相对同数量冻结提议对'])+'，切换时重查相对同数量冻结提议对 '+format_gain(c['切换时重查相对同数量冻结提议对'])+'；G3低安全投影组相对'+METHODS[2]+' '+format_gain(g3[METHODS[2]])+'，去掉时间表生成器的贡献 '+format_gain(g3[METHODS[3]])+'，去掉切换时重查的贡献 '+format_gain(g3[METHODS[4]])+'。')
    from catalog_names import full_type
    focus='；'.join(full_type(t)+' '+format_gain(final['groups'][t]['comparisons']['较好的人定双 η 时间表']) for t in ('transformer_bc','Flow','centralized_mppi'))
    parts.append('相对人定时间表：G4，'+METHODS[0]+' 相对两种人定双η时间表中较好者 '+format_gain(g4)+'；每次bootstrap重新选择较好人定时间表。只有CI下界大于−1pp才达到本轮追平目标。重点类型：'+focus+'。')
    parts.append('本轮主方法 = '+METHODS[0]+'。G5相对'+METHODS[5]+'：'+ '；'.join(f'{k} {format_gain(v)}' for k,v in g['G5'].items())+'。')
    counts={k:dict(requested_unique=v['requested_unique'],actual_DB=v['actual_DB'],actual_executed_this_stage=v['actual_executed_this_stage'],local_inferred=v['local_inferred'],missing=len(v['missing']),unknown=len(v['unknown'])) for k,v in completeness.items()}
    parts.append('执行和预算：'+json.dumps(clean(dict(budget=budget,completeness=counts)),ensure_ascii=False)+'。真实轨迹与本地提前结束等价推断分开计数；推断不写共享DB，unknown不重跑，timeout不直接称deadlock。')
    parts.append('覆盖比较的完整平均成功率、三种均值提议与相对人定时间表的表格见COVERAGE_DETAILS.md；同数量比较与全部区间复用已计算结果，没有新增rollout或重选模型。')
    parts.append('模型：191个源集合分别训练6维时间表生成器，382个三seed critic拟合；19个整类留出折与全部类型最终版本的checkpoint、概率混合及两种gate阈值均由源类型内部折选择。全部选择在新state第一条rollout提交前封存。参数、曲线、未评分比例与混淆矩阵见 SELECTED_MODELS.json、SELECTED_GATES.json、DEVELOPMENT_VALIDATION.json 和各fit目录。')
    parts.append('执行记录：凌晨覆盖bootstrap已完成，但matplotlib缺失导致自动调度中断约8.5小时；修复时复用全部已计算结果，以NumPy直方图和原生SVG补图，没有安装包或重跑轨迹。后续由主agent当前回合监督接续；非关键交付缺口不再阻塞后续rollout。完整过程见ITERATION_LOG.md。')
    parts.append('已知限制：旧类型全部已被研究者使用；训练state只有稀疏44动作标签，选到未标注动作不计成功评分；其余8个学习型类型无新state评估。十个机制特征向量缺失，按各源集合规则填补。候选数原算术笔误已在模型训练前以独立更正封存，配方未变。原工程v1/v2失败与valid v3分别保留，见 ITERATION_LOG.md，不将失败版本当作通过证明。')
    (HERE/'REPORT.md').write_text('\n\n'.join(parts)+'\n')
    gaps=dict(utc=utc(),all_development=True,integrity_passed=audit['passed'],sparse_training_selected_unscored='Reported per source grid and outer fold, never replaced with predictions',missing_mechanism_vectors=10,fresh_unassessed_learning_types=8,training_q1_structural_missing='Only Safety slot26 has two seeds; other seed1 slots are explicitly -2',unknowns={k:len(v['unknown']) for k,v in completeness.items()},unrecoverable_old_state_physical=read(HERE/'NEW_STATE_DEDUP.json')['unrecoverable_by_quality'],failed_engineering_versions='v1/v2 not valid equivalence evidence; v3 all gates passed',arithmetic_correction='PREREGISTRATION_ARITHMETIC_CORRECTION.md',coverage_bootstrap_distribution='COVERAGE_BOOTSTRAP_IMPLEMENTATION.md')
    write(HERE/'GAPS.json',gaps);(HERE/'GAPS.md').write_text('# 证据缺口与边界\n\n'+json.dumps(clean(gaps),ensure_ascii=False,indent=2)+'\n')
    (HERE/'HANDOFF.md').write_text('# 跨类型5交接\n\n本轮主方法 = '+METHODS[0]+'。全部19个类型都是开发证据。\n\n从 REPORT.md、FINAL_RESULTS.json、COVERAGE.md、GAPS.md、INTEGRITY_COMPARISON.json 开始阅读。冻结身份和全部外层/最终模型在 FROZEN_PIPELINE.json；动作及选择在 NEW_EVAL_CHOSEN_ACTIONS.json，收据早于新state rollout。训练原始来源为跨类型4的34动作槽，本轮 TRAINING_BUNDLE.npz 扩为44槽；安全投影有第二seed，其他第二seed为结构性未观测，不当成失败。\n\n可直接复用：schedule_generator.py 与 schedule_generator_fits/、critic_model.py 与 critic_fits/、candidate_menu.py、gate_selection.py 与 SELECTED_GATES.json；实际执行链为 execution_v3.py/requery_execution_v3.py，冻结生成器在独立CPU helper内推理，实际controller进程XLA_FLAGS必须unset，防止旧controller算术改变。已有given动作身份不变。执行证明以 EXECUTION_REPRODUCTION.json、REQUERY_EQUIVALENCE_v3.json 和 EARLY_TERMINATION_EQUIVALENCE_v3.json 为准。\n\n每个阶段 *_BUDGET.json / *_COMPLETENESS.json 分开保存真实执行、精确缓存和本地推断。*_ALL_LOCAL_EQUIVALENCE.json 仅本地，不进入共享DB，不可用作下一层推断来源。后续新问题先查历史、精确preflight与现有预测/checkpoint；不因critic编码改变重跑旧物理标签。\n\n用户持续执行要求见USER_EXECUTION_RULES.md：不影响实验有效性的缺包、绘图或格式化问题必须记缺口并继续训练与评估，跨轮次持续生效，不再次询问。coordinator.py 是本轮顺序调度与恢复入口；排队与claims不自动重试。2026-10-20 09:00新加坡时间起，CPU/GPU上限为18/2（此前24/5）；以RESOURCE_RELEASE_POLICY.json为准，每次提交查全用户占用。旧目录、源码、PoC、官方仓库和venv只读，禁止前缀只能stat。\n')
    write(HERE/'results.json',dict(utc=utc(),evidence='DEVELOPMENT',integrity=audit,goals=g,coverage=coverage['groups'],development_validation=development,budgets=budget,completeness=completeness,gaps=gaps))
    seal('DELIVERY.receipt.json',[HERE/n for n in ('REPORT.md','GAPS.md','GAPS.json','HANDOFF.md','results.json','FINAL_STATISTICS.receipt.json','INTEGRITY_COMPARISON.receipt.json','FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json','PREREGISTRATION_ARITHMETIC_CORRECTION.receipt.json','COVERAGE_DETAILS.md','USER_EXECUTION_RULES.md','RESOURCE_RELEASE_POLICY.json','close_and_deliver.py')],all_development=True)
if __name__=='__main__':audit_end() if len(sys.argv)>1 and sys.argv[1]=='audit' else main()
