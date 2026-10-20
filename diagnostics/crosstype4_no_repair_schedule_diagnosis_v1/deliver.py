"""Self-contained development report and exact evidence handoff."""
from common import *
from catalog_names import full_type
import csv
from resource_guard import queue

def fmt(r):return f"{r['point']:+.2f}pp [{r['CI95'][0]:+.2f}, {r['CI95'][1]:+.2f}]"
def main():
    integrity=read(HERE/'INTEGRITY_COMPARISON.json');A=read(HERE/'A_DIAGNOSIS_RESULTS.json');B=read(HERE/'B_RESULTS.json');g=B['goals'];combined=A['coverage']['类型等权合并'];resources=queue();write(HERE/'FINAL_RESOURCE_STATUS.json',resources)
    budget={s:read(HERE/f'{s}_COMPLETENESS.json') for s in ('A','B_training','B_evaluation')};write(HERE/'EXECUTION_COUNTS.json',dict(utc=utc(),stages=budget,engineering_actual_replays=56,inferred_DB_writes=0,implementation_rerun_rollouts=0))
    lines=[f"完整性审计{'通过' if integrity['passed'] else '未通过'}。所有既有代码、文档、数据与模型的差异详见 INTEGRITY_COMPARISON.json；禁止研究目录只记录stat；本轮不做Git写操作。",'',f"双η差距诊断：{A['conclusion']}；覆盖子集切换方式 {fmt(combined['切换方式'])}，critic 选 η 对 {fmt(combined['critic 选 η 对'])}，生成器覆盖 {fmt(combined['生成器覆盖'])}。",'']
    b1=g['B1_new_states'];b2=g['B2'];status='达到B1和B2的开发目标' if b1['achieved'] and b2['achieved'] else '尚未同时达到B1和B2的开发目标';b1gain=fmt(b1['gain']) if b1['gain'] else 'N/A';b2gain=fmt(b2['gain']) if b2['gain'] else 'N/A'
    lines.extend([f"pipeline 是否学会“不卡就不修”：{status}；新状态只加安全投影≥98%的{b1['controllers']}个controller，学习双 η 含不修正·可选按进度相对只加安全投影为 {b1gain}，伤害数 {b1['harm_count']}；只加安全投影<90%的{b2['controllers']}个controller，相对学习双 η·可选按进度为 {b2gain}。",'', '本轮全部为DEVELOPMENT。19个类型均已被研究者使用；本轮留一类型训练不构成未见类型确认性证据。','', '本轮主方法 = 学习双 η 含不修正·可选按进度。生成器冻结，critic动作输入10→13，其余结构和训练规则沿用跨类型3。候选加入不修正与按进度切换；19折及最终模型全部只在源类型内部选择checkpoint和概率混合权重。','',f"相对两种人定时间表中较好者：{fmt(g['B3'])}；按进度选项贡献：{fmt(g['B4']['相对去掉按进度'])}；不修正选项贡献：{fmt(g['B4']['相对去掉不修正'])}；controller信息贡献：{fmt(g['B5']['gain'])}。",'', 'A部分逐类型拆解、占比及人定切换步敏感性见 DIAGNOSIS.md；全部动作明细见 A_ALL_ACTION_OUTCOMES.csv；B部分完整19行类型表与分组见 AXIS_TABLE.md。','', '真实执行与等价复用分开统计：'])
    for s,r in budget.items():lines.append(f"- {s}：本阶段真实补跑 {r['actual_executed_this_stage']}；本地提前结束等价复用 {r['local_inferred']}；未知请求 {len(r['unknown'])}；未完成 {len(r['missing'])}。")
    lines.extend(['','执行前16条复现、8条前缀/进度检查、32条提前结束等价检查全部通过，后者覆盖全部19类型。推断标签不写共享DB。数值失败保留unknown；timeout不直接标作deadlock。','', '额外说明与缺口见 GAPS.md，复用路径和下一轮入口见 HANDOFF.md。'])
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    gaps=['# 缺口与解释边界','', '全部为开发证据；没有未见controller类型的确认性结论。','', '按prompt跳过其他8个学习型训练类型的新状态评估；AXIS_TABLE对应类型保留N/A。','', '跨类型3能力门中未提供数值安全投影成功率的实例：'+', '.join(B['capability_missing'])+'。不补跑能力门。','', '机制特征完全复用；历史5个和跨类型3新增5个不完整向量保留NaN，只在源类型内填补，共10个。','', '历史DB中无法恢复全部物理坐标的state无法绝对排除物理重合；UID/content全部去重，可恢复坐标也全部去重，见 NEW_STATE_DEDUP.json。','', '数值失败按unknown处理，不自动重跑。未知标签的配对上下界保留相同UID抵消；较好人定时间表在每次bootstrap重选。','', 'A的对角元为选中单η，没有实际切换，代数分解复用同UID使切换效应为零；另列非对角条件结果与支持量。','', '工程修正与冻结前版本修订详见 ITERATION_LOG.md；真实重跑计数见 EXECUTION_COUNTS.json。']
    if not integrity['passed']:gaps.extend(['','完整性审计未通过，必须先评估 INTEGRITY_COMPARISON.json 的差异，本轮结果不得当作审核通过交付。'])
    (HERE/'GAPS.md').write_text('\n'.join(gaps)+'\n')
    handoff=['# 跨类型4交接','',str(HERE),'', '阅读顺序：REPORT.md → DIAGNOSIS.md / AXIS_TABLE.md → GAPS.md → EXECUTION_COUNTS.json。','', 'A使用旧密封选择，不重训：A_OBJECTS.json / A_ACTION_LINKS.json / A_ALL_ACTION_OUTCOMES.csv / A_BOOTSTRAP.npz / A_DIAGNOSIS_RESULTS.json；figures/A_*.svg可直接导出。','', 'B候选为8个冻结提议+零，441唯一动作，13维编码；b_candidates.py、actions.py及CANDIDATES.receipt.json定义身份。TRAINING_ACTIONS.npz保留34槽与原26索引；TRAINING_BUNDLE.npz可复用标签，动作编码改变不使旧执行标签失效。','', '模型：FROZEN_PIPELINE.json及收据包含19留一类型模型和全部类型最终模型；critic_fits/保存全部source-only曲线、概率、权重及checkpoint。inference.py严格恢复h/c float32、controller统计float64。','', '选择：NEW_EVAL_CHOSEN_ACTIONS.json / NEW_EVAL_CHOICES.receipt.json 早于新状态首次提交。新24状态与NEW_EVAL_INPUTS.npz可读，不可再声称全新确认性state。','', '局部等价标签：各阶段*_ALL_LOCAL_EQUIVALENCE.json，均明确实际源UID、轨迹SHA和未切换证明，绝不写DB；共享DB只有真实rollout标签。phase_complete.py分别核查DB和本地证据，未知不重跑。','', '运行：coordinator.py按A优先、B训练标签、19折模型训练、冻结、B新评估执行；每次提交调用submit_aux.py/resource_guard.py，资源全用户18 CPU/2 GPU。监控信息在MONITOR_STATUS.json；异常不会自动重跑未完成claim。','', '审计：INTEGRITY_START/END/COMPARISON及HISTORY_READ_INVENTORY.json；所有代码和输出仅本轮目录，旧源码/PoC/官方仓库/venv只读，未Git提交。']
    (HERE/'HANDOFF.md').write_text('\n'.join(handoff)+'\n');write(HERE/'results.json',dict(A=A,B=B,integrity=integrity,execution_counts=read(HERE/'EXECUTION_COUNTS.json')))
    # Public manifest retains full type names and identifies all types as opened.
    with (D/'crosstype3_unseen_controller_schedule_v1/ZOO_MANIFEST.csv').open() as f:source=list(csv.DictReader(f))
    fields=list(source[0])+['source_cohort','evidence_scope','selector_holdout_rule']
    with (HERE/'ZOO_MANIFEST.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in source:writer.writerow(dict(row,source_cohort=row['cohort'],cohort='development_opened',evidence_scope='DEVELOPMENT_all19_opened',selector_holdout_rule='entire_type_excluded_for_each_outer_fold'))
    seal('DELIVERY.receipt.json',[HERE/n for n in ('REPORT.md','HANDOFF.md','GAPS.md','results.json','ZOO_MANIFEST.csv','EXECUTION_COUNTS.json','FINAL_RESOURCE_STATUS.json','INTEGRITY_COMPARISON.receipt.json','A_DIAGNOSIS.receipt.json','B_RESULTS.receipt.json','deliver.py')],complete=integrity['passed'])
if __name__=='__main__':main()
