"""Make the final user stop decision prominent in the standalone handoff."""
from experiment import *
from closure_scope import FLOW_NAMES,circle_gap


def main():
    original=HERE/'HANDOFF_PRE_USER_CLOSURE.md'
    if not original.exists():original.write_text((HERE/'HANDOFF.md').read_text())
    gap=circle_gap();result=read(HERE/'MIGRATION_RESULTS.json')
    prefix=['# v1最终交接：按用户决定收尾、转入分布式v2','',
        '本节是最终状态，优先于下方保留的历史运行说明。v1不再提交任何新批次，COORDINATOR_STOP永久保留；不得重启协调器或恢复圆形流程版claims。学习型controller与扩展均未启动。', '',
        '流程版结论仅包含窄通道、二维拥挤随机、门口三个完成场景；合并结果在MIGRATION_RESULTS.json的three_scene_fixed_combined中。只有窄通道与门口完成场景内学习的独立评估，二维拥挤随机因8机/14机几何缺口没有学习结果；three_scene_learned_combined为null。四场景合并未计算。', '',
        f"圆形流程版：保留{gap['completed_raw_retained']}条已落盘raw、trace与journal，以及48000条完成的单机能力筛选；{gap['unwritten_discarded']}条未落盘请求直接丢弃，不记unknown、不恢复。部分样本不用于估计迁移效果。见CIRCLE_FLOW_USER_STOP.json。", '',
        'paper_bvc_v2仅是v1内论文版缓冲Voronoi单元修正分支的名字；两个原worker保持原样直到完成。它不等于即将开始的分布式流程v2。论文版四场景的已有记录均保留，完整比较见PAPER_REPRODUCTION。', '',
        '最终入口：REPORT.md、GAPS.md、AXIS_TABLE.md、DB_POSTFLIGHT.json、INTEGRITY_COMPARISON.json、FINAL_NUMERIC_RESULTS.receipt.json、DELIVERY_VALIDATION.receipt.json和V1_FREEZE.receipt.json。严格全量开工审计缺少两个非Python配置的整文件开工hash，不能补造；已使用参数视图与保护源码的比较单列。', '',
        '下一阶段执行docs/prompt_baseline_literature_scenes_distributed_v2_20261020.md，v1目录在v2中只读。特别注意：v1的experiment.py导入会把TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、CUDA_CACHE_PATH指向v1；v2不得直接导入后让缓存写回v1，应按新prompt的源码复用规则处理。仅读取旧raw/trace/checkpoint或复制带来源hash的新文件。', '',
        '最新用户资源授权：若没有其他任务占用，全用户合计最多24个CPU份额、5个GPU份额；每次提交前检查squeue并计入所有现有任务。v1收尾没有利用新增额度启动任何研究批次。', '',
        '下方仅保留构造接口、数据路径、执行语义及历史排障信息，原恢复命令不是继续v1的授权。','', '---','']
    (HERE/'HANDOFF.md').write_text('\n'.join(prefix)+original.read_text())
    (HERE/'PILOT_NOT_STARTED.md').write_text('# 学习型试点最终状态\n\n按用户决定，确定性行为克隆策略、混合密度行为克隆策略及任何十类型扩展均未启动。没有训练checkpoint、试点能力门或迁移成功率；不能把未训练解释为能力不足。PILOT_RESULTS首段描述的是原计划配方，表中“用户决定停止，转入v2”才是最终执行状态。\n')
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n- 最终范围按用户决定更改：协调器退出后取消所有圆形流程worker，保留682条已完成raw/journal，8918条未落盘请求丢弃，不记unknown、不恢复。学习型试点未启动，v1不再提交；论文版BVC修正的原两个worker不动直到完成。最终只合并三个已完成流程场景，结束审计和冻结后进入指定分布式v2，v1只读。\n')
    seal('HANDOFF_FINALIZATION.receipt.json',[HERE/n for n in (
        'HANDOFF.md','HANDOFF_PRE_USER_CLOSURE.md','PILOT_NOT_STARTED.md',
        'ITERATION_LOG.md','finalize_handoff.py','USER_DIRECTED_CLOSURE.receipt.json')],
        final_user_stop_decision_prominent=True,no_more_v1_research_batches=True)


if __name__=='__main__':main()
