"""Clarify final prose/figures without changing any scientific implementation."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ast,difflib

COPIES={'final_reports_supplement_20261022.py':'final_reports_supplement_clarified_20261022.py','optional_delivery_supplement_20261022.py':'optional_delivery_supplement_clarified_20261022.py','correction_continuation_supplement_20261022.py':'correction_continuation_supplement_clarified_20261022.py'}


def main():
    verify_receipt('ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json')
    ready='ANALYSIS_SUPPLEMENT_DELIVERY_CLARIFIED_20261022.receipt.json';assert not (HERE/ready).exists()
    rows=[]
    for old,new in COPIES.items():
        before=(HERE/old).read_text();after=before
        for src,dst in COPIES.items():after=after.replace(Path(src).stem,Path(dst).stem)
        if new.startswith('final_reports_'):
            after=after.replace('窄通道分布式多数低成功实例归为安全层导致的全局失败，保留为死锁候选证据；','窄通道分布式低成功实例归为安全层导致的全局失败并保留在分类分析中；只有实际逐机器人滤波器平衡点死锁计死锁证据，被已到达者阻挡不计死锁证据；')
            needle='\n\n主数值认MIGRATION_NUMBERS'
            insert='\n\n补充分析以ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json和INSTANCE_CLASSES_SUPPLEMENT_20261022为准，原INSTANCE_CLASSES_REVISED完全保留。安全层全局失败的多数计数合并滤波器平衡点死锁与已到达者阻挡；死锁证据仍只计滤波器平衡点死锁。分布式每场景按32个训练state、8类型等权在15菜单中选一个最佳固定动作，精确平局按菜单顺序。TRAIN_SELECTED_FIXED_20261022_<scene>先于对应评估preflight封存；评估严格复用相同执行身份，无结果才按官方preflight补跑。圆形训练标签齐全后才选。主比较在相同独立评估state、controller和seed上计算学习−训练选定固定动作，bootstrap不重选。三种旧固定修正只作附表参照，集中式保留三种固定修正。全部15动作训练排序见MENU_TRAINING_ACTION_RANKING_20261022.md。'+needle
            assert after.count(needle)==1;after=after.replace(needle,insert)
        if new.startswith('optional_delivery_'):
            after=after.replace('plot_correction_revised_20261021.py','plot_training_selected_primary_20261022.py')
        ast.parse(after);p=HERE/new;assert not p.exists();p.write_text(after)
        diff=p.with_suffix('.py.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff)))
    write(HERE/'ANALYSIS_SUPPLEMENT_DELIVERY_MAP_20261022.json',dict(utc=utc(),copies=COPIES,static_copies=rows,execution_or_numerical_changes=False,primary_figure_source='Sealed matched primary evaluation summaries only'))
    seal(ready,[Path(__file__),HERE/'ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json',HERE/'ANALYSIS_SUPPLEMENT_DELIVERY_MAP_20261022.json',HERE/'plot_training_selected_primary_20261022.py']+[p for r in rows for p in (HERE/r['new'],Path(r['diff']))],prose_and_figures_only=True,science_and_statistics_unchanged=True)
    print('PASS: final delivery clarifications sealed;no runtime or numerical changes',flush=True)


if __name__=='__main__':main()
