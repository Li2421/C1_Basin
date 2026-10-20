"""Apply the recorded development ranking, retaining every attempted result."""
from common import *


def main():
    target=HERE/'DEVELOPMENT_FINAL_SELECTION.json';assert not target.exists()
    assert not (HERE/'FROZEN_PIPELINE.json').exists()
    rows=[];inputs=[]
    for folder in sorted((HERE/'iterations').iterdir()):
        receipt=folder/'ITERATION.receipt.json';assert receipt.exists(),folder
        for path,digest in read(receipt)['files'].items():assert sha(path)==digest,path
        r=read(folder/'RESULTS.json');m=r['summaries']['combined']['type_equal']['methods']['pipeline']
        t=r['development_targets']
        rows.append(dict(iteration=folder.name,strategy=r['strategy'],feature_schema=r.get('feature_schema','mechanism_v1'),
            targets=t,regret_pp=m['crossfit_regret_pp'],gain_pp=m['gain_vs_schedule_pp'],gain_CI95_pp=m['gain_CI95_pp'],
            started=read(folder/'RUN_STARTED.json')['utc']))
        inputs.append(receipt)
    assert len(rows)==9,'Three recorded development rounds must finish before final selection'
    rows.sort(key=lambda r:(not r['targets']['passed'],
        not(r['targets']['zero_harm'] and r['targets']['no_type_worse_than_3pp']),
        r['regret_pp'],-r['gain_CI95_pp'][0],r['started']))
    write(target,dict(utc=utc(),selected=rows[0]['iteration'],ranked_iterations=rows,
        ranking_design=read(HERE/'DEVELOPMENT_ITERATION_SELECTION_DESIGN.json'),
        frozen_despite_unmet_targets=not rows[0]['targets']['passed'],
        stopping_reason='Nine fully recorded source-only model/feature comparisons use the complete eight-type evidence. Further identical rollouts are not justified; observed adaptive headroom and comparison with source-only best fixed repair are separately reported. Prompt permits freezing the best achieved version when targets remain unmet.',
        locked_outcomes_read=False))
    seal('DEVELOPMENT_FINAL_SELECTION.receipt.json',[target,HERE/'select_development.py',HERE/'DEVELOPMENT_ITERATION_SELECTION_DESIGN.json',*inputs])
    with (HERE/'ITERATION_LOG.md').open('a') as f:
        f.write(f"\n## 最终开发版本选择\n\n九次模型/特征比较均已完成，按预先记录的排序选中 `{rows[0]['iteration']}`；精确并列取更早的迭代。开发优越性未达到，按主prompt要求冻结目前最好版本并如实报告。完整开发数据已经齐全，没有因改变输入编码而重跑标签；不通过反复补相同证据或改变置信区间种子追求通过。下一步完成同结构规模曲线，再封存全部开发类型拟合的模型；锁定标签仍未读取。\n")
    print(rows[0]['iteration'],flush=True)


if __name__=='__main__':main()
