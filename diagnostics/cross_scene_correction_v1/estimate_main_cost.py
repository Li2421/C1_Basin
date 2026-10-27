"""Submission-time cost forecast from completed pilot and exact historical timings."""
from common import *
from stages import read_result
import numpy as np


def main():
    measured=read(HERE/'PILOT_TIMINGS.json');selected=read(HERE/'SELECTED_CONTROLLERS.json')
    names=('lit_narrow_passage','lit_crowded_random');groups={};excluded=0
    for c in read(HERE/'HISTORY_CASES.json'):
        name=c['state']['scene']
        if name not in names:continue
        ctl=c['controller_payload']['nominal'];chosen=selected[name].get(ctl['type_code'])
        if chosen is None or chosen['controller']!=ctl['controller']:continue
        r=read_result('main',c)
        if r['numerical_failure']:
            excluded+=1;continue
        elapsed=float(r['elapsed_seconds']);assert np.isfinite(elapsed)and elapsed>=0
        groups.setdefault(name+'|'+ctl['type_code'],[]).append(elapsed)
    assert len(groups)==16
    forecast={}
    for key,values in groups.items():
        forecast[key]=dict(historical_observations=len(values),mean_seconds=float(np.mean(values)),
            median_seconds=float(np.median(values)),p95_seconds=float(np.percentile(values,95)),
            missing_requests=16*32,estimated_cpu_hours=16*32*float(np.mean(values))/3600)
    old_total=sum(r['estimated_cpu_hours']for r in forecast.values())
    old_count=sum(r['missing_requests']for r in forecast.values())
    new_full=sum(len(s['states'])*49*sum(c is not None for c in selected[s['name']].values())
        for s in read(HERE/'SCENES.json')if s['family']not in('literature','extrapolation'))
    pilot_count=len(read(HERE/'pilot_CASES.json'))
    preflight=read(HERE/'main_PREFLIGHT.receipt.json')
    assert old_count+new_full-pilot_count==preflight['execution_count']
    pilot_hours=sum(sum(r['all_seconds'])for r in measured['per_scene_controller'].values())/3600
    new_remaining=measured['estimated_new_main_cpu_hours']-pilot_hours
    total=new_remaining+old_total;workers=measured['reference_workers']
    result=dict(utc=utc(),main_submission_utc=preflight['utc'],forecast_basis='Budget at submission, not live remaining-time promise.',
        historical_literature_groups=forecast,historical_literature_backfill_cpu_hours=old_total,
        historical_literature_missing_requests=old_count,historical_numerical_failures_excluded_from_timing_only=excluded,
        new_scene_full_table_cpu_hours=measured['estimated_new_main_cpu_hours'],already_completed_reused_pilot_cpu_hours=pilot_hours,
        new_scene_missing_cpu_hours_estimate=new_remaining,estimated_main_from_submission_cpu_hours=total,
        reference_workers=workers,estimated_main_from_submission_wall_hours=total/workers,
        physical_rollouts_generated_for_forecast=0,new_training=0,
        limitations='New:2 states by2 candidates. Literature:old16 fixed-candidate timings reused to forecast32 new eta; eta-dependent horizon/planning tails and shared hardware can change costs. Excludes extrapolation, audit, and previously discarded attempts. Concurrency varies with other users demand.')
    write(HERE/'TOTAL_MAIN_COST_ESTIMATE.json',result)
    report=HERE/'PILOT_COST.md';text=report.read_text().split('\n## 主标签整体预算')[0]
    text+=f'\n## 主标签整体预算（提交时估计）\n\n文献补缺8192条，利用同执行语义的历史耗时估计约{old_total:.1f} CPU小时；新场景全表估计中扣除已完成并复用的pilot {pilot_hours:.2f} CPU小时，尚需约{new_remaining:.1f} CPU小时。主标签整体从提交起的预算约{total:.1f} CPU小时，按参考{workers}并发约{total/workers:.1f}小时。\n\n这是时长外推，不是保证完成时限，也不是硬上下界；文献新eta的规划耗时可能不同。未含外推、审计和之前撤销的开销，并发会随他人实际资源需求调整。没有为估时补跑或训练。详见TOTAL_MAIN_COST_ESTIMATE.json。\n\n| 文献场景/类型 | 历史耗时样本 | 均值秒 | 补缺请求 | CPU小时估计 |\n|---|---:|---:|---:|---:|\n'
    for key,row in forecast.items():text+=f"| {key} | {row['historical_observations']} | {row['mean_seconds']:.3f} | {row['missing_requests']} | {row['estimated_cpu_hours']:.2f} |\n"
    report.write_text(text)
    print(dict(main_cpu_hours=total,reference_workers=workers,main_wall_hours=total/workers,literature_cpu_hours=old_total),flush=True)


if __name__=='__main__':main()
