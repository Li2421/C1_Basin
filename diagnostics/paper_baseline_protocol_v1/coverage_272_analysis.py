"""Read saved historical coverage evidence; never generate a new rollout."""
import csv
import io
import json
from pathlib import Path
import statistics
import numpy as np
from coverage_272 import HERE, ART, POLICY, load, save, sha
from statistic_rules import hierarchical_interval


def run():
    receipt=load(HERE/'COVERAGE_272_COMPLETE.json')
    assert receipt['complete'] and receipt['frozen_numeric_sources_unchanged']
    summary=load(HERE/'COVERAGE_272_RESULTS.json')
    planned=load(ART/'planned.json')['exact_configs']
    names=load(HERE/'SCENE_NAMES.json')
    rows=[];by_method={}
    for request in planned:
        path=ART/'raw'/(request['rollout_uid']+'.json')
        # This run's preflight found no compatible reusable results.
        assert path.exists()
        result=load(path)
        for key in ('rollout_uid','state_uid','controller_uid','scenario_uid','scene','state_index'):
            assert result[key]==request[key],('Result identity mismatch',key,path)
        for key,field in [('positions','initial_positions'),('velocities','initial_velocities'),('goals','goals')]:
            assert result[field]==request['state'][key]
        trace=Path(result['trace']);assert sha(trace)==result['trace_sha256']
        cfg=request['scenario_payload']['config']
        assert result['horizon_time']==cfg['max_steps']*cfg['dt']
        assert result['success']+result['collision']+result['timeout']+result['numerical_failure']==1
        method=request['controller_payload']['paper_method']
        row=dict(method=method,scene=request['scene'],state=request['state_index'],type_id=request['type_id'],
            success=result['success'],collision=result['collision'],timeout=result['timeout'],
            numerical_unknown=result['numerical_failure'],native_unhandled=int(bool(result['paper_original']['numerical_unknown'])),
            completion_time=result['completion_time'],truncated_time=result['truncated_time'],
            wall_seconds=result['wall_seconds'],episode_length=result['episode_length'],
            minimum_swept_clearance=result.get('minimum_swept_clearance'),
            rollout_uid=result['rollout_uid'],raw=str(path),raw_sha256=sha(path),trace=str(trace),trace_sha256=result['trace_sha256'])
        rows.append(row);by_method.setdefault(method,[]).append(row)
    intervals={};compute={};port_context={}
    gates=load(HERE/'PORT_GATES.json')
    for method,values in by_method.items():
        assert len(values)==136
        blocks={}
        for row in values:blocks.setdefault(str(row['type_id']),{}).setdefault(row['scene'],[]).append(row['success'])
        intervals[method]=hierarchical_interval(blocks)
        intervals[method]['limitation']='Conditional on these fixed17maps; one map per type, not uncertainty over a map population. Exploratory previously used cohort, not generalization.'
        durations=[]
        for row in values:
            with np.load(row['trace'],allow_pickle=False) as trace:
                durations.extend(trace['step_seconds'].tolist())
        compute[method]=dict(recorded_steps=len(durations),step_median_seconds=float(np.median(durations)),
            step_p95_seconds=float(np.percentile(durations,95)),
            episode_mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in values),
            interpretation='One allocated CPU per worker; control plus plant advance per step. Episode wall timing excludes interpreter imports, not total cluster resource time.')
        warnings=sorted({r['benchmark'] for r in gates['pilots'] if r['method']==method and r['port_problem_not_fully_resolved']})
        port_context[method]=dict(port_problem_not_fully_resolved=bool(warnings),
            port_status='端口问题未完全解决（'+','.join(warnings)+'训练试跑）' if warnings else '作者代码移植；原环境复现通过',
            applicable_scenes=sum(r['benchmark']=='B' and r['method']==method and r['applicable']is True and r['scene']in{v['scene']for v in values} for r in gates['applicability']))
    a,b=list(POLICY['methods'])
    left={(r['scene'],r['state']):r for r in by_method[a]}
    right={(r['scene'],r['state']):r for r in by_method[b]}
    both=[k for k in sorted(left) if left[k]['success'] and right[k]['success']]
    paired=dict(method_a=a,method_b=b,common_successful_physical_states=len(both),
        a_mean_seconds=statistics.mean(left[k]['completion_time'] for k in both) if both else None,
        b_mean_seconds=statistics.mean(right[k]['completion_time'] for k in both) if both else None,
        a_minus_b_mean_seconds=statistics.mean(left[k]['completion_time']-right[k]['completion_time'] for k in both) if both else None)
    out=dict(complete=True,rows=rows,conditional_ci95=intervals,paired_success_times=paired,compute=compute,port_context=port_context,
        policy_sha256=sha(HERE/'COVERAGE_272_PARAMETERS_FROZEN.json'),
        code_freeze_sha256=sha(HERE/'COVERAGE_272_CODE_FROZEN.json'),analysis_source_sha256=sha(Path(__file__)),
        comparison='External fixed tuned configurations; no per-case search. Independent generalization pipeline is deferred by user.')
    save(HERE/'COVERAGE_272_ANALYSIS.json',out)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'COVERAGE_272_ROWS.csv').write_text(stream.getvalue())
    table=['', '## 17类逐场景覆盖', '', '| 场景 | '+summary['methods'][a]['name']+' | '+summary['methods'][b]['name']+' |', '|---|---:|---:|']
    rep=load(HERE/POLICY['scenes_source'])
    for entry in rep['selected']:
        name=entry['scene'];display=names.get(name,{}).get('display',entry['title'])
        table.append('| '+display+' | '+str(summary['methods'][a]['by_scene'][name]['success'])+'/8 | '+str(summary['methods'][b]['by_scene'][name]['success'])+'/8 |')
    table.extend(['','## 统计与限制',''])
    for method,v in intervals.items():
        lo,hi=v['ci95'];table.append(f"- {summary['methods'][method]['name']}：覆盖率95%描述性区间 [{100*lo:.2f}%, {100*hi:.2f}%]。")
    table.append('- 区间只反映固定17张地图内初态的重采样变化；旧队列参与过早期方法设计，不是独立泛化成绩。')
    table.append('- 两种方法共同成功的物理初态数：'+str(len(both))+'；配对时间详见COVERAGE_272_ANALYSIS.json，与各自成功条件均值分开。')
    table.append('- 曾未处理异常是过程指标，可能与最终成功、碰撞或超时重叠；超时不等同死锁。')
    table.append('- 外部方法每例使用一个已选配置，不能将其与内部Oracle最多64次的覆盖差直接解释为部署策略优劣。')
    table.extend(['','## 端口状态与计算开销','', '| 方法 | 端口状态 | 单机适用场景 | 控制与推进每步中位数/ms | 每步95分位/ms |', '|---|---|---:|---:|---:|'])
    for method,cost in compute.items():
        context=port_context[method]
        table.append(f"| {summary['methods'][method]['name']} | {context['port_status']} | {context['applicable_scenes']}/17 | {1000*cost['step_median_seconds']:.3f} | {1000*cost['step_p95_seconds']:.3f} |")
    for method,v in summary['methods'].items():
        v.update(port_context[method])
    save(HERE/'COVERAGE_272_RESULTS.json',summary)
    # The required port warning accompanies every method table, including the
    # live summary generated before this evidence-only final analysis.
    report_path=HERE/'COVERAGE_272_REPORT.md'
    text=report_path.read_text()
    for method,context in port_context.items():
        if context['port_problem_not_fully_resolved']:
            text=text.replace(summary['methods'][method]['name'],summary['methods'][method]['name']+'（端口问题未完全解决）')
    report_path.write_text(text)
    with (HERE/'COVERAGE_272_REPORT.md').open('a') as f:f.write('\n'.join(table)+'\n')
    return out


if __name__=='__main__':run()
