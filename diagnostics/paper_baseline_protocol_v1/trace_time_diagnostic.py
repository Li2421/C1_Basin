"""Saved successful trajectories explain observed time differences descriptively."""
import guard
from guard import HERE
import json,collections
from pathlib import Path
import numpy as np


def features(result):
    trace=result.get('trace')
    if not trace or not Path(trace).is_file()or not result['success']or result.get('completion_time')is None:return None
    with np.load(trace)as saved:
        positions=saved['positions'];goals=saved['goals']
    steps=len(positions)-1
    if steps<1:return None
    dt=result['completion_time']/steps;distance=np.linalg.norm(np.diff(positions,axis=0),axis=-1)
    active=np.linalg.norm(positions[:-1]-goals[None,:,:],axis=-1)>.08
    count=int(active.sum());initial=float(np.linalg.norm(positions[0]-goals,axis=-1).sum())
    return dict(completion_time=result['completion_time'],travelled_distance=float(distance.sum()),
        travelled_over_initial_goal_distance=float(distance.sum()/initial)if initial>0 else None,
        active_robot_steps=count,active_wait_fraction=float(((distance/dt<.01)&active).sum()/count)if count else None,
        mean_active_speed=float(distance[active].sum()/(count*dt))if count else None,
        trace=trace)


def run():
    evidence=json.loads((HERE/'NORMALIZED_EVIDENCE.json').read_text())['rows'];cache={};own=collections.defaultdict(list);papers={}
    for row in evidence:
        if row['benchmark']!='A'or row['part']!='test':continue
        method=row['method']
        if not(method=='学习选修正参数'and row['variant']=='frozen'or method in('dac_author_si','mgr_author_unicycle_tracking')and row['variant']=='global_tuned'):continue
        result=row.get('result')
        if not result or not result['success']:continue
        key=(row['scene'],row['state']);uid=result.get('rollout_uid')or result.get('trace')
        if uid not in cache:cache[uid]=features(result)
        item=dict(nominal=row['nominal'],feature=cache[uid])
        if method=='学习选修正参数':own[key].append(item)
        else:papers.setdefault(method,{})[key]=item
    results=[]
    metrics=['completion_time','active_wait_fraction','mean_active_speed','travelled_over_initial_goal_distance']
    for method,by_state in papers.items():
        blocks=[];pairs=0;missing=0
        for key in sorted(own.keys()&by_state.keys()):
            other=by_state[key]['feature'];values=[r['feature']for r in own[key]if r['feature']is not None]
            if other is None or not values:missing+=1;continue
            delta={metric:float(np.mean([v[metric]-other[metric]for v in values if v[metric]is not None and other[metric]is not None]))
                   if other[metric]is not None and any(v[metric]is not None for v in values)else None for metric in metrics}
            blocks.append(dict(scene=key[0],state=key[1],nominal_pairs=len(values),ours_minus_paper=delta));pairs+=len(values)
        results.append(dict(method=method,paired_physical_states=len(blocks),nominal_pairs=pairs,missing_trace_physical_states=missing,
            eligible_success_nominal_pairs=sum(len(own[key])for key in own.keys()&by_state.keys()),
            mean_ours_minus_paper={metric:float(np.mean([b['ours_minus_paper'][metric]for b in blocks if b['ours_minus_paper'][metric]is not None]))
                if any(b['ours_minus_paper'][metric]is not None for b in blocks)else None for metric in metrics},blocks=blocks))
    result=dict(methods=results,unique_successful_trace_features={str(k):v for k,v in cache.items()},new_rollouts=0,
        interpretation='Descriptive observed-trace subset, matched successes only. Mean nominal differences within physical state, then equal physical-state mean. Active means distance to true goal>0.08m; waiting means physical step speed<0.01m/s. Travel/initial straight-line distance is not shortest-path optimality. Associations do not establish a causal explanation; unavailable historical traces remain missing.')
    (HERE/'TRACE_TIME_DIAGNOSTIC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with(HERE/'REPORT.md').open('a')as stream:
        stream.write('\n## 完成时间的轨迹诊断\n\n共同成功且有真实轨迹的子集上，对照任务时间、未到目标时的停等比例、实际行驶速度与行驶距离/初始直线距离；先在物理初态内平均名义控制器，再在物理初态间等权。具体差值与覆盖数见TRACE_TIME_DIAGNOSTIC.json。它们是观察到的关联，不能用缺轨迹子集冒充全队列，也不作机制因果证明。\n')
    return result


if __name__=='__main__':run()
