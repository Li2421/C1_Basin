"""Describe motion of initially arrived agents using saved state0 evidence."""
from experiment import *
import numpy as np
import csv
from states import config


def main():
    scene = 'lit_narrow_passage'
    q = {x['controller'] for x in read(HERE/'capability_v2_lit_narrow_passage_QUALIFICATION.json') if x['qualified']}
    rows = []
    inputs = []
    for stage in ('safety_'+scene, 'whitebox_'+scene):
        for path in (HERE/stage/'raw').glob('*.json'):
            raw = read(path)
            c = raw['case']
            if c['state']['index'] != 0 or c['method'] != 'safety':
                continue
            ctl = c['controller_payload']['nominal']
            cfg = config(scene, c['state']['N'])
            assert sha(raw['trace_path']) == raw['trace_sha256']
            with np.load(raw['trace_path']) as z:
                if not len(z['actions']):
                    continue
                goals = np.asarray(c['state']['goals'])
                initial = np.linalg.norm(z['positions'][0]-goals, axis=-1) <= cfg.goal_tolerance
                excursion = np.linalg.norm(z['positions']-goals, axis=-1).max(0)
                for i in np.flatnonzero(initial):
                    rows.append(dict(
                        controller=f"{ctl['type']}（{ctl['controller']}）", type=ctl['type'],
                        controller_code=ctl['controller'], seed=c['seed'], agent=int(i),
                        qualified=ctl['controller'] in q,
                        whitebox=ctl['type_code']=='goal_reference',
                        nominal_first_speed=float(np.linalg.norm(z['nominal'][0,i])),
                        projected_first_speed=float(np.linalg.norm(z['safe'][0,i])),
                        executed_first_speed=float(np.linalg.norm(z['actions'][0,i])),
                        maximum_goal_distance=float(excursion[i]),
                        left_goal_tolerance=bool(excursion[i]>cfg.goal_tolerance),
                        team_success=bool(raw['success']),
                        rollout_uid=raw['rollout_uid']))
            inputs.append(dict(path=str(path), sha256=sha(path), trace=raw['trace_path'], trace_sha256=raw['trace_sha256']))
    assert rows and all(r['nominal_first_speed']==0 for r in rows)
    summaries = []
    for label in ('qualified_blackbox','all_blackbox','whitebox_background'):
        rr = [r for r in rows if (r['whitebox'] if label=='whitebox_background' else
              not r['whitebox'] and (r['qualified'] if label=='qualified_blackbox' else True))]
        summaries.append(dict(population=label, initially_arrived_agent_episodes=len(rr),
            moved_after_projection_first_step=sum(r['executed_first_speed']>1e-6 for r in rr),
            ever_left_goal_tolerance=sum(r['left_goal_tolerance'] for r in rr),
            maximum_goal_excursion=max((r['maximum_goal_distance'] for r in rr),default=None)))
    write(HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.json',dict(
        evidence='DEVELOPMENT', scope='Narrow passage original state0 only; descriptive mechanism check',
        summaries=summaries, rows=rows, sources=inputs, new_physical_steps=0,
        explanation='Arrival stopping is applied before the centralized projection. An agent at its goal remains a joint decision variable; the projection may move it to satisfy pair constraints. It is not pinned at the goal.'))
    with (HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    text=['# 已到达机器人是否被集中式安全投影移动', '',
        '只读取窄通道场景原始state0的只加安全投影轨迹，逐个检查初始已在目标容差内的机器人。名义输入在到达停止之后确实为零，但它们仍是联合安全投影中的决策变量。此检查是描述性的执行语义证据，不是新增成功终点或死锁因果证明。', '',
        '|统计人群|初始已到达机器人局|第一步投影后非零执行速度|全局曾离开目标容差|最大离目标距离（米）|',
        '|---|---|---|---|---|']
    labels={'qualified_blackbox':'合格黑盒实例','all_blackbox':'全部黑盒实例','whitebox_background':'名义朝目标白盒背景'}
    for r in summaries:text.append('|'+ '|'.join(map(str,[labels[r['population']],r['initially_arrived_agent_episodes'],r['moved_after_projection_first_step'],r['ever_left_goal_tolerance'],r['maximum_goal_excursion']]))+'|')
    text+=['','因此，流程版不能等同于把已到达机器人固定成静态障碍的去中心化控制。中央投影本身可能分配退让运动；它与动力学、安全层替换共同限制论文版死锁证据向流程版的解释。所有实例、seed与机器人数据见CSV/JSON，白盒单列而不进入迁移主结论。']
    (HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.md').write_text('\n'.join(text)+'\n')
    seal('ARRIVAL_RELEASE_DIAGNOSTICS.receipt.json',[
        HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.json',HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.csv',
        HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.md',HERE/'arrival_release_diagnostics.py'],
        saved_trajectories_only=True, descriptive_only=True)
    print(canonical(summaries))


if __name__=='__main__':main()
