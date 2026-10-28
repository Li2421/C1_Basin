"""Matched cached evidence only; no rollout, controller or solver calls."""
import datetime,json,statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'paper_after_four_column'


def read(p):return json.loads(p.read_text())
def mean(values):return statistics.mean(values)if values else None


def main():
    scope=read(HERE/'REPRESENTATIVE17_FROZEN.json')
    selected={(e['type_id'],e['level'])for e in scope['selected']}
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
    completed={x['profile']for x in read(OUT/'EXECUTION_STATUS.json')if x['complete']}
    paper=read(OUT/'RESULTS.json');groups=read(HERE/'COVERAGE.json')['families'][4]['cases_detail']
    snapshots={};c1=[]
    for controller in ('centralized_sampling','gradient_mpc'):
        rows=[g for g in groups if g['controller']==controller and(g['type_id'],g['level'])in selected]
        assert len(rows)==136
        for mode in ('fixed_first_point','oracle64'):
            outcomes={}
            for g in rows:
                s=scenes[(g['type_id'],g['level'])];limit=s['config']['max_steps']*s['config']['dt']
                q=g['queries'][0]if mode=='fixed_first_point'else g.get('solution')
                ok=bool(q and q['success'])
                outcomes[(g['type_id'],g['level'],g['state_index'])]=dict(success=ok,
                    completion_time=q['completion_time']if ok else None,
                    truncated_time=q['completion_time']if ok else limit,
                    rollout_uid=q['rollout_uid']if q else None)
            name=controller+'_'+mode;snapshots[name]=outcomes
            success=[r for r in outcomes.values()if r['success']]
            c1.append(dict(id=name,controller=controller,mode=mode,expected=136,completed=len(outcomes),
                success=len(success),success_rate=len(success)/136,
                success_mean_time=mean([r['completion_time']for r in success]),
                full_cohort_truncated_mean_time=mean([r['truncated_time']for r in outcomes.values()])))
    profiles=[]
    for profile in paper['profiles']:
        if profile['id']not in completed:continue
        labels={}
        for path in sorted((OUT/'labels').glob('*_joint.json'),key=lambda p:p.stat().st_mtime_ns):
            for x in read(path):
                if x['profile']==profile['id']and(x['type_id'],x['level'])in selected:
                    labels[(x['type_id'],x['level'],x['state_index'])]=x
        outcomes={key:read(HERE/'raw'/f"{x['rollout_uid']}.json")for key,x in labels.items()}
        assert len(outcomes)==136
        paired=[]
        for name,cases in snapshots.items():
            both=[k for k,r in outcomes.items()if r['success']and cases[k]['success']]
            c1_only=[k for k,r in outcomes.items()if not r['success']and cases[k]['success']]
            paper_only=[k for k,r in outcomes.items()if r['success']and not cases[k]['success']]
            paired.append(dict(C1=name,common_success=len(both),C1_only_success=len(c1_only),paper_only_success=len(paper_only),
                common_success_C1_mean_time=mean([cases[k]['completion_time']for k in both]),
                common_success_paper_mean_time=mean([outcomes[k]['completion_time']for k in both]),
                paired_mean_time_C1_minus_paper=mean([cases[k]['completion_time']-outcomes[k]['completion_time']for k in both])))
        profiles.append(dict(id=profile['id'],name=profile['name'],options=profile['options'],labels=profile['labels'],
            overall=profile['overall'],single_robot_passed_scenes=sum(x['single_robot_diagnostic_passed']for x in profile['scenes']),
            paired_against_C1=paired,rollout_uids=[x['rollout_uid']for x in labels.values()]))
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),scene_count=17,physical_states=136,
        C1=c1,completed_paper_profiles=profiles,environment_steps=0,solver_calls=0,
        scope='Identical17 geometry-selected scenes and state0..7. C1 nominal controllers each136 cases. Paper fixed profiles and C1 fixed first candidate are deployed policies; Oracle64 is a per-case finite-search envelope, not equal-budget policy performance.',
        limitations=['Paper exceptions use affected-robot one-step stop and recovery; solver/action errors and poor single-agent diagnostics prevent attributing low success to the original paper mechanisms.',
        'Success-conditional means, common-success paired means, and full-cohort truncated means are distinct. Earlier successful Oracle candidates are retained rather than picking fastest.',
        'The C1 fixed first point has C2=0; it does not establish necessity of all four columns.'])
    target=OUT/'COMPLETED_COMPARISON.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 已执行完论文配置与四列族：相同17场景比较\n',result['scope'],
        '\n表中时间均为各自成功样本条件均值，不能直接解释为配对速度优势。共同成功样本的配对时间、仅一方成功数、全队列截断时间与逐UID见JSON。',
        '\n|C1控制器/策略|成功/136|成功率|成功平均时间（秒）|全队列截断均时（秒）|\n|---|---:|---:|---:|---:|']
    lines.extend(f"|{x['id']}|{x['success']}|{x['success_rate']:.1%}|{x['success_mean_time']:.2f}|{x['full_cohort_truncated_mean_time']:.2f}|"for x in c1)
    lines.append('\n|论文固定配置|成功/136|成功平均时间（秒）|全队列截断均时（秒）|出现未处理异常的局数|单机初筛通过/17|\n|---|---:|---:|---:|---:|---:|')
    for x in profiles:
        r=x['overall'];lines.append(f"|{x['name']} {'/'.join(x['labels'])}|{r['success']}|{r['success_mean_time']:.2f}|{r['full_completed_cohort_truncated_mean_time']:.2f}|{r['original_first_error_unknown']}|{x['single_robot_passed_scenes']}|")
    lines.extend('\n- '+x for x in result['limitations'])
    (OUT/'COMPLETED_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('MATCHED_COMPARISON',len(profiles),'completed paper configurations; four C1 rows; zero rollouts')


if __name__=='__main__':main()
