"""Matched physical-state fixed-policy comparisons; keep Oracle upper bounds separate."""
from core import *
from papers import NAMES
import csv

OUT=HERE/'paper_after_four_column'

def metrics(rows,expected):
    success=[r for r in rows if r['success']]
    return dict(expected=expected,completed=len(rows),success=len(success),collision=sum(r['collision']for r in rows),
        timeout=sum(r['timeout']for r in rows),terminal_unknown=sum(r['numerical_failure']for r in rows),
        original_first_error_unknown=sum(r.get('paper_original',{}).get('numerical_unknown',0)for r in rows),
        error_stop_robot_steps=sum(r.get('paper_stop_robot_steps',0)for r in rows),
        success_mean_time=float(np.mean([r['completion_time']for r in success]))if success else None,
        full_completed_cohort_truncated_mean_time=float(np.mean([r['truncated_time']for r in rows]))if rows else None,
        rate=len(success)/expected if expected else None)

def analyze():
    if not(OUT/'PROFILES_FROZEN.json').exists():return
    protocol=read(OUT/'PROFILES_FROZEN.json');scenes=read(HERE/'SCENES.json')['entries']
    labels=[]
    for p in(OUT/'labels').glob('*.json'):labels.extend(read(p))
    # Latest profile implementation version controls a given cohort; old raw
    # revisions remain immutable diagnostics, never mixed into a new policy.
    grouped={}
    for p in sorted((OUT/'labels').glob('*.json'),key=lambda p:p.stat().st_mtime_ns):
        for label in read(p):grouped[(label['profile'],label['phase'],label['scene'],label['state_index'],label['solo'])]=label
    lookup={}
    for label in grouped.values():
        p=HERE/'raw'/f"{label['rollout_uid']}.json"
        if p.exists():lookup[label['rollout_uid']]=read(p)
    profiles=[];table=[]
    for profile in protocol['profiles']:
        pid=profile['id'];joint_labels=[l for l in grouped.values()if l['profile']==pid and l['phase']=='joint']
        joint=[lookup[l['rollout_uid']]for l in joint_labels if l['rollout_uid']in lookup]
        per_scene=[]
        for e in scenes:
            name=e['scene']['name'];jj=[lookup[l['rollout_uid']]for l in joint_labels if l['scene']==name and l['rollout_uid']in lookup]
            ss=[lookup[l['rollout_uid']]for l in grouped.values()if l['profile']==pid and l['phase']=='solo'and l['scene']==name and l['rollout_uid']in lookup]
            solo_expected=e['scene']['config']['num_agents']
            solo=metrics(ss,solo_expected)
            per_scene.append(dict(type_id=e['type_id'],level=e['level'],scene=name,title=e['title'],
                joint=metrics(jj,8),single_robot_diagnostic=solo,
                single_robot_diagnostic_passed=len(ss)==solo_expected and solo['rate']>=.95))
        entry=dict(id=pid,method=profile['method'],name=NAMES[profile['method']],labels=profile['labels'],options=profile['options'],
            overall=metrics(joint,392),scenes=per_scene,
            note='Fixed pre-existing parameters; one paper controller episode per physical state. Single-robot check uses all robots of state0 only; it is a diagnostic, not a full four-state applicability certificate.')
        profiles.append(entry)
        checked=sum(s['single_robot_diagnostic']['completed']==s['single_robot_diagnostic']['expected']for s in per_scene)
        passed=sum(s['single_robot_diagnostic_passed']for s in per_scene)
        row=entry['overall'];table.append([entry['name']+'（'+','.join(profile['labels'])+'）',f"{row['success']}/392",str(row['completed']),
            str(row['collision']),str(row['timeout']),str(row['terminal_unknown']),str(row['original_first_error_unknown']),
            str(row['success_mean_time']),str(row['full_completed_cohort_truncated_mean_time']),f'{passed}/{checked}'])
    result=dict(profiles=profiles,physical_states=392,scene_count=49,
        comparison='Fixed paper profiles are separate from the C1 Oracle envelope; paper failures with bad single-agent diagnostics or implementation/solver errors are not evidence of multi-robot deadlock.',
        no_new_parameter_selection=True)
    dump(OUT/'RESULTS.json',result)
    header=['论文方法与冻结参数','成功 /392','已完成','碰撞','超时','终局unknown','曾未处理异常','成功平均时间','已完成队列截断均时','单机初筛通过/已检查场景']
    with(OUT/'overall.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(header);w.writerows(table)
    text=['# 四列族之后的论文固定参数对照\n',
        '用户追加的对照，49档×8个物理初态，每个论文参数profile392局。论文自带控制器，每个物理初态只运行一次；C1两个冻结nominal分别比较。固定参数对照与C1最多64组的Oracle上限分列，不据此宣称等预算方法优劣。\n',
        '同一物理环境、半径、墙体、速度、成功判据和六倍时限；论文保留自己的控制器和安全求解器。数值与动作异常采用受影响机器人停一步再正常调用，并同时记录首次未处理异常口径。单机初筛用每档状态0的全部机器人，失败场景仍保留在总表，不把导航失败叫死锁。\n',
        '| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']
    text.extend('| '+' | '.join(row)+' |'for row in table)
    text.append('\n作者默认数值可追溯性、代码与公式端口来源见 ../PAPER_REGISTRY.json；Wang与自适应风险方法的参考默认值不冒称作者提供。旧冻结选参不在本轮重新选择。\n')
    if(HERE/'SAFETY_ONLY.json').exists():
        b=read(HERE/'SAFETY_ONLY.json')['cases']
        text.append(f"\n只加安全投影合计 {sum(r['success']for r in b)}/784，两个nominal各392局。\n")
    if(HERE/'COVERAGE.json').exists():
        c1=read(HERE/'COVERAGE.json')['families'][4]['cases_detail']
        rows=[]
        for controller,cn in [('centralized_sampling','集中式随机采样模型预测控制'),('gradient_mpc','梯度下降模型预测控制')]:
            g=[g for g in c1 if g['controller']==controller]
            first=[q['queries'][0]for q in g if q['queries']]
            won=[q for q in first if q['success']]
            rows.append(dict(controller=controller,name=cn,cases=392,first_frozen_point_completed=len(first),
                first_frozen_point_success=sum(q['success']for q in first),
                first_frozen_point_success_mean_time=float(np.mean([q['completion_time']for q in won]))if won else None,
                oracle_success=sum(q['success']for q in g),oracle_pending=sum(not q['success']and q['searched']<64 for q in g)))
        result['c1_four_column_comparison']=rows
        dump(OUT/'RESULTS.json',result)
        text.append('\n四列族对应两个nominal分别列出，每个392个物理初态；第一组冻结参数是单一固定策略，Oracle一列是最多64候选的逐案例上限，不与论文固定参数混作部署性能。\n\n| 控制器 | 第1组冻结参数成功 /392 | 第1组成功平均时间（秒） | Oracle已找到 /392 | 尚未搜完 |\n|---|---:|---:|---:|---:|\n')
        for r in rows:text.append(f"| {r['name']} | {r['first_frozen_point_success']} | {r['first_frozen_point_success_mean_time']} | {r['oracle_success']} | {r['oracle_pending']} |")
    text.append('\n逐场景完整数值与单机初筛结果见 RESULTS.json；报告未完成时的成功数是当前已验证数，缺失不算失败。\n')
    (OUT/'REPORT.md').write_text('\n\n'.join(text).replace('None','—')+'\n')
    return result
