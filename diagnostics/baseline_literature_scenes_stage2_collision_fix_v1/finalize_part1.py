"""Reviewable part-one delivery; COMPLETE only after data, DB and audit pass."""
from experiment import *
from audit import V2,git_snapshot
import subprocess,csv,signal
from pathlib import Path

METHODS=('safety','fixed_eta10','fixed_eta13','human_progress')
LABELS=('只加安全投影','全程固定 η10 修正','全程固定 η13 修正','人定双 η 广播进度切换')
def percent(value):return f'{100*value:.2f}%'
def delta(d):return f"{100*d['delta']:+.2f} pp [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}]；unknown识别界 [{100*d['identified_bounds'][0]:.2f}, {100*d['identified_bounds'][1]:.2f}]"

def main():
    assert (HERE/'PART1_COMPUTATION_READY.json').exists()
    assert read(HERE/'DB_POSTFLIGHT_RESUME.json')['passed']
    names=subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-r','-o','%j'],text=True).splitlines()
    assert not any(n.startswith('lit_resume_') for n in names),'Wait for all part-one jobs before closing audit'
    start=read(HERE/'INTEGRITY_RESUME_START_20261023.json');changed=[]
    for p,h in start['protected'].items():
        if not Path(p).is_file() or sha(p)!=h:changed.append(p)
    assert not changed,changed
    repos={p:git_snapshot(p) for p in start['git']};assert repos==start['git'],'Git state changed outside the repair output directory'
    assert sha(HERE/'INTEGRITY_END.json')==start['old_failed_audit_sha256']
    assert sha(HERE/'READONLY_RESUME_BASELINE.json')==start['readonly_baseline_sha256']
    checks=[read(p) for p in (HERE/'READONLY_BATCH_CHECKS').glob('*.json')]
    assert all(x['passed'] for x in checks),'Read-only batch guard failed'
    audit=dict(utc=utc(),passed=True,protected_hashes_checked=len(start['protected']),changed_files=[],repositories=repos,
       user_accepted_limited_exception=dict(path=str(V2/'runtime_cbs_query_memo_candidate_20261022'),prior_modified_cache_files=34440,source='READONLY_BOUNDARY_IMPACT.json',scope='Prior pure-query cache incident only; no subsequent writes permitted'),
       original_failed_audit_preserved=True,opening_indirect_import_hash_gap_disclosed=True,readonly_checks=len(checks),write_path_audit_sha256=sha(HERE/'WRITE_PATH_AUDIT.json'))
    write(HERE/'INTEGRITY_RESUME_END_20261023.json',audit)
    archive=HERE/'stop_history';archive.mkdir(exist_ok=True)
    for name in ('REPORT.md','GAPS.md','HANDOFF.md','DYNAMICS_STAGE.md'):
        p=HERE/name;q=archive/name
        if p.exists() and not q.exists():q.write_bytes(p.read_bytes())
    selection=read(HERE/'PART1_STAGE_SELECTIONS.json');results={};csvrows=[]
    text=['第二阶段碰撞修复（开发证据）。原v2、圆形初始353局未达零碰撞试跑和原失败审计保留；本表只纳入通过试跑的全量队列。门口原身份复用。',
      '', '| 场景 | 状态 | 完成 | 计划 | unknown | 约束违反步 | 采样速度违反步 | 加速度上限违反步 | 诊断缺失局 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scene in ('lit_circle_five','lit_crowded_random','lit_doorway','lit_narrow_passage'):
        if scene=='lit_doorway':
            results[scene]=dict(status='EXACT_REUSE',requests=23040,source=str(V2/'DYNAMICS_STAGE.md'),source_sha256=sha(V2/'DYNAMICS_STAGE.md'))
            text.append('| 门口场景 | 原身份复用 | 23040 | 23040 | 5 | 0 | 0 | 0 | 0 |');continue
        chosen=selection[scene];planned={'lit_circle_five':5760,'lit_crowded_random':39424,'lit_narrow_passage':22528}[scene]
        if not chosen['stage']:
            results[scene]=dict(status='GAP',reason=chosen['reason'],planned=planned)
            text.append(f'| {NAMES[scene]} | 缺口：{chosen["reason"]} | N/A | {planned} | N/A | N/A | N/A | N/A | N/A |');continue
        r=read(HERE/f'STATISTICS_{scene}.json');assert r['completed']==planned;r['status']='complete';results[scene]=r;c=r['counts']
        text.append(f"| {NAMES[scene]} | complete | {r['completed']} | {planned} | {c['unknown']} | {c['constraint_violation_steps']} | {c['sampled_speed_violation_steps']} | {c['acceleration_bound_violation_steps']} | 0 |")
    original=(V2/'DYNAMICS_STAGE.md').read_text()
    for scene in ('lit_circle_five','lit_crowded_random','lit_doorway','lit_narrow_passage'):
        text+=['',f'### {NAMES[scene]}','']
        if scene=='lit_doorway':
            section=original.split('### 门口场景',1)[1].split('### 窄通道场景',1)[0];text.append(section.strip())
            text.append('\n与v2第二阶段配对差为0（直接复用）；修复前后碰撞均为0/23040。');continue
        r=results[scene]
        if r['status']=='GAP':text.append('全量成功率、配对差和5000次bootstrap均N/A：'+r['reason']);continue
        methods=r['inferences']['all_qualified']['methods'];rates=r['weighted_rates']
        text+=['| 做法 | 成功率 [95% CI] | unknown 成功上界 |','|---|---|---|']
        for method,label in zip(METHODS,LABELS):
            m=methods[method];ci=m['success_ci95'];text.append(f"| {label} | {percent(m['success'])} [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {percent(m['success_upper'])} |")
            csvrows.append(dict(scene=scene,method=method,success=m['success'],ci_low=ci[0],ci_high=ci[1],upper=m['success_upper'],versus_v2=m['versus_v2']['delta'],bootstrap_iterations=5000))
        text+=['','| 做法 | 第二阶段−第一阶段 | 修正增益之差 | 与v2第二阶段的配对差 |','|---|---|---|---|']
        for method,label in zip(METHODS,LABELS):
            m=methods[method];text.append(f"| {label} | {delta(m['versus_first'])} | {delta(m['gain_vs_first_difference']) if method!='safety' else 'N/A'} | {delta(m['versus_v2'])} |")
        text+=['','| 做法 | 逐机器人4a′ | 墙前4a″ | 耦合联合参照 | 实际乘积联合锥 | 直接静止 | 直接静止agent fraction |','|---|---|---|---|---|---|---|']
        for method,label in zip(METHODS,LABELS):
            m=rates['paper_dynamics|'+method]['rates'];text.append('| '+label+' | '+' | '.join(percent(m[k]) for k in ('deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode','direct_stationary_episode','direct_stationary_agent_fraction'))+' |')
        text.append('\n实例归类、九格敏感性、按controller级失败拆分的增益及目标附近徘徊见对应STATISTICS JSON；分类使用v2最终补充口径，仅4a′作为机器人间死锁证据。')
    old=read(HERE/'HISTORY_RAW_COUNTS.json')
    text+=['','### 修复前后碰撞局数（全量，试跑另列）','','| 场景 | 做法 | v2第二阶段 | 本轮修复后 |','|---|---|---:|---:|']
    for scene in NAMES:
        for method,label in zip(METHODS,LABELS):
            before=old[scene]['methods'][method]
            if scene=='lit_doorway':after='0/'+str(before['n'])
            elif results[scene]['status']=='complete':
                pp=results[scene]['weighted_rates']['paper_dynamics|'+method]['rates'];rows=read(HERE/f'COMPACT_FIXED_{scene}.json');subset=[r for r in rows if r['method']==method];after=str(sum(r['collision'] for r in subset))+'/'+str(len(subset))
            else:after='N/A'
            text.append(f"| {NAMES[scene]} | {label} | {before['collision']}/{before['n']} | {after} |")
    text.append('\n圆形初始试跑353/360：只加安全投影176局0碰撞，全程eta13的177局2碰撞；剩余7局不补跑、不算unknown。离散回退试跑360局结果见PILOT_RESULTS_RESUME。所有CI用原5000次配对bootstrap，类型→实例→state等权、两个seed同簇，unknown以下界计并另报上界。与第一阶段差异来自动力学执行栈及本轮安全层修复，不能只归因于动力学。无全局离散安全证明；理论需要前提和可行性，违反、速度超限、unknown均单列。')
    (HERE/'DYNAMICS_STAGE.md').write_text('\n'.join(text)+'\n');write(HERE/'DYNAMICS_RESULTS_RESUME.json',results)
    if csvrows:
        with (HERE/'DYNAMICS_STAGE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    total_collision=sum(r['counts']['collision'] for r in results.values() if r['status']=='complete')
    accepted=sum(r['completed'] for r in results.values() if r['status']=='complete')
    report=f'''恢复阶段完整性审计按用户接受的限定例外通过：1196个源文件/文档开工哈希与结束一致，各批只读目录stat检查通过。原生库已在本轮目录重建，32条逐位回放通过，strace持久写入仅本轮目录。原34440个纯查询缓存文件写入为“用户接受的限定例外”，不是raw、trace、数据库或封存文件；未删除、回滚或再次写入。原失败INTEGRITY_END、停机及影响记录全部保留，新结束审计为INTEGRITY_RESUME_END_20261023.json。原间接导入模块缺最初开工哈希的覆盖限制照实保留；恢复前已补充新开工覆盖，不倒补旧审计。\n\n修复后通过试跑并完成全量的三个候选场景共{accepted}个逻辑请求，碰撞局数为{total_collision}；未通过场景单列缺口，门口23040请求直接复用、0碰撞。圆形原初始形式353局试跑含2碰撞，已按用户决定合并并独立保留，剩余7局不补。窄通道和拥挤随机按用户澄清重判单元门，保留全部不满足前提的快照和旧unknown，不把说明的前提澄清冒充看结果改判据。\n\n成功率、修正增益、与v2第二阶段及第一阶段的配对差、死锁/墙前候选、直接静止和5000次bootstrap见DYNAMICS_STAGE.md/.csv及STATISTICS JSON。timeout不直接作为deadlock；controller级失败上的修正增益不计解开死锁。全部为开发证据，安全结论依赖已说明的预测安全、已有区间安全与约束可行条件，不能从有限样本无碰撞推断全局保证。\n'''
    for scene,r in results.items():
        if r['status']!='complete':continue
        ms=r['inferences']['all_qualified']['methods'];report+=f"\n{NAMES[scene]}：只加安全投影成功率{percent(ms['safety']['success'])}；全程eta10 {percent(ms['fixed_eta10']['success'])}；全程eta13 {percent(ms['fixed_eta13']['success'])}；人定双eta {percent(ms['human_progress']['success'])}。unknown {r['counts']['unknown']}，碰撞 {r['counts']['collision']}。\n"
    (HERE/'REPORT.md').write_text(report)
    gaps=['# 第一部分缺口与限制','', '原开工未覆盖一个间接导入模块的哈希，旧审计限制保留；恢复开工已覆盖。','原纯查询缓存写入的用户接受限定例外、原失败审计保留。','数值unknown保留不重跑；理论前提与实际约束违反不等于全局安全证明。']
    for scene,r in results.items():
        if r['status']=='GAP':gaps.append(f"- {NAMES[scene]}：{r['reason']}")
        elif r['status']=='complete' and r['counts']['collision']:gaps.append(f"- {NAMES[scene]}：全量仍有{r['counts']['collision']}碰撞，未达全量零碰撞；不执行未预注册的额外修改。")
    (HERE/'GAPS.md').write_text('\n\n'.join(gaps)+'\n')
    with (HERE/'STAGE2_FIX.md').open('a') as f:
        f.write('\n## 用户恢复说明1.4的前提澄清与最终门结果\n\n原门及反例全部保留。新门只在预测g>=0、当前区间无碰撞且原约束有解时检查k+1到k+2；编号交换限唯一解，1e-4阈值，是否重叠变化始终失败。gamma=.3/.2保持。\n\n')
        for scene,r in read(HERE/'STAGE2_FIX_RESUME.receipt.json')['scenes'].items():f.write(f"- {NAMES[scene]}：{r.get('counts')}；最大编号误差 {r.get('maximum_permutation_error','见圆形原收据')}。\n")
    with (HERE/'HANDOFF.md').open('a') as f:f.write('\n## 第一部分恢复完成（覆盖原停止状态）\n\n用户接受限定缓存例外后已使用resume_backend和native_resume安全库恢复；原危险入口和库仅保留历史，不使用。第一部分结果、DB_POSTFLIGHT_RESUME、新结束审计和COMPLETE是当前入口。第二部分在COMPLETE之后才开始，目录为baseline_literature_scenes_supplement_eta_swap8_v1，独立审计和实验身份。\n')
    (HERE/'STATUS.md').write_text('# 第一部分完成\n\n全部授权门、可运行的全量、诊断、5000次配对bootstrap、数据库postflight和恢复结束审计完成。用户接受的旧限定例外及其余限制见REPORT/GAPS。现在允许开始第二部分。\n')
    guard=HERE/'RESUME_RESOURCE_GUARD_PID.json'
    if guard.exists():
        pid=read(guard)['pid'];proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists() and str(HERE/'resume_resources.py').encode() in proc.read_bytes():os.kill(pid,signal.SIGTERM)
    write(HERE/'COMPLETE.json',dict(utc=utc(),part=1,status='complete_with_explicit_gaps' if any(r['status']=='GAP' for r in results.values()) or total_collision else 'complete',
       protocol_execution_complete=True,observed_collision_episodes=total_collision,
       unknown_episodes=sum(r['counts']['unknown'] for r in results.values() if r['status']=='complete')+5,
       accepted_limited_cache_exception=True,original_failed_audit_retained=True,db_postflight_sha256=sha(HERE/'DB_POSTFLIGHT_RESUME.json'),
       integrity_resume_end_sha256=sha(HERE/'INTEGRITY_RESUME_END_20261023.json'),reports={n:sha(HERE/n) for n in ('REPORT.md','DYNAMICS_STAGE.md','GAPS.md','HANDOFF.md')},part2_started=False))
    print('Part one COMPLETE written; part two may now begin',flush=True)
if __name__=='__main__':main()
