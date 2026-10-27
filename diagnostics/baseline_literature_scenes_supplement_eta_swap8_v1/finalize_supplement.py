"""Never substitutes pending computation for a completed deliverable."""
import bootstrap
from experiment import *
from audit_base import git_snapshot
from audit_supplement import H
from collections import Counter
import subprocess,signal

def pct(x):return f'{100*x:.2f}%'
def fmt_delta(x):return f"{100*x['delta']:+.2f} pp [{100*x['ci95'][0]:.2f}, {100*x['ci95'][1]:.2f}]"
def main():
    assert read(HERE/'DB_POSTFLIGHT.json')['passed'];assert read(HERE/'DIAGNOSTIC_COMPLETENESS.json')['passed']
    names=subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-r','-o','%j'],text=True).splitlines();assert not any(n.startswith('lit_supp_') for n in names)
    start=read(HERE/'INTEGRITY_START.json');changed=[]
    for p,h in start['protected'].items():
        if not Path(p).is_file() or sha(p)!=h:changed.append(p)
    assert not changed,changed
    inputs={}
    for line in (HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():
        r=json.loads(line);p=r['path'];inputs[p]=r['sha256']
    changedinputs=[p for p,h in inputs.items() if sha(p)!=h];assert not changedinputs,changedinputs
    git={p:git_snapshot(p) for p in start['git']}
    # New user prompt files are explicitly allowed metadata-only additions.
    # Existing prompt files are still protected by the opening hash list.
    def normalized(snapshot):
        out=dict(snapshot);lines=[]
        for line in out['status'].splitlines():
            rel=line[3:].strip()
            if rel.startswith('docs/prompt_') and str(ROOT/rel) not in start['protected']:continue
            lines.append(line)
        out['status']='\n'.join(lines);out['status_sha256']=hashlib.sha256(out['status'].encode()).hexdigest();return out
    assert {p:normalized(r) for p,r in git.items()}=={p:normalized(r) for p,r in start['git'].items()},'Repository state changed outside supplement'
    for p,h in read(HERE/'EXTERNAL_SOURCES.json')['files'].items():assert sha(p)==h,p
    assert sha(H/'COMPLETE.json')==start['part1_complete_sha256']
    for receipt in ('IMPLEMENTATION_FROZEN.receipt.json','PREREGISTRATION.receipt.json','SCENES.receipt.json','WRITE_PATH_READY.receipt.json'):
        for p,h in read(HERE/receipt)['files'].items():assert sha(p)==h,p
    wa=read(HERE/'WRITE_PATH_AUDIT.json')
    for r in wa['libraries']:assert sha(r['path'])==r['sha256'],r['path']
    checks=[read(p) for p in (HERE/'READONLY_BATCH_CHECKS').glob('*.json')];assert checks and all(r['passed'] for r in checks)
    audit=dict(utc=utc(),passed=True,protected_files=len(start['protected']),reused_inputs_checked=len(inputs),changed_files=[],git=git,part1_complete_unchanged=True,stat_checks=len(checks),write_path_passed=True,procedural_exception='Replay batch28322 was submitted before its scan completed; stopped,29 completed replays retained,3 recovered only after complete scan. No new read-only writes observed.',new_prompt_files_and_parallel_task_changes='Metadata only as protocol allows')
    write(HERE/'INTEGRITY_END.json',audit)
    eta=read(HERE/'ETA_SCREEN.json');swap=read(HERE/'SWAP8_RESULTS.json');selection=read(HERE/'SCREEN_SELECTION.json');gaps=[]
    et=['# 粗筛η：修订SI分布式语义','', '只使用冻结训练子集选择；评估不重选；全部为开发证据。unknown成功下界0另报上界。圆形原训练MPPI类型无合格实例的缺口见下。']
    for scene,r in eta['scenes'].items():
        et+=['',f'## {scene}','']
        chosen=r['selection']
        if 'best_fixed' not in chosen:et.append('N/A：没有可用合格训练类型或评估实例。');gaps.append(scene+'：η选择/评估缺少合格训练或评估实例');continue
        et.append(f"最佳固定：{chosen['best_fixed']}；最佳配对：{chosen['best_pair']}；缺训练类型：{chosen.get('missing_types',[])}。")
        for label,key in [('16个固定候选','fixed_ranking'),('6个有序配对','pair_ranking')]:
            if scene=='lit_swap8_moving':continue
            et+=['',label,'','| 排名 | 动作 | 类型等权成功下界 | 上界 | 类型数 |','|---:|---|---:|---:|---:|']
            for i,rr in enumerate(chosen.get(key,[]),1):et.append(f"| {i} | {rr['action']} | {pct(rr['type_equal_lower'])} | {pct(rr['type_equal_upper'])} | {len(rr['per_type'])} |")
        # Descriptive per-type best over fixed and paired training actions.
        if scene!='lit_swap8_moving':
            both=chosen.get('fixed_ranking',[])+chosen.get('pair_ranking',[]);types=sorted({t for rr in both for t in rr['per_type']});et+=['','每类仅1个训练实例，以下只作描述：','']
            for t in types:
                opts=[rr for rr in both if t in rr['per_type']];opts.sort(key=lambda rr:(-rr['per_type'][t]['success_lower'],int(rr['action'][3:].split('_to_')[0]),rr['action']));et.append(f"- {t}：{opts[0]['action']}，成功下界 {pct(opts[0]['per_type'][t]['success_lower'])}。")
        ev=r['evaluation']
        if ev['status']=='complete':
            et+=['','| 评估动作 | 成功下界 [95% CI] | 上界 | 相对η13 | 相对人定双η |','|---|---|---|---|---|']
            for a,m in ev['methods'].items():
                et.append(f"| {a} | {pct(m['success_lower'])} [{100*m['ci95'][0]:.2f}, {100*m['ci95'][1]:.2f}] | {pct(m['success_upper'])} | {fmt_delta(m['versus']['eta13'])} | {fmt_delta(m['versus']['eta13_to_10'])} |")
        else:et.append('评估N/A：'+ev['reason'])
        if chosen.get('missing_types'):gaps.append(scene+'：原训练池缺合格类型 '+','.join(chosen['missing_types'])+'；保留可用类型的部分筛选，不是完整8类结论')
    et+=['',f"最佳固定η跨场景不同：{eta['best_fixed_differs']}；最佳配对不同：{eta['best_pair_differs']}。移动版不参与筛选，直接转移free选择。原四文献的门口按指令不做。"]
    (HERE/'ETA_SCREEN.md').write_text('\n'.join(et)+'\n')
    ss=['# 八机对径交换结果','', '静态单机有路不等于联合可解或controller成功。timeout不是deadlock。4a′为轨迹滤波器平衡点候选；只有通过能力门实例上的候选可作机器人间死锁证据。4a″按静态/移动障碍分别记录，controller级失败上的增益单列。未合格实例仍计全部48 baseline，候选统计不冒充经能力门支持的死锁结论。']
    for scene,r in swap['scenes'].items():
        ss+=['',f'## {scene}','',f"合格 {r['qualified']}/48；主线{r['main_requests']}逻辑请求。区分度：{r['discrimination']}。",'', '| 做法 | 成功局 | unknown | timeout | 碰撞agent/static/moving | 4a′局 | 静态4a″机器人·局 | 移动4a″机器人·局 |','|---|---:|---:|---:|---|---:|---:|---:|']
        for a,c in r['counts'].items():
            col=r['collision_types'][a];ss.append(f"| {a} | {c['success']} | {c['unknown']} | {c['timeout']} | {col['agent']}/{col['static']}/{col['moving']} | {c['deadlock_episode']} | {c['static_wall_agents']} | {c['moving_wall_agents']} |")
        ss+=['', '合格实例的类型→实例→state等权诊断率（seed同簇；全48 baseline原始计数在上表）：', '', '| 做法 | 4a′局率 | 4a″局率 | 耦合联合参照 | 实际乘积联合锥 | 静态4a″机器人比例 | 移动4a″机器人比例 |', '|---|---|---|---|---|---|---|']
        for key,weighted in r['weighted_rates_qualified'].items():
            rates=weighted['rates'];ss.append('| '+key.split('|')[-1]+' | '+' | '.join(pct(rates[k]) for k in ('deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode','static_wall_agent_fraction','moving_wall_agent_fraction'))+' |')
        for scope,inf in r['inferences'].items():
            ss+=['',f'### {scope}','']
            if inf['status']!='complete':ss.append('N/A：'+inf['reason']);continue
            ss+=['| 做法 | 成功下界 [95% CI] | 上界 | 相对只加安全投影 |','|---|---|---|---|']
            for a,m in inf['methods'].items():ss.append(f"| {a} | {pct(m['success_lower'])} [{100*m['ci95'][0]:.2f}, {100*m['ci95'][1]:.2f}] | {pct(m['success_upper'])} | {fmt_delta(m['versus']['safety'])} |")
        if not r['qualified']:gaps.append(scene+'：48个实例均未通过能力门，三种固定修正与η评估不可运行；只加安全投影仍按全部48报告')
        if any(v['collision'] for v in r['counts'].values()):gaps.append(scene+'：存在物理碰撞，分类型见SWAP8_RESULTS；没有额外调安全层或布局')
    ss+=['','## 移动障碍与无障碍的匹配比较','', '保留相同机器人state/RNG，障碍轨迹不受机器人影响。移动版能力门0/48，以下4a′变化仅原始候选诊断。消失还须结合成功/碰撞/timeout判断，不能把被撞出静止算解死锁。']
    for a,effect in swap['moving_vs_free'].items():
        inf=effect['inference'];d=fmt_delta(inf['methods']['moving']['versus']['free']) if inf['status']=='complete' else 'N/A'
        ss.append(f"\n{a}：匹配{effect['matched_episodes']}局，移动新增原始4a′候选 {effect['deadlock_created']}局、4a′消失候选{effect['deadlock_removed_candidate']}局；成功率moving−free {d}。")
    (HERE/'SWAP8_RESULTS.md').write_text('\n'.join(ss)+'\n')
    moving_audit=HERE/'MOVING_CAPABILITY_COLLISION_AUDIT.json'
    if moving_audit.exists():
        mc=read(moving_audit)['counts'];gaps.append(f"移动版单机{mc['collision']}/{mc['requests']}局真实移动障碍碰撞；其中{mc['collision_without_constraint_violation']}局未记录约束违反，{mc['collision_without_restoration']}局未触发无解回退。这是当前动态安全层时间模型的证据缺口，非correction效果试验；本轮没有合格实例，correction仍未评估。详见MOVING_CAPABILITY_COLLISION_AUDIT.json。")
    gaps.extend(['圆形/新场景缺训练类型的部分筛选不能代表完整8类型比较；每类型仅1个训练实例，描述性最佳没有普遍最优含义。','论文实景非凸布局缺可靠米制标定，采用用户预先声明4L；二维SI迁移不是HJB-GNN硬件复现。','保留数值unknown；速度放松、安全minimax及有限感知不构成全局安全证明。','先前提前提交回放的程序性失序原样披露，未倒补扫描时点；本轮未发现新的只读写入。'])
    (HERE/'GAPS.md').write_text('# 缺口与限制\n\n'+'\n\n'.join(gaps)+'\n')
    ax=['| 轴 | 本轮 | 证据边界 |','|---|---|---|','| controller类型 | 8类48冻结实例 | 能力门/部分类型缺口单列 |','| state | 每新场景32，训练/评估各16 | 开发证据；不对新N作泛化声明 |','| η | 16固定→top3的6配对 | 仅训练筛选；评估前冻结 |','| 动态障碍 | free同机器人state+独立外生圆盘 | 不交换名义控制器，不改变碰撞判据 |','| 安全层 | SI本地感知，互惠pair、障碍全责 | 修正位置广播假设仍保留 |','| 几何 | 无障碍/4L/移动圆盘 | 几何有效、单机有路、联合可解、成功和死锁分开 |']
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(ax)+'\n')
    first=f"完整性审计通过：{len(start['protected'])}个开工保护文件、{len(inputs)}个复用输入与结束一致，两仓库Git状态/diff一致；第一部分COMPLETE保持原SHA。全部写路径已审计，32条SI逐位回放通过，strace持久写入仅补充目录，默认journal及官方单merger符合规则。提前提交回放批28322的程序性失序已停止并保留29条，完整扫描后仅补3条；未发现本轮越界写入，时序限制照实披露。\n\n"
    report=first+f"η粗筛已按冻结训练子集与评估分离完成，固定赢家是否跨场景不同：{eta['best_fixed_differs']}，配对赢家不同：{eta['best_pair_differs']}。16固定/6配对完整排名、每类型描述性最佳、与η13及人定双η的配对差和5000次bootstrap见ETA_SCREEN.md/.json。缺合格训练类型只报告部分结果，不伪称完整8类；不同场景训练类型组成不同，赢家差异不能纯归因于场景几何。\n\n八机对径交换free/static/moving已分别做几何、单机能力门、全部48实例baseline和合格实例固定修正。三类碰撞、4a′、静态/移动4a″、联合参照、九格敏感性、实例归类、分组修正增益及移动−free匹配比较见SWAP8_RESULTS.md/.json。timeout不是deadlock，controller级失败上的增益不是解死锁。\n\n缺口：见GAPS.md。新场景为用户定义二维迁移开发证据，非硬件/原策略复现；有限样本不能替代全局安全证明。\n"
    (HERE/'REPORT.md').write_text(report)
    (HERE/'HANDOFF.md').write_text('第二部分按原顺序完成；当前入口为COMPLETE/REPORT/ETA_SCREEN/SWAP8_RESULTS/DB_POSTFLIGHT/INTEGRITY_END。源码、raw和trace保持不可变，不重启旧协调器；数值unknown保留。精确批次数见PREREGISTRATION_REQUESTS_ADDENDUM_20261023.md和REQUEST_LEDGER.jsonl；完整来源见EVIDENCE_INPUTS及EVIDENCE_SOURCES_END；用户资源上限18CPU总份额、2GPU，当前作业已结束。\n')
    (HERE/'STATUS.md').write_text('第二部分全部授权计算、复用、诊断、统计、DB核查与结束审计完成。存在的协议/数值/几何限制见GAPS；完成不表示全局无碰撞证明。\n')
    for line in (HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():json.loads(line)
    guard=HERE/'RESOURCE_GUARD_PID.json'
    if guard.exists():
        pid=read(guard)['pid'];proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists() and str(HERE/'resources.py').encode() in proc.read_bytes():os.kill(pid,signal.SIGTERM)
    local_guard=HERE/'LOCAL_CPU_GUARD_PID.json'
    if local_guard.exists():
        pid=read(local_guard)['pid'];proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists() and str(HERE/'local_cpu_guard.py').encode() in proc.read_bytes():os.kill(pid,signal.SIGTERM)
    write(HERE/'COMPLETE.json',dict(utc=utc(),part=2,status='complete_with_explicit_gaps' if gaps else 'complete',protocol_execution_complete=True,db_postflight_sha256=sha(HERE/'DB_POSTFLIGHT.json'),integrity_end_sha256=sha(HERE/'INTEGRITY_END.json'),reports={name:sha(HERE/name) for name in ('REPORT.md','ETA_SCREEN.md','SWAP8_RESULTS.md','PARAMETERS.md','HISTORY_CHECK.md','GAPS.md','AXIS_TABLE.md','HANDOFF.md')},gaps=gaps))
    print('Supplement COMPLETE written after actual evidence and audits',flush=True)
if __name__=='__main__':main()
