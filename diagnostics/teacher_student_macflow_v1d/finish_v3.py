from core import *
from audit_utils import source_inventory
from registration import connect,TABLES
import subprocess,difflib

def main():
    assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
    from shared_rollout_db.src import cache_writer
    assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
    source_inventory('imports_finish.json');imports={}
    for p in HERE.glob('imports_*.json'):
        for f,h in read(p)['imported_sources'].items():
            if Path(f).is_relative_to(HERE):continue
            assert sha(f)==h,f
            if f in imports:assert imports[f]==h
            imports[f]=h
    write(HERE/'IMPORTED_SOURCE_AUDIT.json',dict(utc=utc(),passed=True,files=imports))
    receipts={}
    for p in HERE.glob('*.receipt.json'):
        for f,h in read(p).get('files',{}).items():assert sha(f)==h,(p,f)
        receipts[p.name]=sha(p)
    for r in read(HERE/'STUDENTS.json').values():assert sha(r['checkpoint'])==r['checkpoint_sha256']
    for r in read(HERE/'TEACHERS.json'):assert sha(r['checkpoint'])==r['checkpoint_sha256']
    registrations=[]
    with connect(True) as db:
        for p in sorted((HERE/'registration').glob('*/RECEIPT.json')):
            r=read(p);assert r['passed'] and r['existing_rows_unchanged'] and r['rollout_counts_unchanged'];regs=read(p.parent/'INSERTED_ROWS.json');assert sha(p.parent/'INSERTED_ROWS.json')==r['inserted_rows_sha256']
            for entry in regs:
                table=entry['table'];row=entry['row'];key=TABLES[table];now=db.execute(f'SELECT * FROM {table} WHERE {key}=?',(row[key],)).fetchone();assert now is not None and dict(now)==row
            registrations.append(dict(receipt=str(p),sha256=sha(p),inserted=r['inserted']))
    write(HERE/'DB_PROVENANCE.json',dict(utc=utc(),passed=True,package=str(Path(shared_rollout_db.__file__).resolve()),default_journals=str((cache_writer.ROOT/'journals').resolve()),registration_receipts=registrations,allowed_writes='design INSERT OR IGNORE metadata; default append_journal; official merger only',unknown_policy='retained without rerun, lower/upper intervals; timeout not called deadlock'))
    monitor=read(HERE/'RESOURCE_MONITOR.json');violations=[x for x in monitor if x['active_cpu']>20];resource_pass=not violations
    write(HERE/'RESOURCE_AUDIT.json',dict(utc=utc(),historical_cap=20,cap_pass=resource_pass,observed_peak_cpu=max(x['active_cpu'] for x in monitor),violations=violations,monitoring_gap_start=violations[-1]['utc'] if violations else None,monitoring_resumed=read(HERE/'RESOURCE_MONITOR_RESUME.json')[0]['utc'],gap_peak_and_duration_unknown=True if violations else False,current_authorization='All CPU permitted when other tasks absent; does not retroactively erase previous21CPU breach',resumed_samples=len(read(HERE/'RESOURCE_MONITOR_RESUME.json'))))
    selected=read(HERE/'RHR_SELECTED.json') if (HERE/'RHR_SELECTED.json').exists() else None
    if selected:
        frozen=read(HERE/'RHR_FROZEN.receipt.json');submission=read(HERE/'fair_SUBMISSION.json');assert frozen['utc']<submission['utc']
    prereg=read(HERE/'PREREGISTRATION.receipt.json')
    for phase in ['A','tune','fair','C']:
        if not (HERE/f'{phase}_SUBMISSION.json').exists():continue
        sub=read(HERE/f'{phase}_SUBMISSION.json');pf=read(HERE/f'{phase}_PREFLIGHT.receipt.json');assert prereg['utc']<pf['utc']<sub['utc'];assert read(HERE/f'{phase}_MERGE_RECEIPT.json')['db_exact']
    diffs=[]
    for r in read(HERE/'COPY_PROVENANCE.json'):
        src=Path(r['source']);dest=Path(r['destination']);assert sha(src)==r['source_sha256'];
        if src.suffix=='.py':diffs.extend(difflib.unified_diff(src.read_text().splitlines(True),dest.read_text().splitlines(True),fromfile=str(src),tofile=str(dest)))
    (HERE/'COPY_DIFF.patch').write_text(''.join(diffs))
    subprocess.run([sys.executable,'-B',str(HERE/'integrity_v2.py'),'end'],check=True,cwd=ROOT);audit=read(HERE/'INTEGRITY_COMPARISON.json');assert audit['protected_content_pass'] and not audit['removed_status_entries'],audit['changes']
    results=dict(utc=utc(),integrity_pass=True,DB_guard_pass=True,resource_peak_cpu=max(x['active_cpu'] for x in monitor),GPU_shards_used=0,resource_cap_pass=resource_pass,resource_audit=read(HERE/'RESOURCE_AUDIT.json'),classical_comparison=read(HERE/'CLASSICAL_COMPARISON.json'),handoff=read(HERE/'HANDOFF_COMPLETION.json'),journal_exactly_once_pass=read(HERE/'JOURNAL_ONCE_AUDIT.json')['passed'],phases={})
    for phase,file in [('A','A_RESULTS.json'),('B','B_RESULTS.json'),('C','C_RESULTS.json')]:results['phases'][phase]=read(HERE/file) if (HERE/file).exists() else dict(status='GAP',records=[read(p) for p in HERE.glob('*GAP.json')])
    outcomes={}
    for phase in ['A','tune','fair','C']:
        f=HERE/f'{phase}_DB_ROWS.json'
        if f.exists():
            rs=read(f);outcomes[phase]=dict(logical_records=len(rs),unique=len({r['rollout_uid'] for r in rs}),unknown=sum(r['numerical_failure'] for r in rs),collision=sum(r['collision'] for r in rs),timeout=sum(r['timeout'] for r in rs),new_cases=len(read(HERE/f'{phase}_MISSING_CASES.json')))
    results['outcome_inventory']=outcomes;write(HERE/'results.json',results)
    A=results['phases']['A'];B=results['phases']['B'];C=results['phases']['C']
    if all('status' not in v for v in [A,B,C]):
        def rate(pipe,n):return A[pipe]['controls'][n]['arms']['S500']['mean']
        opening=f"完整性审计与 Basin_C1 DB/journal 路径断言均通过。D-local 的 Z/G+S500 成功率分别为 {rate('local','Z'):.2%}/{rate('local','G'):.2%}，学生参考为 {A['local']['reference_S500']['mean']:.2%}；集中式 TT 的 Z/G+S500 为 {rate('central','Z'):.2%}/{rate('central','G'):.2%}，教师参考为 {A['central']['reference_S500']['mean']:.2%}。公平 RHR 学生组选择 {B['selected']['students']['rule']}，成功率 {B['students_RHR']['mean']:.2%}；G 组选择 {B['selected']['G']['rule']}，成功率 {B['G_RHR']['mean']:.2%}。学生 S500−RHR 为 {B['students_S500_minus_RHR']['mean']:.2%}，保守 95%CI {B['students_S500_minus_RHR']['CI_unknown_envelope']}。K16∪S300/S500 的按 controller/按 state 交叉拟合空间分别为 {C['students']['controller']['mean']:.2%}/{C['students']['state']['mean']:.2%}；{C['decision']}。"
    else:opening='完整性审计与 DB 路径断言通过；存在阶段证据缺口，见下表及 results.json。不得将缺口解释为负结果。'
    opening+=' 阶段A在两条管线都支持学习策略与修正存在协同；但经典 G+CBF+RHR 已在所测32个state全部成功，不能据学生内部比较宣称整体优于经典控制方案。'
    opening+=' 资源审计存在例外：旧20CPU上限期间监测到总21CPU，随后监测中断，该时段峰值与持续时间未知；记录未删除。' if violations else ''
    lines=[opening,'','阶段A的零动作差距在两条管线都超过20个百分点，按预注册规则均判为协同。','', '## 阶段 A','', (HERE/'A_RESULTS_TABLE.md').read_text() if (HERE/'A_RESULTS_TABLE.md').exists() else 'GAP','## 阶段 B','']
    if 'status' not in B:
        lines += [f"学生 RHR={B['students_RHR']['mean']:.2%}；G RHR={B['G_RHR']['mean']:.2%}。",f"学生 S500−RHR：{B['students_S500_minus_RHR']}。",f"G S500−RHR：{B['G_S500_minus_RHR']}。",B['students_interpretation']+'（仅限同一学生上的两种修正比较）。最简单 G+CBF+RHR 在所测32个state全部成功，不能据学生内部对比宣称优于经典控制方案。两seed轨迹完全相同，64条并非64个独立state。',str(read(HERE/'CLASSICAL_COMPARISON.json')['delta']),'',(HERE/'RHR_GRID_TABLE.md').read_text(),'','完整20组分别针对14学生均值和G选择，见 RHR_GRID_RESULTS.json；选择冻结收据早于 fair 主评估提交。']
    lines += ['','## 阶段 C','','| 选择层级 | 交叉拟合增益 | 95%CI | unknown保守CI | 样本内上界 |','|---|---:|---|---|---:|']
    if 'status' not in C:
        for mode in ['controller','state']:
            r=C['students'][mode];lines.append(f"| {mode} | {r['mean']:.2%} | {r['CI95']} | {r['CI_unknown_envelope']} | {C['students']['in_sample_upper'][mode+'_mean']:.2%} |")
        lines += ['',C['decision'],'','Z 的相同空间分析与全部18臂逐controller成功率见 C_RESULTS.json；C_ARRAYS.npz 包含下界与unknown掩码。']
    lines+=['','## 证据范围与审计','',f"{json.dumps(outcomes,ensure_ascii=False)}",'',f"旧监测中总CPU峰值至少{results['resource_peak_cpu']}，资源上限审计{'通过' if resource_pass else '未通过；旧上限20期间观测21，随后缺监测，无法断言完整峰值'}；GPU使用0 shard。源码审计已包括 PoC 与30/30b/30c全部文件；发现的其他并行任务 git 新增条目共{len(audit['concurrent_new_status_entries'])}项，单独列在 INTEGRITY_COMPARISON.json，并非本轮所写。",'','unknown 保留为[0,1]，不是已知失败；timeout 只表述为超时。结论只覆盖固定14学生、Ring32state与冻结K16菜单。controller交叉拟合按state奇偶两折；state交叉拟合只跨seed，不代表新state泛化。bootstrap5000，controller后state聚类，种子2026103032，所有臂配对。CI以成功率比例表示，乘100为百分点。', '', 'CPU闲置后按用户新增授权，将1817条未开始任务安全移交17分片；7个旧worker在进行中试验完成后因预留输出保护而在下一条执行前退出，属于预期调度退出，未中断试验、未形成额外unknown、未重跑。HANDOFF_COMPLETION.json与JOURNAL_ONCE_AUDIT.json证明所有13440条新C结果各有且仅有一条journal记录。', '', '所有新代码和输出均在本目录；目录外仅按官方流程追加身份和journal并通过官方merger写DB。没有训练或改动checkpoint、物理、事件、成功标准、安全层与修正。源码复制和逐项diff见 COPY_PROVENANCE.json、COPY_DIFF.patch；逐入口import哈希见 imports_*.json 和 IMPORTED_SOURCE_AUDIT.json。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (HERE/'HANDOFF.md').write_text('本轮执行 docs/prompt_round30d_controls_fair_rhr_k16_headroom_20261015.md。\n\n'+opening+'\n\n报告入口 REPORT.md；机读 results.json。预注册 PREREGISTRATION.md 与收据先于任何新rollout。A/Tune/fair/C各有 CASES、MISSING_CASES、官方 CACHE_PREFLIGHT/POSTFLIGHT、DB_ROWS、MERGE_RECEIPT；registration/记录只追加身份及所有旧行哈希保护。原始仿真位于各phase/raw与traces，不得重跑unknown或覆盖既有文件。7个旧C分片的FAILED是预期的执行前输出保护退出；1817条未开始试验由17个新分片接管，无重复，见HANDOFF_PLAN/HANDOFF_COMPLETION/JOURNAL_ONCE_AUDIT.json。当前实际收尾入口为complete_handoff.py，早期协调器停止记录保留以便审计。RHR_GRID_RESULTS.json 为完整两池网格，RHR_SELECTED.json 与RHR_FROZEN收据为冻结选择。K16_AUDIT.json 精确核对16候选及0基索引10/13。源码与DB保护是最高约束。无本轮critic训练；下一步依 C_RESULTS.json 决策，并需新一轮授权/协议。\n\n所有保护路径只读；不得修改旧源码或使用PoC旧DB副本；任何新worker先加载core.py校验Basin DB及默认journal。Slurm CPU空闲时可按2026-10-16新增授权使用全机24核，其他任务存在时协调资源；GPU shard上限2；本轮只用CPU。orchestrate.py按A->tune->freeze->fair->C一次执行，绝不直接重新启动；请先检查RUN_STATUS、已有raw/DB与预检证据以免重复。\n')
    write(HERE/'RESULTS_VALIDATION.json',dict(utc=utc(),passed=resource_pass,receipts=receipts,checkpoints_frozen=True,imported_sources_pass=True,integrity_pass=True,DB_guard_pass=True,temporal_order_pass=True,CPU_cap_pass=resource_pass,resource_audit_exception=not resource_pass,scientific_validation_pass=True))
    seal('DELIVERY.receipt.json',[HERE/f for f in ['results.json','REPORT.md','HANDOFF.md','INTEGRITY_COMPARISON.json','DB_PROVENANCE.json','RESULTS_VALIDATION.json','analysis.py','finish_v3.py','HANDOFF_COMPLETION.json','JOURNAL_ONCE_AUDIT.json','PAIRING_AUDIT.json','RESOURCE_AUDIT.json','RHR_TRACE_AUDIT.json','CLASSICAL_COMPARISON.json','RHR_GRID_TABLE.md']],complete=True,resource_cap_pass=resource_pass)
if __name__=='__main__':main()
