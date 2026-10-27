"""Final content review; preserve and disclose external repository differences."""
from common import *
from audit_scope import metadata,sha as stream_sha,OFFICIAL
import subprocess,sqlite3

def git(parent,*args):
    return subprocess.check_output(['git',*args],cwd=parent,text=True,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))

def main():
    start=read(HERE/'INTEGRITY_START.json');protected=dict(start['input_sha256'])
    for p in sorted(HERE.glob('INPUT_ADDITIONS_*.json')):
        for r in read(p)['files']:
            assert r['path'] not in protected or protected[r['path']]==r['sha256']
            protected[r['path']]=r['sha256']
    differences=[]
    for p,h in protected.items():
        actual=stream_sha(p) if Path(p).exists() else None
        if actual!=h:differences.append(dict(path=p,before=h,after=actual))
    docchanges=[]
    for p,h in start['existing_document_sha256'].items():
        actual=stream_sha(p) if Path(p).exists() else None
        if actual!=h:docchanges.append(dict(path=p,before=h,after=actual))
    newdocs=[]
    for p in (ROOT/'docs').rglob('*'):
        if p.is_file() and str(p) not in start['existing_document_sha256']:
            s=p.stat();newdocs.append(dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,stat_only=True))
    repositories={}
    for parent in (ROOT,OFFICIAL):
        old=start['repositories'][str(parent)]['head'];head=git(parent,'rev-parse','HEAD').strip()
        repositories[str(parent)]=dict(opening_head=old,current_head=head,head_changed=old!=head,status_outside_diagnostics=git(parent,'status','--porcelain','--','.',':(exclude)diagnostics'),committed_names_outside_diagnostics=git(parent,'diff','--name-status',old,head,'--','.',':(exclude)diagnostics'),recent_commit_metadata=git(parent,'log','-3','--format=%h %ai %an %s'))
    patch=git(ROOT,'diff',start['repositories'][str(ROOT)]['head'],'HEAD','--','.gitignore')
    (HERE/'EXTERNAL_REPOSITORY_GITIGNORE_DIFF.patch').write_text(patch)
    checked=[]
    for receipt in ('DELIVERY.receipt.json','NUMERIC_RESULTS.receipt.json','INFORMATION_GROUPED_BARS.receipt.json'):
        for p,h in read(HERE/receipt)['files'].items():assert stream_sha(p)==h,(receipt,p)
        checked.append(receipt)
    model_files={}
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('checkpoint_path','transform_path','fit_receipt_path'):
                    h=x[k[:-5]+'_sha256'];assert v not in model_files or model_files[v]==h;model_files[v]=h
                elif isinstance(v,(dict,list)):visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(read(HERE/'FROZEN_PIPELINE.json'))
    for p,h in model_files.items():assert stream_sha(p)==h,p
    meta=metadata(HERE/'AUDIT_METADATA_REVIEW.sqlite')
    db=sqlite3.connect(HERE/'AUDIT_METADATA_REVIEW.sqlite');db.execute('PRAGMA cache_size=-32768');db.execute('ATTACH DATABASE ? AS original',(str(HERE/'AUDIT_METADATA_START.sqlite'),));counts={}
    query='SELECT e.root,COUNT(*) FROM file e LEFT JOIN original.file o ON e.path=o.path WHERE o.path IS NULL OR e.size!=o.size OR e.mtime_ns!=o.mtime_ns GROUP BY e.root'
    for root,n in db.execute(query):counts[root]=n
    deleted={root:n for root,n in db.execute('SELECT o.root,COUNT(*) FROM original.file o LEFT JOIN file e ON e.path=o.path WHERE e.path IS NULL GROUP BY o.root')};db.close()
    result=dict(utc=utc(),status='PROTECTED_CONTENT_MATCH_WITH_EXTERNAL_REPOSITORY_CHANGES',protected_contents_match=not differences and not docchanges,strict_repository_state_unchanged=False,protected_input_count=len(protected),protected_input_differences=differences,existing_document_differences=docchanges,new_documents_stat_only=newdocs,repositories=repositories,original_strict_difference_record='INTEGRITY_FINAL_COMPARISON.json',original_pre_housekeeping_audit='INTEGRITY_COMPARISON.json',main_round_git_writes_performed=False,original_receipts_verified=checked,selected_model_files_verified=len(model_files),parallel_stat_metadata=meta,parallel_stat_changes_by_directory=counts,parallel_stat_deleted_entries_by_directory=deleted,prohibited_diagnostic_content_not_opened=True,git_status_and_diff_prune_all_diagnostics=True,scientific_results_unchanged=True)
    write(HERE/'INTEGRITY_RECONCILIATION.json',result)
    assert result['protected_contents_match'],'Protected content changed: stop and report; do not waive it'
    seal('INTEGRITY_RECONCILIATION.receipt.json',[HERE/n for n in ('INTEGRITY_RECONCILIATION.json','EXTERNAL_REPOSITORY_GITIGNORE_DIFF.patch','INTEGRITY_START.receipt.json','INTEGRITY_COMPARISON.receipt.json','INTEGRITY_FINAL_COMPARISON.receipt.json','readiness_review_c6.py')],protected_contents_match=True,strict_repository_state_unchanged=False,external_changes_disclosed=True)
    r=read(HERE/'T1_T7_RESULTS.json');failure=read(HERE/'FAILURE_DIAGNOSIS.json');methods=list(r['groups']['类型等权合并']['methods'])
    lines=['未达到90%的学习型类型及失败诊断如下。计数单位为 controller-state-seed；重查提议占比和阈值分布按开局选择计，两个seed共用同一选择。','', '| 类型 | 方法全称 | 成功率及95% CI | 已知失败 / unknown | 未触发修正的已知失败 | 触发≥200步的已知失败 | 第一层成功而该方法失败 | 重查提议占比 | 所选θ分布 |','|---|---|---|---|---|---|---|---|---|']
    for t,row in failure['types'].items():
        for m,v in row['branches'].items():
            if v['point_at_least90']:continue
            z=v['success'];lo,hi=z['CI95'];counts=row['selected_thresholds'][m];total=sum(counts.values());theta='；'.join(('无作用' if float(k)==0 else k)+f'：{n/total:.1%}' for k,n in sorted(counts.items(),key=lambda a:float(a[0])))
            lines.append(f"| {t} | {m} | {z['point']:.2f}% [{lo:.2f}, {hi:.2f}] | {v['known_failed_controller_state_seed_cases']} / {v['unknown_cases']} | {v['nonzero_schedule_never_triggered_known_failures']} | {v['trigger_at_or_after200_known_failures']} | {v['first_layer_success_online_layer_known_failure']} | {v['chosen_requery_fraction']:.1%} | {theta} |")
    lines+=['','这些观察不能直接识别因果：不同方法可能同时改变η对、θ与触发设置。人定双η成功而五个已评估学习选择均已知失败的案例，分别为：'+ '；'.join(t+' '+str(next(iter(v['branches'].values()))['human_success_all_five_learned_choices_known_failure']) for t,v in failure['types'].items() if any(not x['point_at_least90'] for x in v['branches'].values()))+'。它说明当前pipeline有缺口；完整提议池未逐项标注，η来源覆盖不足与critic选错目前不能分开判定。下一轮优先对象为上述未达类型，结合已存η来源、触发时间和高阈值选择检查；本轮不追加训练、候选或rollout。','', '探测中不停滞组为0个controller，其比较为N/A。700步探测关闭终止，达到目标后的进度平台也可能满足停滞公式；早/晚停滞是预注册诊断分组，不能直接作为死锁判定。','']
    optim=r['groups']['优化型']['comparisons']['T2_online'];lines.append(f"需要保留的负结果：优化型组加入在线停滞触发的配对差为{optim['point']:+.2f}个百分点，95% CI [{optim['CI95'][0]:+.2f}, {optim['CI95'][1]:+.2f}]；高安全投影组目标达到，但低安全投影组的CI下界未达到大于−1个百分点的要求。")
    lines+=['',f"[信息价值分组条形图]({HERE/'figures/information_value_grouped_bars_v2.png'})；同名SVG可编辑。原三组图及原收据全部保留，新增图从已封存数值读取，未重新计算统计。"]
    first='内容完整性检查通过：全部受保护实验输入、已有文档和所选模型文件的内容哈希均一致。仓库状态检查存在外部差异：17:32–17:33发生仓库快照与缓存迁移提交，改变Git HEAD、追踪状态、.gitignore，并新增迁移说明文档；本轮未执行写Git操作。原始差异保留在INTEGRITY_FINAL_COMPARISON.json，逐项内容复核见INTEGRITY_RECONCILIATION.json，严格仓库状态未变化标志为false；不把该项差异改写为完全无变化。'
    original=(HERE/'REPORT.md').read_text().split('\n\n',1)[1]
    (HERE/'REPORT_FINAL.md').write_text(first+'\n\n'+original+'\n\n'+'\n'.join(lines)+'\n')
    (HERE/'GAPS_FINAL.md').write_text((HERE/'GAPS.md').read_text()+'\n\n外部仓库状态差异：见INTEGRITY_RECONCILIATION.json；受保护输入内容保持一致，严格Git状态不一致。新增docs/hdd_migration_20261020.md只做stat，未作为实验输入。探测中不停滞组为空，相关比较N/A。\n')
    (HERE/'HANDOFF_FINAL.md').write_text((HERE/'HANDOFF.md').read_text()+'\n\n交付后补充复核入口：REPORT_FINAL.md、GAPS_FINAL.md、INTEGRITY_RECONCILIATION.json、DELIVERY_REVIEW.receipt.json。信息价值分组条形图为figures/information_value_grouped_bars_v2.png及.svg，原图保留。原REPORT.md/HANDOFF.md和原DELIVERY收据均未改写；最终复核区分内容一致与外部Git状态变化。每个学习型类型≥90%的前提未达到，未进行3个最终未见类型测试。\n')
    (HERE/'STATUS_FINAL.md').write_text('跨类型6已完成全部实验与数值、图和最终交付复核。\n\n受保护实验内容一致；外部Git状态及.gitignore/新增迁移文档差异已单列，未声称仓库状态完全未变。\n\n首读REPORT_FINAL.md、HANDOFF_FINAL.md、GAPS_FINAL.md。\n')
    seal('DELIVERY_REVIEW.receipt.json',[HERE/n for n in ('REPORT_FINAL.md','HANDOFF_FINAL.md','GAPS_FINAL.md','STATUS_FINAL.md','DELIVERY.receipt.json','NUMERIC_RESULTS.receipt.json','INFORMATION_GROUPED_BARS.receipt.json','INTEGRITY_RECONCILIATION.receipt.json','readiness_review_c6.py')],complete=True,protected_contents_match=True,strict_repository_state_unchanged=False,external_changes_disclosed=True,no_science_rerun=True)
    print('Final review complete; protected contents match; external repository changes disclosed',flush=True)

if __name__=='__main__':main()
