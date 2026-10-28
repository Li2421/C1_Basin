"""Read-only evidence analysis and final provenance; never create a rollout."""
import guard
from guard import HERE,ROOT
import sys
sys.path.insert(0,str(ROOT))
import json,hashlib,subprocess,datetime,os,time,re
from pathlib import Path
from shared_rollout_db.src import rollout_db as db
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reconcile_completed_frontier():
    """A hard cap can interrupt a batch after some real results were saved."""
    state_path=HERE/'SEARCH_STATE.json'
    if not state_path.exists():return
    state=read(state_path);groups=state['groups'];changed=False
    pending=[]
    for p in HERE.glob('oracle_f*_k*_reuse.json'):
        stage=p.name[:-len('_reuse.json')]
        if not(HERE/f'{stage}_labels.json').exists():pending.append(stage)
    for p in HERE.glob('oracle_f*_r*_frontier.json'):
        stage=p.name[:-len('_frontier.json')]
        if not(HERE/f'{stage}_labels.json').exists():pending.append(stage)
    if not pending:return
    from runner import case,controllers
    from acceleration import accelerated_case
    from coverage import write as coverage_write
    make=accelerated_case if read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']and read(HERE/'ACCELERATION_KEY_CHAIN_EXTENSION.json')['passed']else case
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
    ctls={c['type_code']:c for c in controllers()}
    menus=read(HERE/'FAMILIES_64.json')['families'];reuse=read(HERE/'REUSE_MANIFEST.json')['records']
    for stage in pending:
        frontier_path=HERE/f'{stage}_frontier.json'
        if frontier_path.exists():
            todo=read(frontier_path)
        else:
            match=re.fullmatch(r'oracle_f(\d+)_k(\d+)',stage)
            if not match:continue
            fi,ki=map(int,match.groups())
            todo=[dict(family=fi,type_id=g['type_id'],level=g['level'],controller=g['controller'],state_index=g['state_index'],candidate_index=ki)for g in groups if g['family']==fi]
        group_map={(g['family'],g['type_id'],g['level'],g['controller'],g['state_index']):g for g in groups}
        for item in todo:
            fi,ki=item['family'],item['candidate_index'];sp=menus[fi]['candidate_pool'][ki-1]
            g=group_map[(fi,item['type_id'],item['level'],item['controller'],item['state_index'])]
            if g['success']or g['skipped']or g.get('screened_out')or g['searched']!=ki-1:continue
            c=make(scenes[(g['type_id'],g['level'])],ctls[g['controller']],g['state_index'],sp,purpose='v3_oracle_coverage')
            uid=c['rollout_uid'];p=HERE/'raw'/f'{uid}.json';r=None
            if p.exists():r=read(p)
            elif uid in reuse and reuse[uid]['action']=='reuse_terminal_outcome_after_prefix_certificate':r=read(reuse[uid]['v1_path'])
            if r is None:continue
            query=dict(index=ki,target_query_uid=uid,rollout_uid=r['rollout_uid'],success=r['success'],
                       outcome=r['outcome'],source=r['source'],completion_time=r['completion_time'],
                       episode_length=r['episode_length'],trace=r['trace'])
            g['queries'].append(query);g['searched']=ki;changed=True
            if r['success']:g['success']=True;g['first_success']=ki;g['solution']=dict(query,spec=sp)
    if changed:
        write(state_path,state);coverage_write(groups,state['baseline'])
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n最终只读重算：把硬上限/异常发生时最后一批已保存的真实结局纳入覆盖表；没有启动任何新rollout或跳过候选顺序。\n')

def consume_available_without_rollouts():
    """A numerical budget limit does not discard already existing exact evidence."""
    p=HERE/'SEARCH_STATE.json'
    if not p.exists()or(HERE/'PROTECTION_VIOLATION.jsonl').exists():return
    state=read(p)
    if all(g['success']or g['searched']==64 or g['skipped']for g in state['groups']):return
    from runner import case,sources
    from acceleration import accelerated_case
    from reuse import cached,flush
    from main import consume
    from representative_scope import in_requested_scope
    planned=read(HERE/'safety_only_planned.json')['exact_configs']
    if not planned or sources()!=planned[0]['controller_payload']['execution_sources']:
        with(HERE/'GAPS.md').open('a')as f:f.write('\n执行源码指纹变化，跳过最终缓存推进，保留原始已执行记录。\n')
        return
    make=accelerated_case if read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']and read(HERE/'ACCELERATION_KEY_CHAIN_EXTENSION.json')['passed']else case
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
    ctls={c['type_code']:c for c in read(HERE/'CONTROLLERS_FROZEN.json')['controllers']}
    menus=read(HERE/'FAMILIES_64.json')['families'];extra=[]
    with db.connect(True)as con:
        for g in state['groups']:
            if not in_requested_scope(g)or g.get('screened_out'):continue
            while not g['success']and not g['skipped']and g['searched']<64:
                ki=g['searched'];sp=menus[g['family']]['candidate_pool'][ki]
                c=make(scenes[(g['type_id'],g['level'])],ctls[g['controller']],g['state_index'],sp,purpose='v3_oracle_coverage')
                r=cached(c,'final_cache_only_no_rollouts',con)
                if r is None:break
                extra.append(consume(g,c,r,ki+1))
            if g['success']or g['searched']==64:g['gap']=None
    flush()
    if extra:
        write(p,state);write(HERE/'FINAL_CACHE_CONSUMPTION.json',dict(environment_steps=0,queries=extra))
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write(f'\n数值运行停止后的只读补全：按原候选顺序消化{len(extra)}条已有精确/前缀认证标签，遇首个缺失即停；没有新增rollout，未跳过旧超时，也未越过首次成功。\n')

def main():
    from repair_stage_counts import main as repair_accounting
    repair_accounting()
    reconcile_completed_frontier()
    consume_available_without_rollouts()
    # Rebuild presentation from actual query records, including pending counts.
    if(HERE/'SEARCH_STATE.json').exists():
        from coverage import write as coverage_write
        search=read(HERE/'SEARCH_STATE.json');coverage_write(search['groups'],search['baseline'])
    state=read(HERE/'COMPLETE.json');coverage=read(HERE/'COVERAGE.json')
    paper_required=(HERE/'BASELINE_AFTER_FAMILY4.json').exists()
    paper_path=HERE/'paper_after_four_column/COMPLETE.json'
    paper_complete=read(paper_path).get('complete',False)if paper_path.exists()else not paper_required
    if paper_required:
        from baseline_analysis import analyze as paper_analyze
        paper_analyze()
    for family in coverage['families']:
        for g in family['cases_detail']:
            assert [q['index']for q in g['queries']]==list(range(1,g['searched']+1))
            assert 0<=g['searched']<=64
            assert not any(q['success']for q in g['queries'][:-1])
            assert bool(g['success'])==bool(g['queries']and g['queries'][-1]['success'])
    raw=list((HERE/'raw').glob('*.json'));m=read(HERE/'REUSE_MANIFEST.json')
    seen={};counters=dict(success=0,collision=0,timeout=0,numerical_failure=0)
    with db.connect(True)as con:
        for p in raw:
            r=read(p);uid=r['rollout_uid']
            row=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
            assert row and row['success']==r['success']and row['numerical_failure']==r['numerical_failure']and not row['conflict_quarantined']
            assert Path(r['trace']).exists()
            seen[uid]=r
            for k in counters:counters[k]+=r[k]
    target=HERE/'candidates.jsonl';tmp=target.with_suffix('.tmp')
    with tmp.open('w')as out:
        for uid,r in sorted(seen.items()):out.write(json.dumps(r,ensure_ascii=False)+'\n')
        for query,item in sorted(m['records'].items()):
            if item['action']!='reuse_terminal_outcome_after_prefix_certificate':continue
            r=read(item['v1_path']);r=dict(r,logical_query_uid=query,source='v1_prefix_certified_horizon_transfer')
            out.write(json.dumps(r,ensure_ascii=False)+'\n')
    tmp.replace(target)
    with(HERE/'budget_curves.csv').open('w')as out:
        out.write('family,type_id,K,covered,denominator\n')
        for family in coverage['families']:
            for point in family['curve']:out.write(f"{family['index']},all,{point['K']},{point['covered']},{family['cases']}\n")
        for t in coverage['by_type']:
            for fi,curve in t['curves'].items():
                for k,value in enumerate(curve,1):out.write(f"{fi},{t['type_id']},{k},{value},{t.get('denominator_by_family',{}).get(fi,t['denominator'])}\n")
    analysis=[]
    for family in coverage['families']:
        cases=family['cases_detail'];success=[g for g in cases if g['success']]
        truncated=[]
        for g in cases:
            scene=next(e['scene']for e in read(HERE/'SCENES.json')['entries']if e['type_id']==g['type_id']and e['level']==g['level'])
            truncated.append(g['solution']['completion_time']if g['success']else scene['config']['max_steps']*scene['config']['dt'])
        analysis.append(dict(family=family['index'],name=family['name'],successful_cases=len(success),
            success_condition_mean_time=sum(g['solution']['completion_time']for g in success)/len(success)if success else None,
            full_cohort_truncated_mean_time=sum(truncated)/len(truncated),search_distribution=family['distribution']))
    paired=[]
    base={(g['type_id'],g['level'],g['controller'],g['state_index']):g for g in coverage['families'][0]['cases_detail']}
    for family in coverage['families'][1:]:
        pairs=[]
        for g in family['cases_detail']:
            other=base.get((g['type_id'],g['level'],g['controller'],g['state_index']))
            if other is None:continue
            if g['success']and other['success']:pairs.append([g['solution']['completion_time'],other['solution']['completion_time']])
        paired.append(dict(family=family['index'],paired_success_n=len(pairs),
            method_mean=sum(p[0]for p in pairs)/len(pairs)if pairs else None,
            original_three_mean=sum(p[1]for p in pairs)/len(pairs)if pairs else None))
    write(HERE/'TIMING.json',dict(families=analysis,paired_success=paired,
        distinction='Success-conditional time, paired common-success time and full-cohort truncated time are separate. No fastest-eta selection after first success.'))
    write(HERE/'COMPUTE_COSTS.json',dict(new_complete_unique=len(seen),
        actual_attempts=read(HERE/'BUDGET.json')['actual_attempts'],
        total_new_rollout_wall_seconds=sum(r['wall_seconds']for r in seen.values()),
        dictionary_seconds=sum(r.get('dictionary',{}).get('seconds',0.)for r in seen.values()),
        matrix_decomposition_seconds=sum(r.get('dictionary',{}).get('svd_seconds',0.)for r in seen.values()),
        note='Unique new requests only; shared/reused requests are not charged as new computation. Worker import/startup, coordinator, calibration and queue elapsed differ from rollout wall time.'))
    plot_error=None
    try:
        subprocess.run(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python',str(HERE/'plot.py')],
                       cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
    except Exception as exc:plot_error=repr(exc)
    observed={};changed=[]
    for p in(HERE/'import_manifests').glob('*.json'):
        for name,checksum in read(p).items():
            path=Path(name)
            if path.is_relative_to(HERE):continue
            observed.setdefault(name,set()).add(checksum)
    for name,checksums in observed.items():
        path=Path(name)
        if path.exists():
            current=sha(path)
            if checksums!={current}:changed.append(dict(path=name,observed_sha256=sorted(checksums),current_sha256=current))
    imports={name:sorted(values)for name,values in observed.items()}
    diff=subprocess.check_output(['git','diff','--binary','--','.',':(exclude)diagnostics/scene_types_oracle_coverage_v3'])
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'])
    initial=read(HERE/'INTEGRITY_START.json')
    pins=read(HERE/'CONTROLLERS_FROZEN.json')
    controller_unchanged=sha(pins['source'])==pins['sha256']
    integrity=dict(time=datetime.datetime.now().astimezone().isoformat(),status=status.decode(),
        status_sha256=hashlib.sha256(status).hexdigest(),external_diff_sha256=hashlib.sha256(diff).hexdigest(),
        initial_diff_sha256=initial['diff_sha256'],external_diff_unchanged=hashlib.sha256(diff).hexdigest()==initial['diff_sha256'],
        controller_profile_unchanged=controller_unchanged,
        imported_files_sha256=imports,changed_imports=changed,
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='No writes to old experiment directories; first-observed worker import hashes checked against current files. Git metadata and official journals/SQLite are authorized writes.')
    write(HERE/'INTEGRITY_END.json',integrity)
    write(HERE/'DELIVERY_AUDIT.json',dict(official_rows_verified=len(raw),raw_outcomes=counters,
        actual_attempts=read(HERE/'BUDGET.json')['actual_attempts'],horizon_transfer_records=len(m['records']),
        candidates_sha256=sha(target),integrity_passed=not changed and integrity['external_diff_unchanged']and controller_unchanged,
        candidate_order_verified=True,no_queries_after_first_success=True))
    gaps=[]
    if plot_error:gaps.append('曲线绘图失败（非主线，覆盖数保存在JSON/CSV）：'+plot_error)
    if not coverage['complete']:gaps.append('搜索未全部完成；未运行候选与基础设施缺失均单列，不能当作64次失败。')
    for family in coverage['families']:
        if family['distribution'].get('screened_out'):gaps.append(f"{family['name']}：{family['distribution']['screened_out']} 个案例因族筛选提前停止，未运行候选仍未知，不算64组失败。")
        if family['distribution']['incomplete']:gaps.append(f"{family['name']}：{family['distribution']['incomplete']} 个案例未完成。")
    if changed:gaps.append('导入文件发生变化，见 INTEGRITY_END.json。')
    if paper_required and not paper_complete:gaps.append('用户追加的论文固定参数对照存在未完成项，见 paper_after_four_column/EXECUTION_STATUS.json。')
    if not controller_unchanged:gaps.append('控制器配置源发生变化，见 CONTROLLERS_FROZEN.json 与 INTEGRITY_END.json。')
    (HERE/'GAPS.md').write_text('# 证据缺口\n\n'+('\n'.join('- '+g for g in gaps)if gaps else'规定的有限候选覆盖与追加固定论文对照已完成。未找到不构成连续参数空间不可解证明。')+'\n\n不重新标定；按用户追加要求筛选候选族，结果用于开发集选族，不作为独立验证。论文固定参数对照由用户后续追加，与C1 Oracle上限分开报告。\n')
    report=HERE/'REPORT.md';report.write_text(report.read_text().replace('None','—'))
    with report.open('a')as out:
        out.write('\n\n每档8个物理初态分别接入集中式随机采样模型预测控制和梯度下降模型预测控制，合计16个案例；两控制器使用匹配的物理初态与随机流。\n')
        out.write('\n代码分支：'+subprocess.check_output(['git','branch','--show-current'],text=True).strip()+'；记录时提交：'+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()+'。新代码与结果仅在 diagnostics/scene_types_oracle_coverage_v3/。\n')
        out.write('\n\n逐候选原始记录见 candidates.jsonl；每个旧结局的显式迁移见 REUSE_MANIFEST.json；时间口径见 TIMING.json；全部搜索曲线见 budget_curves.csv。\n')
        out.write('\n| 修正族 | 成功条件平均完成时间（秒） | 全队列截断平均时间（秒） |\n|---|---:|---:|\n')
        for row in analysis:out.write(f"| {row['name']} | {row['success_condition_mean_time']} | {row['full_cohort_truncated_mean_time']} |\n")
        out.write('\n共同成功案例的配对时间另见 TIMING.json，不与成功条件均值混合。\n')
        names=[f['name']for f in coverage['families']]
        out.write('\n\n## 逐类型合计覆盖\n\n| 类型 | 案例数 | '+' | '.join(names)+' |\n|---|---:|'+'---:|'*7+'\n')
        for t in coverage['by_type']:out.write('| '+t['name']+' | 按族分母 | '+' | '.join(f"{v}/{t.get('denominator_by_family',{}).get(str(fi),t['denominator'])}"for fi,v in enumerate(t['coverage64']))+' |\n')
        out.write('\n\n## 首次成功序号与搜索成本\n\n| 修正族 | 中位数 | 四分位 | 成功案例平均搜索数 | 完整案例平均搜索数 | 64次未找到 | 尚未完成 | 筛选停止 |\n|---|---:|---|---:|---:|---:|---:|---:|\n')
        for f in coverage['families']:
            d=f['distribution']
            out.write(f"| {f['name']} | {d['median']} | {d['quartiles']} | {d['average_queries_success']} | {d['average_queries_completed']} | {d['not_found_in64']} | {d['incomplete']} | {d.get('screened_out',0)} |\n")
        out.write('\n完整案例平均搜索数包含64次未找到的案例；成功案例均值只对找到解的案例计算，两者不混用。\n')
        out.write('\n\n## 逐场景四倍与六倍时限\n\n下表各格为四倍时限成功数→六倍时限成功数，分母均为16。沿用六倍时限首次成功即停；未运行的候选不作为四倍时限失败证据。\n\n| 场景 | 只加安全投影 | '+' | '.join(names)+' |\n|---|---:|'+'---:|'*7+'\n')
        for e in coverage['scenes']:
            values=['—（后续不运行此档）'if v.get('out_of_scope')else f"{v['coverage_4x']['64']}→{v['coverage']['64']}"for v in e['families']]
            out.write('| '+e['name']+f" | {e['baseline_success_4x']}→{e['baseline_success']} | "+' | '.join(values)+' |\n')
        if not plot_error:
            out.write('\n每族总体与逐类型搜索曲线见 figures/（14张独立矢量图），图像只读取冻结结果。\n')
        if paper_required:
            from representative_scope import selected_scenes
            out.write(f'\n\n用户追加的五篇论文六版本固定参数对照见 [单独报告](paper_after_four_column/REPORT.md)。每个参数profile{len(selected_scenes())*8}个物理初态，不把同一论文控制器重复成两个nominal。该对照和本轮C1 Oracle上限分别解释。\n')
        if(HERE/'REPRESENTATIVE17_FROZEN.json').exists():out.write('\n按用户最新要求，四列族完整49场景收尾后，其余族与论文方法仅运行纯几何参数中档的17代表场景。相同272案例的族间比较见 REPRESENTATIVE17_COMPARISON.json；未运行的其他档位不算失败。\n')
        if(HERE/'zero_coverage_audit/REPORT.md').exists():out.write('\n零覆盖场景的几何与原CBF联合可行轨迹核验见 [补充审计](zero_coverage_audit/REPORT.md)，诊断轨迹不混入Oracle或论文方法成功率。\n')
    state.update(complete=coverage['complete']and paper_complete and not(HERE/'PROTECTION_VIOLATION.jsonl').exists(),
        full_search_complete=coverage['full_search_complete'],adaptive_screening=True,
        delivery_analysis_complete=True,official_rows_verified=len(raw),
        paper_comparison_required=paper_required,paper_comparison_complete=paper_complete,
        integrity_passed=not changed and integrity['external_diff_unchanged']and controller_unchanged)
    if not state['integrity_passed']:state['complete']=False
    write(HERE/'COMPLETE.json',state)
    # Normalize empty statistics in user-facing tables without changing JSON nulls.
    report.write_text(report.read_text().replace('None','—'))
    (HERE/'STATUS.md').write_text('# 当前状态\n\n'+datetime.datetime.now().astimezone().isoformat()+'：'+
        ('有限候选覆盖与交付分析已全部完成，预计剩余时间0。'if state['complete']else'已停止并保存部分交付；未完成项见 GAPS.md，不能记作64次失败。')+
        f"\n\n实际新rollout尝试 {state['actual_new_rollouts']}/90000；官方索引核验 {len(raw)} 条。\n\n"+
        '；'.join(f"{f['name']} {f['coverage']['64']}/{f['cases']}"for f in coverage['families'])+'。\n')
    (HERE/'RESOURCE_WATCH_STOP').write_text('Numerical pipeline and final provenance analysis finished.\n')
    print('FINALIZED',state,flush=True)

if __name__=='__main__':main()
