"""Read-only evidence analysis and final provenance; never create a rollout."""
import guard
from guard import HERE,ROOT
import sys
sys.path.insert(0,str(ROOT))
import json,hashlib,subprocess,datetime,os,time,re
from pathlib import Path
from shared_rollout_db.src import rollout_db as db

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
    if not pending:return
    from runner import case,controllers
    from acceleration import accelerated_case
    from coverage import write as coverage_write
    make=accelerated_case if read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']and read(HERE/'ACCELERATION_KEY_CHAIN_EXTENSION.json')['passed']else case
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
    ctls={c['type_code']:c for c in controllers()}
    menus=read(HERE/'FAMILIES_64.json')['families'];reuse=read(HERE/'REUSE_MANIFEST.json')['records']
    for stage in pending:
        match=re.fullmatch(r'oracle_f(\d+)_k(\d+)',stage)
        if not match:continue
        fi,ki=map(int,match.groups());sp=menus[fi]['candidate_pool'][ki-1]
        for g in groups:
            if g['family']!=fi or g['success']or g['skipped']or g['searched']!=ki-1:continue
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

def main():
    reconcile_completed_frontier()
    state=read(HERE/'COMPLETE.json');coverage=read(HERE/'COVERAGE.json')
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
            for point in family['curve']:out.write(f"{family['index']},all,{point['K']},{point['covered']},784\n")
        for t in coverage['by_type']:
            for fi,curve in t['curves'].items():
                for k,value in enumerate(curve,1):out.write(f"{fi},{t['type_id']},{k},{value},{t['denominator']}\n")
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
            other=base[(g['type_id'],g['level'],g['controller'],g['state_index'])]
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
    integrity=dict(time=datetime.datetime.now().astimezone().isoformat(),status=status.decode(),
        status_sha256=hashlib.sha256(status).hexdigest(),external_diff_sha256=hashlib.sha256(diff).hexdigest(),
        initial_diff_sha256=initial['diff_sha256'],external_diff_unchanged=hashlib.sha256(diff).hexdigest()==initial['diff_sha256'],
        imported_files_sha256=imports,changed_imports=changed,
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='No writes to old experiment directories; first-observed worker import hashes checked against current files. Git metadata and official journals/SQLite are authorized writes.')
    write(HERE/'INTEGRITY_END.json',integrity)
    write(HERE/'DELIVERY_AUDIT.json',dict(official_rows_verified=len(raw),raw_outcomes=counters,
        actual_attempts=read(HERE/'BUDGET.json')['actual_attempts'],horizon_transfer_records=len(m['records']),
        candidates_sha256=sha(target),integrity_passed=not changed and integrity['external_diff_unchanged']))
    gaps=[]
    if plot_error:gaps.append('曲线绘图失败（非主线，覆盖数保存在JSON/CSV）：'+plot_error)
    if not coverage['complete']:gaps.append('搜索未全部完成；未运行候选与基础设施缺失均单列，不能当作64次失败。')
    for family in coverage['families']:
        if family['distribution']['incomplete']:gaps.append(f"{family['name']}：{family['distribution']['incomplete']} 个案例未完成。")
    if changed:gaps.append('导入文件发生变化，见 INTEGRITY_END.json。')
    (HERE/'GAPS.md').write_text('# 证据缺口\n\n'+('\n'.join('- '+g for g in gaps)if gaps else'规定的有限候选覆盖搜索已完成。未找到不构成连续参数空间不可解证明。')+'\n\n本轮不执行论文方法、重新标定、选族或维度推荐。\n')
    with(HERE/'REPORT.md').open('a')as out:
        out.write('\n\n每档8个物理初态分别接入集中式随机采样模型预测控制和梯度下降模型预测控制，合计16个案例；两控制器使用匹配的物理初态与随机流。\n')
        out.write('\n代码分支：'+subprocess.check_output(['git','branch','--show-current'],text=True).strip()+'；记录时提交：'+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()+'。新代码与结果仅在 diagnostics/scene_types_oracle_coverage_v3/。\n')
        out.write('\n\n逐候选原始记录见 candidates.jsonl；每个旧结局的显式迁移见 REUSE_MANIFEST.json；时间口径见 TIMING.json；全部搜索曲线见 budget_curves.csv。\n')
        out.write('\n| 修正族 | 成功条件平均完成时间（秒） | 全队列截断平均时间（秒） |\n|---|---:|---:|\n')
        for row in analysis:out.write(f"| {row['name']} | {row['success_condition_mean_time']} | {row['full_cohort_truncated_mean_time']} |\n")
        out.write('\n共同成功案例的配对时间另见 TIMING.json，不与成功条件均值混合。\n')
        if not plot_error:
            out.write('\n每族总体与逐类型搜索曲线见 figures/（14张独立矢量图），图像只读取冻结结果。\n')
    state.update(complete=coverage['complete']and not(HERE/'PROTECTION_VIOLATION.jsonl').exists(),
        delivery_analysis_complete=True,official_rows_verified=len(raw),
        integrity_passed=not changed and integrity['external_diff_unchanged'])
    if not state['integrity_passed']:state['complete']=False
    write(HERE/'COMPLETE.json',state)
    print('FINALIZED',state,flush=True)

if __name__=='__main__':main()
