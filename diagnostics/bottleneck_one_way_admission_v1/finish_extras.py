"""Post-main integrity, optional horizon sensitivity, and geometry figure."""
from dispatch import *
from acceleration import accelerated_case
import datetime

def sensitivity():
    from reporting import failure_type
    from evidence import run_requests
    from validation import KEYS
    if(HERE/'SENSITIVITY.json').exists():return
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
    controls={c['type_code']:c for c in controllers()};spec=read(HERE/'FAMILIES_THIS_ROUND.json')['families'][0]['candidate_pool'][0]
    make=accelerated_case if read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']else case
    selected=[]
    for g in read(HERE/'ORACLE_STATE.json')['groups']:
        if g['family']!=0 or not g['hard']or g['success']or g['searched']!=64:continue
        r=g['queries'][0]['result'];sc=scenes[tuple(g['identity'][:2])]
        diag=failure_type(r,sc)
        if not diag.get('flags',{}).get('progressing_timeout'):continue
        sc=read_copy(sc);sc['config']['max_steps']=int(np.ceil(8*sc['longest_initial_route']/sc['config']['max_speed']/sc['config']['dt']))
        c=make(sc,controls[g['identity'][2]],g['identity'][3],spec,purpose='optional_eight_times_horizon')
        selected.append((g,r,c))
    from budget import used
    if used()+len(selected)>15000:
        dump(HERE/'SENSITIVITY.json',dict(complete=False,reason='hard budget',cases=len(selected)));return
    results=run_requests([c for g,r,c in selected],'optional_horizon8',allow_no_passage=False)if selected else{}
    checks=[]
    for g,old,c in selected:
        new=results.get(c['rollout_uid'])
        if new is None:checks.append(dict(identity=g['identity'],complete=False));continue
        with np.load(old['trace'])as a,np.load(new['trace'])as b:
            detail={k:a[k].dtype==b[k].dtype and a[k].shape==b[k][:len(a[k])].shape and a[k].tobytes()==b[k][:len(a[k])].tobytes()for k in KEYS}
        checks.append(dict(identity=g['identity'],prefix_equal=all(detail.values()),arrays=detail,old_outcome=old['outcome'],new_result=new))
    dump(HERE/'SENSITIVITY.json',dict(complete=len(results)==len(selected),cases=len(selected),checks=checks,excluded_from_primary_tables=True,horizon_definition='ceil(8 * frozen longest initial route / max speed / dt)'))

def read_copy(x):return json.loads(json.dumps(x))

def integrity():
    changed={};imports={}
    for p in(HERE/'import_manifests').glob('*.json'):
        for path,expected in read(p).items():
            f=Path(path)
            if not f.is_file()or f.is_relative_to(HERE):continue
            imports.setdefault(path,expected)
    for path,expected in imports.items():
        actual=sha(path)
        if actual!=expected:changed[path]=dict(expected=expected,actual=actual)
    # Active v3 content was not integrity-hashed. Snapshot manifests preserve
    # copied inputs; only stat records describe its concurrent output changes.
    old=ROOT/'diagnostics/scene_types_oracle_coverage_v3'
    stats={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)for p in old.glob('*')if p.is_file()}
    external=subprocess.check_output(['git','diff','--binary','--','.',':(exclude)diagnostics/bottleneck_one_way_admission_v1',':(exclude)diagnostics/scene_types_oracle_coverage_v3'])
    start=read(HERE/'INTEGRITY_START.json');same=hashlib.sha256(external).hexdigest()==start['external_diff_excluding_concurrent_sha256']
    freeze='DESIGN_FREEZE_v2.json'if(HERE/'DESIGN_FREEZE_v2.json').exists()else'DESIGN_FREEZE.json'
    frozen=read(HERE/freeze)['files'];design={n:dict(expected=h,actual=sha(HERE/n))for n,h in frozen.items()if sha(HERE/n)!=h}
    passed=not changed and same and not design and not(HERE/'PROTECTION_VIOLATION.jsonl').exists()
    dump(HERE/'INTEGRITY_END.json',dict(passed=passed,changed_external_imports=changed,external_diff_unchanged=same,design_changes=design,external_import_hashes=imports,concurrent_directory_stats=stats,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    (HERE/'GIT_END_STATUS.txt').write_text(subprocess.check_output(['git','status','--porcelain'],text=True))
    return passed

def main():
    for action in [sensitivity]:
        try:action()
        except Exception as exc:
            with(HERE/'GAPS.md').open('a')as f:f.write('\n非主线敏感性未完成：'+repr(exc)+'\n')
    try:subprocess.run(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python',str(HERE/'plot_geometry.py')],check=True)
    except Exception as exc:
        with(HERE/'GAPS.md').open('a')as f:f.write('\n俯视图未完成：'+repr(exc)+'\n')
    passed=integrity()
    result=read(HERE/'COMPLETE.json');result.update(integrity_passed=passed,complete=result['required_numeric_work_complete']and passed,finished=datetime.datetime.now().astimezone().isoformat())
    result['optional_sensitivity']=read(HERE/'SENSITIVITY.json')if(HERE/'SENSITIVITY.json').exists()else dict(complete=False)
    dump(HERE/'COMPLETE.json',result)
    gaps=[]
    if not result['required_numeric_work_complete']:gaps.append('规定数值主线存在未完成请求，不能记作控制失败。')
    if not passed:gaps.append('结束完整性未通过，见INTEGRITY_END.json。')
    if not result['optional_sensitivity'].get('complete'):gaps.append('可选8倍时限敏感性未完成。')
    if not(HERE/'passages_overview.png').exists():gaps.append('可选通道俯视图未生成。')
    (HERE/'GAPS.md').write_text('# 证据缺口\n\n'+('\n'.join('- '+g for g in gaps)if gaps else'要求的主线和已执行附表已完成；有限候选未成功不构成连续参数空间不可解证明。')+'\n')
    from reporting import write_report
    write_report()
    with(HERE/'REPORT.md').open('a')as f:
        f.write('\n结束完整性：'+str(passed)+'。详见INTEGRITY_END.json。\n\n可选8倍时限结果见SENSITIVITY.json，只用于诊断，不进入主表。通道俯视图见passages_overview.png。\n')
    (HERE/'STATUS.md').write_text('# 当前状态\n\n'+('本轮已完成，详见REPORT.md。'if result['complete']else'本轮结束但存在证据缺口，详见GAPS.md与COMPLETE.json。')+'\n')

if __name__=='__main__':main()
