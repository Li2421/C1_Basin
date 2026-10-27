"""Closing audit with prefix pruning and stat-only parallel exceptions."""
from experiment import *
import subprocess
from datetime import datetime
from audit import safe_walk

FORBIDDEN=('decentralized_','teacher_student_','multiscene_','baseline1_')

def stat(path):
    try:s=path.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    except FileNotFoundError:return dict(missing=True)

def main():
    start=read(HERE/'INTEGRITY_START.json');start_epoch=datetime.fromisoformat(start['utc'].replace('Z','+00:00')).timestamp();parallel=set(start['parallel_stat_only']);skipped={};sources={};differences=[]
    def under_parallel(rel):return any(rel==p or rel.startswith(p+'/') for p in parallel)
    def protected_hash(path,expected,label):
        value=sha(path) if path.exists() else None
        if value!=expected:differences.append(dict(kind=label,path=str(path),opening=expected,closing=value))
        return value
    for rel,h in start['sources'].items():
        path=ROOT/rel;st=stat(path)
        if under_parallel(rel) and (st.get('missing') or st['mtime_ns']>start_epoch*1e9):skipped[rel]=st;continue
        sources[rel]=protected_hash(path,h,'source')
    poc={}
    for p,h in start['poc_python'].items():poc[p]=protected_hash(Path(p),h,'poc')
    # Detect added PoC Python paths without touching prohibited diagnostics.
    pocroot=ROOT.parent/'Basin_C1_flow_field_poc_20261004/poc'
    for p in safe_walk(pocroot):
        if p.suffix=='.py' and str(p) not in start['poc_python']:differences.append(dict(kind='new_poc_python',path=str(p)))
    docs={rel:protected_hash(ROOT/rel,h,'existing_document') for rel,h in start['docs'].items()};newprompts={}
    for p in (ROOT/'docs').iterdir():
        rel=str(p.relative_to(ROOT))
        if rel in start['docs']:continue
        if p.name.startswith('prompt_'):newprompts[rel]=stat(p)
        else:differences.append(dict(kind='new_nonprompt_document',path=rel,stat=stat(p)))
    parallel_stat={}
    for p in (ROOT/'diagnostics').iterdir():
        if p==HERE:continue
        rel=str(p.relative_to(ROOT))
        if rel in parallel or p.name.startswith(FORBIDDEN):parallel_stat[rel]=stat(p)
    repositories={}
    def ignored(rel):
        return rel==str(HERE.relative_to(ROOT)) or rel.startswith(str(HERE.relative_to(ROOT))+'/') or under_parallel(rel) or rel in newprompts
    for root,opening in start['repositories'].items():
        def git(*args):return subprocess.check_output(['git','-C',root,*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
        status=git('status','--porcelain').decode();head=git('rev-parse','HEAD').decode().strip();names=git('diff','--name-only').decode().splitlines();restricted=[]
        if Path(root)==ROOT:
            restricted=[n for n in names if under_parallel(n) or n.startswith(tuple('diagnostics/'+p for p in FORBIDDEN))]
        if restricted:
            # Never obtain or hash changed parallel/prohibited file contents.
            args=['diff','--','.']+[f':(exclude){p}' for p in restricted];diffhash=hashlib.sha256(git(*args)).hexdigest();full=None
        else:diffhash=hashlib.sha256(git('diff')).hexdigest();full=diffhash
        repositories[root]=dict(head=head,status=status,diff_sha256=full,protected_diff_sha256=diffhash,parallel_diff_paths_stat_only={p:stat(ROOT/p) for p in restricted})
        def normalized(text):
            return sorted(line for line in text.splitlines() if Path(root)!=ROOT or not ignored(line[3:].split(' -> ')[-1].strip('"').rstrip('/')))
        if head!=opening['head']:differences.append(dict(kind='git_head',root=root))
        if normalized(status)!=normalized(opening['status']):differences.append(dict(kind='git_status',root=root,opening=normalized(opening['status']),closing=normalized(status)))
        if diffhash!=opening['diff_sha256']:differences.append(dict(kind='git_diff',root=root,opening=opening['diff_sha256'],closing=diffhash))
    external={}
    for name,expected in read(HERE/'EXTERNAL_SOURCES.json').items():
        folder=HERE/'external'/name;files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and '.git' not in p.parts}
        external[name]=dict(commit=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD']).decode().strip(),files=files)
        if files!=expected['files'] or external[name]['commit']!=expected['commit']:differences.append(dict(kind='external_source',name=name))
    supplement=read(HERE/'SOURCE_INPUT_AUDIT_SUPPLEMENT.json');extra={}
    for r in supplement['entries']:extra[r['path']]=protected_hash(Path(r['path']),r['whole_file_sha256'],'supplemental_input')
    historical=[]
    for r in read(HERE/'STRICT_PROJECTION_GATE.json')['records']:
        p=Path(r['source']);h=protected_hash(p,r['source_sha256'],'historical_raw');old=read(p);trace=Path(old['trace_path']);th=protected_hash(trace,r['trace_sha256'],'historical_trace');historical.append(dict(path=str(p),sha256=h,trace=str(trace),trace_sha256=th))
    end=dict(utc=utc(),repositories=repositories,poc_python=poc,sources=sources,docs=docs,user_new_prompts_stat_only=newprompts,parallel_stat_only=parallel_stat,parallel_source_changes_stat_only=skipped,external=external,supplemental_inputs=extra,historical_reuse=historical)
    write(HERE/'INTEGRITY_PRECLOSE.json',end)
    passed=not differences;coverage=supplement['coverage_limitation'];summary=('已记录保护源码、PoC、两个仓库Git与外部源码的前后比较一致' if passed else '完整性发现差异，详见INTEGRITY_PRECLOSE_COMPARISON.json')+'；并行目录/用户新增提示词按要求仅stat、单列例外；实际复用参数字段与开工封存副本一致，但原非Python配置全文件开工hash覆盖不完整，不能宣称严格全量审计通过'
    write(HERE/'INTEGRITY_PRECLOSE_COMPARISON.json',dict(utc=utc(),covered_sources_pass=passed,strict_full_coverage_pass=False,coverage_limitation=coverage,differences=differences,parallel_stat_only_exceptions=skipped,user_new_prompts_stat_only=newprompts,summary=summary))
    seal('INTEGRITY_PRECLOSE.receipt.json',[HERE/'INTEGRITY_START.json',HERE/'INTEGRITY_PRECLOSE.json',HERE/'INTEGRITY_PRECLOSE_COMPARISON.json',HERE/'SOURCE_INPUT_AUDIT_SUPPLEMENT.json',HERE/'audit_preclose.py'],no_start_snapshot_rewritten=True)
    print(summary,flush=True)

if __name__=='__main__':main()
