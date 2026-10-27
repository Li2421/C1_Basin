"""Streaming protected hashes and on-disk stat comparison; never open prohibited data."""
from audit_scope import HERE,ROOT,D,OFFICIAL,utc,sha,read,write,metadata
import subprocess,hashlib,sqlite3,os
os.environ['GIT_OPTIONAL_LOCKS']='0'
from pathlib import Path

def main():
    start=read(HERE/'INTEGRITY_START.json');protected=dict(start['input_sha256']);declarations=[]
    for p in sorted(HERE.glob('INPUT_ADDITIONS_*.json')):
        x=read(p);declarations.append(dict(path=str(p),sha256=sha(p)))
        for r in x['files']:
            if r['path'] in protected:assert protected[r['path']]==r['sha256'],r['path']
            protected[r['path']]=r['sha256']
    changes=[]
    for p,expected in protected.items():
        q=Path(p);actual=sha(q) if q.exists() else None
        if actual!=expected:changes.append(dict(path=p,before=expected,after=actual))
    docs=start['existing_document_sha256'];docchanges=[]
    for p,expected in docs.items():
        q=Path(p);actual=sha(q) if q.exists() else None
        if actual!=expected:docchanges.append(dict(path=p,before=expected,after=actual))
    newprompts=[];othernewdocs=[]
    for p in (ROOT/'docs').rglob('*'):
        if not p.is_file() or str(p) in docs:continue
        s=p.stat();record=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,stat_only=True)
        (newprompts if p.name.startswith('prompt_') and p.suffix=='.md' else othernewdocs).append(record)
    newprompt_paths={r['path'] for r in newprompts};repos={};repochanges=[]
    def filtered_status(status,parent):
        rows=[]
        for line in status.splitlines():
            relative=line[3:]
            if parent==ROOT and (relative.startswith('diagnostics/') or relative=='diagnostics/' or str(ROOT/relative) in newprompt_paths):continue
            rows.append(line)
        return rows
    for parent in (ROOT,OFFICIAL):
        old=start['repositories'][str(parent)];run=lambda *args:subprocess.check_output(['git',*args],cwd=parent);excludes=[':(exclude)diagnostics'] if parent==ROOT else []
        if parent==ROOT:excludes += [':(exclude)'+str(Path(p).relative_to(ROOT)) for p in newprompt_paths]
        now=dict(head=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain').decode(),diff_sha256=hashlib.sha256(run('diff','HEAD','--','.',*excludes)).hexdigest());repos[str(parent)]=now
        if old['head']!=now['head'] or old['diff_sha256']!=now['diff_sha256'] or filtered_status(old['status'],parent)!=filtered_status(now['status'],parent):repochanges.append(dict(repository=str(parent),before=old,after=now))
    end_db=HERE/'AUDIT_METADATA_FINAL.sqlite'
    if end_db.exists():end_db.rename(HERE/('AUDIT_METADATA_FINAL.previous_attempt_'+utc().replace(':','-')+'.sqlite'))
    meta=metadata(end_db);db=sqlite3.connect(end_db);db.execute('PRAGMA cache_size=-32768');db.execute('ATTACH DATABASE ? AS original',(str(HERE/'AUDIT_METADATA_START.sqlite'),));query='SELECT e.path,e.root,o.size old_size,e.size new_size,o.mtime_ns old_mtime_ns,e.mtime_ns new_mtime_ns FROM file e LEFT JOIN original.file o ON e.path=o.path WHERE o.path IS NULL OR e.size!=o.size OR e.mtime_ns!=o.mtime_ns';counts={};sample=[]
    for row in db.execute(query):
        counts[row[1]]=counts.get(row[1],0)+1
        if len(sample)<200:sample.append(row)
    deleted={}
    for root,count in db.execute('SELECT o.root,COUNT(*) FROM original.file o LEFT JOIN file e ON e.path=o.path WHERE e.path IS NULL GROUP BY o.root'):deleted[root]=count
    db.close();write(HERE/'INTEGRITY_FINAL_END.json',dict(utc=utc(),protected_inputs=len(protected),input_additions=declarations,repositories=repos,new_user_prompts_stat_only=newprompts,other_new_documents_stat_only=othernewdocs,diagnostic_metadata=meta))
    passed=not changes and not docchanges and not repochanges and not othernewdocs;comparison=dict(utc=utc(),passed=passed,protected_input_differences=changes,existing_document_differences=docchanges,repository_differences=repochanges,unexpected_new_documents_stat_only=othernewdocs,new_user_prompts_stat_only=newprompts,parallel_diagnostic_stat_changes_by_directory=counts,parallel_diagnostic_deleted_stat_entries_by_directory=deleted,parallel_stat_difference_sample=sample,noninput_diagnostic_changes_do_not_fail_audit=True,prohibited_content_never_opened=True,read_only_git_only=True,no_git_commit=True)
    write(HERE/'INTEGRITY_FINAL_COMPARISON.json',comparison)
    from core import seal
    seal('INTEGRITY_FINAL_COMPARISON.receipt.json',[HERE/'INTEGRITY_START.receipt.json',HERE/'INTEGRITY_FINAL_END.json',HERE/'INTEGRITY_FINAL_COMPARISON.json',HERE/'closing_audit_c6_final.py'],passed=passed)
    print('Closing audit',passed,'protected differences',len(changes),'document differences',len(docchanges),'repository differences',len(repochanges),flush=True)

if __name__=='__main__':main()
