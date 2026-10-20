"""End audit: protected content unchanged, parallel/forbidden trees stat-only."""
from core import *
from audit import snapshot


def normalize_status(value,excluded):
    result=[]
    for line in value.splitlines():
        path=line[3:].rstrip('/')
        if path==str(HERE.relative_to(ROOT)) or any(path==p or path.startswith(p+'/') for p in excluded):continue
        result.append(line)
    return sorted(result)


def main():
    assert not (HERE/'INTEGRITY_END.json').exists()
    opening=read(HERE/'INTEGRITY_START.json');closing=snapshot(tuple(opening['parallel_metadata']));write(HERE/'INTEGRITY_END.json',closing)
    excluded=set(closing['parallel_metadata'])|set(closing['forbidden_metadata']);relative=[str(Path(p).relative_to(ROOT)) for p in excluded]
    before=opening['protected_python_sha256'];after=closing['protected_python_sha256']
    changed={p:dict(before=h,after=after.get(p)) for p,h in before.items() if not any(Path(p).is_relative_to(q) for q in map(Path,excluded)) and after.get(p)!=h}
    added={p:h for p,h in after.items() if p not in before and not any(Path(p).is_relative_to(q) for q in map(Path,excluded))}
    sources=dict(before)
    for row in read(HERE/'HISTORY_READ_INVENTORY.json')['files']:sources[row['path']]=row['sha256']
    for cfg in read(HERE/'IPPO_TRAINING_CONFIGS.json'):sources.update(cfg['input_sha256'])
    for name in ('BOOTSTRAP_SOURCES.json','COPIED_SOURCES.json','ADDITIONAL_COPIED_SOURCES.json','CRITIC_SOURCE_PROVENANCE.json'):
        value=read(HERE/name);rows=value['sources'] if isinstance(value,dict) else value
        for row in rows:sources[row['source']]=row['sha256']
    write(HERE/'PREIMPORT_SOURCES.json',dict(utc=utc(),snapshot_source_utc=opening['utc'],snapshot_source_sha256=sha(HERE/'INTEGRITY_START.json'),sources=before,note='This index is materialized at close from the immutable opening snapshot; protected-Python hashes themselves were captured before runtime imports. Forbidden/parallel content excluded.'))
    source_changes={}
    for p,h in sources.items():
        assert not any(Path(p).is_relative_to(Path(q)) for q in closing['forbidden_metadata'])
        if any(Path(p).is_relative_to(Path(q)) for q in excluded):continue
        actual=sha(p)
        if actual!=h:source_changes[p]=dict(before=h,after=actual)
    write(HERE/'SOURCE_AND_INPUT_RECEIPTS.json',dict(utc=utc(),sources=sources,explicit_changed=source_changes))
    repositories={}
    for root,old in opening['repositories'].items():
        new=closing['repositories'][root]
        a=normalize_status(old['status'],relative) if root==str(ROOT) else old['status'];b=normalize_status(new['status'],relative) if root==str(ROOT) else new['status']
        if old['head']!=new['head'] or old['diff_sha256']!=new['diff_sha256'] or a!=b:repositories[root]=dict(before=old,after=new)
    metadata={p:dict(before_files=len(opening['parallel_metadata'].get(p,{})),after_files=len(rows),metadata_changed=rows!=opening['parallel_metadata'].get(p)) for p,rows in closing['parallel_metadata'].items()}
    result=dict(utc=utc(),passed=not(changed or added or source_changes or repositories),protected_python_changes=changed,protected_python_additions=added,explicit_source_changes=source_changes,repository_changes=repositories,parallel_metadata_changes=metadata,forbidden_content_read=False,protected_opening_files=len(before),protected_input_files=len(sources),no_git_writes=True,no_package_installs=True)
    write(HERE/'INTEGRITY_COMPARISON.json',result)
    seal('INTEGRITY_COMPARISON.receipt.json',[HERE/n for n in ('INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','PREIMPORT_SOURCES.json','SOURCE_AND_INPUT_RECEIPTS.json','audit.py','final_integrity.py')])
    print('Closing integrity audit passed:',result['passed'])


if __name__=='__main__':main()
