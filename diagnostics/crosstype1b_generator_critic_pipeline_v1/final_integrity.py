"""Closing protected-source comparison; parallel/forbidden trees remain metadata only."""
from core import *
from audit import snapshot


def normalized_status(value,excluded):
    out=[]
    for line in value.splitlines():
        path=line[3:].rstrip('/')
        if path==str(HERE.relative_to(ROOT)) or any(path==p or path.startswith(p+'/') for p in excluded):continue
        out.append(line)
    return sorted(out)


def main():
    assert not (HERE/'INTEGRITY_END.json').exists()
    opening=read(HERE/'INTEGRITY_START.json');closing=snapshot(tuple(opening['parallel_metadata']))
    write(HERE/'INTEGRITY_END.json',closing)
    excluded=set(closing['parallel_metadata'])|set(closing['forbidden_metadata'])
    relative=[str(Path(p).relative_to(ROOT)) for p in excluded]
    old=opening['protected_python_sha256'];new=closing['protected_python_sha256']
    changed={p:dict(before=digest,after=new.get(p)) for p,digest in old.items()
        if not any(Path(p).is_relative_to(Path(q)) for q in excluded) and new.get(p)!=digest}
    added={p:digest for p,digest in new.items() if p not in old
        and not any(Path(p).is_relative_to(Path(q)) for q in excluded)}
    source_changes={}
    for name in ('PREIMPORT_SOURCES.json','REFERENCE_CONDITIONING_SOURCES.json'):
        for path,digest in read(HERE/name)['sources'].items():
            # Explicit source records must never include forbidden content.
            assert not any(Path(path).is_relative_to(Path(q)) for q in closing['forbidden_metadata'])
            actual=sha(path)
            if actual!=digest:source_changes[path]=dict(before=digest,after=actual)
    repo_changes={}
    for root,before in opening['repositories'].items():
        after=closing['repositories'][root]
        status_before=normalized_status(before['status'],relative) if root==str(ROOT) else before['status']
        status_after=normalized_status(after['status'],relative) if root==str(ROOT) else after['status']
        if before['head']!=after['head'] or before['diff_sha256']!=after['diff_sha256'] or status_before!=status_after:
            repo_changes[root]=dict(before=before,after=after)
    metadata_changes={p:dict(before_files=len(opening['parallel_metadata'].get(p,{})),after_files=len(rows),
        metadata_changed=rows!=opening['parallel_metadata'].get(p)) for p,rows in closing['parallel_metadata'].items()}
    result=dict(utc=utc(),passed=not(changed or added or source_changes or repo_changes),
        protected_python_changes=changed,explicit_source_changes=source_changes,repository_changes=repo_changes,
        protected_python_additions=added,
        parallel_metadata_changes=metadata_changes,forbidden_content_read=False,protected_opening_files=len(old),
        scope='Source content, repository tracked diff and normalized status; active/forbidden trees metadata only')
    write(HERE/'INTEGRITY_COMPARISON.json',result)
    seal('INTEGRITY_COMPARISON.receipt.json',[HERE/'INTEGRITY_START.json',HERE/'INTEGRITY_END.json',
        HERE/'INTEGRITY_COMPARISON.json',HERE/'final_integrity.py',HERE/'audit.py'])
    print('Closing integrity audit:',result['passed'],flush=True)


if __name__=='__main__':main()
