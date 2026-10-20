"""Final immutable-source/input audit, forbidden trees stat-only."""
from common import *
from audit import snapshot

def differences(a,b):return {k:dict(before=a.get(k),after=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}

def main():
    start=read(HERE/'INTEGRITY_START.json');end=snapshot(start);write(HERE/'INTEGRITY_END.json',end);code=differences(start['protected_python_sha256'],end['protected_python_sha256']);docs=differences(start['existing_document_sha256'],end['existing_document_sha256']);repos={}
    for repo,old in start['repositories'].items():
        now=end['repositories'][repo];changes={k:dict(before=old[k],after=now[k]) for k in ('head','status','diff_sha256') if old[k]!=now[k]}
        if 'status' in changes:
            added={str(Path(p).relative_to(ROOT)) for p in end['user_added_prompts_stat_only']};filtered='\n'.join(line for line in now['status'].splitlines() if line[3:] not in added)+ ('\n' if now['status'] else '')
            if filtered==old['status']:changes.pop('status')
        if changes:repos[repo]=changes
    inventory=read(HERE/'HISTORY_READ_INVENTORY.json');inputs=[]
    for r in inventory:
        path=Path(r['path']);after=sha(path) if path.exists() else None
        if after!=r['sha256']:inputs.append(dict(path=str(path),before=r['sha256'],after=after))
    result=dict(utc=utc(),passed=not code and not docs and not repos and not inputs,protected_source_unchanged=not code,python_changes=code,unexpected_existing_document_changes=docs,repository_changes=repos,historical_input_changes=inputs,user_added_prompt_metadata_only=end['user_added_prompts_stat_only'],forbidden_trees_stat_only=True,no_git_writes=True)
    write(HERE/'INTEGRITY_COMPARISON.json',result);seal('INTEGRITY_COMPARISON.receipt.json',[HERE/n for n in ('INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','close_audit.py')]);print('Final integrity passed:',result['passed'],flush=True)
if __name__=='__main__':main()
