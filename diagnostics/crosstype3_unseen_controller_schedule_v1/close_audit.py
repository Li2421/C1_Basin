"""Final protected-source audit; user-authored prompt revisions stay explicit."""
from common import *
from audit import snapshot

def differences(a,b):return {k:dict(before=a.get(k),after=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
def main():
    start=read(HERE/'INTEGRITY_START.json');end=snapshot(start);write(HERE/'INTEGRITY_END.json',end)
    code=differences(start['protected_python_sha256'],end['protected_python_sha256']);docs=differences(start['existing_document_sha256'],end['existing_document_sha256']);path=str(ROOT/'docs/prompt_crosstype3_unseen_controller_two_eta_schedule_20261019.md');authorized={}
    if path in docs:
        change=docs.pop(path);assert change['after']==read(HERE/'PROMPT_REVISION_LATEST.json')['prompt_new_sha256'];authorized[path]=dict(**change,reason='User explicitly edited and asked full reread; both revised snapshots/receipts preserved')
    repositories={}
    for repo,old in start['repositories'].items():
        now=end['repositories'][repo];changes={k:dict(before=old[k],after=now[k]) for k in ('head','status','diff_sha256') if old[k]!=now[k]}
        if 'status' in changes:
            new=set(end['user_added_prompts_stat_only']);relative={str(Path(p).relative_to(ROOT)) for p in new}
            filtered='\n'.join(line for line in now['status'].splitlines() if line[3:] not in relative)+('\n' if now['status'] else '')
            if filtered==old['status']:changes.pop('status')
        if changes:repositories[repo]=changes
    inventory=read(HERE/'HISTORY_READ_INVENTORY.json');inputs=[dict(path=r['path'],before=r['sha256'],after=sha(r['path'])) for r in inventory if sha(r['path'])!=r['sha256']]
    result=dict(utc=utc(),protected_source_unchanged=not code,unexpected_existing_document_changes=docs,user_authorized_prompt_changes=authorized,python_changes=code,repository_changes=repositories,historical_input_changes=inputs,user_added_prompt_metadata_only=end['user_added_prompts_stat_only'],forbidden_trees_stat_only=True,passed=not code and not docs and not repositories and not inputs,audit_description='All protected code and inputs compared exactly; the existing prompt modified by the user is listed as an authorized document difference, never hidden')
    write(HERE/'INTEGRITY_COMPARISON.json',result)
    seal('INTEGRITY_COMPARISON.receipt.json',[HERE/'INTEGRITY_START.json',HERE/'INTEGRITY_END.json',HERE/'INTEGRITY_COMPARISON.json',HERE/'close_audit.py',HERE/'PROMPT_REVISION_LATEST.receipt.json'])
    print('Integrity audit passed:',result['passed'],'authorized prompt differences:',len(authorized),flush=True)
if __name__=='__main__':main()
