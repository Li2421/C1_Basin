"""Closing read-only audit, with prefix pruning and stat-only new prompts."""
from audit import *

def differences(before,after):
    return dict(added=sorted(set(after)-set(before)),deleted=sorted(set(before)-set(after)),changed=sorted(p for p in set(before)&set(after) if before[p]!=after[p]))
def main():
    assert not (HERE/'INTEGRITY_END.receipt.json').exists()
    opening=json.loads((HERE/'INTEGRITY_START.json').read_text());closing=snapshot(opening)
    dump(HERE/'INTEGRITY_END_SNAPSHOT.json',closing)
    protected=differences(opening['protected_python_sha256'],closing['protected_python_sha256'])
    docs=differences(opening['existing_document_sha256'],closing['existing_document_sha256'])
    repositories={}
    def visible_status(value,excluded):
        result=[]
        for row in value.splitlines():
            name=row[3:].strip().rstrip('/')
            p=ROOT/name
            if any(p==q or p.is_relative_to(q) for q in excluded):continue
            result.append(row)
        return sorted(result)
    excluded=[HERE,*[Path(p) for p in closing['forbidden_metadata']],*[Path(p) for p in closing['user_added_prompts_stat_only']]]
    for root,old in opening['repositories'].items():
        new=closing['repositories'][root]
        repositories[root]=dict(head_unchanged=old['head']==new['head'],tracked_diff_unchanged=old['diff_sha256']==new['diff_sha256'],status_unchanged=visible_status(old['status'],excluded)==visible_status(new['status'],excluded))
    inventory=json.loads((HERE/'HISTORY_READ_INVENTORY.json').read_text())
    expected={r['path']:r['sha256'] for r in inventory['files']};expected.update(inventory['controller_inputs'])
    for row in json.loads((HERE/'BOOTSTRAP_SOURCES.json').read_text())['sources']:expected[row['source']]=row['sha256']
    row=json.loads((HERE/'REGISTRY_SOURCE.json').read_text());expected[row['source']]=row['sha256']
    expected.update(json.loads((HERE/'LEGACY_COMPARATOR_INPUTS.json').read_text())['files'])
    input_differences=[]
    for path,h in expected.items():
        p=Path(path)
        # Explicitly authorized historical documents are the only crosstype2 reads.
        for folder in closing['forbidden_metadata']:
            if p.is_relative_to(Path(folder)):
                assert p.parent.name=='crosstype2_mixed_zoo_selector_v1' and p.name in ('REPORT.md','HANDOFF.md','GAPS.md'),path
        if not p.exists() or sha(p)!=h:input_differences.append(path)
    forbidden_changes={}
    for folder in set(opening['forbidden_metadata'])|set(closing['forbidden_metadata']):
        d=differences(opening['forbidden_metadata'].get(folder,{}),closing['forbidden_metadata'].get(folder,{}))
        if any(d.values()):forbidden_changes[folder]=d
    passed=not any(protected.values()) and not any(docs.values()) and not input_differences and all(all(r.values()) for r in repositories.values())
    result=dict(utc=utc(),passed=passed,protected_python_differences=protected,existing_document_differences=docs,repositories=repositories,historical_inputs_checked=len(expected),historical_input_differences=input_differences,user_added_prompts_stat_only=closing['user_added_prompts_stat_only'],forbidden_and_parallel_metadata_changes=forbidden_changes,forbidden_changes_are_stat_only_not_attributed_to_this_round=True,write_scope='This round directory and official shared database append lifecycle only',git_writes=0,package_installs=0,unit_replay_process_caveat='See UNIT_REPLAY_DOCUMENTATION_V2.json; two additional engineering replays lacked independently sealed preflight before replay. No experimental labels/DB writes; original mistaken wording retained with correction.')
    dump(HERE/'INTEGRITY_END.json',result)
    files=[HERE/n for n in ('INTEGRITY_START.json','INTEGRITY_END_SNAPSHOT.json','INTEGRITY_END.json','final_integrity.py','UNIT_REPLAY_DOCUMENTATION_V2.receipt.json')]
    dump(HERE/'INTEGRITY_END.receipt.json',dict(utc=utc(),files={str(p):sha(p) for p in files}))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert passed,'Closing integrity differences require attribution before delivery'
if __name__=='__main__':main()
