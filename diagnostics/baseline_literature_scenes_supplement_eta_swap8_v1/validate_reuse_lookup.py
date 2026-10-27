"""Validate cross-stage exact reuse with immutable completed evidence only."""
import bootstrap
from experiment import *
from evidence_store import load,locate,sources_index
if __name__=='__main__':
    selected=read(HERE/'SAVED_DIAGNOSTIC_VALIDATION.json')['cases'];checked=[]
    for item in selected:
        raw=read(item['raw_path']);c=raw['case'];r=load('nonexistent_evaluation_lookup_only',c);p=locate('nonexistent_evaluation_lookup_only',c)
        assert r['rollout_uid']==raw['rollout_uid'] and p.resolve()==Path(item['raw_path']).resolve()
        checked.append(dict(rollout_uid=r['rollout_uid'],source=str(p),source_sha256=sha(p)))
    assert not (HERE/'nonexistent_evaluation_lookup_only').exists()
    write(HERE/'EXACT_REUSE_LOOKUP_VALIDATION.json',dict(utc=utc(),passed=True,checked=checked,local_artifacts_indexed=len(sources_index()),new_rollouts=0,source_manifests_never_modified=True))
    print('Cross-stage exact reuse lookup passed',len(checked),flush=True)
