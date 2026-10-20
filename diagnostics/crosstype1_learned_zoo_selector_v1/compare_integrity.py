"""Compare final protected coverage, source hashes, git records and metadata."""
from pathlib import Path
import hashlib,json,datetime

HERE=Path(__file__).resolve().parent
PREFIXES=('decentralized_','teacher_student_','multiscene_')


def main():
    initial=json.loads((HERE/'INTEGRITY_START.json').read_text())
    final=json.loads((HERE/'INTEGRITY_END.json').read_text())
    before=initial['protected_python_sha256'];after=final['protected_python_sha256']
    changed=[p for p,h in before.items() if p in after and after[p]!=h]
    # Newly recent directories must not silently remove initial protected files
    # from the audit. Coverage loss is reported as unverified, never as a pass.
    unverified=sorted(set(before)-set(after))
    preimports=json.loads((HERE/'PREIMPORT_SOURCES.json').read_text())['sources'];import_changes=[]
    for name,digest in preimports.items():
        path=Path(name)
        assert not any(part.startswith(PREFIXES) for part in path.parts),'Do not read forbidden source content'
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:import_changes.append(name)
    repositories={}
    for path,record in initial['repositories'].items():
        end=final['repositories'][path]
        repositories[path]=dict(head_unchanged=record['head']==end['head'],status_unchanged=record['status']==end['status'],
            diff_unchanged=record['diff_sha256']==end['diff_sha256'],exclusions_unchanged=record['exclusions']==end['exclusions'])
    metadata={}
    for category in ('parallel_metadata','forbidden_metadata'):
        metadata[category]={}
        for folder in set(initial[category])|set(final[category]):
            a=initial[category].get(folder,{});b=final[category].get(folder,{})
            metadata[category][folder]=dict(changed_paths=sum(a.get(p)!=b.get(p) for p in set(a)|set(b)),
                added_paths=len(set(b)-set(a)),removed_paths=len(set(a)-set(b)),content_opened=False)
    passed=not changed and not unverified and not import_changes and all(
        r['head_unchanged'] and r['status_unchanged'] and r['diff_unchanged'] for r in repositories.values())
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=passed,
        protected_initial_count=len(before),protected_final_count=len(after),protected_python_changes=changed,
        initially_protected_files_no_longer_hashed=unverified,preimport_source_changes=import_changes,
        repositories=repositories,metadata_changes=metadata,forbidden_contents_read=False,final_experiment_audit=True)
    target=HERE/'INTEGRITY_FINAL_COMPARISON.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='metadata_changes'},indent=2))


if __name__=='__main__':main()
