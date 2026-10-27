"""Full exact-result, immutable trace and scoped repository integrity audit."""
from common import *
from stages import source_path
import subprocess


def database():
    stages=('replay','capability','pilot','main','extrapolation');seen={};files={};counts={};failures=[];checkpoint_receipts=set()
    with dbm.connect(True)as db:
        for stage in stages:
            p=HERE/f'{stage}_CASES.json'
            if not p.exists():continue
            cases=read(p);count=0
            for c in cases:
                rid=c['rollout_uid']
                if rid in seen:count+=1;continue
                if c.get('cross_scene_new'):
                    for p,h in c['controller_payload']['execution_sources'].items():
                        if p not in files:files[p]=sha(p)
                        assert files[p]==h,('EXECUTION_INPUT_CHANGED',p)
                p=source_path(stage,c);r=read(p);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                assert row is not None,('DB_MISSING',rid)
                for k in ('state_uid','eta_uid','controller_uid','seed_key'):
                    assert row[k]==c[k]==r[k],(rid,k)
                for k in ('success','deadlock','timeout','collision','numerical_failure'):
                    assert row[k]==r.get(k,0),(rid,k,row[k],r.get(k))
                assert not row['conflict_quarantined'],rid
                trace=Path(r['trace_path'])
                if str(trace)not in files:files[str(trace)]=sha(trace)
                assert files[str(trace)]==r['trace_sha256'],('TRACE_CHANGED',rid,trace)
                operational=r.get('operational_resume')
                if operational:
                    for name,expected in operational['sources'].items():
                        assert sha(name)==expected,('CHECKPOINT_SOURCE_CHANGED',name)
                        files[name]=expected
                    proof=operational['validation_receipt']
                    assert sha(proof)==operational['validation_receipt_sha256']
                    assert read(proof)['passed'];files[proof]=sha(proof)
                    for name,expected in read(proof)['implementation_sources'].items():
                        assert sha(name)==expected,('CHECKPOINT_VALIDATOR_SOURCE_CHANGED',name)
                        files[name]=expected
                    receipts=list(operational['saved_checkpoint_receipts'])
                    receipts.extend(str(p)for p in(HERE/'operational_checkpoints'/rid).glob('*.receipt.json'))
                    prior=operational.get('resumed_from')
                    if prior:
                        assert sha(prior['receipt'])==prior['receipt_sha256']
                        receipts.append(prior['receipt'])
                    for receipt in sorted(set(receipts)):
                        saved=read(receipt);assert saved['scientific_label']is None
                        assert saved['case_hash']==uid('checkpoint_case',c)
                        assert saved['sources']==operational['sources']
                        assert sha(saved['path'])==saved['sha256']
                        files[receipt]=sha(receipt);files[saved['path']]=saved['sha256'];checkpoint_receipts.add(receipt)
                if p.resolve().is_relative_to(HERE):pass
                else:files[str(p)]=sha(p)
                seen[rid]=dict(stage=stage,raw_path=str(p),raw_sha256=sha(p),trace_sha256=files[str(trace)]);count+=1
            counts[stage]=count
    unmerged=list((cache_writer.ROOT/'journals'/EXP).glob('*.jsonl'))+list((cache_writer.ROOT/'journals'/EXP).glob('*.jsonl.tmp'))
    assert not unmerged,unmerged
    write(HERE/'DB_POSTFLIGHT.json',dict(passed=True,utc=utc(),stage_requests=counts,unique_requests=len(seen),
        trace_files=sum(p.endswith('.npz')for p in files),all_identities_and_outcomes_equal=True,all_trace_hashes_equal=True,
        no_unmerged_experiment_journals=True,unknowns_retained=True))
    audit=read(HERE/'DB_POSTFLIGHT.json');audit.update(verified_operational_checkpoint_receipts=len(checkpoint_receipts),
        checkpoints_not_counted_as_rollout_results=True);write(HERE/'DB_POSTFLIGHT.json',audit)
    write(HERE/'VERIFIED_RESULT_PROVENANCE.json',seen)
    write(HERE/'REUSED_EVIDENCE_SHA256_END.json',files)


def integrity():
    start=read(HERE/'INTEGRITY_START.json');repos={}
    freeze=read(HERE/'MODEL_FROZEN.receipt.json')
    frozen_changed={p:dict(before=h,after=sha(p))for p,h in freeze['files'].items()if sha(p)!=h}
    for repo,old in start['repositories'].items():
        status=subprocess.check_output(['git','--no-optional-locks','status','--porcelain'],cwd=repo).decode()
        diff=subprocess.check_output(['git','--no-optional-locks','diff','--no-ext-diff','--binary'],cwd=repo)
        normalized='\n'.join(x for x in status.splitlines()if x!='?? diagnostics/cross_scene_correction_v1/')+'\n'
        repos[repo]=dict(status=status,status_sha256=hashlib.sha256(status.encode()).hexdigest(),
            diff_sha256=hashlib.sha256(diff).hexdigest(),normalized_status_equal=normalized==old['status'],
            diff_equal=hashlib.sha256(diff).hexdigest()==old['diff_sha256'])
    baseline=read(HERE/'EVIDENCE_SHA256_START.json')['files'];after={p:sha(p)for p in baseline}
    expected={}
    for line in (HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():
        row=json.loads(line);prior=expected.setdefault(row['path'],row['sha256']);assert prior==row['sha256'],('INPUT_CHANGED_DURING_RUN',row['path'])
    for receipt in ('BACKEND_BUILD.json','EXTENSION_BUILD.json'):
        value=read(HERE/receipt)
        for row in value.get('sources',value.get('files',[])):
            expected[row['source']]=row['source_sha256']
    expected[str(HERE/'native_resume/cbs_kernel_resume.so')]=read(HERE/'BACKEND_BUILD.json')['native_sha256']
    reused=read(HERE/'REUSED_EVIDENCE_SHA256_START.json')
    expected.update(reused['raw_files']);expected.update(reused['trace_files'])
    proof=read(HERE/'DETERMINISM_CHECK.json')
    expected[proof['historical_proof']]=proof['sha256']
    additional=HERE/'ADDITIONAL_HISTORICAL_EVIDENCE_SHA256_START.json'
    if additional.exists():expected.update(read(additional)['files'])
    imported=read(HERE/'IMPORTED_SOURCE_AUDIT.json');assert imported['passed']
    expected.update({row['path']:row['baseline_sha256']for row in imported['files']})
    strict_path=HERE/'CHECKPOINT_STRICT_BITWISE_AUDIT.json'
    if strict_path.exists():
        strict=read(strict_path);assert strict['passed']
        for row in strict['rows']:
            expected[row['reference']]=row['reference_sha256']
            expected[row['resumed']]=row['resumed_sha256']
        for name,key in(('REPLAY_CHECK.json','original_replay_proof_sha256'),
                         ('history.py','original_comparison_source_sha256'),
                         ('CHECKPOINT_REPLAY_CHECK.json','checkpoint_proof_sha256'),
                         ('CHECKPOINT_PLANNER_SCOPE_CHECK.json','planner_scope_proof_sha256')):
            expected[str(HERE/name)]=strict[key]
    changed={p:dict(before=h,after=sha(p))for p,h in {**baseline,**expected}.items()if sha(p)!=h}
    write(HERE/'EVIDENCE_SHA256_END.json',dict(files={p:sha(p)for p in {**baseline,**expected}}))
    ok=not changed and not frozen_changed and all(r['normalized_status_equal']and r['diff_equal']for r in repos.values())
    write(HERE/'INTEGRITY_END.json',dict(utc=utc(),repositories=repos,passed=ok))
    write(HERE/'INTEGRITY_COMPARE.json',dict(passed=ok,scope='User amendment: remove only this new experiment status entry',
        repositories=repos,input_file_count=len({**baseline,**expected}),changed_inputs=changed,
        frozen_files_unchanged=not frozen_changed,frozen_changed=frozen_changed))
    assert ok,('INTEGRITY_FAILED',changed,frozen_changed,repos)


if __name__=='__main__':
    from audit_metadata_uid_content import main as audit_metadata_uid_content
    audit_metadata_uid_content();database();integrity()
