"""Audit first landed PA-memo cases against existing frozen requests/journals."""
from experiment import *
from registry import verify_receipt
from atomic_write import write


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    prefix='PA_SPATIAL_MASK_FIRST5_PRODUCTION_20261022';stage='learning_labels_revised_20261021_lit_circle_five'
    if (HERE/(prefix+'.receipt.json')).exists():verify_receipt(prefix+'.receipt.json');return
    for proof in ('PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json','PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json',stage+'_PREFLIGHT.receipt.json'):verify_receipt(proof)
    planned={c['rollout_uid']:c for c in read(HERE/f'{stage}_MISSING_CASES.json')};rows=[]
    for path in (HERE/stage/'raw').glob('*.json'):
        if path.stat().st_mtime<1792641400:continue
        raw=read(path)
        if raw.get('execution_backend',{}).get('actual')=='runtime_correction_pa_memo_candidate_spatial_mask_20261022.py':rows.append((raw['timestamp'],path,raw))
    rows.sort(key=lambda v:v[0]);assert len(rows)>=5,len(rows);out=[];proof_paths=[]
    for timestamp,path,raw in rows[:5]:
        rid=raw['rollout_uid'];assert raw['case']==planned[rid]
        assert sha(raw['trace_path'])==raw['trace_sha256']
        assert raw['spatial_mask_cache_equivalence_receipt_sha256']==sha(HERE/'PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json')
        marker=HERE/stage/'journal_receipts'/f'{rid}.json';receipt=read(marker);assert receipt['raw_sha256']==sha(path)
        journal=Path(receipt['journal']);journal=journal if journal.exists() else journal.with_suffix('.merged');assert journal.exists()
        journal_rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        assert any(row.get('rollout_uid')==rid for row in journal_rows),(rid,journal_rows[:1])
        out.append(dict(rollout_uid=rid,raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],timestamp=timestamp,outcome=raw['outcome'],seconds=raw['elapsed_seconds'],cache=raw['spatial_mask_cache'],journal_receipt=str(marker),journal=str(journal),journal_sha256=sha(journal)))
        proof_paths.extend([path,marker])
    write(HERE/(prefix+'.json'),dict(utc=utc(),passed=True,rows=out,all_exact_requests_in_sealed_genuinely_missing_manifest=True,all_official_journals_verified=True,no_scientific_uid_change=True,not_a_comparative_success_or_speed_estimate=True))
    seal(prefix+'.receipt.json',[Path(__file__),HERE/(prefix+'.json'),HERE/'PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json',HERE/'PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json',HERE/f'{stage}_PREFLIGHT.receipt.json',*proof_paths],passed=True)
    print('PASS:first5 PA spatial-mask production requests,trace hashes and official journals verified',flush=True)


if __name__=='__main__':main()
