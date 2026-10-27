"""Read-only final checks after all requested numerical execution and delivery."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    complete = read('COMPLETE.json')
    assert complete['status'] == 'COMPLETE'
    for name, expected in complete['reports'].items():
        assert sha(HERE / name) == expected, name
    db = read('DB_POSTFLIGHT.json')
    integrity = read('INTEGRITY_COMPARE.json')
    reconstruction = read('REQUEST_RECONSTRUCTION_CHECK.json')
    assert db['passed'] and integrity['passed'] and reconstruction['passed']
    assert reconstruction['cases'] == 10
    assert all(c['canonical_json_bytes_equal'] and c['uid_equal'] for c in reconstruction['checks'])
    assert complete['retraining_after_resume'] == 0
    resume = read('RESUME_START.json')
    for path, expected in resume['frozen_files'].items():
        assert sha(path) == expected, path
    fix = read('REQUEST_FIX_RESUME.receipt.json')
    assert sha(HERE / 'STOP.receipt.json') == fix['prior_stop_sha256']
    for path, expected in fix['active_execution_files'].items():
        assert sha(path) == expected, path
    events = [json.loads(line) for line in (HERE / 'SCHEDULER_EVENTS.jsonl').read_text().splitlines()]
    post_resume_training = [e for e in events if e['utc'] >= resume['utc']
                            and str(e.get('stage', '')).startswith('train')]
    assert not post_resume_training
    expected = {c['rollout_uid'] for stage in ('pilot', 'main')
                for c in read(f'{stage}_CASES.json')}
    assert len(expected) == db['unique_requests'] == 1773
    missing = read('main_MISSING_CASES.json')
    assert len(missing) == 1729
    assert all((HERE / 'main/raw' / (c['rollout_uid'] + '.json')).exists()
               and (HERE / 'main/journal_receipts' / (c['rollout_uid'] + '.json')).exists()
               for c in missing)
    jobs = subprocess.check_output(['squeue', '-h', '-u', 'zhihan', '-o', '%i %j %T'], text=True)
    assert not any('xpl_main' in line or 'xpl_tail' in line for line in jobs.splitlines()), jobs
    assert not (HERE / 'TAIL_FATAL.json').exists()
    if (HERE / 'TAIL_COMPLETION_CHECK.json').exists():
        assert read('TAIL_COMPLETION_CHECK.json')['passed']
    if (HERE / 'TAIL_CHECKPOINT_HANDOFF.receipt.json').exists():
        handoff = read('TAIL_CHECKPOINT_HANDOFF_CHECK.json')
        assert handoff['passed'] and handoff['no_prefix_reexecuted']
    for name in ('METHOD_COMPARISON.md', 'LOFO_METHOD_COMPARISON.md', 'FULL18_METHOD_COMPARISON.md'):
        rows = [line for line in (HERE / name).read_text().splitlines()
                if line.startswith('| ') and '（' in line and '） |' in line]
        count = sum('（lit_' in line or '（xsc_' in line for line in rows)
        assert count == {'METHOD_COMPARISON.md': 11, 'LOFO_METHOD_COMPARISON.md': 8,
                         'FULL18_METHOD_COMPARISON.md': 18}[name], (name, count)
    proof = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True,
                 reports_verified=len(complete['reports']), unique_new_rollouts=len(expected),
                 main_missing_requests_completed=len(missing),
                 original_frozen_files_verified=len(resume['frozen_files']),
                 reconstruction_checks=10, training_events_after_resume=0,
                 original_failure_stop_preserved=True, no_experiment_workers_running=True,
                 integrity_and_database_audits_passed=True,
                 complete_sha256=sha(HERE / 'COMPLETE.json'),
                 report_sha256=sha(HERE / 'REPORT.md'))
    (HERE / 'FINAL_DELIVERY_VERIFY.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == '__main__':
    main()
