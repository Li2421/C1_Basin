"""Seal completed numbers and verify their immutable evidence before plotting.

Only this experiment's outputs are enumerated. Referenced protected inputs are
read solely when an existing receipt already names them. No rollout is run.
"""
from experiment import *
import subprocess

RECEIPTS = [
    'MIGRATION_RESULTS', 'PAPER_REPRODUCTION', 'SAFETY_DIAGNOSTICS',
    'BACKGROUND_RESULTS', 'PILOT_RESULTS', 'REFERENCE_ENDPOINTS',
    'SWITCHING_DIAGNOSTICS', 'ARRIVAL_RELEASE_DIAGNOSTICS',
    'TRAINING_MENU_REFERENCE', 'USER_DIRECTED_CLOSURE', 'HANDOFF_FINALIZATION', 'DB_POSTFLIGHT', 'INTEGRITY_END',
]
ARTIFACTS = [
    'HISTORY_CHECK.md', 'PARAMETERS.md', 'PREREGISTRATION.md',
    'PAPER_TARGETS.csv', 'PAPER_REPRODUCTION.md', 'PAPER_REPRODUCTION.csv',
    'SCENES.md', 'CONTROLLER_PORTING.md', 'SAFETY_BASELINE.csv',
    'FAILURE_DECOMPOSITION.md', 'GEOMETRY_CENSUS.md',
    'MIGRATION_RESULTS.json', 'results.json', 'MIGRATION_RESULTS.csv',
    'AXIS_TABLE.md', 'AXIS_TABLE.csv', 'ITERATION_LOG.md',
    'INTEGRITY_START.json', 'INTEGRITY_END.json', 'HANDOFF.md',
]
FORBIDDEN = ('decentralized_', 'teacher_student_', 'multiscene_', 'baseline1_')


def allowed(path):
    p = Path(path).resolve()
    try:
        rel = p.relative_to(ROOT / 'diagnostics')
    except ValueError:
        return p
    assert not rel.parts[0].startswith(FORBIDDEN), ('forbidden_reference', str(p))
    return p


def receipts():
    # External repositories and dependency trees are covered by their own
    # source audits, and never traversed as experiment receipts here.
    pruned = {'external', '.venv-ext', 'cache', 'tmp', 'cuda_cache',
              'mplconfig', 'raw', 'traces', 'logs', 'claims', 'groups'}
    for folder, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in pruned and not d.startswith('.')]
        for name in files:
            if name.endswith('.receipt.json'):
                yield Path(folder) / name


def idle():
    jobs = subprocess.check_output([
        'squeue', '-u', os.environ['USER'], '-h', '-r', '-o', '%i|%j'
    ]).decode().splitlines()
    assert not any('baseline2-' in j for j in jobs), jobs


def numeric():
    idle()
    assert not (HERE / 'FINAL_NUMERIC_RESULTS.receipt.json').exists()
    for name in ARTIFACTS:
        assert (HERE / name).is_file(), name
    assert (HERE / 'external_patches').is_dir()
    for name in RECEIPTS:
        assert (HERE / (name + '.receipt.json')).is_file(), name
    assert read(HERE / 'DB_POSTFLIGHT.json')['all_active_stage_identities_indexed']
    compared = read(HERE / 'INTEGRITY_COMPARISON.json')
    assert compared['covered_sources_pass'], compared['differences']
    # A disclosed missing opening whole-file snapshot must never be relabelled
    # as a strict full-coverage pass by the final packaging step.
    assert compared['strict_full_coverage_pass'] is False
    checked = {}
    receipt_index = []
    for p in sorted(receipts()):
        r = read(p)
        for name, expected in r.get('files', {}).items():
            q = Path(name)
            q = allowed(q if q.is_absolute() else p.parent / q)
            if str(q) not in checked:
                checked[str(q)] = sha(q)
            assert checked[str(q)] == expected, (str(p), str(q))
        receipt_index.append(dict(path=str(p), sha256=sha(p),
                                  referenced_files=len(r.get('files', {}))))
    trace_hashes = {}
    raw_count = 0
    stages = {}
    identities = {}
    evidence_index = HERE / 'RAW_EVIDENCE_INDEX.jsonl'
    with evidence_index.open('w') as out:
        for folder in sorted(HERE.iterdir()):
            rawdir = folder / 'raw'
            if not folder.is_dir() or not rawdir.is_dir():
                continue
            n = 0
            for p in sorted(rawdir.glob('*.json')):
                r = read(p)
                assert p.stem == r['rollout_uid'], str(p)
                trace = allowed(r['trace_path'])
                assert trace.is_relative_to(HERE), str(trace)
                if str(trace) not in trace_hashes:
                    trace_hashes[str(trace)] = sha(trace)
                assert trace_hashes[str(trace)] == r['trace_sha256'], str(p)
                fingerprint = [r.get(k) for k in
                               ('state_uid', 'eta_uid', 'controller_uid', 'seed_key',
                                'success', 'collision', 'numerical_failure', 'trace_sha256')]
                rid = r['rollout_uid']
                if rid in identities:
                    assert identities[rid] == fingerprint, ('duplicate_conflict', str(p))
                else:
                    identities[rid] = fingerprint
                out.write(canonical(dict(path=str(p), sha256=sha(p),
                                         rollout_uid=rid, trace=str(trace),
                                         trace_sha256=r['trace_sha256'])) + '\n')
                n += 1
                raw_count += 1
            stages[folder.name] = n
            print(folder.name, n, 'trace hashes verified', flush=True)
    write(HERE / 'RECEIPT_VALIDATION.json', receipt_index)
    code = {str(p): sha(p) for pattern in ('*.py', '*.sbatch', '*.cpp', '*.hpp', '*.so')
            for p in HERE.glob(pattern)}
    write(HERE / 'FINAL_CODE_MANIFEST.json', code)
    write(HERE / 'EVIDENCE_VALIDATION.json', dict(
        utc=utc(), raw_records=raw_count, unique_rollout_identities=len(identities),
        unique_trace_files=len(trace_hashes), stages=stages,
        all_receipt_hashes_match=True, all_trace_hashes_match=True,
        strict_full_opening_audit_coverage=False,
        obsolete_zero_step_branch_preserved=True, new_physical_steps=0))
    files = {HERE / name for name in ARTIFACTS}
    files.update(Path(p) for p in code)
    files.update(Path(p) for p in checked if Path(p).is_relative_to(HERE))
    files.update(Path(r['path']) for r in receipt_index)
    files.update(HERE / name for name in (
        'RAW_EVIDENCE_INDEX.jsonl', 'RECEIPT_VALIDATION.json',
        'FINAL_CODE_MANIFEST.json', 'EVIDENCE_VALIDATION.json'))
    seal('FINAL_NUMERIC_RESULTS.receipt.json', sorted(files),
         evidence='DEVELOPMENT', controllers_and_simulation_not_called=True,
         raw_records=raw_count, unique_trace_files=len(trace_hashes))


def delivery():
    idle()
    receipt = read(HERE / 'FINAL_NUMERIC_RESULTS.receipt.json')
    for name, expected in receipt['files'].items():
        assert sha(allowed(name)) == expected, name
    for name in [*ARTIFACTS, 'REPORT.md', 'GAPS.md', 'STATUS.md',
                 'DELIVERY_SUMMARY.json']:
        assert (HERE / name).is_file(), name
    result = read(HERE / 'DELIVERY_SUMMARY.json')
    assert result['all_authorized_computable_work_processed']
    assert result['pending_research_jobs'] == 0
    seal('DELIVERY_VALIDATION.receipt.json',
         [HERE / name for name in ('REPORT.md', 'GAPS.md', 'STATUS.md',
                                   'DELIVERY_SUMMARY.json',
                                   'FINAL_NUMERIC_RESULTS.receipt.json',
                                   'validate_delivery.py')],
         figures_optional=True, documented_gaps_preserved=True)


if __name__ == '__main__':
    {'numeric': numeric, 'delivery': delivery}[sys.argv[1]]()
