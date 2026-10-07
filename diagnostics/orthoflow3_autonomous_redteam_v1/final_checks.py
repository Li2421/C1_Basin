"""Only narrow regression, artifact hashes and read-only cache reconciliation."""
from audit import ROOT,OUT,read,save,sha,canonical,conn
import importlib.util
import io
import unittest
from shared_rollout_db.src.planner import preflight

def main():
    refs=read(OUT/'H2_rollout_refs.json')
    requests=[{k:r[k] for k in ('state_uid','eta_uid','controller_uid')} | {
        'seed_keys':[canonical({'future_index':i}) for i in range(16)]} for r in refs]
    save('cache_requests.json',{'purpose':'read-only redteam frozen artifact semantic verification',
        'new_execution_authorized_by_this_manifest':False,'requests':requests})
    checked=preflight(OUT/'cache_requests.json');save('standard_cache_preflight.json',checked)
    with conn() as c:
        # Targeted persistence, not another global DB integrity sweep.
        expected=[r for x in refs for r in x['rollout_uids']]
        missing=[]
        for rid in expected:
            if c.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(rid,)).fetchone() is None:missing.append(rid)
    assert not missing
    save('DB_CACHE_REPORT.json',{'scientific_new_continuations':0,'new_training_labels':0,
        'standard_preflight':checked['summary'],'required_seed_records':len(expected),
        'required_seed_records_present':len(expected)-len(missing),
        'certified_reuse':16214,'numerically_uncertified_preserved':106,
        'missing_physical_records':len(missing),'retries_executed_here':0,
        'note':'Planner excludes numerical records and calls them genuinely_missing. All106 are present uncertified, not absent data. This audit resolves reporting bounds without re-executing exhausted retries.',
        'database_write_operations':0,'journal_merger_new_entries':0,
        'prior_general_integrity_evidence':'orthoflow3_data_hygiene_v2:75/75 labels and695/695 outcomes replay exactly; not repeated'})
    import test_evidence
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_evidence))
    basis=[]
    spec=importlib.util.spec_from_file_location('redteam_basis_regression',ROOT/'tests/test_basis_families.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in sorted(n for n in vars(module) if n.startswith('test_')):
        getattr(module,name)();basis.append(name)
    save('regression_results.json',{'evidence_tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'basis_regressions':basis,'basis_failures':0,'stdout':stream.getvalue(),
        'full_symmetry_suite_repeated':False,'all_passed':result.wasSuccessful()})
    before=read(OUT/'hypothesis_test_manifest.json')['frozen_files']
    verification={p:sha(ROOT/p)==h for p,h in before.items()}
    # Additional independently recorded current-control hashes from the concurrent
    # task are checked, but its mutable results are not scientific evidence here.
    extra=read(ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/runtime_artifacts.json')['files']
    extra_matches={p:sha(ROOT/p)==h for p,h in extra.items()}
    save('hash_verification.json',{'frozen_files':verification,'runtime_addendum':extra_matches,
        'all_match':all(verification.values()) and all(extra_matches.values()),
        'canonical_sources_changed_by_redteam':False})
    assert all(verification.values()) and all(extra_matches.values())
    print({'tests':result.testsRun+len(basis),'all_passed':result.wasSuccessful(),
        'frozen_file_checks':len(verification),'runtime_hash_checks':len(extra_matches),
        'new_rollouts':0,'DB_records_present':len(expected)})
if __name__=='__main__':main()
