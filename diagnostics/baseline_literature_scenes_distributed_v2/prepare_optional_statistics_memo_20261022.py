"""Reuse the production-verified exact bootstrap memo in optional postprocessing."""
from experiment import *
from registry import verify_receipt
import ast

READY = 'OPTIONAL_STATISTICS_EXACT_MEMO_READY_20261022.receipt.json'


def same_functions(a, b, names):
    first = {n.name:n for n in ast.parse(a).body if isinstance(n, ast.FunctionDef)}
    second = {n.name:n for n in ast.parse(b).body if isinstance(n, ast.FunctionDef)}
    for name in names:
        assert ast.dump(first[name], include_attributes=False) == ast.dump(second[name], include_attributes=False), name


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert not (HERE/READY).exists()
    for proof in ('MIGRATION_NUMBERS.receipt.json', 'BOOTSTRAP_EXACT_MEMO_READY_20261022.receipt.json',
                  'BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json', 'FINAL_DELIVERY_PROTOCOL_ORDER_20261022.receipt.json'):
        verify_receipt(proof)
    stats = 'statistics_bootstrap_memo_reader_20261022.py'
    original = (HERE/'statistics.py').read_text()
    cached = (HERE/stats).read_text()
    same_functions(original, cached, ['summarize', 'contrast', 'crossfit_oracle', 'controller_census'])
    oa = {n.name:n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    ca = {n.name:n for n in ast.parse(cached).body if isinstance(n, ast.FunctionDef)}
    ca['bootstrap_uncached'].name = 'bootstrap'
    assert ast.dump(oa['bootstrap'], include_attributes=False) == ast.dump(ca['bootstrap_uncached'], include_attributes=False)
    mappings = [
        ('optional_revision_diagnostics_causeaware_20261022.py', 'optional_revision_diagnostics_memo_20261022.py'),
        ('optional_results_causeaware_20261022.py', 'optional_results_memo_20261022.py'),
        ('optional_delivery_protocol_order_20261022.py', 'optional_delivery_statistics_memo_20261022.py'),
    ]
    copies = []
    for source, target in mappings:
        path = HERE/target
        assert not path.exists()
        text = (HERE/source).read_text()
        if source.startswith('optional_revision'):
            needle = 'from statistics import summarize,FIXED'
            assert text.count(needle) == 1
            text = text.replace(needle, 'from statistics_bootstrap_memo_reader_20261022 import summarize,FIXED')
            same_functions((HERE/source).read_text(), text, ['baseline_failure_modes', 'report'])
        elif source.startswith('optional_results'):
            needle = 'from statistics import summarize,bootstrap,contrast,FIXED'
            assert text.count(needle) == 1
            text = text.replace(needle, 'from statistics_bootstrap_memo_reader_20261022 import summarize,bootstrap,contrast,FIXED')
            text = text.replace('from optional_revision_diagnostics_causeaware_20261022 import report', 'from optional_revision_diagnostics_memo_20261022 import report')
            text = text.replace('optional_revision_diagnostics_causeaware_20261022.py', 'optional_revision_diagnostics_memo_20261022.py')
            text = text.replace('optional_results_causeaware_20261022.py', target)
            anchor = 'def main():\n'
            assert text.count(anchor) == 1
            text = text.replace(anchor, anchor+"    from registry import verify_receipt\n    verify_receipt('"+READY+"')\n")
            anchor = "'optional_results_memo_20261022.py')],main_numbers_unchanged=True)"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "'optional_results_memo_20261022.py','"+stats+"','bootstrap_exact_memo_20261022.py','"+READY+"','OPTIONAL_BOOTSTRAP_CACHE_STATS_20261022.json')],main_numbers_unchanged=True)")
            anchor = "    seal('OPTIONAL_NUMBERS.receipt.json'"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "    from bootstrap_exact_memo_20261022 import snapshot\n    write(HERE/'OPTIONAL_BOOTSTRAP_CACHE_STATS_20261022.json',dict(utc=utc(),cache=snapshot(),iterations_per_unique_input=5000,mathematical_kernel_unchanged=True))\n"+anchor)
            same_functions((HERE/source).read_text(), text, ['collect', 'paired_stage_comparison'])
        else:
            needle = "action('optional_numbers','optional_results_causeaware_20261022.py')"
            assert text.count(needle) == 1
            text = text.replace(needle, "action('optional_numbers','optional_results_memo_20261022.py')")
            anchor = "    verify_receipt('FINAL_DELIVERY_PROTOCOL_ORDER_20261022.receipt.json')"
            assert text.count(anchor) == 1
            text = text.replace(anchor, anchor+"\n    verify_receipt('"+READY+"')")
            same_functions((HERE/source).read_text(), text, ['submit', 'gap', 'run', 'prepare'])
        ast.parse(text)
        path.write_text(text)
        copies.append(dict(source=source, source_sha256=sha(HERE/source), target=target, target_sha256=sha(path)))
    source = HERE/'activate_delivery_protocol_order_20261022.py'
    target = HERE/'activate_optional_statistics_memo_20261022.py'
    assert not target.exists()
    text = source.read_text().replace('FINAL_DELIVERY_PROTOCOL_ORDER_20261022.receipt.json', READY)
    text = text.replace('FINAL_DELIVERY_PROTOCOL_ORDER_DEPLOYMENT_20261022', 'OPTIONAL_STATISTICS_EXACT_MEMO_DEPLOYMENT_20261022')
    text = text.replace('optional_delivery_protocol_order_20261022.py', 'optional_delivery_statistics_memo_20261022.py')
    text = text.replace('protocol_order_receipt=ready', 'optional_statistics_memo_receipt=ready')
    ast.parse(text);target.write_text(text)
    write(HERE/'OPTIONAL_STATISTICS_EXACT_MEMO_HISTORY_20261022.json', dict(
        utc=utc(), existing_evidence=['BOOTSTRAP_EXACT_MEMO_READY_20261022.receipt.json', 'BOOTSTRAP_EXACT_MEMO_TEST_20261022.json', 'MIGRATION_NUMBERS.receipt.json'],
        compatibility='Existing memo reader summarize/contrast/crossfit/census AST identical to original;uncached5000-draw kernel AST identical. Original full5000 bitwise,capacity/FP/callback gates reused;main production44 statistics tasks complete.',
        actual_gap='Optional results and taxonomy still imported original uncached statistics;static imports now use the already sealed production reader',
        copies=copies, new_rollouts=0, new_training=0, new_bootstrap_validation_runs=0,
        scope_changes=False, bootstrap_iterations=5000, per_process_payload_limit_bytes=64*1024*1024,
        performance_claim='Only repeated identical matrices are memo hits;new unique inputs still execute original5000 draws. No whole-pipeline speedup factor assumed.'))
    seal(READY, [Path(__file__), *[HERE/t for _,t in mappings], target,
                 HERE/'OPTIONAL_STATISTICS_EXACT_MEMO_HISTORY_20261022.json', HERE/stats,
                 HERE/'bootstrap_exact_memo_20261022.py', HERE/'BOOTSTRAP_EXACT_MEMO_READY_20261022.receipt.json',
                 HERE/'BOOTSTRAP_EXACT_MEMO_CANDIDATE_20261022.receipt.json', HERE/'FINAL_DELIVERY_PROTOCOL_ORDER_20261022.receipt.json'],
         science_unchanged=True, original5000_draws_unchanged=True, existing_bitwise_gates_reused=True,
         original_mathematical_functions_AST_identical=True, new_rollouts=0, new_training=0)
    print('Prepared optional exact statistics cache;existing full5000-draw gates reused', flush=True)


if __name__ == '__main__':
    main()
