"""Source-supported adapter repair; author implementations stay read-only."""
import guard
from guard import HERE
import ast
import hashlib
import json

BASE = HERE / 'inputs/diagnostics/scene_types_oracle_coverage_v3/paper_methods'


def repaired_source(source):
    tree = ast.parse(source)
    candidates = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install_status_checks']
    assert len(candidates) == 1
    node = candidates[0]
    original = ast.get_source_segment(source, node)
    assert 'result.get(\'status\')!=\'optimal\'' in original
    assert "get('ExitFlag')not in(None,0)" in original
    replacement = '''def install_status_checks(context):
    """Leave solver returns to the unchanged author call sites.

    SolverMonitor still records nonoptimal returns. Native exceptions still
    propagate to the per-agent error wrapper. No solver option is changed.
    """
    return None'''
    assert source.count(original) == 1
    return source.replace(original, replacement)


def apply():
    assert (HERE / 'PRIOR_TASK_DELIVERED.json').exists(), 'Port application is queued behind previous-study delivery.'
    assert not (HERE / 'PARAMETERS_FROZEN.json').exists(), 'No port edits after test parameter freeze.'
    from execution_policy import active_methods
    from addendum_gates import native_methods
    active = set(active_methods()) & set(native_methods())
    if 'mgr_author_unicycle_tracking' in active:
        from mgr_parameters import apply_base
        apply_base()
    changes = []
    skipped = []
    for source in sorted(BASE.glob('*/adapter.py')):
        if source.parent.name not in active:
            skipped.append(dict(method=source.parent.name, reason='withdrawn_or_native_reproduction_failed'))
            continue
        target = HERE / 'paper_methods' / source.parent.name / source.name
        value = repaired_source(source.read_text())
        if source.parent.name == 'mgr_author_unicycle_tracking':
            from mgr_parameters import expose_method
            value = expose_method(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_text() == value
        else:
            target.write_text(value)
        changes.append(dict(method=source.parent.name, source=str(source.relative_to(HERE)),
                            target=str(target.relative_to(HERE)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            change='Remove adapter-added status-code rejection; preserve author return-value handling.',
                            direct_effect=source.parent.name in ('dac_author_si', 'mgr_author_unicycle_tracking', 'chen_impc_ob_si_tracking')))
    assert len(changes) == len(active)
    (HERE / 'PORT_REPAIR_APPLIED.json').write_text(json.dumps(dict(changes=changes, skipped=skipped, new_rollouts=0,
        solver_settings_changed=False, author_source_changed=False, unit_checks='PORT_RETURN_UNIT_CHECKS.json'), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    apply()
