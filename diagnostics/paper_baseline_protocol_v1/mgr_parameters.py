"""Expose the author's existing C_R parameter; default numerical value unchanged."""
import guard
from guard import HERE
import ast
import hashlib
import json


def expose_method(source):
    old = "obj=base.MGRNative(env,seed,dict(tau=options.get('tau',.5)));obj.v2_context=context"
    new = "obj=base.MGRNative(env,seed,dict(tau=options.get('tau',.5),circle_radius=options.get('circle_radius',.3)));obj.v2_context=context"
    assert source.count(old) == 1
    return source.replace(old, new)


def expose_base(source):
    anchor = '        from simulator import Simulator\n'
    addition = (anchor + '        import simulator as _mgr_simulator\n'
                "        _mgr_simulator.C_R=float(options.get('circle_radius',_mgr_simulator.C_R))\n")
    if addition in source:
        return source
    assert source.count(anchor) == 1
    return source.replace(anchor, addition)


def apply_base():
    assert (HERE / 'PRIOR_TASK_DELIVERED.json').exists()
    path = HERE / 'ports/paper_adapters.py'
    previous = path.read_text()
    modified = expose_base(previous)
    if modified != previous:
        path.write_text(modified)
    output = dict(target=str(path.relative_to(HERE)), before_sha256=hashlib.sha256(previous.encode()).hexdigest(),
                  after_sha256=hashlib.sha256(modified.encode()).hexdigest(),
                  author_parameter='C_R', adapter_name='circle_radius', author_default=.3,
                  provenance=['inputs/diagnostics/paper_baselines_20261028/vendor/mgr/params.py:8',
                              'inputs/diagnostics/paper_baselines_20261028/vendor/mgr/simulator.py:36-37'],
                  reason='Existing geometric circle-size parameter, alongside prediction horizon and orbit gain; expose for the protocol parameter menu without a new coordination mechanism.',
                  author_code_modified=False, solver_settings_modified=False, new_rollouts=0)
    (HERE / 'MGR_PARAMETER_EXPOSURE.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


def check_defaults():
    """Execute the literal author constructor without Qt/controller imports."""
    root = HERE / 'inputs/diagnostics/paper_baselines_20261028/vendor/mgr'
    tree = ast.parse((root / 'simulator.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Simulator')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    shell = ast.ClassDef(name='ConstructorCheck', bases=[], keywords=[], body=[init], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    params = ast.parse((root / 'params.py').read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in params.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in ('C_R', 'K')}
    assert values == dict(C_R=.3, K=.1)
    namespace = dict(values)
    exec(compile(module, str(root / 'simulator.py'), 'exec'), namespace)
    constructor = namespace['ConstructorCheck']
    original = constructor().__dict__
    checks = []
    for value in (.3, .15, .6):
        namespace['C_R'] = float(value)
        actual = constructor().__dict__
        assert actual['c_r'] == actual['max_radius'] == value
        expected = dict(original, c_r=value, max_radius=value)
        assert actual == expected
        if value == .3:
            assert actual == original and actual['c_r'].hex() == original['c_r'].hex()
        checks.append(dict(circle_radius=value, exactly_author_constructor_values=True))
    result = dict(passed=True, checks=checks, author_default_unchanged=True, new_rollouts=0,
                  limitation='Constructor/source-interface check only. Default trajectory equivalence and nondefault performance still require actual training-scene runs.')
    (HERE / 'MGR_PARAMETER_UNIT_CHECKS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Author C_R constructor checks passed:3; no rollout')


if __name__ == '__main__':
    check_defaults()
