"""Regression for the adapter-added failure mode; zero simulation episodes."""
import guard
from guard import HERE
import ast
import json
import sys
import types
from repair_ports import BASE, repaired_source


def function(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'install_status_checks']
    namespace = dict(MethodFailure=RuntimeError)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<isolated_status_wrapper>', 'exec'), namespace)
    return namespace['install_status_checks']


def fake_modules(result, raises=False):
    def call(*args, **kwargs):
        if raises:
            raise ValueError('native solver error')
        return result
    cvxopt = types.ModuleType('cvxopt')
    solvers = types.ModuleType('cvxopt.solvers')
    solvers.qp = solvers.coneqp = call
    cvxopt.solvers = solvers
    swift = types.ModuleType('qpSWIFT')
    swift.run = call
    return {'cvxopt': cvxopt, 'cvxopt.solvers': solvers, 'qpSWIFT': swift}


def main():
    source = (BASE / 'dac_author_si/adapter.py').read_text()
    repaired = repaired_source(source)
    old, new = function(source), function(repaired)
    checks = []
    names = ('cvxopt', 'cvxopt.solvers', 'qpSWIFT')
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for solver in ('qp', 'coneqp', 'swift'):
            result = dict(status='unknown', x=None, basicInfo=dict(ExitFlag=2), sol=None)
            modules = fake_modules(result)
            sys.modules.update(modules)
            old(None)
            try:
                (modules['qpSWIFT'].run if solver == 'swift' else getattr(modules['cvxopt.solvers'], solver))()
                rejected = False
            except RuntimeError:
                rejected = True
            assert rejected, solver
            modules = fake_modules(result)
            sys.modules.update(modules)
            new(None)
            returned = (modules['qpSWIFT'].run if solver == 'swift' else getattr(modules['cvxopt.solvers'], solver))()
            assert returned is result
            checks.append(dict(solver=solver, old_adapter_rejected=True, repaired_returns_original_object=True))
            modules = fake_modules(None, raises=True)
            sys.modules.update(modules)
            new(None)
            try:
                (modules['qpSWIFT'].run if solver == 'swift' else getattr(modules['cvxopt.solvers'], solver))()
                propagated = False
            except ValueError as exc:
                propagated = str(exc) == 'native solver error'
            assert propagated
            checks.append(dict(solver=solver, native_exception_propagated=True))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    # The repair replaces exactly one function; all other syntax remains literal.
    before = ast.parse(source); after = ast.parse(repaired)
    before.body = [n for n in before.body if getattr(n, 'name', None) != 'install_status_checks']
    after.body = [n for n in after.body if getattr(n, 'name', None) != 'install_status_checks']
    assert ast.dump(before) == ast.dump(after)
    output = dict(passed=True, checks=checks, unchanged_other_ast=True, new_rollouts=0,
                  limitation='Tests adapter return semantics only. Native author decisions, validity of nonoptimal candidates and task success require actual training-scene runs.')
    (HERE / 'PORT_RETURN_UNIT_CHECKS.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('Adapter return-handling checks passed:', len(checks), '; no rollout or optimizer-setting change')


if __name__ == '__main__':
    main()
