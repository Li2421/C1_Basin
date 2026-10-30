"""User-directed withdrawal and reproduction checks, independent of scientific success labels."""
import guard
from guard import HERE
import json


def assert_method_enabled(method):
    path = HERE / 'METHOD_EXECUTION_POLICY.json'
    if path.exists():
        policy = json.loads(path.read_text())
        if method in policy.get('native_excluded_methods',[]):
            raise RuntimeError('NATIVE_REPRODUCTION_FAILED: '+method)
        if method in policy.get('withdrawn_methods',[]):
            raise RuntimeError('USER_WITHDRAWN_METHOD: ' + method + '; no reproducible author code/parameters')
        if method in policy.get('paused_methods',[]):
            raise RuntimeError('USER_PAUSED_METHOD: ' + method + '; waiting for author parameters')
        if method in policy['active_external_methods'] and policy.get('native_reproduction_required'):
            from addendum_gates import native_methods
            assert method in native_methods(), 'Method did not pass native author-scene reproduction: '+method


def active_methods():
    methods=json.loads((HERE / 'METHOD_EXECUTION_POLICY.json').read_text())['active_external_methods']
    path=HERE/'NATIVE_REPRODUCTION_RESULTS.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):
        results=json.loads(path.read_text())['methods']
        methods=[m for m in methods if results[m]['reproducible']]
    return methods
