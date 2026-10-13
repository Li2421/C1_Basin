"""Correct cache ownership for optional legacy H20 diagnostics.

H20 is NOT an input to the new static-query critic. This adapter exists to
prevent reuse of the verified legacy cache defect during diagnostic work.
The controller program must stay frozen for the lifetime of its runtime.
"""
import types
import weakref

_FUNCTIONS = weakref.WeakKeyDictionary()


def h20_context(runtime, state, eta):
    from field_pipeline_v1.learn import h20_context as legacy
    if runtime not in _FUNCTIONS:
        namespace = dict(legacy.__globals__)
        namespace['NOMINAL_CACHE'] = {}
        function = types.FunctionType(legacy.__code__, namespace,
            legacy.__name__, legacy.__defaults__, legacy.__closure__)
        function.__kwdefaults__ = legacy.__kwdefaults__
        _FUNCTIONS[runtime] = function
    return _FUNCTIONS[runtime](runtime, state, eta)


def verify_real_case():
    import sys
    import numpy as np
    import jax
    from .study import HERE, FIELD, read, write
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1.ring_t0_source import runtime
    from field_pipeline_v1.held import HeldRuntime
    case=read(HERE/'legacy_context_cache_audit.json')
    state=next(s for s in read(HERE/'protocol.json')['states'] if s['uid']==case['case_state_uid'])
    expected=np.load(HERE/'legacy_context_probe_vectors.npz')
    base=runtime();held=HeldRuntime('ring_exchange');jax.config.update('jax_enable_x64',False)
    a=h20_context(base,state,np.asarray(case['case_eta']))
    b=h20_context(held,state,np.asarray(case['case_eta']))
    np.testing.assert_array_equal(a,expected['base'])
    np.testing.assert_array_equal(b,expected['held_correct'])
    assert np.max(np.abs(b-expected['held_polluted']))>1e-5
    write(HERE/'controller_scoped_cache_check.json',dict(
        same_state_eta_two_controllers=True,base_matches_original=True,
        held_matches_fresh_controller_specific_context=True,
        cache_clear_between_controllers=False,old_bug_reproduced_separately=True,
        new_Q_labels=0,physical_steps_for_feature_regression=160))


if __name__=='__main__':verify_real_case()
