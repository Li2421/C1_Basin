"""Reject the internal fixed-B workflow removed by the user on2026-10-30."""


def assert_request_allowed(request):
    payload=request.get('controller_payload',{})
    if payload.get('engine')!='original_admission_dictionary':
        return
    purpose=request.get('purpose','')
    if purpose=='symmetric_two_round_tuning' or purpose in {
        'fixed_global_tuned','fixed_type_tuned','fixed_first_point',
    }:
        raise RuntimeError('USER_REMOVED_FIXED_B_WORKFLOW: this saved internal-B global/type-specific/first-point request must not execute. Use per-case Oracle or the actual generalizing pipeline.')
