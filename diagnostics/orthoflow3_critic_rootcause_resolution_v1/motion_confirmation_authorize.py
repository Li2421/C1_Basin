"""Pre-submission audit for the two preregistered 2048-continuation batches."""
import argparse
from .motion_confirmation_eval import SOURCE, ROOT, folder, read, sha, write
from shared_rollout_db.src.rollout_db import connect, eta_identity, canonical, uid


def main(replica):
    out = folder(replica)
    pre = read(out/'cache_preflight.json')
    execution = read(out/'execution_preflight_held.json')
    audit = read(out/'identity_preflight_audit.json')
    frozen = read(out/'prediction_freeze.json')
    protocol = read(out/'protocol.json')
    profile = protocol['profiles'][0]
    assert audit['identity_checks_passed'] and audit['actual_semantic_incompatibilities'] == 0
    assert pre['summary']['ambiguous'] == 0 and pre['summary']['total_requested'] == 2048
    assert frozen['models_frozen_sha256'] == protocol['models_frozen_sha256'] == sha(SOURCE/'models_frozen.json')
    assert frozen['predictions_sha256'] == sha(out/'frozen_predictions.npz')
    assert frozen['target_labels_read'] is False
    assert sha(profile['path']) == profile['sha256']
    source_profile = read(SOURCE/'protocol.json')['profiles'][0]
    with connect(True) as db:
        src = db.execute('SELECT * FROM controller_config WHERE controller_uid=?', (source_profile['controller_uid'],)).fetchone()
        dst = db.execute('SELECT * FROM controller_config WHERE controller_uid=?', (profile['controller_uid'],)).fetchone()
        import json
        expected = json.loads(src['config_json'])
        expected['flow_checkpoint_sha256'] = profile['sha256']
        assert canonical(expected) == canonical(json.loads(dst['config_json']))
        assert uid('ctl', expected) == profile['controller_uid']
        for field in ('scenario_uid','orthoflow3_sha256','safety_config_hash','horizon','dt',
                      'success_semantics_version','conditioning_version','rng_semantics_version'):
            assert dst[field] == src[field], field
    for pair in read(out/'pairs.json'):
        assert eta_identity(pair['eta'])[0] == pair['eta_uid']
    seeds = {canonical({'future_index':k}) for k in range(16)}
    for request in read(out/'planned_rollouts.json')['requests']:
        assert request['controller_uid'] == profile['controller_uid']
        assert len(request['seed_keys']) == 16 and set(request['seed_keys']) == seeds
    missing = execution['summary']['genuinely_missing']
    assert 0 <= missing <= 2048
    reported = dict(controller_seed=protocol['target_controller_seed'], requested=2048,
        exact_reuse=pre['summary']['exact_reusable'], partial_reuse=pre['summary']['partial_reusable'],
        aggregate_reuse=pre['summary']['aggregate_reusable'], truly_missing=missing,
        ambiguous=0, semantic_incompatibilities=0,
        legacy_empty_cache_INCOMPATIBLE=pre['summary']['incompatible'])
    print(reported, flush=True)
    marker = dict(preflight_sha256=sha(out/'cache_preflight.json'),
        planned_sha256=sha(out/'planned_rollouts.json'),
        prediction_freeze_sha256=sha(out/'prediction_freeze.json'),
        models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
        reported_before_submission=True, new_continuation_limit=missing,
        semantics_audit='Only future frozen Flow SHA differs from the compatible source controller; no safety/success/RNG/basis changes',
        preflight_summary=reported, authorization='Existing user authorization; preregistered two-batch confirmation',
        generator_changed=False, code_sha256=sha(__file__))
    path = out/'execution_authorization.json'
    if path.exists():
        assert read(path) == marker
    else:
        write(path, marker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate', type=int, required=True)
    args = parser.parse_args()
    main(args.replicate)
