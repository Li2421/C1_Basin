"""Certified semantic aliases, kept separate from exact official cache hits."""
from core import *
from index_ours_history import signature

_INPUTS=None


def inputs():
    global _INPUTS
    certificate_path=HERE/'ENV_REUSE_PHASE_COMPLETE.json'
    if not certificate_path.exists():return None
    certificate=read(certificate_path)
    if not certificate.get('complete')or not certificate.get('historical_B_reuse_eligible'):return None
    if sha(HERE/'EXECUTION_FROZEN.json')!=certificate['execution_release_sha256']:return None
    if _INPUTS is not None and _INPUTS[0]['execution_release_sha256']!=certificate['execution_release_sha256']:
        _INPUTS=None
    if _INPUTS is None:
        assert sha(HERE/'EXECUTION_FROZEN.json')==certificate['execution_release_sha256']
        assert sha(HERE/'OURS_HISTORY_INDEX.json')==certificate['history_index_sha256']
        assert sha(HERE/'HISTORY_FINGERPRINT_SCREEN.json')==certificate['fingerprint_screen_sha256']
        history=read(HERE/'OURS_HISTORY_INDEX.json')
        allowed=set(read(HERE/'HISTORY_FINGERPRINT_SCREEN.json')['eligible_for_prefix_verification'])
        contexts={r['source_rollout_uid']:r for r in map(json.loads,(HERE/'inputs/ours_history/db_context.jsonl').open())}
        _INPUTS=(certificate,history,allowed,contexts)
    return _INPUTS


def lookup(requests):
    candidates=[c for c in requests if c['benchmark']=='B' and c['phase']=='training' and c['solo']is None
                and c['controller_payload']['engine']=='original_admission_dictionary']
    if not candidates:return{}
    source=inputs()
    if source is None:return{}
    certificate,history,allowed,contexts=source
    manifest_path=HERE/'REUSE_MANIFEST.json'
    manifest=read(manifest_path)if manifest_path.exists()else dict(records={})
    results={}
    for request in candidates:
        assert request['controller_payload']['execution_sources']['freeze_sha256']==certificate['execution_release_sha256']
        key=signature(request['scene'],request['controller_payload']['nominal']['type_code'],request['state_index'],request['spec'])
        available=sorted(uid for uid in history['signature_index'].get(key,[])if uid in allowed)
        if not available:continue
        # UID order is deterministic and independent of outcome/success.
        uid=available[0];old=history['records'][uid]['result'];context=contexts[uid]
        source_controller=json.loads(context['controller']['config_json'])
        assert db.canonical(old['spec'])==db.canonical(request['spec'])
        assert db.canonical(source_controller['nominal'])==db.canonical(request['controller_payload']['nominal'])
        scenario=json.loads(context['state']['goals_geometry_json'])
        for name in ('config','walls','bottleneck_config','factor','route_guidance','scene_definition_sha256'):
            assert db.canonical(scenario.get(name))==db.canonical(request['scenario_payload'].get(name)),('Reuse configuration mismatch',name)
        physical={name:request['state'][name]for name in ('positions','velocities','goals')}
        assert db.canonical(physical)==db.canonical(json.loads(context['state']['physical_state_json']))
        assert request['rng_state_uid']==db.uid('rng_physical',physical)
        target=request['rollout_uid']
        results[target]=dict(old,target_query_uid=target,source_rollout_uid=uid,
            source='certified_archived_semantic_reuse',reuse_certificate=str(HERE/'HISTORY_PREFIX_CHECK.json'))
        record=dict(target_query_uid=target,source_rollout_uid=uid,
                    kind='certified_semantic_alias_not_exact_new_UID_hit',
                    certificate_sha256=sha(HERE/'HISTORY_PREFIX_CHECK.json'),
                    source_stream=history['records'][uid]['source_stream'],
                    source_line=history['records'][uid]['source_line'],
                    official_target_preflight_still_required=True)
        if target in manifest['records']:assert manifest['records'][target]==record
        manifest['records'][target]=record
    if results:dump(manifest_path,manifest)
    return results
