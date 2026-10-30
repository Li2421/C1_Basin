"""Prospective compatibility screen; reuse still needs fresh prefix checks."""
import guard
from guard import HERE,ROOT
import collections
import hashlib
import json

BASE=HERE/'inputs/diagnostics'


def read(path):return json.loads(path.read_text())


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index=read(HERE/'OURS_HISTORY_INDEX.json')
    contexts={r['source_rollout_uid']:r for r in map(json.loads,(HERE/'inputs/ours_history/db_context.jsonl').open())}
    scenes={e['scene']['name']:e['scene'] for e in read(BASE/'scene_types_oracle_coverage_v3/SCENES.json')['entries']}
    controllers=read(BASE/'cross_scene_correction_v1/SELECTED_CONTROLLERS.json')['xsc_swap_n6']
    old=ROOT/'diagnostics/scene_types_oracle_coverage_v3'
    expected_sources={str(old/name):sha(BASE/'scene_types_oracle_coverage_v3'/name)
                      for name in ('runtime.py','routes.py','recording_fix.py')}
    expected_sources.update({str(ROOT/'shared_control'/name):sha(HERE/'private/Basin_C1/shared_control'/name)
                            for name in ('basis_dictionary.py','basis_families.py','hard_projection.py','correction_b5.py')})
    expected_sources['private_harness']=read(HERE/'BACKEND_COMPLETENESS.json')['canonical_manifest_sha256']
    eligible=[];excluded={};versions=collections.Counter()
    for uid,item in index['records'].items():
        reasons=[]
        context=contexts.get(uid)
        if context is None:
            excluded[uid]=['No official source context']
            continue
        row=item['result'];scene=scenes[row['scene']]
        payload=json.loads(context['controller']['config_json'])
        scenario=json.loads(context['state']['goals_geometry_json'])
        state=scene['states'][row['state_index']]
        physical={k:state[k]for k in ('positions','velocities','goals')}
        if canonical(physical)!=canonical(json.loads(context['state']['physical_state_json'])):reasons.append('physical_state')
        for key,wanted in dict(config=scene['config'],walls=scene['walls'],bottleneck_config=scene.get('bottleneck_config'),
                               factor=scene['type_id'],route_guidance=bool(scene['route_guidance']),
                               scene_definition_sha256=scene['definition_sha256']).items():
            if canonical(scenario.get(key))!=canonical(wanted):reasons.append('scenario.'+key)
        if canonical(payload.get('nominal'))!=canonical(controllers[row['controller']]):reasons.append('nominal_controller')
        if canonical(payload.get('dictionary'))!=canonical(row['spec']):reasons.append('dictionary')
        if payload.get('dictionary',{}).get('one_way'):reasons.append('admission_not_covered_by_v3_source')
        if payload.get('rng')!='2026102004/matched_physical_state/seed/t':reasons.append('rng_recipe')
        if payload.get('precision')!='jax_x64_false' or row.get('jax_enable_x64'):reasons.append('precision')
        if payload.get('safety_layer')!='distributed' or row.get('responsibility')!=.5:reasons.append('safety')
        for path,expected in expected_sources.items():
            if payload.get('execution_sources',{}).get(path)!=expected:reasons.append('source.'+path)
        accelerated=payload.get('exact_acceleration',{})
        if not accelerated or accelerated.get('check')is not False:
            reasons.append('uncertified_old_acceleration_variant')
        else:
            hashes={__import__('pathlib').Path(r['target']).name:r['sha256']for r in accelerated['inputs']['files']}
            for name in ('fastproj.py','accel.py','np_threefry.py'):
                if hashes.get(name)!=sha(HERE/'accel_inputs'/name):reasons.append('acceleration.'+name)
        if reasons:excluded[uid]=reasons
        else:eligible.append(uid)
        versions[(scenario.get('version'),payload.get('correction_semantics'))]+=1
    output=dict(eligible_for_prefix_verification=sorted(eligible),excluded=excluded,
        exclusions_by_reason=dict(collections.Counter(reason for reasons in excluded.values()for reason in reasons)),
        eligible=len(eligible),total=len(index['records']),new_rollouts=0,certified_for_reuse=False,
        kernel_expected_sources=expected_sources,
        note='Exact physical/config/native-controller/RNG/source screen only. The new no-admission wrapper and acceleration still require fresh trajectory checks before any archived label is used; source UID is never relabeled as an exact new execution UID.')
    (HERE/'HISTORY_FINGERPRINT_SCREEN.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Historical candidates passing fingerprint screen',len(eligible),'/',len(index['records']),'; prefix verification pending.')


if __name__=='__main__':main()
