"""Literal copied scene records and immutable protocol splits; no resampling."""
import guard
from guard import HERE
import copy
import json

BASE = HERE/'inputs/diagnostics'
HETERO = BASE/'hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/hetset'


def read(path):
    return json.loads(path.read_text())


def catalog(benchmark):
    assert (HERE/'PRIOR_TASK_DELIVERED.json').exists()
    split = read(HERE/'SPLITS_FROZEN.json')['benchmark_'+benchmark]
    if benchmark == 'A':
        all_scenes = {s['name']: s for s in read(HETERO/'scenes_all2.json')}
        names = split['train'] + split['validation'] + split['test']
        result = {name: copy.deepcopy(all_scenes[name]) for name in names}
    else:
        assert benchmark == 'B'
        result = {e['scene']['name']: copy.deepcopy(e['scene'])
                  for e in read(BASE/'scene_types_oracle_coverage_v3/SCENES.json')['entries']}
        continued = read(HERE/'NEW_STATES.json')
        if continued['fresh']:
            for item in continued['scenes']:
                states = result[item['scene']]['states']
                assert len(states) == 12 and len(item['states']) == 8
                assert [s['index'] for s in item['states']] == list(range(12,20))
                states.extend(copy.deepcopy(item['states']))
    return result


def cohort(benchmark, part):
    split = read(HERE/'SPLITS_FROZEN.json')['benchmark_'+benchmark]
    scenes = catalog(benchmark)
    return [scenes[name] for name in split[part]]


def frozen_eta(scene_name):
    first = read(BASE/'hetero_set_critic_selfcheck_v1/FINAL_FROZEN.json')
    second = read(BASE/'hetero_set_learned_planners_v1/FROZEN_T2.json')
    matches = [f['choices'][scene_name] for f in (first,second) if scene_name in f['choices']]
    assert len(matches) == 1, ('Frozen scene choice must be unique', scene_name)
    chosen = matches[0]
    index = chosen['ens16']
    assert 0 <= index < 16 and len(chosen['proposals']) == 16
    return 'p'+str(index), list(chosen['proposals'][index])


def validate_common_environment():
    """Check actual instantiated plant geometry as well as copied input fields."""
    from core import RN, dump, sha, np
    scenes = catalog('A')
    archived = BASE/'hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/scene_search/harness'
    source_checks = {name: dict(archived=sha(archived/name), current=sha(HERE/'generated'/name))
                     for name in ('runtime_new.py','scene_environment.py','controllers_new.py')}
    assert all(x['archived']==x['current'] for x in source_checks.values())
    rows = []
    for name, scene in scenes.items():
        env = RN.SwapEnv(scene['states'][0], RN.LiteratureConfig(**scene['config']))
        actual = env.config.to_dict()
        differences = []
        geometry_config=scene.get('bottleneck_config')or scene['config']
        expected_margin=geometry_config['agent_radius']+geometry_config['wall_radius']+geometry_config['wall_collision_margin']
        if env.instance.geometry.margin!=expected_margin:
            differences.append(dict(field='actual_wall_collision_margin',original=expected_margin,common=env.instance.geometry.margin))
        for key, value in scene['config'].items():
            if actual[key] != value:
                differences.append(dict(field=key, original=value, common=actual[key]))
        walls = np.asarray(scene['walls'],float).reshape(-1,2,2)
        actual_walls = np.asarray(env.walls,float).reshape(-1,2,2)
        if walls.shape != actual_walls.shape or walls.tobytes() != actual_walls.tobytes():
            differences.append(dict(field='wall_segments', original_shape=list(walls.shape), common_shape=list(actual_walls.shape)))
        for key in ('positions','velocities','goals'):
            expected = np.asarray(scene['states'][0][key],float)
            actual_array = getattr(env,key)
            if expected.shape != actual_array.shape or expected.tobytes() != actual_array.tobytes():
                differences.append(dict(field='state.'+key))
        rows.append(dict(scene=name, category=scene['factor'], differences=differences,
                         same_step_implementation='inherited SwapEnv.step',
                         frozen_wall_centerline_threshold=expected_margin,
                         frozen_horizon_steps=scene['config']['max_steps']))
    common = dict(scenes=list(scenes.values()), conversion='No physical conversion: literal config, wall segments, bottleneck config and states retained.',
                  source_sha256=sha(HETERO/'scenes_all2.json'))
    dump(HERE/'HETERO_SCENES_COMMON.json',common)
    output = dict(passed=all(not r['differences'] for r in rows), scenes=rows,
                  source_runtime_byte_identical_to_archived_heterogeneous=True, source_checks=source_checks,
                  inherited_geometry_limitation='STATIC_WALL_MARGIN_AUDIT.json',
                  note='Field and instantiated geometry check only. Mandatory full-episode label/time checks remain separate in ENV_CHECK.json.')
    dump(HERE/'ENV_DIFF.json',output)
    assert output['passed'], 'Common environment differs from frozen heterogeneous inputs'
    return output


if __name__ == '__main__':
    validate_common_environment()
