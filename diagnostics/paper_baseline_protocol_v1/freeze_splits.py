"""Freeze declared cohorts after reproducible state continuation; no rollout."""
import guard
from guard import HERE
import collections
import datetime
import hashlib
import json

BASE = HERE / 'inputs/diagnostics'
HETERO = BASE / 'hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/hetset'


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def a_name(scene):
    config, params = scene['config'], scene['params']
    robots, radius = config['num_agents'], config['agent_radius']
    factor = scene['factor']
    if factor == 'plus':
        text = f"十字路口·路宽 {2*params['w']:g} m"
    elif factor == 'box':
        text = '方形场地'
    elif factor == 'obst':
        text = f"带障碍对穿·起始圆半径 {params['R']:g} m"
    elif factor == 'swap':
        text = f"开阔对穿·起始圆半径 {params['R']:g} m"
    else:
        raise ValueError(('Unknown frozen heterogeneous factor', factor))
    text += f"，{robots}台，机器人半径 {radius:g} m"
    if 'f' in params:
        text += f"，时限系数 {params['f']:g}"
    return text


def b_name(scene):
    t, value, widen = scene['type_id'], scene['knob'], scene['widen']
    robots = scene['config']['num_agents']
    detail = f'{robots}台'
    # Literal construction rules from the copied v1/v3 build() functions.
    if t in (2, 8, 10, 11, 12):
        detail += f'，每侧 {int(value)}台'
    elif t in (5, 13):
        detail += f'，每向 {int(value)}台'
    elif t == 6:
        detail += f"，机器人半径 {scene['config']['agent_radius']:g} m"
    elif t == 7:
        detail += f'，{int(value)}个障碍块'
    widths = {3: (.9 + widen, '走廊宽'), 5: (.6 + widen, '支路宽'),
              8: (.3 + widen, '门宽'), 9: (.3 + widen, '出口宽'),
              10: (.3 + widen, '走廊宽'), 11: (.3 + widen, '主走廊宽'),
              12: (.5 + widen, '支路宽'), 13: (.6 + widen, '支路宽'),
              14: (float(value) + widen, '路宽'), 15: (.4 + widen, '门宽'),
              16: (.4 + widen, '巷宽')}
    if t in widths:
        width, label = widths[t]
        detail += f'，{label} {width:g} m'
    return f"{scene['title']}·{detail}"


def immutable(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, ('Frozen cohort changed', path)
    else:
        path.write_text(text)


def main():
    assert (HERE / 'PRIOR_TASK_DELIVERED.json').exists()
    destination = HERE / 'SPLITS_FROZEN.json'
    if destination.exists():
        # A frozen cohort cannot be regenerated with a new timestamp.
        print('Existing frozen split:', sha(destination))
        return
    split = read(BASE / 'hetero_set_critic_selfcheck_v1/SPLIT.json')
    first = read(BASE / 'hetero_set_critic_selfcheck_v1/FINAL_FROZEN.json')
    second = read(BASE / 'hetero_set_learned_planners_v1/FROZEN_T2.json')
    old_scenes = read(HETERO / 'scenes_all.json')
    all_scenes = read(HETERO / 'scenes_all2.json')
    scene_map = {s['name']: s for s in all_scenes}
    assert len(scene_map) == len(all_scenes)
    for scene in old_scenes:
        assert canonical(scene) == canonical(scene_map[scene['name']]), scene['name']
    a = dict(train=list(split['train']), validation=list(first['val']),
             test=list(first['test']) + list(second['test2']),
             train_states=list(range(8)), validation_states=list(range(8, 24)), test_states=list(range(8, 24)))
    assert list(map(lambda key: len(a[key]), ('train', 'validation', 'test'))) == [28, 8, 24]
    assert not set(a['train']) & (set(a['validation']) | set(a['test']))
    assert not set(a['validation']) & set(a['test'])
    included = a['train'] + a['validation'] + a['test']
    assert len(included) == len(set(included)) == 60
    assert all(len(scene_map[name]['states']) >= 24 for name in included)
    expected_categories = dict(plus=6, box=10, obst=9, swap=3)
    assert dict(collections.Counter(scene_map[name]['factor'] for name in a['train'])) == expected_categories
    representative = read(BASE / 'scene_types_oracle_coverage_v3/REPRESENTATIVE17_FROZEN.json')
    selected = {entry['scene'] for entry in representative['selected']}
    entries = read(BASE / 'scene_types_oracle_coverage_v3/SCENES.json')['entries']
    continued = read(HERE / 'NEW_STATES.json')
    assert continued['test_state_indices'] == (list(range(12, 20)) if continued['fresh'] else list(range(8, 12)))
    b = dict(train=[e['scene']['name'] for e in entries if e['scene']['name'] not in selected],
             test=[e['scene']['name'] for e in entries if e['scene']['name'] in selected],
             train_states=list(range(8)), test_states=continued['test_state_indices'], fresh_states=continued['fresh'])
    assert len(b['train']) == 32 and len(b['test']) == 17
    assert set(b['test']) == {e['scene'] for e in continued['scenes']}
    names = {}
    for name in included:
        scene = scene_map[name]
        names[name] = dict(benchmark='异质场景集', display=a_name(scene), category=scene['factor'],
                           parameters=scene['params'], robots=scene['config']['num_agents'])
    for entry in entries:
        scene = entry['scene']
        names[scene['name']] = dict(benchmark='十七类场景集', display=b_name(scene), type_id=scene['type_id'],
                                   knob=scene['knob'], level=scene['level'], widen=scene['widen'],
                                   robots=scene['config']['num_agents'], source_title=scene['title'])
    output = dict(frozen_at=datetime.datetime.now().astimezone().isoformat(), benchmark_A=a, benchmark_B=b,
                  source_hashes={str(path.relative_to(HERE)): sha(path) for path in (
                      BASE / 'hetero_set_critic_selfcheck_v1/SPLIT.json',
                      BASE / 'hetero_set_critic_selfcheck_v1/FINAL_FROZEN.json',
                      BASE / 'hetero_set_learned_planners_v1/FROZEN_T2.json',
                      BASE / 'scene_types_oracle_coverage_v3/SCENES.json',
                      BASE / 'scene_types_oracle_coverage_v3/REPRESENTATIVE17_FROZEN.json', HERE / 'NEW_STATES.json')},
                  leakage='通道单向放行设计使用过代表17场景的状态0–7，基准乙存在场景层面的设计泄漏；并列报告原四列。',
                  fallback_limitation=continued['limitation'], new_rollouts=0)
    immutable(HERE / 'SCENE_NAMES.json', names)
    immutable(destination, output)
    print('Frozen split:', sha(destination), '; A test384 physical states; B test', 17 * len(b['test_states']))


if __name__ == '__main__':
    main()
