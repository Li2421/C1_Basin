"""Audit inherited scene config versus the actual frozen wall-collision formula."""
import guard
from guard import HERE
import collections
import hashlib
import json

BASE=HERE/'inputs/diagnostics'


def read(path):return json.loads(path.read_text())


def main():
    original=BASE/'hetero_set_critic_selfcheck_v1'
    split=read(original/'SPLIT.json');first=read(original/'FINAL_FROZEN.json')
    second=read(BASE/'hetero_set_learned_planners_v1/FROZEN_T2.json')
    roles={name:part for part,names in [('train',split['train']),('validation',first['val']),
                                       ('test',first['test']+second['test2'])]for name in names}
    all_a=read(original/'workspace_snapshot_20261027/hetset/scenes_all2.json')
    scenes_a=[s for s in all_a if s['name']in roles]
    scenes_b=[e['scene']for e in read(BASE/'scene_types_oracle_coverage_v3/SCENES.json')['entries']]
    rows=[]
    for benchmark,scenes in [('A',scenes_a),('B',scenes_b)]:
        for scene in scenes:
            # NewEnv constructs Geometry from the state-level config. Verify
            # every state carries the same value before using the scene copy.
            assert all(s.get('bottleneck_config')==scene.get('bottleneck_config')for s in scene['states'])
            outer=scene['config'];geometry=scene.get('bottleneck_config')or outer
            fields=('agent_radius','wall_radius','wall_collision_margin')
            mismatched=[k for k in fields if outer[k]!=geometry[k]]
            nominal_threshold=outer['agent_radius']+outer['wall_radius']+outer['wall_collision_margin']
            plant_threshold=geometry['agent_radius']+geometry['wall_radius']+geometry['wall_collision_margin']
            rows.append(dict(benchmark=benchmark,scene=scene['name'],part=roles.get(scene['name'],'breadth'),
                category=scene.get('factor',scene.get('type_id')),outer={k:outer[k]for k in fields},
                geometry={k:geometry[k]for k in fields},mismatched_parameters=mismatched,
                outer_centerline_threshold=nominal_threshold,frozen_plant_centerline_threshold=plant_threshold,
                implied_wall_width_if_robot_radius_preserved=plant_threshold-outer['agent_radius'],
                all_state_geometry_configs_verified=True))
    sources=[HERE/'generated/scene_environment.py',HERE/'generated/backend/swap_geometry.py',
             HERE/'inputs/production/bottleneck_family/scenario.py']
    output=dict(new_rollouts=0,physical_changes=0,controller_changes=0,records=rows,
        summary={benchmark:dict(scenes=sum(r['benchmark']==benchmark for r in rows),
            parameter_mismatches=sum(r['benchmark']==benchmark and bool(r['mismatched_parameters'])for r in rows),
            radius_mismatches=sum(r['benchmark']==benchmark and 'agent_radius'in r['mismatched_parameters']for r in rows),
            negative_implied_width=sum(r['benchmark']==benchmark and r['implied_wall_width_if_robot_radius_preserved']<0 for r in rows),
            mismatches_by_part=dict(collections.Counter(r['part']for r in rows if r['benchmark']==benchmark and r['mismatched_parameters'])))
            for benchmark in ('A','B')},
        source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},
        source_lines=['generated/scene_environment.py:24-29 uses state.bottleneck_config',
                      'generated/backend/swap_geometry.py:63 uses instance.geometry.margin for swept wall collision',
                      'inputs/production/bottleneck_family/scenario.py:116 defines geometry.margin from its own Config'],
        decision='Preserve the frozen engine, nominal controllers, safety and physical input records. Do not force equality by shrinking robots, negative wall widths, moving walls or relaxing collision checks. Record this inherited limitation in the main report and applicability analysis.',
        evidence_level='Static code/config audit; mandatory instantiated-environment and trajectory checks remain separate.')
    (HERE/'STATIC_WALL_MARGIN_AUDIT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output['summary'],ensure_ascii=False))


if __name__=='__main__':main()
