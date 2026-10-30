"""Independent width/turn diagnostics, with original paper solvers and plant."""
from core import *
from dispatch import execute, status
from execution_policy import active_methods
from freeze_menus import DEFAULTS
from request_factory import paper
import copy
import datetime

WIDTHS = [.30, .32, .34, .36, .38, .40, .45, .50, .60]


def prepare():
    from physical_geometry import geometry
    template = next(e['scene'] for e in read(HERE/'inputs/diagnostics/scene_types_oracle_coverage_v3/SCENES.json')['entries']
                    if e['scene'].get('bottleneck_config'))
    rows = []
    for shape in ('straight', 'right_angle'):
        for width in WIDTHS:
            for offset in (0., .05, -.05):
                bc = copy.deepcopy(template['bottleneck_config'])
                bc.update(num_agents=2, half_length=2., half_height=2., agent_radius=.1,
                          wall_radius=template['config']['wall_radius'],
                          wall_collision_margin=template['config']['wall_collision_margin'],
                          barrier_x=[], openings=[], barrier_thickness=.1, extra_rectangles=[])
                if shape == 'straight':
                    bc.update(barrier_x=[0.], openings=[[[0., width]]], barrier_thickness=1.)
                    start, goal = [-1.3, offset], [1.3, 0.]
                else:
                    bc['extra_rectangles'] = [
                        [-2., .4, -.5, .5],
                        [-.5, width/2, -width/2, .5],
                        [width/2, -2., 2., .5],
                        [-.5, -2., width/2, -width/2],
                    ]
                    start, goal = [-1.3, offset], [0., 1.3]
                geom = geometry(bc)
                name = f'paper_diag_{shape}_w{width:.2f}_o{offset:+.2f}'
                config = dict(template['config'], scene=name, num_agents=1, agent_radius=.1,
                              max_steps=800, goal_tolerance=.08)
                state = dict(index=0, scene=name, N=1, positions=[start], velocities=[[0., 0.]], goals=[goal],
                             paper_geometry=dict(config=bc, walls=geom.walls.tolist()))
                assert geom.valid_points(np.asarray([start, goal])).all()
                scene = dict(name=name, config=config, states=[state], walls=geom.walls.tolist(),
                             bottleneck_config=None, route_guidance=True, type_id=100,
                             definition_sha256=digest(dict(geometry=bc, shape=shape, width=width, offset=offset)),
                             diagnostic=dict(shape=shape, width=width, offset=offset))
                rows.append(scene)
    dump(HERE/'SYNTHETIC_NARROW_SCENES.json', dict(scenes=rows, separate_from_all_protocol_splits=True,
         straight_length=1., turn_centerline_leg_lengths=[.5, .5], entry_exit_staging_length=.8,
         horizon_seconds=40., robot_radius=.1, route_rule='Existing benchmark visibility guidance'))
    return rows


def diagnostic(version, scenes):
    path = HERE/f'NARROW_{version}.json'
    if path.exists() and read(path).get('complete'):
        return read(path)
    requests = []; owners = []
    for method in active_methods():
        for scene in scenes:
            request = paper('B', scene, 0, method, DEFAULTS[method],
                            phase='diagnostic', purpose='synthetic_narrow_'+version)
            requests.append(request)
            owners.append(dict(method=method, scene=scene['name'], rollout_uid=request['rollout_uid'], **scene['diagnostic']))
    rows = {r['rollout_uid']: r for r in execute(requests, 'synthetic_narrow_'+version)}
    records = []
    for owner in owners:
        result = rows.get(owner['rollout_uid'])
        record = dict(owner, result=result)
        if result is not None:
            with np.load(result['trace']) as trace:
                last = trace['positions'][-1]
            scene = next(s for s in scenes if s['name'] == owner['scene'])
            from single_integrator.environment import point_segment_distance
            walls = np.asarray(scene['walls'])
            distances = point_segment_distance(last[:, None], walls[None, :, 0], walls[None, :, 1])
            record.update(final_position=last.tolist(), nearest_wall_centerline_distance=float(distances.min()),
                          final_wall_clearance=float(distances.min()-.1-scene['config']['wall_radius']-scene['config']['wall_collision_margin']))
        records.append(record)
    summaries = []
    for method in active_methods():
        for shape in ('straight', 'right_angle'):
            successes = [r for r in records if r['method']==method and r['shape']==shape and r['result'] and r['result']['success']]
            all_offsets_widths = [w for w in WIDTHS if all(any(r['width']==w and r['offset']==o for r in successes) for o in (0., .05, -.05))]
            summaries.append(dict(method=method, shape=shape, minimum_width_any_offset=min((r['width'] for r in successes), default=None),
                                  minimum_width_all_three_offsets=min(all_offsets_widths, default=None)))
    output = dict(complete=len(rows)==len(requests), version=version, records=records, summaries=summaries,
                  missing=[o['rollout_uid'] for o in owners if o['rollout_uid'] not in rows])
    dump(path, output)
    return output


def run():
    # This stage follows matched training solo tests; no tested execution source
    # is edited until their workers have exited and their journals are merged.
    from physical_geometry import install
    install(RN)
    core_path = HERE/'core.py'
    source = core_path.read_text()
    hook = '\nfrom physical_geometry import install as _install_physical_geometry\n_install_physical_geometry(RN)\n'
    if hook not in source:
        core_path.write_text(source+hook)
    manifest = HERE/'freeze_execution.py'
    source = manifest.read_text()
    if "'physical_geometry.py'" not in source:
        manifest.write_text(source.replace("'core.py','guard.py'", "'core.py','physical_geometry.py','guard.py'"))
    from freeze_execution import main as release
    release()
    scenes = prepare()
    before = diagnostic('before_alignment', scenes)
    assert before['complete'], 'Synthetic diagnostic has infrastructure gaps'
    from margin_alignment import apply
    changes = apply()
    release()
    after = diagnostic('after_alignment', scenes)
    assert after['complete']
    output = dict(complete=True, before=before, after=after, physical_changes=changes,
                  algorithm_buffer_tuning=False, same54scenes_per_method=True)
    dump(HERE/'NARROW_PASSAGE_DIAGNOSTIC.json', output)
    return output
