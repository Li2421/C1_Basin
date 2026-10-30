"""Semantic requests: analysis labels never change execution/cache identity."""
from core import *
from functools import lru_cache


@lru_cache(None)
def sources():
    frozen = read(HERE / 'EXECUTION_FROZEN.json')
    for relative, expected in frozen['files'].items():
        assert sha(HERE / relative) == expected, ('Execution source changed', relative)
    # Hash-address the complete manifest instead of repeating hundreds of
    # identical source entries in every case and every worker queue.
    manifest=HERE/frozen['immutable_manifest']
    assert manifest.read_bytes()==(HERE/'EXECUTION_FROZEN.json').read_bytes()
    return dict(freeze_sha256=sha(manifest), manifest=str(manifest))


def legacy_a_rng(scene, state, config):
    # Literal payload from the archived heterogeneous make_new/make_case pair.
    scenario = dict(version='cross_scene_si_v1', config=config, walls=scene['walls'],
                    bottleneck_config=scene.get('bottleneck_config'), factor=scene['factor'],
                    plant='frozen_swap_step', scene_definition_sha256=scene['definition_sha256'])
    physical = {key: state[key] for key in ('positions', 'velocities', 'goals')}
    return db.uid('state', dict(scenario=db.uid('scenario', scenario), physical=physical))


@lru_cache(None)
def certified_acceleration(certificate_mtime,release_sha):
    if certificate_mtime is None:return False
    cert=read(HERE/'ACCELERATION_CERTIFICATE.json')
    return bool(cert.get('passed'))and cert.get('execution_release_sha256')==release_sha


def case(benchmark, scene, controller, index, spec=None, solo=None, purpose='main', phase='training', accelerated=None):
    assert benchmark in ('A', 'B')
    if phase == 'test':
        assert all((HERE / name).exists() for name in ('PARAMETERS_FROZEN.json', 'OURS_FIXED_FROZEN.json', 'ANALYSIS_PLAN_FROZEN.json'))
    state = json.loads(json.dumps(scene['states'][index]))
    spec = spec or dict(eta=[0., 0., 0.], coefficients={})
    if solo is not None:
        state.update(N=1, positions=[state['positions'][solo]], velocities=[state['velocities'][solo]],
                     goals=[state['goals'][solo]], solo_parent_agent=solo)
    config = dict(scene['config'], num_agents=state['N'])
    factor=scene['factor'] if benchmark=='A' else scene['type_id']
    scenario = dict(version='paper_baseline_protocol_v1', benchmark=benchmark, config=config,
                    bottleneck_config=scene.get('bottleneck_config'), walls=scene['walls'],
                    factor=factor, route_guidance=bool(scene.get('route_guidance', False) if benchmark == 'B' else False),
                    plant='frozen_swap_step', scene_definition_sha256=scene['definition_sha256'])
    physical = {key: state[key] for key in ('positions', 'velocities', 'goals')}
    scenario_uid = db.uid('scenario', scenario)
    state_uid = db.uid('state', dict(scenario=scenario_uid, physical=physical))
    rng_uid = legacy_a_rng(scene, state, config) if benchmark == 'A' else db.uid('rng_physical', physical)
    payload = dict(nominal=controller, dictionary=spec, execution_sources=sources(), safety_layer='distributed',
                   precision='jax_x64_false', engine='heterogeneous_plain_original' if benchmark == 'A' else 'original_admission_dictionary',
                   rng='2026102004/preserved_A_state_uid_or_B_physical_uid/seed/t')
    if accelerated is None:
        certificate=HERE/'ACCELERATION_CERTIFICATE.json'
        stamp=certificate.stat().st_mtime_ns if certificate.exists()else None
        accelerated=certified_acceleration(stamp,payload['execution_sources']['freeze_sha256'])
    if accelerated:
        payload['exact_acceleration']=dict(check=False,loader_sha256=sha(HERE/'acceleration.py'),
            inputs={name:sha(HERE/'accel_inputs'/name)for name in ('fastproj.py','accel.py','np_threefry.py')})
    controller_uid = db.uid('ctl', dict(scenario=scenario_uid, payload=payload))
    eta_uid = db.eta_identity(spec['eta'])[0]
    seed_key = db.canonical(dict(future_index=0))
    return dict(state=state, scenario_payload=scenario, controller_payload=payload,
                scenario_uid=scenario_uid, state_uid=state_uid, controller_uid=controller_uid,
                eta_uid=eta_uid, seed_key=seed_key,
                rollout_uid=db.uid('roll', dict(state=state_uid, eta=eta_uid, controller=controller_uid, seed=seed_key)),
                seed=0, eta_a=spec['eta'], eta_b=spec['eta'], theta=None, spec=spec,
                scene=scene['name'], type_id=factor, state_index=index,
                solo=solo, purpose=purpose, phase=phase, benchmark=benchmark, rng_state_uid=rng_uid,
                state_provenance=dict(experiment=EXP, scene_definition_sha256=scene['definition_sha256'],
                                      state_index=index, solo_agent=solo, benchmark=benchmark, source='frozen_scene_and_declared_continuation',
                                      rng_state_uid=rng_uid))


def paper(benchmark, scene, index, method, options, mechanism=True, solo=None, purpose='paper_main', phase='training'):
    from execution_policy import assert_method_enabled
    assert_method_enabled(method)
    request = case(benchmark, scene, controllers()[0], index, solo=solo, purpose=purpose, phase=phase,accelerated=False)
    payload = request['controller_payload']
    adapter = HERE / 'paper_methods' / method / 'adapter.py'
    episode = HERE / 'paper_episode.py'
    assert adapter.exists() and episode.exists()
    payload.update(nominal=dict(type_code=method, parameters=db.canonical(options), checkpoint=''),
                   dictionary=None, safety_layer='paper_own_controller_safety', engine='paper_own_algorithm',
                   paper_method=method, paper_options=options,
                   # Frozen A learned observations already include a map-based
                   # navigation field. Do not remove that input from ours; give
                   # non-planning paper controllers the protocol's existing v3
                   # route hook on A as well. Chen keeps its own planner.
                   paper_route_guidance=bool((benchmark == 'A' or request['scenario_payload']['route_guidance'])
                                             and method != 'chen_impc_ob_si_tracking'),
                   paper_root_diagnostic_only=False, paper_mechanism=bool(mechanism),
                   paper_method_adapter=str(adapter), paper_episode_entry=str(episode),
                   error_policy='failed_agent_zero_one_step_then_normal_call',
                   planner_stopping='fixed_termination_condition_checks400_per_original0.04_retry_budget')
    request['controller_uid'] = db.uid('ctl', dict(scenario=request['scenario_uid'], payload=payload))
    request['rollout_uid'] = db.uid('roll', dict(state=request['state_uid'], eta=request['eta_uid'],
                                               controller=request['controller_uid'], seed=request['seed_key']))
    return request
