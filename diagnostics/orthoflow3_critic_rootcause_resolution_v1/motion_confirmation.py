"""Independent natural-controller confirmation; reuse audited cache executor."""
import argparse
import os
import shutil
import time
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from . import function_confirmation as parent
from .function_support import read, write, sha

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'motion_final_source_models'
ASSETS = ROOT/'controller_function_support'
RULE = ROOT/'motion_independent_confirmation_protocol.json'


def bridge():
    assert (SOURCE/'models_frozen.json').exists(), 'Models must be frozen first'
    copied = {}
    for name in ('protocol.json', 'states.json', 'pairs.json'):
        source, dest = ASSETS/name, SOURCE/name
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert sha(dest) == sha(source)
        copied[name] = sha(dest)
    confirmation = SOURCE/'independent_confirmation_protocol.json'
    if not confirmation.exists():
        shutil.copyfile(RULE, confirmation)
    assert sha(confirmation) == sha(RULE)
    gate = dict(passed=True, interpretation='Eligible for a mechanistic independent test only; not a generalization success gate',
        state_specific_selection_established=False, source_only=True,
        evidence_sha256=sha(ROOT/'motion_eta_control/source_selection.json'),
        primary_claim='Source held-controller matched-eta comparison improved across seeds; independent replication not yet done.')
    if (SOURCE/'source_gate.json').exists():
        assert read(SOURCE/'source_gate.json') == gate
    else:
        write(SOURCE/'source_gate.json', gate)
    dependencies = [
        ROOT/'motion_final_fit.py', ROOT/'motion_factorial_train.py', ROOT/'goal_velocity_train.py',
        ROOT.parent/'orthoflow3_controller_information_probe_v1/probe.py',
        ROOT.parent/'orthoflow3_unified_representation_v1/representation.py',
        ROOT.parent/'orthoflow3_controller_training_repair_v1/data.py',
        ROOT.parent/'orthoflow3_controller_intervention_generalization_v1/rich_context.py',
        ROOT/'goal_response.py', ROOT/'goal_velocity_response.py',
        ROOT/'function_confirmation.py', ROOT/'seed_replication.py',
        ROOT/'motion_confirmation.py',
        ROOT.parent/'orthoflow3_source_contrast_interaction_v1/pipeline.py',
    ]
    audit = dict(source_metadata_hashes=copied, confirmation_rule_sha256=sha(RULE),
                 models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
                 dependency_hashes={str(p):sha(p) for p in dependencies},
                 new_task_rollouts=0, target_labels_read=False)
    dest = SOURCE/'confirmation_bridge.json'
    if dest.exists():
        assert read(dest) == audit, 'Dependency changed after confirmation setup'
    else:
        write(dest, audit)


def configure(replica):
    assert replica in (0, 1)
    rule = read(RULE)
    target = rule['target_rule'][replica]
    assert target['families'] == 64, 'Audited parent contract is 64 families'
    parent.SOURCE, parent.RULE = SOURCE, RULE
    parent.TARGET_SEED = target['controller_seed']
    parent.FAMILY_SEED = target['initial_state_seed_start']
    parent.OUT = ROOT/f'motion_independent_confirmation_{parent.TARGET_SEED}'
    parent.FLOW = SOURCE/f'target_flow_seed{parent.TARGET_SEED}'
    parent.EXP = f'exp_orthoflow3_motion_fit_confirmation{parent.TARGET_SEED}_v1'
    parent.configure()
    return parent.OUT


def response_inputs():
    import jax
    from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
    from .goal_response import measurement as rest
    from .goal_velocity_response import measurement as motion
    assert jax.default_backend() == 'gpu'
    out = parent.OUT
    profiles = read(out/'protocol.json')['profiles'] + [read(SOURCE/'protocol.json')['profiles'][0]]
    records, pairs = read(out/'physical.json'), read(out/'pairs.json')
    for profile in profiles:
        dest = out/f'goal_both_{profile["name"]}.npz'
        if dest.exists():
            assert read(out/f'goal_both_audit_{profile["name"]}.json')['controller_sha256']==profile['sha256']
            with np.load(dest) as existing:
                assert existing['valid'].all(), 'Invalid cached physical response'
            continue
        assert sha(profile['path']) == profile['sha256']
        rt = RichRuntime('ring_exchange', profile['path'])
        values, errors, times = [], [], []
        for i, p in enumerate(pairs):
            record = records[p['state_index']]
            assert record['state_uid'] == p['state_uid']
            start = time.perf_counter()
            try:
                value = np.concatenate((rest(rt, record['physical'], np.array(p['eta'], float)),
                                        motion(rt, record['physical'], np.array(p['eta'], float))))
                assert np.isfinite(value).all()
            except Exception as exc:
                value = np.full(32, np.nan)
                errors.append(dict(pair=i, error=f'{type(exc).__name__}: {exc}'))
            values.append(value); times.append(time.perf_counter()-start)
        values = np.array(values, np.float32)
        np.savez_compressed(dest, goal_response=values[:, :16], goal_motion_response=values[:, 16:],
                            valid=np.isfinite(values).all(1), seconds=np.array(times))
        write(out/f'goal_both_audit_{profile["name"]}.json', dict(invalid=errors,
            controller_sha256=profile['sha256'], models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
            task_labels_read=False, new_task_rollouts=0, no_environment_steps=True,
            rest_code_sha256=sha(ROOT/'goal_response.py'), motion_code_sha256=sha(ROOT/'goal_velocity_response.py')))
        assert not errors, 'Stop before task rollouts if controller inputs are invalid'


def validate_inputs():
    out = parent.OUT
    profiles = read(out/'protocol.json')['profiles']
    profiles += [read(SOURCE/'protocol.json')['profiles'][0]]
    pairs = read(out/'pairs.json')
    assert len(pairs) == 128
    for profile in profiles:
        name = profile['name']
        assert sha(profile['path']) == profile['sha256']
        with np.load(out/f'inputs_{name}.npz') as initial, np.load(out/f'goal_both_{name}.npz') as goal:
            assert initial['valid'].shape == goal['valid'].shape == (128,)
            assert initial['valid'].all() and goal['valid'].all()
            for values in (initial['context'], initial['agent_response'],
                           goal['goal_response'], goal['goal_motion_response']):
                assert np.isfinite(values).all()
        assert read(out/f'goal_both_audit_{name}.json')['controller_sha256'] == profile['sha256']


def preflight():
    from . import state_support
    from shared_rollout_db.src.planner import preflight as query
    validate_inputs()
    state_support.OUT = parent.OUT
    state_support.validate_plan()
    doc = query(parent.OUT/'planned_rollouts.json')
    for name in ('cache_preflight.json', 'cache_preflight_held.json', 'execution_preflight_held.json'):
        write(parent.OUT/name, doc)
    # Numeric rows must not be silently retried.
    parent.engine.filter_numerical()
    print(dict(target=parent.TARGET_SEED, **doc['summary']), flush=True)


def main(action, replica, index):
    configure(replica)
    if action == 'bridge':
        bridge()
    elif action == 'flow':
        bridge(); parent.flow()
    elif action == 'prepare':
        bridge(); parent.prepare()
    elif action == 'physical':
        parent.physical()
    elif action == 'inputs':
        # Cache only the target and predeclared source-alt wrong-controller input.
        for i, name in ((0, 'held'), (1, 'alt')):
            if not (parent.OUT/f'inputs_{name}.npz').exists():
                parent.features(i)
        response_inputs()
        validate_inputs()
    elif action == 'preflight':
        preflight()
    elif action == 'run':
        # Explicit invocation only after inspected/reported preflight; never
        # called automatically by the setup or feature-construction stages.
        gate = read(parent.OUT/'execution_authorization.json')
        assert gate['preflight_sha256'] == sha(parent.OUT/'cache_preflight.json')
        assert gate['planned_sha256'] == sha(parent.OUT/'planned_rollouts.json')
        assert gate['prediction_freeze_sha256'] == sha(parent.OUT/'prediction_freeze.json')
        assert gate['models_frozen_sha256'] == sha(SOURCE/'models_frozen.json')
        assert gate['reported_before_submission'] is True
        assert gate['new_continuation_limit'] <= 2048
        parent.engine.run('held', index)
    elif action == 'postflight':
        from . import postprocess
        postprocess.run('motion_fit_confirmation')
    elif action == 'dataset':
        parent.dataset()
    else:
        raise ValueError(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('bridge','flow','prepare','physical','inputs','preflight','run','postflight','dataset'))
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    main(args.action, args.replicate, args.index)
