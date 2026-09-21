"""Frozen prediction and early-action intervention probe for VI R_CERT."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.action_probe import calibrate_offsets, execute_offsets
from single_integrator.c1.rollout_vi_r_cert import rollout
from single_integrator.c1.train_deadlock_union import setup, noise, digest
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv


CHECKPOINT_STEP = 100
INTERVENTION_STOP = 120
PREFIX_SEED = 84200
RISK_SEEDS = (84210, 84211)
EVALUATION_SEEDS = (84220, 84221)
EXTENDED_EVALUATION_SEEDS = (84220,)
RID_OFFSET = 74000


def noise_steps(seed, rid, steps):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), rid)
    return jax.vmap(lambda t: jax.random.normal(
        jax.random.fold_in(key, t), (4,), dtype=jnp.float32))(
            jnp.arange(steps, dtype=jnp.uint32))


def conditioned_noise(seed, rid, steps=850):
    prefix = noise_steps(PREFIX_SEED, rid, steps)
    future = noise_steps(seed, rid, steps)
    return jnp.concatenate((prefix[:CHECKPOINT_STEP],
                            future[CHECKPOINT_STEP:]), axis=0)


def classify(row, trace, goals, dt):
    original = ('safe_deadlock' if row['any_deadlock'] else
                'success' if row['success'] else 'other_timeout')
    label, details = classify_timeout_trace(dict(
        max_speed=np.linalg.norm(trace['applied'].reshape(-1, 2, 2),
                                 axis=-1).max(axis=-1),
        goal_errors=np.linalg.norm(trace['positions_after']-goals,
                                   axis=-1)), original, dt)
    return original, label, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('new output directory required')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    audit = json.loads((ROOT/'results/c1_vi_r_cert_check_v1/report.json').read_text())
    if not audit['passed']:
        raise RuntimeError('implementation audit did not pass')
    params, field, plant, cbf, baseline = setup()
    if not all((plant.terminate_on_collision, plant.terminate_on_success,
                plant.terminate_on_deadlock)):
        raise RuntimeError('frozen evaluation flags changed')
    goals = np.asarray(GiveWayEnv(plant).goals)
    rng = np.random.default_rng(2026091896)
    x = rng.uniform(.30, .55, (25, 2))
    y = rng.uniform(-.025, .025, (25, 2))
    starts = np.stack([np.c_[-x[:, 0], y[:, 0]],
                       np.c_[x[:, 1], y[:, 1]]], axis=1)
    arms = ['reference'] + [
        f'{kind}_{direction}' for kind in ('direct', 'augmented')
        for direction in ('negative', 'positive', 'random_0', 'random_1')]
    args.out.mkdir(parents=True)
    sources = {p: digest(ROOT/p) for p in (
        'scripts/probe_c1_vi_r_cert.py',
        'scripts/analyze_c1_vi_r_cert.py',
        'single_integrator/c1/rollout_vi_r_cert.py',
        'single_integrator/c1/risk/vi_r_cert.py',
        'single_integrator/c1/termination.py',
        'single_integrator/c1/action_probe.py')}
    protocol = dict(
        scope='fixed VI-augmented R_CERT validation; no training',
        starts=starts.tolist(), start_distribution=(
            '25 held-out initial conditions: independent |x_i| U(.30,.55), '
            'y_i U(-.025,.025); selected before any future continuation'),
        rid_offset=RID_OFFSET, checkpoint_step=CHECKPOINT_STEP,
        checkpoint_seconds=CHECKPOINT_STEP*plant.dt,
        intervention_actions=[CHECKPOINT_STEP, INTERVENTION_STOP-1],
        intervention_seconds=[CHECKPOINT_STEP*plant.dt,
                              INTERVENTION_STOP*plant.dt],
        prefix_seed=PREFIX_SEED, risk_seeds=list(RISK_SEEDS),
        evaluation_seeds=list(EVALUATION_SEEDS),
        extended_evaluation_seeds=list(EXTENDED_EVALUATION_SEEDS),
        future_randomness=(
            'iid standard-normal Flow latent per action; common prefix through '
            'action99; independently seeded suffixes from action100'),
        risk_continuations=25*len(RISK_SEEDS),
        scalar_reverse_sweeps=25*len(RISK_SEEDS)*2,
        main_paired_scenarios=25*len(EVALUATION_SEEDS),
        main_outcome_replays=25*len(EVALUATION_SEEDS)*len(arms),
        extended_paired_scenarios=25*len(EXTENDED_EVALUATION_SEEDS),
        extended_outcome_replays=25*len(EXTENDED_EVALUATION_SEEDS)*len(arms),
        random_directions_per_risk=2, random_seed=2026091897,
        random_aggregation='arithmetic mean; never best-direction selection',
        target_projected_rms=.0025, component_cap=.025,
        risk=dict(kappa=1., sigma=1., control_scale=.5,
                  direct_counts=dict(strict=303, stalled=42),
                  augmented_counts=dict(strict=2727, stalled=378)),
        initial_G_phi='frozen zero-output initialization',
        environment=plant.to_dict(), cbf=cbf.to_dict(),
        extended_horizon_steps=1000,
        baseline_sha256=digest(baseline), sources=sources,
        test_set_opened=False)
    atomic_save(args.out/'protocol.json', protocol)
    for path, sha in sources.items():
        assert digest(ROOT/path) == sha
        target = args.out/'source_snapshot'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    (args.out/'reference_params.pkl').write_bytes(pickle.dumps(
        jax.tree_util.tree_map(np.asarray, params)))

    length = INTERVENTION_STOP-CHECKPOINT_STEP
    zero = jnp.zeros((length, 4), jnp.float64)
    def objective(local_offsets, initial, draws, augmented):
        full = jnp.zeros((850, 4), jnp.float64).at[
            CHECKPOINT_STEP:INTERVENTION_STOP].set(local_offsets)
        terms, trace = rollout(params, field, initial, draws, plant, cbf, full)
        risk = jax.lax.select(augmented, terms['augmented_R_CERT'],
                              terms['direct_R_CERT'])
        aux = (trace['w'][CHECKPOINT_STEP:INTERVENTION_STOP],
               trace['A'][CHECKPOINT_STEP:INTERVENTION_STOP],
               trace['b'][CHECKPOINT_STEP:INTERVENTION_STOP],
               trace['before'][CHECKPOINT_STEP],
               trace['applied'][CHECKPOINT_STEP-1],
               trace['alive_pre'][CHECKPOINT_STEP],
               trace['latch_pre'][CHECKPOINT_STEP],
               terms['primary_deadlock'], terms['historical_strict_deadlock'],
               terms['stalled_deadlock'], terms['terminal_safe_deadlock'],
               terms['success'], terms['original_other_timeout'],
               terms['action_count'], terms['eligible'],
               terms['active_event_index'])
        return risk, aux
    direct_vg = jax.jit(jax.value_and_grad(
        lambda z, x, d: objective(z, x, d, False), has_aux=True))
    augmented_vg = jax.jit(jax.value_and_grad(
        lambda z, x, d: objective(z, x, d, True), has_aux=True))

    random_rng = np.random.default_rng(2026091897)
    diagnostics, records, extended = [], [], []
    begun = time.monotonic()
    for rid, initial in enumerate(starts):
        rollout_id = RID_OFFSET+rid
        direct_samples, augmented_samples = [], []
        for seed in RISK_SEEDS:
            draws = conditioned_noise(seed, rollout_id)
            direct_samples.append(direct_vg(zero, jnp.asarray(initial), draws))
            augmented_samples.append(augmented_vg(
                zero, jnp.asarray(initial), draws))
        checkpoint_positions = [np.asarray(s[0][1][3])
                                for s in augmented_samples]
        checkpoint_velocities = [np.asarray(s[0][1][4])
                                 for s in augmented_samples]
        checkpoint_ok = bool(
            all(np.array_equal(checkpoint_positions[0], x)
                for x in checkpoint_positions[1:]) and
            all(np.array_equal(checkpoint_velocities[0], x)
                for x in checkpoint_velocities[1:]) and
            all(bool(s[0][1][5]) and not bool(s[0][1][6])
                for s in augmented_samples))
        direct_values = np.array([float(s[0][0]) for s in direct_samples])
        augmented_values = np.array([float(s[0][0])
                                     for s in augmented_samples])
        direct_grads = np.asarray([s[1] for s in direct_samples])
        augmented_grads = np.asarray([s[1] for s in augmented_samples])
        defined = bool(checkpoint_ok and np.isfinite(direct_values).all() and
                       np.isfinite(augmented_values).all() and
                       np.isfinite(direct_grads).all() and
                       np.isfinite(augmented_grads).all())
        gradients = dict(
            direct=(direct_grads.mean(axis=0) if defined else
                    np.zeros((length, 4))),
            augmented=(augmented_grads.mean(axis=0) if defined else
                       np.zeros((length, 4))))
        prediction_samples = [tuple(np.asarray(x) for x in s[0][1][:3])
                              for s in augmented_samples]
        offsets = {'reference': np.zeros((length, 4))}
        calibration = {}
        for kind in ('direct', 'augmented'):
            gradient = gradients[kind]
            directions = dict(negative=-gradient, positive=gradient)
            for random_index in range(2):
                directions[f'random_{random_index}'] = (
                    random_rng.normal(size=(length, 4)) if
                    np.linalg.norm(gradient) > 0 else np.zeros((length, 4)))
            for direction, vector in directions.items():
                arm = f'{kind}_{direction}'
                if defined:
                    offsets[arm], calibration[arm] = calibrate_offsets(
                        vector, prediction_samples, cbf,
                        target_rms=.0025, component_cap=.025)
                else:
                    offsets[arm] = np.zeros((length, 4))
                    calibration[arm] = dict(
                        zero_direction=True, matched=False, rms=0., scale=0.,
                        undefined_risk_or_checkpoint=True)
        diagnostic = dict(
            rid=rid, rollout_id=rollout_id, checkpoint_ok=checkpoint_ok,
            defined=defined,
            checkpoint_positions=checkpoint_positions[0].tolist(),
            checkpoint_velocity=checkpoint_velocities[0].tolist(),
            direct_prediction_risks=direct_values.tolist(),
            augmented_prediction_risks=augmented_values.tolist(),
            direct_prediction_risk=(float(direct_values.mean()) if
                                    np.isfinite(direct_values).all() else None),
            augmented_prediction_risk=(float(augmented_values.mean()) if
                                       np.isfinite(augmented_values).all() else None),
            prediction_primary_events=[bool(s[0][1][7])
                                       for s in augmented_samples],
            prediction_historical_strict=[bool(s[0][1][8])
                                          for s in augmented_samples],
            prediction_stalled=[bool(s[0][1][9])
                                for s in augmented_samples],
            prediction_terminal_safe_deadlock=[bool(s[0][1][10])
                                               for s in augmented_samples],
            prediction_success=[bool(s[0][1][11])
                                for s in augmented_samples],
            prediction_original_timeout=[bool(s[0][1][12])
                                         for s in augmented_samples],
            prediction_action_counts=[int(s[0][1][13])
                                      for s in augmented_samples],
            prediction_eligible=[bool(s[0][1][14])
                                 for s in augmented_samples],
            direct_gradient_norm=float(np.linalg.norm(gradients['direct'])),
            augmented_gradient_norm=float(np.linalg.norm(
                gradients['augmented'])), calibration=calibration)
        diagnostics.append(diagnostic)
        atomic_save(args.out/'diagnostics.json', diagnostics)
        np.savez_compressed(args.out/f'offsets_{rid}.npz',
                            direct_gradient=gradients['direct'],
                            augmented_gradient=gradients['augmented'], **offsets)

        for seed in EVALUATION_SEEDS:
            draws = conditioned_noise(seed, rollout_id)
            for arm in arms:
                row, trace = execute_offsets(
                    params, field, initial, draws, plant, cbf, offsets[arm],
                    start_step=CHECKPOINT_STEP)
                original, label, stalled_details = classify(
                    row, trace, goals, plant.dt)
                changes = trace['direct_action_changes']
                row.update(
                    rid=rid, rollout_id=rollout_id, seed=seed, arm=arm,
                    original_outcome=original, outcome=label,
                    historical_strict=bool(row['any_deadlock']),
                    terminal_label_deadlock=original == 'safe_deadlock',
                    stalled_deadlock=label == 'stalled_deadlock',
                    primary_deadlock=(bool(row['any_deadlock']) or
                                      label == 'stalled_deadlock'),
                    ordinary_timeout=label == 'other_timeout',
                    collision=False, stalled_details=stalled_details,
                    direct_action_sum_squares=float(np.sum(changes**2)),
                    direct_action_components=int(changes.size),
                    zero_executed_effect=bool(changes.size == 0 or
                                              np.all(changes == 0.)))
                records.append(row)
                np.savez_compressed(
                    args.out/f'{arm}_{rid}_{seed}.npz', **trace)
                atomic_save(args.out/'records.json', records)

        plant_extended = replace(plant, max_steps=1000)
        for seed in EXTENDED_EVALUATION_SEEDS:
            draws = conditioned_noise(seed, rollout_id, steps=1000)
            for arm in arms:
                row, trace = execute_offsets(
                    params, field, initial, draws, plant_extended, cbf,
                    offsets[arm], start_step=CHECKPOINT_STEP)
                original, label, stalled_details = classify(
                    row, trace, goals, plant.dt)
                extended.append(dict(
                    rid=rid, rollout_id=rollout_id, seed=seed, arm=arm,
                    action_count=int(row['steps']), original_outcome=original,
                    outcome=label, historical_strict=bool(row['any_deadlock']),
                    primary_deadlock=(bool(row['any_deadlock']) or
                                      label == 'stalled_deadlock'),
                    success=bool(row['success']),
                    ordinary_timeout=label == 'other_timeout',
                    stalled_details=stalled_details))
                atomic_save(args.out/'extended_records.json', extended)
        print(json.dumps(dict(scenarios_completed=rid+1,
                              main_replays=len(records),
                              extended_replays=len(extended),
                              elapsed=time.monotonic()-begun)), flush=True)

    assert all(digest(ROOT/p) == sha for p, sha in sources.items())
    atomic_save(args.out/'complete.json', dict(
        prediction_continuations=25*len(RISK_SEEDS),
        scalar_reverse_sweeps=25*len(RISK_SEEDS)*2,
        main_replays=len(records), extended_replays=len(extended),
        elapsed=time.monotonic()-begun, test_set_opened=False,
        full_training_started=False))


if __name__ == '__main__':
    main()
