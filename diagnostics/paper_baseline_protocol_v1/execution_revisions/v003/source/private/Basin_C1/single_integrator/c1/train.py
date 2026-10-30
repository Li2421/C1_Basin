"""Train only the C1 residual against an approved 309/314 baseline checkpoint."""
import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from single_integrator.c1.differentiable_rollout import ResidualFlowField, barrier_constraints, bounded_nominal
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.risk.risk_function import RiskConfig, risk_diagnostics, trajectory_risk
from single_integrator.c1.risk.soft_activity import SoftRiskConfig, soft_risk_diagnostics
from single_integrator.c1.risk.risk_v1 import RiskV1Config, trajectory_risk_v1
from single_integrator.c1.risk.risk_v2 import RiskV2Config
from single_integrator.c1.socp import ExactProjection
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update, primal_loss, deviation_cost
from single_integrator.c1.training.selection import better_feasible_candidate
from single_integrator.c1.training.calibration import calibrate_fixed_set, reuse_fixed_calibration
from single_integrator.c1.training.dataset_starts import load_dataset_starts, load_fixed_inputs, require_population
from single_integrator.c1.training.persistence import atomic_save, restored_history
from single_integrator.c1.training.step_control import guarded_step
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import ROOT, load_policy


APPROVED = {
    "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32",
    "3f58b12266d19358a8ef5673e3e4018c1248c526de7d71877218ea176bd53383",
}


def observation(positions, velocity, goals):
    return jnp.stack([
        jnp.concatenate((positions[:, 0], velocity[:, 0], goals[0] - positions[:, 0],
                         positions[:, 1] - positions[:, 0], velocity[:, 1] - velocity[:, 0]), axis=-1),
        jnp.concatenate((positions[:, 1], velocity[:, 1], goals[1] - positions[:, 1],
                         positions[:, 0] - positions[:, 1], velocity[:, 0] - velocity[:, 1]), axis=-1),
    ], axis=1)


def rollout_terms(params, field, projection, initial, noise, plant, cbf, risk, return_details=False, return_mask=False):
    """C1 V0 rollout with same-state, same-noise Safety references.

    Baseline weights are closed-over constants; all state dependence survives.
    Active-set selection alone is discrete. Safety is validated in the solver.
    """
    if isinstance(risk,RiskV2Config):
        from single_integrator.c1.episode_rollout import rollout_episode_terms
        return rollout_episode_terms(params,field,projection,initial,noise,plant,cbf,risk,
                                      observation,return_details,return_mask)
    positions = jnp.asarray(initial, jnp.float64)
    velocity = jnp.zeros_like(positions)
    env = GiveWayEnv(plant)
    goals, walls = jnp.asarray(env.goals), jnp.asarray(env.walls)
    executed, baseline_safe, risks, position_history = [], [], [], [positions]
    details = []
    for step in range(noise.shape[1]):
        obs = observation(positions, velocity, goals)
        A, b, h = jax.vmap(lambda p: barrier_constraints(p, walls, plant.to_dict(), cbf))(positions)
        control = field.control(params, obs, noise[:, step], A, b, plant.max_speed, projection)
        u, u_base = control['applied'], control['safe']
        # Current pre-final-CBF candidate is the task control for C1 V0.
        nominal = control['candidate']
        residual = jnp.einsum('bij,bj->bi', A, u) - b
        active = jax.lax.stop_gradient((h <= risk.rho) & (jnp.abs(residual) <= risk.active_tol))
        blocks = A.reshape(len(positions), 17, 2, 2)
        if isinstance(risk, SoftRiskConfig):
            diagnostic = jax.vmap(lambda f,g,h,s: soft_risk_diagnostics(f,g,h,s,risk))(
                nominal.reshape(-1,2,2),blocks,h,residual)
        else:  # Historical V0 regression tests and hard diagnostics only.
            diagnostic = jax.vmap(lambda f, g, m: risk_diagnostics(f, g, m, risk))(
                nominal.reshape(-1, 2, 2), blocks, active)
        executed.append(u); baseline_safe.append(u_base); risks.append(diagnostic['risk'])
        if return_details:
            details.append(dict(**control, **diagnostic, positions=positions, A=A, b=b, h=h))
        positions, velocity = positions + plant.dt * u.reshape(-1, 2, 2), u.reshape(-1, 2, 2)
        position_history.append(positions)
    cone_steps = jnp.stack(risks, 1)
    episode_mask=jnp.ones(cone_steps.shape,bool)
    if isinstance(risk, RiskV1Config):
        v1 = trajectory_risk_v1(cone_steps, jnp.stack(position_history, 1), goals, risk,
                                risk.window_steps(plant.dt))
        trajectory = v1['trajectory_risk']
    else:
        v1 = None
        trajectory = trajectory_risk(cone_steps, risk)
    result = (jnp.stack(executed, 1), jnp.stack(baseline_safe, 1), trajectory)
    if return_details:
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 1), *details)
        if v1 is not None:
            stacked = dict(stacked, **v1)
        stacked['episode_mask']=episode_mask
        return (*result, stacked)
    if return_mask:
        return (*result,episode_mask)
    return result


def approved_checkpoint(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in APPROVED:
        raise ValueError('C1 accepts only a baseline_309_314 25k checkpoint')
    return digest


def risk_start_pool(data_root, plant, rho):
    """Safe train-trajectory states inside the pairwise TRO look-ahead band."""
    d_safe = 2 * plant.agent_radius + plant.agent_collision_margin + CBFConfig().separation_buffer
    starts = []
    for pair in range(200):
        with np.load(data_root / f'episode_{2 * pair:04d}.npz') as item:
            positions = np.asarray(item['positions'])
            delta = positions[:, 0] - positions[:, 1]
            h = np.sum(delta * delta, axis=-1) - d_safe * d_safe
            starts.extend(positions[index] for index in np.flatnonzero((h > 0) & (h <= rho)))
    if not starts:
        # The certified expert corpus stays farther than rho from the CBF
        # boundary. Construct legal short-scene near-conflict starts from the
        # same plant constants; these are training coverage, never test starts.
        separation = np.sqrt(d_safe * d_safe + .5 * rho)
        offsets = np.linspace(-.015, .015, 33)
        starts = [np.array([[-separation / 2, offset], [separation / 2, -offset]]) for offset in offsets]
    return np.asarray(starts)


def sample_batch(rng, data_root, batch_size, horizon, key, pool, risk_start_fraction, dataset_starts=None):
    """Mix ordinary train starts with safe in-distribution risk-coverage starts."""
    if dataset_starts is not None:
        initial = dataset_starts[rng.integers(0, len(dataset_starts), size=batch_size)]
        return jnp.asarray(initial, jnp.float64), jax.random.normal(key, (batch_size, horizon, 4), dtype=jnp.float32)
    initial = []
    risk_count = int(round(batch_size * risk_start_fraction))
    for index in range(batch_size):
        if index < risk_count:
            initial.append(pool[int(rng.integers(0, len(pool)))])
        else:
            pair = int(rng.integers(0, 200))
            with np.load(data_root / f'episode_{2 * pair:04d}.npz') as item:
                initial.append(item['initial_positions'])
    return jnp.asarray(initial, jnp.float64), jax.random.normal(key, (batch_size, horizon, 4), dtype=jnp.float32)


def main():
    jax.config.update('jax_enable_x64', True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--horizon', type=int, default=None,
                   help='v2 defaults to full 850-step episodes; legacy versions default to 120-step segments')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dual-lr', type=float, default=1e-2)
    p.add_argument('--max-backtracks', type=int, default=8,
                   help='halve the Adam proposal until the same-batch fixed-dual objective does not increase')
    p.add_argument('--epsilon-fraction', type=float, default=.8)
    p.add_argument('--start-distribution', choices=('legacy_mixture', 'dataset'), default='legacy_mixture')
    p.add_argument('--risk-start-fraction', type=float, default=None)
    for name, default in [('rho', .05), ('active-tol', 1e-7), ('D0', .1), ('kappa', 8.), ('alpha', .5), ('p', 4.)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--tau-h', type=float, default=.01)
    p.add_argument('--tau-s', type=float, default=.01)
    p.add_argument('--candidate-sigmas', type=float, default=6.)
    p.add_argument('--risk-version', choices=('v0_1', 'v1', 'v1_1', 'v2'), default='v2')
    p.add_argument('--terminal-distance-scale',type=float,default=.1)
    p.add_argument('--window-seconds', type=float, default=4.)
    p.add_argument('--delta-prog', type=float, default=.02)
    p.add_argument('--tau-prog', type=float, default=.005)
    p.add_argument('--task-distance-epsilon', type=float, default=1e-5)
    p.add_argument('--task-mask-temperature', type=float, default=.02)
    p.add_argument('--calibration-set', type=Path,
                   help='Frozen starts/noise NPZ; average all episodes to calibrate epsilon.')
    p.add_argument('--calibration-reference', type=Path,
                   help='Explicit prior config.json with identical zero-residual calibration; verify provenance and replay first batch')
    p.add_argument('--validation-set', type=Path,
                   help='Fixed NPZ with initial_positions and Flow noise for v1 validation.')
    p.add_argument('--validation-every', type=int, default=0,
                   help='Evaluate the fixed validation set every K updates; 0 disables it.')
    p.add_argument('--resume', action='store_true',
                   help='continue an interrupted run in out-dir, preserving phi, Adam, dual, and RNG state')
    args = p.parse_args()
    if args.calibration_reference and not args.calibration_set:
        raise ValueError('calibration reference requires a frozen calibration set')
    if args.risk_start_fraction is None:
        args.risk_start_fraction = 0. if args.start_distribution == 'dataset' else .5
    if args.start_distribution == 'dataset' and (args.risk_start_fraction != 0 or args.calibration_set is None or args.validation_set is None):
        raise ValueError('dataset starts require zero risk-start-fraction and frozen calibration/validation sets')
    if args.horizon is None:
        args.horizon=850 if args.risk_version=='v2' else 120
    if min(args.updates, args.batch_size, args.horizon) < 1 or args.validation_every < 0 or args.max_backtracks < 0 or not 0 < args.epsilon_fraction < 1 or not 0 <= args.risk_start_fraction <= 1:
        raise ValueError('invalid C1 training configuration')
    if any(not np.isfinite(rate) or rate <= 0 for rate in (args.lr, args.dual_lr)):
        raise ValueError('learning rates must be finite and positive')
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
        raise FileExistsError('out-dir must be empty unless --resume is supplied')
    checkpoint_hash = approved_checkpoint(args.checkpoint)
    baseline, provenance = load_policy(args.checkpoint)
    plant = Config(**provenance['evaluation_environment'])
    if plant.corridor_half_length != 1.3:
        raise ValueError('selected baseline must use the short 1.3m scene')
    model = ResidualCorrection(hidden_dims=tuple(baseline.config['actor_hidden_dims']), layer_norm=baseline.config['actor_layer_norm'])
    params = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 4)), jnp.zeros((1, 1)), jnp.zeros((1, 20)))
    field, projection, cbf = ResidualFlowField(baseline, model), ExactProjection(), CBFConfig()
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    risk_args = dict(rho=args.rho, active_tol=args.active_tol, D0=args.D0, kappa=args.kappa, alpha=args.alpha, p=args.p,
                     tau_h=args.tau_h, tau_s=args.tau_s, candidate_sigmas=args.candidate_sigmas)
    risk_class=RiskV2Config if args.risk_version=='v2' else RiskV1Config
    if args.risk_version=='v2':risk_args['terminal_distance_scale']=args.terminal_distance_scale
    risk = (risk_class(**risk_args, unfinished_mode='joint_sum' if args.risk_version == 'v1' else 'per_agent',
                         window_seconds=args.window_seconds, delta_prog=args.delta_prog,
                         tau_prog=args.tau_prog, task_distance_epsilon=args.task_distance_epsilon,
                         task_mask_temperature=args.task_mask_temperature, goal_tolerance=plant.goal_tolerance)
            if args.risk_version in ('v1', 'v1_1','v2') else SoftRiskConfig(**risk_args))
    if isinstance(risk,RiskV2Config) and args.horizon<plant.max_steps:
        raise ValueError('v2 episode training requires horizon >= environment max_steps; short prefixes are censored')
    if isinstance(risk, RiskV1Config) and not isinstance(risk,RiskV2Config) and args.horizon <= risk.window_steps(plant.dt):
        raise ValueError('v1 horizon must exceed the progress window in steps')
    if args.validation_every and args.validation_set is None:
        raise ValueError('--validation-every requires --validation-set')
    validation = None
    validation_digest = None
    if args.validation_set:
        starts, draws, validation_digest = load_fixed_inputs(args.validation_set, args.horizon)
        for start in starts:
            GiveWayEnv(plant).reset(start)
        validation = (jnp.asarray(starts, jnp.float64), jnp.asarray(draws))
    data_root = ROOT / 'datasets/give_way_si_short_v1/raw'
    rng = np.random.default_rng(args.seed)
    dataset_starts = None
    dataset_metadata = None
    if args.start_distribution == 'dataset':
        dataset_starts, dataset_metadata = load_dataset_starts(data_root.parent, 'train', plant)
        val_starts, _ = load_dataset_starts(data_root.parent, 'val', plant)
        calibration_starts, _, _ = load_fixed_inputs(args.calibration_set, args.horizon)
        require_population(calibration_starts, dataset_starts, 'calibration')
        require_population(np.asarray(validation[0]), val_starts, 'validation')
    pool = risk_start_pool(data_root, plant, risk.rho) if args.risk_start_fraction else np.empty((0, 2, 2))
    key = jax.random.PRNGKey(args.seed + 1000)
    calibration = None
    if args.calibration_set:
        def evaluate_calibration(starts, draws):
            return rollout_terms(
                params, field, projection, jnp.asarray(starts), jnp.asarray(draws),
                plant, cbf, risk)[2]
        if args.calibration_reference:
            expected = dict(method='c1_v0', baseline_checkpoint_sha256=checkpoint_hash,
                            baseline_provenance=provenance, environment=plant.to_dict(), cbf=cbf.to_dict(),
                            risk_version='R_risk_'+args.risk_version, risk=asdict(risk),
                            architecture=dict(hidden_dims=list(model.hidden_dims),layer_norm=model.layer_norm,
                                              insertion='post_safety_double_projection'))
            baseline_risk, calibration = reuse_fixed_calibration(args.calibration_reference,
                args.calibration_set,args.horizon,args.batch_size,expected,evaluate_calibration)
            print(dict(calibration_reused=str(args.calibration_reference), baseline_J_live=baseline_risk),flush=True)
        else:
            baseline_risk, calibration = calibrate_fixed_set(
                args.calibration_set, args.horizon, args.batch_size, evaluate_calibration,
                progress=lambda done, total: print(dict(calibration_completed=done, calibration_total=total), flush=True))
        initial, noise = sample_batch(rng, data_root, args.batch_size, args.horizon,
                                      key, pool, args.risk_start_fraction, dataset_starts)
    else:
        initial, noise = sample_batch(rng, data_root, args.batch_size, args.horizon, key, pool, args.risk_start_fraction)
        _, _, initial_risk = rollout_terms(params, field, projection, initial, noise, plant, cbf, risk)
        baseline_risk = float(jnp.mean(initial_risk))
        # A liveness constraint with zero measured baseline risk has no useful
        # calibration. Search independent train batches before declaring failure.
        for attempt in range(7):
            if baseline_risk > 0:
                break
            key, draw_key = jax.random.split(key)
            initial, noise = sample_batch(rng, data_root, args.batch_size, args.horizon, draw_key, pool, args.risk_start_fraction)
            _, _, initial_risk = rollout_terms(params, field, projection, initial, noise, plant, cbf, risk)
            baseline_risk = float(jnp.mean(initial_risk))
        if baseline_risk <= 0:
            raise RuntimeError('no active TRO risk found in eight train batches; revise risk coverage before training')
    epsilon = args.epsilon_fraction * baseline_risk
    optimizer, opt_state, dual = optax.adam(args.lr), None, PrimalDualState()
    opt_state = optimizer.init(params)
    history, start_update = [], 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    risk_metadata = asdict(risk)
    if args.risk_version == 'v1':
        risk_metadata.pop('unfinished_mode')  # Preserve historical v1 metadata.
    metadata = dict(method='c1_v0', risk_version='R_risk_'+args.risk_version, baseline_checkpoint=str(args.checkpoint.resolve()),
                    baseline_checkpoint_sha256=checkpoint_hash, baseline_provenance=provenance,
                    environment=plant.to_dict(), cbf=cbf.to_dict(), risk=risk_metadata,
                    training=dict(seed=args.seed, updates=args.updates, batch_size=args.batch_size, horizon=args.horizon,
                                  lr=args.lr, dual_lr=args.dual_lr, epsilon=epsilon, epsilon_fraction=args.epsilon_fraction,
                                  step_control=dict(protocol='fixed_batch_fixed_dual_nonincrease_v1',
                                                    max_backtracks=args.max_backtracks, contraction=.5,
                                                    rejected_optimizer_state='rollback', dual_risk='pre_step'),
                                  baseline_J_live=baseline_risk, risk_start_fraction=args.risk_start_fraction,
                                  validation_set=(str(args.validation_set.resolve()) if args.validation_set else None),
                                  validation_every=args.validation_every,
                                  risk_start_pool_size=len(pool)), architecture=dict(hidden_dims=list(model.hidden_dims),
                                  layer_norm=model.layer_norm, insertion='post_safety_double_projection'))
    if validation_digest is not None:
        metadata['training']['validation_sha256'] = validation_digest
    if dataset_metadata is not None:
        metadata['training']['start_distribution'] = dataset_metadata
    if calibration is not None:
        metadata['training']['calibration'] = calibration
    if args.calibration_reference:
        metadata['training']['calibration_reference'] = dict(path=str(args.calibration_reference.resolve()),
            sha256=hashlib.sha256(args.calibration_reference.read_bytes()).hexdigest(),
            verification='full_provenance_input_hash_and_first_batch_replay')
    if isinstance(risk,RiskV2Config):
        metadata['training']['rollout_protocol']='first_terminal_event_masked_full_episode'
    def save_checkpoint(completed):
        atomic_save(args.out_dir/'residual.pkl', dict(params=jax.device_get(params),
                    optimizer_state=jax.device_get(opt_state), dual=dual.dual,
                    rng_state=rng.bit_generator.state, jax_key=jax.device_get(key),
                    metadata=metadata, completed_updates=completed, history=history), binary=True)
        atomic_save(args.out_dir/'history.json', history)
    if args.resume:
        with (args.out_dir/'residual.pkl').open('rb') as handle:
            saved = pickle.load(handle)
        if saved['metadata'] != metadata:
            raise ValueError('resume configuration differs from the saved C1 run')
        params = jax.tree_util.tree_map(jnp.asarray, saved['params'])
        opt_state = jax.tree_util.tree_map(jnp.asarray, saved['optimizer_state'])
        dual = PrimalDualState(float(saved['dual']))
        rng.bit_generator.state = saved['rng_state']
        key = jnp.asarray(saved['jax_key'])
        history = restored_history(saved, args.out_dir/'history.json')
        start_update = int(saved['completed_updates'])
        if start_update != len(history):
            raise ValueError('resume checkpoint/history disagreement')
        atomic_save(args.out_dir/'history.json', history)
    else:
        atomic_save(args.out_dir/'config.json', metadata)
        save_checkpoint(0)
    validation_path = args.out_dir / 'validation.json'
    validation_rows = (json.loads(validation_path.read_text())
                       if args.resume and validation_path.exists() else [])
    selection_path = args.out_dir / 'selection.json'
    best_validation = (json.loads(selection_path.read_text())
                       if args.resume and selection_path.exists() else None)
    def validate(tag, phi, dual_value):
        nonlocal best_validation
        if validation is None:
            return
        current, safe, risks, details = rollout_terms(phi, field, projection, validation[0], validation[1], plant, cbf, risk, return_details=True)
        if isinstance(risk,RiskV2Config):
            mask=details['episode_mask'];count=jnp.maximum(mask.sum(),1)
            cone=jnp.sum(jnp.where(mask,details['cone_risk_t'],0.))/count
            stall=jnp.sum(jnp.where(mask,details['stall_risk_t'],0.))/count
        elif isinstance(risk, RiskV1Config):
            cone = jnp.mean(details['cone_risk_t'][:, risk.window_steps(plant.dt):])
            stall = jnp.nanmean(details['stall_risk_t'])
        else:
            cone, stall = jnp.mean(details['risk']), jnp.nan
        row=dict(update=tag, J_def=float(deviation_cost(current,safe,step_mask=details['episode_mask'])),
                 J_live=float(jnp.mean(risks)), R_cone=float(cone), R_stall=float(stall) if np.isfinite(float(stall)) else None, lambda_value=float(dual_value),
                 epsilon=epsilon, constraint=float(jnp.mean(risks))-epsilon,
                 constraint_satisfied=bool(jnp.mean(risks) <= epsilon))
        if isinstance(risk,RiskV2Config):
            from single_integrator.c1.termination import EVENTS
            codes=np.asarray(details['terminal_codes'])
            row.update(outcome_counts={name:int(np.sum(codes==code)) for name,code in EVENTS.items()},
                       mean_episode_steps=float(jnp.mean(details['episode_lengths'])),
                       mean_terminal_risk=float(jnp.mean(details['terminal_risk'])))
            if np.any(codes==0):
                raise RuntimeError('censored episode in full-task validation')
        validation_rows.append(row)
        atomic_save(args.out_dir/'validation.json', validation_rows)
        if better_feasible_candidate(row, best_validation):
            protocol=('first_event_validation_feasible_min_J_def' if isinstance(risk,RiskV2Config)
                      else 'fixed_horizon_validation_feasible_min_J_def')
            best_validation = dict(row, protocol=protocol)
            atomic_save(args.out_dir/'best_feasible.pkl', dict(params=jax.device_get(phi), metadata=metadata,
                        selection=best_validation), binary=True)
            atomic_save(selection_path, best_validation)
        print(dict(validation=row),flush=True)
    if not args.resume:
        validate('before_training', params, dual.dual)
    elif start_update == 0:
        validation_rows = [r for r in validation_rows if r['update'] != 'before_training']
        validate('before_training', params, dual.dual)
    elif validation is not None and start_update and (
            start_update == args.updates or (args.validation_every and start_update % args.validation_every == 0)):
        # A process can stop after checkpoint publication but before validation
        # or selection publication. Re-evaluate this same checkpoint on resume.
        validation_rows = [r for r in validation_rows if r['update'] != start_update]
        validate(start_update, params, dual.dual)
    for update in range(start_update, args.updates):
        if update:
            key, draw_key = jax.random.split(key)
            initial, noise = sample_batch(rng, data_root, args.batch_size, args.horizon, draw_key, pool, args.risk_start_fraction, dataset_starts)
        def objective(phi):
            current, safe, risks,mask = rollout_terms(phi, field, projection, initial, noise, plant, cbf, risk,return_mask=True)
            return primal_loss(current, safe, risks, dual.dual, epsilon,step_mask=mask)
        (loss, values), gradient = jax.value_and_grad(objective, has_aux=True)(params)
        live_gradient = jax.grad(lambda phi: jnp.mean(rollout_terms(phi, field, projection, initial, noise, plant, cbf, risk)[2]))(params)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((loss, values, gradient, live_gradient))):
            raise FloatingPointError('nonfinite objective/gradient; update was not applied')
        lambda_used = dual.dual
        params, opt_state, (after_loss, after_values), step = guarded_step(
            params, opt_state, gradient, optimizer, objective, (loss, values), args.max_backtracks)
        dual = dual_update(dual, values['J_live'], epsilon, args.dual_lr)
        row = dict(update=update, loss=float(loss), J_def=float(values['J_def']), J_live=float(values['J_live']),
                   constraint=float(values['constraint']), lambda_value=dual.dual,
                   lambda_used=lambda_used,
                   step_control=step,
                   post_step=dict(loss=float(after_loss), **{k:float(v) for k,v in after_values.items()}),
                   gradient_norm=float(optax.global_norm(gradient)),
                   live_gradient_norm=float(optax.global_norm(live_gradient)))
        history.append(row); print(row, flush=True)
        save_checkpoint(update + 1)
        if validation is not None and ((args.validation_every and (update+1) % args.validation_every == 0)
                                       or update+1 == args.updates):
            validate(update+1, params, dual.dual)


if __name__ == '__main__':
    main()
