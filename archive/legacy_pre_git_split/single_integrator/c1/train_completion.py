"""Versioned temporal-certificate primal-dual experiment, with frozen validation.

Run --prepare before training; keep validation and test noise separate from
fresh on-policy training noise. The held-out test is never read by training.
Legacy scores/checkpoints and the fixed-lambda experiments are untouched.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.rollout_completion import rollout
from single_integrator.c1.train import approved_checkpoint
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update
from single_integrator.c1.training.selection import better_feasible_candidate
from single_integrator.c1.training.step_control import guarded_step, finite
from single_integrator.cbf import CBFConfig
from single_integrator.environment import GiveWayEnv
from single_integrator.evaluate import ROOT, load_policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = ['single_integrator/c1/train_completion.py', 'single_integrator/c1/rollout_completion.py',
             'single_integrator/c1/risk/completion_certificate.py', 'single_integrator/c1/risk/progress_debt.py', 'single_integrator/c1/risk/temporal_certificate.py', 'single_integrator/c1/risk/joint_frozen.py',
             'single_integrator/c1/differentiable_rollout.py', 'single_integrator/c1/models/residual.py',
             'single_integrator/c1/termination.py', 'single_integrator/environment.py',
             'single_integrator/cbf.py', 'single_integrator/c1/training/step_control.py',
             'single_integrator/c1/training/primal_dual.py', 'single_integrator/c1/training/selection.py',
             'flowbc/giveway_flowbc_agent.py', 'scripts/audit_c1_joint_witness_risk.py']
    return {p: digest(ROOT/p) for p in paths}


def setup(seed=0, baseline_seed=0):
    jax.config.update('jax_enable_x64', True)
    checkpoint = ROOT/f'baseline_309_314/checkpoints/seed{baseline_seed}/ckpt_0025000.pkl'
    approved_checkpoint(checkpoint)
    baseline, provenance = load_policy(checkpoint)
    from single_integrator.environment import Config
    plant = replace(Config(**provenance['evaluation_environment']), terminate_on_deadlock=False)
    model = ResidualCorrection(hidden_dims=tuple(baseline.config['actor_hidden_dims']),
                               layer_norm=baseline.config['actor_layer_norm'])
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,4)), jnp.zeros((1,1)), jnp.zeros((1,20)))
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), params)
    field = ResidualFlowField(baseline, model)
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    return params, field, plant, CBFConfig(), checkpoint


def noise(seed, rid):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), rid)
    return jax.vmap(lambda t: jax.random.normal(jax.random.fold_in(key, t), (4,), dtype=jnp.float32))(
        jnp.arange(850, dtype=jnp.uint32))


def prepare(args):
    if args.sets.exists():
        raise FileExistsError('use a new sets path; frozen data must not be replaced')
    _, _, plant, _, _ = setup(args.seed, args.baseline_seed)
    rng = np.random.default_rng(args.data_seed)
    n = args.n_train+args.n_val+args.n_test
    x = rng.uniform(.55, 1.05, n)
    y = rng.uniform(-.025, .025, (n,2))
    starts = np.stack([np.c_[-x,y[:,0]], np.c_[x,y[:,1]]], 1)
    for x in starts:
        GiveWayEnv(plant).reset(x)
    if len(np.unique(starts.reshape(n,4), axis=0)) != n:
        raise ValueError('duplicated starts')
    # Explicitly exclude original benchmark and recent development/test sets.
    excluded = []
    suite = ROOT/'baseline_309_314/planning/wide_initial_states_200.npz'
    with np.load(suite) as z:
        for k in z.files:
            if z[k].ndim == 3 and z[k].shape[1:] == (2,2):
                excluded.extend(z[k].tolist())
    for name, keys in [('c1_four_objectives_multiseed', ('train_initials','test_initials')),
                       ('c1_focused_p_vs_pg_6seeds', ('training_initials','evaluation_initials')),
                       ('c1_frozen_unseen_64', ('initials',))]:
        path = ROOT/'results'/name/'protocol.json'
        if path.exists():
            p = json.loads(path.read_text())
            for k in keys:
                excluded.extend(p[k])
    distance = float(np.min(np.max(np.abs(starts[:,None]-np.asarray(excluded)[None]), axis=(2,3))))
    if distance == 0:
        raise ValueError('overlap with prior starts')
    pools = {}
    offset = 0
    for name, count in [('train',args.n_train), ('validation',args.n_val), ('test',args.n_test)]:
        pools[name] = [dict(rid=i, initial=starts[i].tolist()) for i in range(offset, offset+count)]
        offset += count
    data = dict(version='c1_completion_certificate_v1', data_seed=args.data_seed,
        distribution='shared |x| U(.55,1.05); independent y U(-.025,.025)',
        environment=plant.to_dict(), prefix_steps=0, score_steps=[0,850],
        calibration_noise_seeds=[70101,70102], validation_noise_seeds=[70201,70202],
        test_noise_seeds=[70301,70302], min_distance_from_checked_prior_starts=distance,
        pools=pools, purpose='Frozen split; tiny sets are engineering smoke only')
    args.sets.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.sets, data)
    print(dict(frozen_sets=str(args.sets), sha256=digest(args.sets)), flush=True)


def train(args):
    if args.out is None:
        raise ValueError('--out is required for training')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('use an empty V3 output directory; no implicit resume')
    params, field, plant, cbf, checkpoint = setup(args.seed, args.baseline_seed)
    data = json.loads(args.sets.read_text())
    if data['version'] != 'c1_completion_certificate_v1' or data['environment'] != plant.to_dict():
        raise ValueError('unsupported V3 sets/environment')
    fn = jax.jit(lambda phi,x,n: rollout(phi,field,x,n,plant,cbf,prefix_steps=0)[0])
    def measure(phi, pool, seeds):
        rows = []
        for item in pool:
            for draw in seeds:
                row = fn(phi, jnp.asarray(item['initial']), noise(draw,item['rid']))
                if not finite(row):
                    raise FloatingPointError('undefined/nonfinite risk; episode cannot be silently excluded')
                rows.append(dict(rid=item['rid'], noise_seed=draw,
                                 **{k:float(v) for k,v in row.items()}))
        means = {k:float(np.mean([r[k] for r in rows])) for k in ('J_live','J_def','timeout_bound','deadlock_bound','success','steps')}
        return dict(**means, episodes=rows)
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(version=data['version'], source_hashes=source_hashes(),
        baseline_sha256=digest(checkpoint), baseline_seed=args.baseline_seed, seed=args.seed,
        sets_sha256=digest(args.sets), sets=str(args.sets.resolve()), environment=plant.to_dict(),
        updates=args.updates, batch_size=args.batch_size, lr=args.lr,
        batch_sampling='uniform without replacement when batch <= pool', dual_lr=args.dual_lr,
        epsilon=args.epsilon, objective=args.objective, initial_dual=args.initial_dual,
        warm_start_sha256=digest(args.warm_start) if args.warm_start else None,
        warm_start_path=str(args.warm_start.resolve()) if args.warm_start else None,
        risk='max(failure_gate*deadline_debt, temporal_deadlock_certificate); full 42.5s; residual from step0',
        protocol='recoverable_deadlock; not the original first-event 400-case benchmark',
        feasibility_probe=args.objective=='risk_only')
    atomic_save(args.out/'config.json', config)
    calibration = measure(params, data['pools']['train'], data['calibration_noise_seeds'])
    epsilon = args.epsilon
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('nonpositive calibration risk')
    atomic_save(args.out/'calibration.json', dict(calibration, epsilon=epsilon))
    if args.warm_start:
        saved = pickle.loads(args.warm_start.read_bytes())
        previous = saved['config']
        if previous['baseline_sha256'] != digest(checkpoint) or previous['environment'] != plant.to_dict():
            raise ValueError('warm start baseline/environment mismatch')
        params = jax.tree_util.tree_map(jnp.asarray, saved['params'])
        if not finite(params):
            raise ValueError('invalid warm-start parameters')
        atomic_save(args.out/'initialization.json', dict(source_sha256=digest(args.warm_start),
            prior_objective=previous['objective'], prior_updates=saved.get('completed_updates'),
            prior_config=previous, note='Additional phase-I interaction budget; not a feasible C1 solution'))
    optimizer = optax.adam(args.lr)
    state, dual = optimizer.init(params), PrimalDualState(args.initial_dual)
    rng = np.random.default_rng(args.seed)
    history, validations, best = [], [], None
    def validate(step):
        nonlocal best
        row = measure(params, data['pools']['validation'], data['validation_noise_seeds'])
        row.update(update=step, epsilon=epsilon, constraint=row['J_live']-epsilon)
        validations.append(row)
        atomic_save(args.out/'validation.json', validations)
        if args.objective == 'constrained' and better_feasible_candidate(row, best):
            best = row
            atomic_save(args.out/'best_feasible.pkl', dict(params=jax.device_get(params),
                config=config, selection=row), binary=True)
        print(dict(validation={k:v for k,v in row.items() if k!='episodes'}), flush=True)
    def save(step):
        atomic_save(args.out/'checkpoint.pkl', dict(params=jax.device_get(params),
            optimizer=jax.device_get(state), dual=dual.dual, rng_state=rng.bit_generator.state,
            config=config, epsilon=epsilon, completed_updates=step, history=history), binary=True)
        atomic_save(args.out/'history.json', history)
    save(0)
    validate(0)
    for step in range(args.updates):
        indices = rng.choice(len(data['pools']['train']), size=args.batch_size,
                             replace=args.batch_size > len(data['pools']['train']))
        # Fresh on-policy noise; frozen calibration/validation noise is never trained on.
        draw = int(rng.integers(1000000, 2**31-1))
        items = [data['pools']['train'][int(i)] for i in indices]
        noises = [noise(draw, item['rid']) for item in items]
        def objective(phi):
            terms = [fn(phi,jnp.asarray(item['initial']),n) for item,n in zip(items,noises)]
            live = jnp.mean(jnp.stack([r['J_live'] for r in terms]))
            deviation = jnp.mean(jnp.stack([r['J_def'] for r in terms]))
            loss = live if args.objective == 'risk_only' else deviation+dual.dual*(live-epsilon)
            return loss, dict(J_live=live,J_def=deviation,constraint=live-epsilon)
        before, gradient = jax.value_and_grad(objective, has_aux=True)(params)
        used = dual.dual
        params, state, after, decision = guarded_step(params,state,gradient,optimizer,objective,before)
        if args.objective == 'constrained':
            dual = dual_update(dual,before[1]['J_live'],epsilon,args.dual_lr)
        history.append(dict(update=step, rids=[i['rid'] for i in items], noise_seed=draw,
            lambda_used=used, lambda_after=dual.dual, gradient_norm=float(optax.global_norm(gradient)),
            pre=dict(loss=float(before[0]),**{k:float(v) for k,v in before[1].items()}),
            post=dict(loss=float(after[0]),**{k:float(v) for k,v in after[1].items()}), step_control=decision))
        save(step+1)
        print(history[-1], flush=True)
        if (step+1) % args.validation_every == 0 or step+1 == args.updates:
            validate(step+1)
    # Final whole-train evaluation separates optimization from validation gap.
    atomic_save(args.out/'final_train.json', measure(params,data['pools']['train'],data['calibration_noise_seeds']))
    atomic_save(args.out/'complete.json', dict(updates=len(history), validation_feasible=best is not None,
        objective=args.objective, warning='Completion is not a liveness or feasibility certificate'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--initial-dual', type=float, default=1.)
    p.add_argument('--sets', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--baseline-seed', type=int, choices=(0,1), default=0)
    p.add_argument('--data-seed', type=int, default=2026091617)
    p.add_argument('--n-train', type=int, default=32)
    p.add_argument('--n-val', type=int, default=16)
    p.add_argument('--n-test', type=int, default=64)
    p.add_argument('--updates', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--dual-lr', type=float, default=.01)
    p.add_argument('--epsilon', type=float, default=.05)
    p.add_argument('--validation-every', type=int, default=10)
    p.add_argument('--objective', choices=('constrained','risk_only'), default='constrained',
                   help='risk_only diagnoses empirical attainability; never a C1 feasible model')
    args = p.parse_args()
    if min(args.n_train,args.n_val,args.n_test,args.updates,args.batch_size,args.validation_every)<1:
        p.error('counts must be positive')
    if not all(np.isfinite(x) and x>0 for x in (args.lr,args.dual_lr)) or not 0<args.epsilon<1:
        p.error('invalid learning rate or absolute epsilon')
    if not np.isfinite(args.initial_dual) or args.initial_dual < 0:
        p.error('invalid initial dual')
    prepare(args) if args.prepare else train(args)


if __name__ == '__main__':
    main()



