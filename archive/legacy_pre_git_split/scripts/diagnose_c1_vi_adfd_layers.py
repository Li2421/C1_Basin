"""Focused layer probes for the archived VI R_CERT AD/FD mismatch.

Diagnostic only.  The production controller, rollout, monitor, and risk are
imported unchanged.  Frozen-branch probes are labeled and are not validation
of the original nonsmooth function.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.joint_frozen import controller_projection
from single_integrator.c1.risk.vi_r_cert import strict_atom_margins
from single_integrator.c1.rollout_vi_r_cert import rollout
from single_integrator.c1.train_deadlock_union import setup, noise, digest
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.cbf import project_velocity
from single_integrator.environment import GiveWayEnv


INITIAL = np.array([[-.4, .005], [.4, -.005]], np.float64)
NOISE_SEED = 84001
ROLLOUT_ID = 73000
DIRECTION_SEED = 2026091895
COTANGENT_SEED = 2026091903
ACTION = 100
MARGIN_ACTION = 200
EPSILON = 1e-10
SIGN_TOLERANCE = 1e-8
H_VALUES = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4,
                     1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign(value):
    return 0 if abs(value) <= SIGN_TOLERANCE else (1 if value > 0 else -1)


def relative(ad, fd):
    return abs(ad-fd) / max(abs(ad), abs(fd), EPSILON)


def scalar_table(fn, base, direction):
    compiled = jax.jit(fn)
    f0 = float(compiled(base))
    gradient = jax.jit(jax.grad(fn))(base)
    ad = float(jnp.vdot(gradient, direction))
    repeats = [float(compiled(base)) for _ in range(5)]
    rows = []
    for h in H_VALUES:
        fp = float(compiled(base+h*direction))
        fm = float(compiled(base-h*direction))
        central = (fp-fm)/(2*h)
        forward = (fp-f0)/h
        backward = (f0-fm)/h
        rows.append(dict(h=float(h), baseline=f0, plus=fp, minus=fm,
                         ad=ad, central_fd=central,
                         forward_one_sided=forward,
                         backward_one_sided=backward,
                         absolute_error=abs(ad-central),
                         relative_error=relative(ad, central),
                         ad_sign=sign(ad), fd_sign=sign(central)))
    return dict(ad=ad, gradient_norm=float(jnp.linalg.norm(gradient)),
                repeated_values=repeats,
                repeated_range=max(repeats)-min(repeats), rows=rows)


def projection_signature(target, A, b, cbf):
    projected, status = project_velocity(np.asarray(target), np.asarray(A),
                                         np.asarray(b), .5, cbf)
    p = projected.reshape(4)
    linear = np.asarray(A)@p-np.asarray(b)
    speed = np.linalg.norm(p.reshape(2, 2), axis=-1)
    slack = np.r_[linear, (.25-speed*speed)/2]
    active = slack < 1e-8
    return dict(status=status, projected=p.tolist(),
                min_linear_slack=float(linear.min()),
                max_speed=float(speed.max()),
                active_indices=np.flatnonzero(active).tolist(),
                active_count=int(active.sum()),
                active_rank=int(np.linalg.matrix_rank(np.vstack((
                    np.asarray(A),
                    np.array([[-p[0], -p[1], 0., 0.],
                              [0., 0., -p[2], -p[3]]])))[active])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('new diagnostic output required')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')

    params, field, plant, cbf, checkpoint = setup()
    env = GiveWayEnv(plant)
    goals = jnp.asarray(env.goals)
    draws = noise(NOISE_SEED, ROLLOUT_ID)
    offsets = jnp.zeros((850, 4), jnp.float64)
    terms, trace = jax.jit(lambda z: rollout(
        params, field, jnp.asarray(INITIAL), draws, plant, cbf, z))(offsets)

    rng = np.random.default_rng(DIRECTION_SEED)
    full_direction = rng.normal(size=(20, 4))
    full_direction /= np.linalg.norm(full_direction)
    actual_direction = jnp.asarray(full_direction[0])
    cot_rng = np.random.default_rng(COTANGENT_SEED)
    state_cotangent = cot_rng.normal(size=(2, 2))
    state_cotangent /= np.linalg.norm(state_cotangent)
    state_cotangent = jnp.asarray(state_cotangent)
    random_cotangent = cot_rng.normal(size=4)
    random_cotangent /= np.linalg.norm(random_cotangent)
    random_cotangent = jnp.asarray(random_cotangent)

    y = trace['w'][ACTION]
    A = trace['A'][ACTION]
    b = trace['b'][ACTION]
    p = trace['applied'][ACTION]
    before = trace['before'][ACTION]
    position_scalar = lambda value: jnp.vdot(
        before + plant.dt*controller_projection(value, A, b).reshape(2, 2),
        state_cotangent)
    random_scalar = lambda value: jnp.vdot(
        controller_projection(value, A, b), random_cotangent)

    projection = dict(
        target=np.asarray(y).tolist(), projected=np.asarray(p).tolist(),
        actual_direction=np.asarray(actual_direction).tolist(),
        state_cotangent=np.asarray(state_cotangent).tolist(),
        random_cotangent=np.asarray(random_cotangent).tolist(),
        baseline_signature=projection_signature(y, A, b, cbf),
        state_output=scalar_table(position_scalar, y, actual_direction),
        random_output=scalar_table(random_scalar, y, actual_direction),
        branches=[])
    for h in H_VALUES:
        projection['branches'].append(dict(
            h=float(h),
            plus=projection_signature(y+h*actual_direction, A, b, cbf),
            minus=projection_signature(y-h*actual_direction, A, b, cbf)))

    # Move along the active row normal to construct two diagnostic smooth points.
    # These alter only the isolated probe input and are not alternative controls.
    row = A[0]
    shift = .005*row/jnp.vdot(row, row)
    inside = y+shift
    outside = y-shift
    projection['smooth_inside_signature'] = projection_signature(inside, A, b, cbf)
    projection['smooth_outside_signature'] = projection_signature(outside, A, b, cbf)
    projection['smooth_inside'] = scalar_table(random_scalar, inside,
                                                actual_direction)
    projection['smooth_outside'] = scalar_table(random_scalar, outside,
                                                 actual_direction)

    # Strict-margin probes at one action in the winning event window.  Inputs
    # are varied independently to localize derivatives; the branch is explicit.
    t = MARGIN_ACTION
    states = jnp.concatenate((trace['before'][:1], trace['after']), axis=0)
    anchors = jnp.linalg.norm(states[t-39]-goals, axis=-1)
    base = dict(before=trace['before'][t], after=trace['after'][t],
                w=trace['w'][t], applied=trace['applied'][t],
                witnesses=trace['witnesses'][t], anchors=anchors)
    def margins(**updates):
        values = dict(base)
        values.update(updates)
        return strict_atom_margins(
            values['before'], values['after'], values['w'], values['applied'],
            values['witnesses'], goals, values['anchors'])

    c_direct = cot_rng.normal(size=3); c_direct /= np.linalg.norm(c_direct)
    c_vi = cot_rng.normal(size=(3, 8)); c_vi /= np.linalg.norm(c_vi)
    c_direct, c_vi = jnp.asarray(c_direct), jnp.asarray(c_vi)
    directions = {}
    def unit(name, shape):
        value = cot_rng.normal(size=shape); value /= np.linalg.norm(value)
        directions[name] = value.tolist()
        return jnp.asarray(value)
    anchor_direction = unit('anchor', (2,))
    w_direction = unit('w', (4,))
    witness_direction = unit('witness', (8, 4))
    state_direction = unit('state', (2, 2, 2))
    applied_direction = unit('applied', (4,))
    state_pair = jnp.stack((base['before'], base['after']))

    layer_tables = dict(
        direct_anchor=scalar_table(
            lambda z: jnp.vdot(margins(anchors=z)[:, 0], c_direct),
            base['anchors'], anchor_direction),
        vi_anchor=scalar_table(
            lambda z: jnp.vdot(margins(anchors=z)[:, 1:], c_vi),
            base['anchors'], anchor_direction),
        vi_preprojection_w=scalar_table(
            lambda z: jnp.vdot(margins(w=z)[:, 1:], c_vi),
            base['w'], w_direction),
        vi_witness_inputs=scalar_table(
            lambda z: jnp.vdot(margins(witnesses=z)[:, 1:], c_vi),
            base['witnesses'], witness_direction),
        direct_state=scalar_table(
            lambda z: jnp.vdot(margins(before=z[0], after=z[1])[:, 0],
                               c_direct), state_pair, state_direction),
        vi_state=scalar_table(
            lambda z: jnp.vdot(margins(before=z[0], after=z[1])[:, 1:],
                               c_vi), state_pair, state_direction),
        direct_executed_control=scalar_table(
            lambda z: jnp.vdot(margins(applied=z)[:, 0], c_direct),
            base['applied'], applied_direction))

    all_margins = np.asarray(margins())
    costs = np.logaddexp(0., -all_margins)/np.log(2.)
    sigmoid = 1/(1+np.exp(all_margins))
    branch_geometry = dict(
        margin_action=t, margins=all_margins.tolist(),
        positive=int(np.sum(all_margins > 1e-10)),
        negative=int(np.sum(all_margins < -1e-10)),
        zero=int(np.sum(np.abs(all_margins) <= 1e-10)),
        nonfinite=int(np.sum(~np.isfinite(all_margins))),
        cost_min=float(costs.min()), cost_max=float(costs.max()),
        sigmoid_min=float(sigmoid.min()), sigmoid_max=float(sigmoid.max()),
        direct_cotangent=np.asarray(c_direct).tolist(),
        vi_cotangent=np.asarray(c_vi).tolist(), directions=directions,
        anchor_state_index=t-39)

    sources = [
        'single_integrator/c1/risk/joint_frozen.py',
        'single_integrator/c1/differentiable_rollout.py',
        'single_integrator/c1/risk/vi_r_cert.py',
        'single_integrator/c1/rollout_vi_r_cert.py',
        'single_integrator/c1/termination.py',
        'single_integrator/cbf.py',
        'flowbc/giveway_flowbc_agent.py',
        'scripts/diagnose_c1_vi_adfd_layers.py']
    report = dict(protocol=dict(
        initial=INITIAL.tolist(), noise_seed=NOISE_SEED, rollout_id=ROLLOUT_ID,
        action=ACTION, margin_action=MARGIN_ACTION,
        direction_seed=DIRECTION_SEED, cotangent_seed=COTANGENT_SEED,
        h_values=H_VALUES.tolist(), epsilon=EPSILON,
        sign_tolerance=SIGN_TOLERANCE, jax_x64=True,
        flow_bc_internal='float32 by frozen baseline_sample',
        solver=cbf.to_dict(), environment=plant.to_dict(),
        backend=jax.default_backend(), checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        source_hashes={name: sha(ROOT/name) for name in sources}),
        baseline=dict(direct=float(terms['direct_R_CERT']),
                      augmented=float(terms['augmented_R_CERT']),
                      first_deadlock_step=int(terms['first_deadlock_step']),
                      active_event_index=np.asarray(
                          terms['active_event_index']).tolist()),
        projection_action_100=projection,
        frozen_branch_margin_layers=layer_tables,
        margin_branch_geometry=branch_geometry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.out, report)
    print(json.dumps(dict(
        out=str(args.out),
        kink_state_ad=projection['state_output']['ad'],
        kink_state_best=min(x['relative_error'] for x in
                            projection['state_output']['rows']),
        layer_best={name:min(x['relative_error'] for x in table['rows'])
                    for name, table in layer_tables.items()}), indent=2))


if __name__ == '__main__':
    main()
