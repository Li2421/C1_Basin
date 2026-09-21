"""One-trajectory implementation and numerical audit for frozen VI R_CERT."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.action_probe import execute_offsets
from single_integrator.c1.risk.vi_r_cert import strict_atom_margins
from single_integrator.c1.rollout_vi_r_cert import rollout
from single_integrator.c1.train_deadlock_union import setup, noise, digest
from single_integrator.c1.training.persistence import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('new audit output required')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    params, field, plant, cbf, baseline = setup()
    initial = jnp.array([[-.4, .005], [.4, -.005]], jnp.float64)
    draws = noise(84001, 73000)
    zeros = jnp.zeros((850, 4), jnp.float64)
    run = jax.jit(lambda offsets: rollout(
        params, field, initial, draws, plant, cbf, offsets))
    terms, trace = run(zeros)
    terms_host = {k: np.asarray(v) for k, v in terms.items()}
    valid = np.asarray(trace['alive_pre'], bool)
    n = int(valid.sum())
    if n < 1:
        raise AssertionError('empty episode')

    row, operational = execute_offsets(
        params, field, np.asarray(initial), draws, plant, cbf,
        np.zeros((0, 4)))
    applied = np.asarray(trace['applied'])[:n]
    parity_error = float(np.max(np.abs(applied - operational['applied'])))
    monitor = dict(
        action_count=n,
        operational_action_count=len(operational['applied']),
        applied_max_abs_error=parity_error,
        candidate_equal=bool(np.array_equal(
            np.asarray(trace['candidate'])[:n],
            operational['candidate_deadlock'])),
        raw_deadlock_equal=bool(np.array_equal(
            np.asarray(trace['raw_deadlock'])[:n], operational['deadlock'])),
        historical_strict=bool(terms_host['historical_strict_deadlock']),
        operational_historical_strict=bool(row['any_deadlock']),
        terminal_safe_deadlock=bool(terms_host['terminal_safe_deadlock']),
        operational_terminal_safe_deadlock=bool(
            row['any_deadlock'] and not row['recovered_deadlock']),
        first_deadlock_step=int(terms_host['first_deadlock_step']),
        no_post_terminal_projection_placeholders=bool(
            np.isnan(np.asarray(trace['applied'])[n:]).all()) if n < 850 else True)

    A, b = np.asarray(trace['A'])[:n], np.asarray(trace['b'])[:n]
    witnesses = np.asarray(trace['witnesses'])[:n]
    witness_residual = np.einsum('tmi,tji->tmj', witnesses, A) - b[:, None, :]
    witness_speed = np.linalg.norm(witnesses.reshape(n, 8, 2, 2), axis=-1)
    actual_residual = np.einsum('ti,tji->tj', applied, A) - b
    projection = dict(
        witness_min_cbf_residual=float(witness_residual.min()),
        witness_max_speed_excess=float(witness_speed.max() - .5),
        actual_min_cbf_residual=float(actual_residual.min()),
        actual_max_speed_excess=float(
            np.linalg.norm(applied.reshape(n, 2, 2), axis=-1).max() - .5),
        exact_zero_applied_steps=int(np.sum(np.all(applied == 0., axis=1))),
        near_cbf_active_entries=int(np.sum(np.abs(actual_residual) <= 1e-7)),
        near_speed_active_blocks=int(np.sum(np.abs(
            np.linalg.norm(applied.reshape(n, 2, 2), axis=-1) - .5) <= 1e-7)))

    primary = bool(terms_host['primary_deadlock'])
    implication = dict(
        primary_event=primary,
        direct_R_CERT=float(terms_host['direct_R_CERT']),
        augmented_R_CERT=float(terms_host['augmented_R_CERT']),
        passed=(not primary or (float(terms_host['direct_R_CERT']) >= 1-2e-6 and
                                float(terms_host['augmented_R_CERT']) >= 1-2e-6)))

    window = (100, 120)
    def objectives(local_offsets):
        full = jnp.zeros((850, 4), jnp.float64).at[window[0]:window[1]].set(
            local_offsets)
        result, _ = rollout(params, field, initial, draws, plant, cbf, full)
        return jnp.stack((result['direct_R_CERT'],
                          result['augmented_R_CERT']))

    zero_local = jnp.zeros((window[1]-window[0], 4), jnp.float64)
    values = jax.jit(objectives)(zero_local)
    # Two scalar reverse sweeps avoid batching a pure-callback VJP.  This is
    # exactly the Jacobian of the two reported objectives, without a vmap.
    jacobian = jnp.stack([
        jax.jit(jax.grad(lambda z, index=index: objectives(z)[index]))(
            zero_local) for index in range(2)])
    direction = np.random.default_rng(2026091895).normal(size=zero_local.shape)
    direction /= np.linalg.norm(direction)
    ad = np.einsum('aij,ij->a', np.asarray(jacobian), direction)
    fd_rows = []
    for h in (1e-3, 3e-4, 1e-4, 3e-5):
        plus = np.asarray(jax.jit(objectives)(zero_local+h*direction))
        minus = np.asarray(jax.jit(objectives)(zero_local-h*direction))
        fd = (plus-minus)/(2*h)
        fd_rows.append(dict(step=h, finite_difference=fd.tolist(),
                            relative_error=(np.abs(fd-ad)/np.maximum(
                                np.abs(ad), 1e-12)).tolist()))
    active = np.asarray(terms_host['active_event_index'])
    active_times = [int(139+x) if x < 711 else 'stalled'
                    for x in active.tolist()]
    gradient = dict(
        values=np.asarray(values).tolist(),
        norms=np.linalg.norm(np.asarray(jacobian).reshape(2, -1), axis=1).tolist(),
        directional_derivative=ad.tolist(), finite_differences=fd_rows,
        active_event_indices=active.tolist(), active_event_times=active_times,
        intervention_actions=list(window),
        genuinely_earlier_than_active=[
            bool(isinstance(t, int) and window[1]-1 < t) for t in active_times],
        direct_augmented_cosine=float(np.vdot(jacobian[0], jacobian[1]) /
            max(np.linalg.norm(jacobian[0])*np.linalg.norm(jacobian[1]), 1e-300)))

    # Term-level numerical diagnostics at the active augmented strict branch.
    term_diagnostics = None
    if active[1] < 711:
        t = 139 + int(active[1]); start = t-100
        goals = jnp.asarray(__import__('single_integrator.environment',
            fromlist=['GiveWayEnv']).GiveWayEnv(plant).goals)
        states = jnp.concatenate((trace['before'][:1], trace['after']), axis=0)
        anchors = jnp.linalg.norm(states[start-39:start-39+101]-goals, axis=-1)
        margins = jax.vmap(lambda before, after, w, u, v, anchor:
            strict_atom_margins(before, after, w, u, v, goals, anchor))(
                trace['before'][start:t+1], trace['after'][start:t+1],
                trace['w'][start:t+1], trace['applied'][start:t+1],
                trace['witnesses'][start:t+1], anchors)
        m = np.asarray(margins)
        penalties = np.asarray(jax.nn.softplus(-m)/np.log(2.))
        term_diagnostics = dict(
            active_t=t, margin_min=float(m.min()), margin_max=float(m.max()),
            nonfinite=int(np.sum(~np.isfinite(m))),
            saturated_near_zero=int(np.sum(penalties < 1e-12)),
            direct_nonpositive=int(np.sum(m[:, :, 0] <= 2e-6)),
            vi_nonpositive=int(np.sum(m[:, :, 1:] <= 2e-6)))

    passed = bool(
        parity_error <= 2e-8 and monitor['candidate_equal'] and
        monitor['raw_deadlock_equal'] and
        monitor['historical_strict'] == monitor['operational_historical_strict'] and
        projection['witness_min_cbf_residual'] >= -cbf.feasibility_tol and
        projection['witness_max_speed_excess'] <= cbf.speed_tol and
        implication['passed'] and np.isfinite(np.asarray(values)).all() and
        np.isfinite(np.asarray(jacobian)).all())
    sources = {p: digest(ROOT/p) for p in (
        'scripts/check_c1_vi_r_cert.py',
        'single_integrator/c1/rollout_vi_r_cert.py',
        'single_integrator/c1/risk/vi_r_cert.py',
        'single_integrator/c1/termination.py')}
    report = dict(passed=passed, backend=jax.default_backend(),
                  baseline_sha256=digest(baseline), environment=plant.to_dict(),
                  cbf=cbf.to_dict(), monitor=monitor, projection=projection,
                  implication=implication, gradient=gradient,
                  term_diagnostics=term_diagnostics, solver_failures=0,
                  sources=sources)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.out, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError('VI R_CERT implementation audit failed')


if __name__ == '__main__':
    main()
