"""One-sided local utility test of the current VJP for frozen augmented R_CERT.

Diagnostic only.  This imports the production rollout and VJP unchanged and
evaluates the archived one-trajectory failing scalar with fixed randomness.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import nnls

from single_integrator.c1.risk.vi_r_cert import strict_atom_margins
from single_integrator.c1.rollout_vi_r_cert import rollout, WITNESS_TARGETS
from single_integrator.c1.train_deadlock_union import setup, noise, digest
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.cbf import project_velocity
from single_integrator.environment import GiveWayEnv


INITIAL = np.array([[-.4, .005], [.4, -.005]], np.float64)
NOISE_SEED = 84001
ROLLOUT_ID = 73000
WINDOW = (100, 120)
H_VALUES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
REPEATS = 3
TAU_S = 1e-8
ACTIVE_TOL = 1e-8
TIE_TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(items):
    state = hashlib.sha256()
    for value in items:
        x = np.ascontiguousarray(value)
        state.update(str(x.dtype).encode())
        state.update(np.asarray(x.shape, np.int64).tobytes())
        state.update(x.tobytes())
    return state.hexdigest()


def json_ready(value):
    """Convert NumPy containers/scalars without changing numeric values."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def active_arrays(trace, n):
    A = np.asarray(trace['A'])[:n]
    b = np.asarray(trace['b'])[:n]

    def one(values):
        values = np.asarray(values)[:n]
        linear = np.einsum('tmi,ti->tm', A, values)-b
        speed_slack = (.25-np.sum(values.reshape(n, 2, 2)**2, axis=-1))/2
        return np.concatenate((linear < ACTIVE_TOL,
                               speed_slack < ACTIVE_TOL), axis=-1)

    def many(values):
        values = np.asarray(values)[:n]
        linear = np.einsum('tmi,tji->tjm', A, values)-b[:, None, :]
        speed_slack = (.25-np.sum(values.reshape(n, 8, 2, 2)**2,
                                  axis=-1))/2
        return np.concatenate((linear < ACTIVE_TOL,
                               speed_slack < ACTIVE_TOL), axis=-1)

    return dict(first=one(trace['safe']), execution=one(trace['applied']),
                witness=many(trace['witnesses']))


def kkt_one(target, p, A, b):
    target, p, A, b = map(np.asarray, (target, p, A, b))
    p = p.reshape(4)
    linear = A@p-b
    speed_slack = (.25-np.sum(p.reshape(2, 2)**2, axis=-1))/2
    ball_rows = np.zeros((2, 4))
    for agent in range(2):
        ball_rows[agent, 2*agent:2*agent+2] = -p[2*agent:2*agent+2]
    J = np.vstack((A, ball_rows))
    slack = np.r_[linear, speed_slack]
    active = slack < ACTIVE_TOL
    Ja = J[active]
    if len(Ja):
        multiplier = nnls(Ja.T, p-target, maxiter=10000)[0]
        stationarity = np.linalg.norm(p-target-Ja.T@multiplier, ord=np.inf)
        complementarity = np.max(np.abs(multiplier*slack[active]))
    else:
        stationarity = np.linalg.norm(p-target, ord=np.inf)
        complementarity = 0.
    return (float(linear.min()),
            float(np.linalg.norm(p.reshape(2, 2), axis=-1).max()-.5),
            float(stationarity), float(complementarity),
            int(active.sum()), int(np.linalg.matrix_rank(Ja)))


def solver_summary(trace, n, cbf):
    A, b = np.asarray(trace['A'])[:n], np.asarray(trace['b'])[:n]
    w = np.asarray(trace['w'])[:n]
    applied = np.asarray(trace['applied'])[:n]
    witnesses = np.asarray(trace['witnesses'])[:n]
    targets = np.asarray(WITNESS_TARGETS, np.float64)
    rows = []
    failures = 0
    for t in range(n):
        try:
            rows.append(kkt_one(w[t], applied[t], A[t], b[t]))
            for j in range(8):
                rows.append(kkt_one(targets[j], witnesses[t, j], A[t], b[t]))
        except Exception:
            failures += 1
    values = np.asarray(rows) if rows else np.empty((0, 6))
    action_status = None
    try:
        replay, status = project_velocity(w[WINDOW[0]], A[WINDOW[0]],
                                          b[WINDOW[0]], .5, cbf)
        action_status = dict(
            status=status,
            replay_max_abs_error=float(np.max(np.abs(
                replay.reshape(4)-applied[WINDOW[0]]))))
    except Exception as exc:
        failures += 1
        action_status = dict(status='failure', error=repr(exc))
    return dict(
        projection_count=int(len(rows)), diagnostic_failures=failures,
        production_solver_failures=0,
        min_linear_slack=float(values[:, 0].min()) if len(values) else None,
        max_speed_excess=float(values[:, 1].max()) if len(values) else None,
        max_stationarity=float(values[:, 2].max()) if len(values) else None,
        max_complementarity=float(values[:, 3].max()) if len(values) else None,
        max_active_count=int(values[:, 4].max()) if len(values) else None,
        min_active_rank=int(values[:, 5].min()) if len(values) else None,
        action_100_execution_replay=action_status)


def margin_branch_arrays(trace, terms, goals):
    winner = int(np.asarray(terms['active_event_index'])[1])
    if winner >= 711:
        return dict(kind='stalled', winner=winner, arrays=None,
                    summary=dict(certificate_count=378))
    t = 139+winner
    start = t-100
    before = np.asarray(trace['before'])[start:t+1]
    after = np.asarray(trace['after'])[start:t+1]
    w = np.asarray(trace['w'])[start:t+1]
    applied = np.asarray(trace['applied'])[start:t+1]
    witnesses = np.asarray(trace['witnesses'])[start:t+1]
    states = np.concatenate((np.asarray(trace['before'])[:1],
                             np.asarray(trace['after'])), axis=0)
    anchors = np.linalg.norm(states[start-39:start-39+101]-goals, axis=-1)
    margins = np.asarray(jax.vmap(
        lambda x, y, z, u, v, anchor: strict_atom_margins(
            x, y, z, u, v, jnp.asarray(goals), anchor))(
                jnp.asarray(before), jnp.asarray(after), jnp.asarray(w),
                jnp.asarray(applied), jnp.asarray(witnesses),
                jnp.asarray(anchors)))
    sign_code = np.where(margins > 1e-10, 1,
                         np.where(margins < -1e-10, -1, 0)).astype(np.int8)

    distances = np.linalg.norm(after-goals, axis=-1)
    direct_goal_arg = np.argmax(distances, axis=-1).astype(np.int8)
    direct_goal_tie = (np.abs(distances[:, 0]-distances[:, 1]) <= 1e-10)
    progress_abs = np.abs(anchors-distances)
    direct_progress_arg = np.argmax(progress_abs, axis=-1).astype(np.int8)
    direct_progress_tie = (np.abs(progress_abs[:, 0]-progress_abs[:, 1])
                           <= 1e-10)
    direct_speed_outside = (np.linalg.norm(
        applied.reshape(101, 2, 2), axis=-1) > .025)

    midpoint = (w[:, None, :]+witnesses)/2
    blocks = midpoint.reshape(101, 8, 2, 2)
    centers = (goals[None, :, :]-before[:, None, :, :])/.05
    radial = np.linalg.norm(blocks-centers, axis=-1)
    goal_block_distance = np.maximum(.08/.05-radial, 0.)
    goal_squared = goal_block_distance**2
    vi_goal_arg = np.argmin(goal_squared, axis=-1).astype(np.int8)
    vi_goal_tie = (np.abs(goal_squared[:, :, 0]-goal_squared[:, :, 1])
                   <= 1e-10)
    lower = np.maximum(anchors-.01, 0.)/.05
    upper = (anchors+.01)/.05
    vi_progress_region = np.where(
        radial < lower[:, None, :], -1,
        np.where(radial > upper[:, None, :], 1, 0)).astype(np.int8)
    vi_speed_outside = (np.linalg.norm(blocks, axis=-1) > .025)
    arrays = dict(
        margin_sign=sign_code, direct_goal_arg=direct_goal_arg,
        direct_goal_tie=direct_goal_tie,
        direct_progress_arg=direct_progress_arg,
        direct_progress_tie=direct_progress_tie,
        direct_speed_outside=direct_speed_outside,
        vi_goal_arg=vi_goal_arg, vi_goal_tie=vi_goal_tie,
        vi_progress_region=vi_progress_region,
        vi_speed_outside=vi_speed_outside)
    summary = dict(
        active_t=t, certificate_count=2727,
        margin_positive=int(np.sum(sign_code == 1)),
        margin_negative=int(np.sum(sign_code == -1)),
        margin_zero=int(np.sum(sign_code == 0)),
        direct_goal_ties=int(direct_goal_tie.sum()),
        direct_progress_ties=int(direct_progress_tie.sum()),
        vi_goal_ties=int(vi_goal_tie.sum()),
        branch_hash=array_hash(arrays.values()))
    return dict(kind='strict', winner=winner, arrays=arrays, summary=summary)


def trace_summary(terms, trace, goals, cbf):
    n = int(np.asarray(terms['action_count']))
    active = active_arrays(trace, n)
    branch = margin_branch_arrays(trace, terms, goals)
    strict = np.asarray(terms['strict_augmented_R_e'])
    all_risk = np.r_[strict, float(terms['stalled_augmented_R_e'])]
    finite = np.isfinite(all_risk)
    maximum = np.max(all_risk[finite])
    winners = np.flatnonzero(finite & (np.abs(all_risk-maximum) <= TIE_TOL))
    ordered = np.sort(all_risk[finite])
    core = [np.asarray(trace[name])[:n] for name in
            ('before', 'after', 'w', 'safe', 'applied', 'witnesses', 'A', 'b')]
    core.extend(np.asarray(trace[name])[:n] for name in
                ('alive_pre', 'latch_pre', 'candidate', 'raw_deadlock',
                 'event_code'))
    return dict(
        n=n, active=active, margin=branch,
        summary=dict(
            action_count=n, terminal_code=int(terms['terminal_code']),
            first_deadlock_step=int(terms['first_deadlock_step']),
            outer_winner=int(np.asarray(terms['active_event_index'])[1]),
            outer_tie_indices=winners.tolist(),
            outer_tie_count=int(len(winners)),
            outer_top_gap=(float(ordered[-1]-ordered[-2])
                           if len(ordered) > 1 else None),
            strict_guard_indices=(139+np.flatnonzero(
                np.asarray(terms['strict_guards']))).tolist(),
            latch_true_indices=np.flatnonzero(
                np.asarray(trace['latch_pre'])[:n]).tolist(),
            candidate_indices=np.flatnonzero(
                np.asarray(trace['candidate'])[:n]).tolist(),
            raw_deadlock_indices=np.flatnonzero(
                np.asarray(trace['raw_deadlock'])[:n]).tolist(),
            active_hashes={name:array_hash((value,))
                           for name, value in active.items()},
            margin=branch['summary'], trace_hash=array_hash(core),
            nonfinite_counts={name:int(np.sum(~np.isfinite(
                np.asarray(trace[name])[:n]))) for name in
                ('before', 'after', 'w', 'safe', 'applied', 'witnesses',
                 'A', 'b')},
            solver=solver_summary(trace, n, cbf)))


def differences(left, right):
    n = min(left['n'], right['n'])
    result = dict(
        common_action_count=n,
        unequal_termination_lengths=left['n'] != right['n'],
        action_count_delta=left['n']-right['n'],
        terminal_code_changed=(left['summary']['terminal_code'] !=
                               right['summary']['terminal_code']),
        outer_winner_changed=(left['summary']['outer_winner'] !=
                              right['summary']['outer_winner']),
        guard_indices_changed=(left['summary']['strict_guard_indices'] !=
                               right['summary']['strict_guard_indices']),
        latch_indices_changed=(left['summary']['latch_true_indices'] !=
                               right['summary']['latch_true_indices']),
        candidate_indices_changed=(left['summary']['candidate_indices'] !=
                                   right['summary']['candidate_indices']),
        raw_deadlock_indices_changed=(
            left['summary']['raw_deadlock_indices'] !=
            right['summary']['raw_deadlock_indices']))
    for name in ('first', 'execution'):
        changed = np.any(left['active'][name][:n] !=
                         right['active'][name][:n], axis=-1)
        result[name+'_active_changed_steps'] = np.flatnonzero(changed).tolist()
    witness_changed = np.any(left['active']['witness'][:n] !=
                             right['active']['witness'][:n], axis=-1)
    result['witness_active_changed_steps'] = np.flatnonzero(
        np.any(witness_changed, axis=-1)).tolist()
    result['witness_active_changed_pairs'] = np.argwhere(
        witness_changed).tolist()
    if (left['margin']['kind'] == right['margin']['kind'] == 'strict' and
            left['margin']['winner'] == right['margin']['winner']):
        margin_changes = {}
        for name in left['margin']['arrays']:
            margin_changes[name] = int(np.sum(
                left['margin']['arrays'][name] !=
                right['margin']['arrays'][name]))
        result['margin_branch_changes'] = margin_changes
    else:
        result['margin_branch_changes'] = None
    return result


def npz_trace(prefix, terms, trace, target):
    n = int(np.asarray(terms['action_count']))
    for name in ('before', 'after', 'w', 'safe', 'applied', 'witnesses',
                 'A', 'b', 'alive_pre', 'latch_pre', 'candidate',
                 'raw_deadlock', 'event_code'):
        target[prefix+'__'+name] = np.asarray(trace[name])[:n]
    for name in ('strict_augmented_R_e', 'strict_guards',
                 'active_event_index', 'terminal_code',
                 'first_deadlock_step', 'action_count',
                 'augmented_R_CERT'):
        target[prefix+'__term__'+name] = np.asarray(terms[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('new output directory required')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')

    params, field, plant, cbf, checkpoint = setup()
    draws = noise(NOISE_SEED, ROLLOUT_ID)
    initial = jnp.asarray(INITIAL)
    goals = np.asarray(GiveWayEnv(plant).goals)
    zero = jnp.zeros((WINDOW[1]-WINDOW[0], 4), jnp.float64)

    def full(local):
        offsets = jnp.zeros((850, 4), jnp.float64).at[
            WINDOW[0]:WINDOW[1]].set(local)
        return rollout(params, field, initial, draws, plant, cbf, offsets)

    def objective(local):
        return full(local)[0]['augmented_R_CERT']

    compiled_full = jax.jit(full)
    gradient = jax.jit(jax.grad(objective))(zero)
    gradient_host = np.asarray(gradient)
    gradient_norm = float(np.linalg.norm(gradient_host))
    if not np.isfinite(gradient_host).all():
        gradient_status = 'nonfinite'
        directions = {}
    elif gradient_norm == 0.:
        gradient_status = 'zero'
        directions = {}
    else:
        gradient_status = 'finite_nonzero'
        directions = dict(minus=-gradient_host/gradient_norm,
                          plus=gradient_host/gradient_norm)

    args.out_dir.mkdir(parents=True)
    trace_store = {}

    def evaluate_group(name, local):
        values, raw = [], []
        for repeat in range(REPEATS):
            terms, trace = compiled_full(jnp.asarray(local))
            value = float(terms['augmented_R_CERT'])
            values.append(value)
            raw.append((jax.tree_util.tree_map(np.asarray, terms),
                        jax.tree_util.tree_map(np.asarray, trace)))
        median = float(np.median(values))
        selected = min(range(REPEATS), key=lambda index:
                       (abs(values[index]-median), index))
        repeat_summaries = []
        for repeat, (terms, trace) in enumerate(raw):
            n = int(terms['action_count'])
            repeated_core = [np.asarray(trace[key])[:n] for key in
                             ('before', 'after', 'w', 'safe', 'applied',
                              'witnesses', 'A', 'b', 'alive_pre',
                              'latch_pre', 'candidate', 'raw_deadlock',
                              'event_code')]
            repeat_summaries.append(dict(
                repeat=repeat, F=values[repeat], action_count=n,
                terminal_code=int(terms['terminal_code']),
                first_deadlock_step=int(terms['first_deadlock_step']),
                outer_winner=int(terms['active_event_index'][1]),
                trace_hash=array_hash(repeated_core)))
        terms, trace = raw[selected]
        detail = trace_summary(terms, trace, goals, cbf)
        npz_trace(name, terms, trace, trace_store)
        if detail['margin']['arrays'] is not None:
            for key, value in detail['margin']['arrays'].items():
                trace_store[name+'__margin__'+key] = value
        return dict(name=name, values=values, median=median,
                    within_range=float(max(values)-min(values)),
                    selected_repeat=selected,
                    repeat_summaries=repeat_summaries,
                    terms=terms, trace=trace, detail=detail)

    baseline = evaluate_group('baseline', np.zeros_like(gradient_host))
    cases = []
    if gradient_status == 'finite_nonzero':
        for direction_name, direction in directions.items():
            for h in H_VALUES:
                cases.append(evaluate_group(
                    f'{direction_name}_h_{h:.0e}', h*direction))

    all_groups = [baseline]+cases
    spread = max(group['within_range'] for group in all_groups)
    observed = [value for group in all_groups for value in group['values']]
    scale = max(1., max(abs(value) for value in observed))
    machine_epsilon = np.finfo(np.float64).eps
    e_F = max(10*spread, 100*machine_epsilon*scale)

    baseline_applied = np.asarray(baseline['trace']['applied'])
    baseline_n = baseline['detail']['n']
    public_rows = []
    case_lookup = {}
    for case in cases:
        direction_name = case['name'].split('_h_')[0]
        h = float(case['name'].split('_h_')[1])
        delta = case['median']-baseline['median']
        slope = delta/h
        uncertainty = 2*e_F/h
        resolved = uncertainty <= .1*max(abs(slope), TAU_S)
        lower, upper = slope-uncertainty, slope+uncertainty
        if not resolved:
            slope_sign = 'unresolved'
        elif upper < -TAU_S:
            slope_sign = 'negative'
        elif lower > TAU_S:
            slope_sign = 'positive'
        elif lower >= -TAU_S and upper <= TAU_S:
            slope_sign = 'zero'
        else:
            slope_sign = 'indeterminate'
        n = case['detail']['n']
        common_n = min(n, baseline_n)
        stop = min(WINDOW[1], n, baseline_n)
        if stop > WINDOW[0]:
            window_change = (np.asarray(case['trace']['applied'])[
                WINDOW[0]:stop]-baseline_applied[WINDOW[0]:stop])
        else:
            window_change = np.empty((0, 4))
        prefix_change = (np.asarray(case['trace']['applied'])[:common_n]-
                         baseline_applied[:common_n])
        window_norm = float(np.linalg.norm(window_change))
        prefix_norm = float(np.linalg.norm(prefix_change))
        row = dict(
            direction=direction_name, h=h,
            F0=baseline['median'], Fh=case['median'],
            F0_repeats=baseline['values'], Fh_repeats=case['values'],
            delta_F=delta, slope=slope,
            slope_uncertainty=uncertainty,
            slope_interval=[lower, upper], resolved=resolved,
            sign=slope_sign, within_input_range=case['within_range'],
            intervention_executed_change_l2=window_norm,
            intervention_change_over_h=window_norm/h,
            common_prefix_executed_change_l2=prefix_norm,
            common_prefix_change_over_h=prefix_norm/h,
            exact_zero_executed_effect=bool(
                window_change.size == 0 or np.all(window_change == 0.)),
            surrogate_only_preprojection=bool(
                delta < 0 and (window_change.size == 0 or
                               np.all(window_change == 0.))),
            unequal_termination_lengths=n != baseline_n,
            branch_vs_baseline=differences(case['detail'],
                                           baseline['detail']),
            trace=case['detail']['summary'],
            repeats=case['repeat_summaries'])
        public_rows.append(row)
        case_lookup[(direction_name, h)] = (case, row)

    for direction_name in directions:
        for larger, smaller in zip(H_VALUES[:-1], H_VALUES[1:]):
            larger_case, larger_row = case_lookup[(direction_name, larger)]
            smaller_case, _ = case_lookup[(direction_name, smaller)]
            larger_row['branch_vs_next_smaller_h'] = differences(
                larger_case['detail'], smaller_case['detail'])
        case_lookup[(direction_name, H_VALUES[-1])][1][
            'branch_vs_next_smaller_h'] = None

    stability = {}
    if gradient_status == 'finite_nonzero':
        for direction_name in directions:
            rows = [row for row in public_rows
                    if row['direction'] == direction_name and row['resolved']]
            selected = sorted(rows, key=lambda row: row['h'])[:3]
            if len(selected) < 3:
                stability[direction_name] = dict(
                    stable=False, blocker='fewer than three resolved h values',
                    selected_h=[row['h'] for row in selected])
                continue
            slopes = np.asarray([row['slope'] for row in selected])
            median_slope = float(np.median(slopes))
            slope_spread = float(slopes.max()-slopes.min())
            threshold = .05*max(abs(median_slope), TAU_S)
            stable = slope_spread <= threshold
            signs = [row['sign'] for row in selected]
            stable_sign = signs[0] if stable and len(set(signs)) == 1 else None
            stability[direction_name] = dict(
                stable=stable, selected_h=[row['h'] for row in selected],
                selected_slopes=slopes.tolist(), median_slope=median_slope,
                slope_spread=slope_spread, stability_threshold=threshold,
                signs=signs, stable_sign=stable_sign,
                blocker=(None if stable else
                         'three-smallest-resolved slope spread exceeds 5%'))

    if gradient_status == 'zero':
        verdict = 'B: no VJP direction / B'
    elif gradient_status != 'finite_nonzero':
        verdict = 'C: nonfinite or numerically unresolved gradient'
    else:
        minus = stability['minus']; plus = stability['plus']
        if (minus['stable'] and minus['stable_sign'] == 'negative' and
                plus['stable'] and plus['stable_sign'] == 'positive'):
            verdict = ('A: local surrogate-descent utility at this tested '
                       'checkpoint')
        elif (minus['stable'] and minus['stable_sign'] in
              ('zero', 'positive')):
            verdict = 'B: no first-order local descent evidence'
        elif (minus['stable'] and minus['stable_sign'] == 'negative' and
              plus['stable'] and plus['stable_sign'] in ('zero', 'negative')):
            verdict = ('minus-direction descent observed; '
                       'opposite-direction control not confirmed')
        else:
            blockers = [f'{name}: {value.get("blocker") or "sign not stable"}'
                        for name, value in stability.items()
                        if not value['stable'] or value['stable_sign'] is None]
            verdict = 'C: '+('; '.join(blockers) or 'slopes not stabilized')

    sources = [
        'single_integrator/c1/risk/joint_frozen.py',
        'single_integrator/c1/differentiable_rollout.py',
        'single_integrator/c1/risk/vi_r_cert.py',
        'single_integrator/c1/rollout_vi_r_cert.py',
        'single_integrator/c1/termination.py',
        'single_integrator/cbf.py',
        'single_integrator/environment.py',
        'flowbc/giveway_flowbc_agent.py',
        'scripts/check_c1_vi_r_cert.py',
        'scripts/diagnose_c1_vi_vjp_one_sided.py']
    report = dict(
        verdict=verdict,
        scope=('current production VJP one-sided local augmented R_CERT; '
               'no outcome study or training'),
        protocol=dict(
            original_scalar='one fixed continuation augmented_R_CERT',
            continuation_count=1, initial=INITIAL.tolist(),
            noise_seed=NOISE_SEED, rollout_id=ROLLOUT_ID,
            intervention_actions=[WINDOW[0], WINDOW[1]-1],
            h_values=list(H_VALUES), repeats=REPEATS,
            direction_norm='Euclidean norm of concatenated 20x4 residual action',
            tau_s=TAU_S, active_tolerance=ACTIVE_TOL,
            outer_tie_tolerance=TIE_TOL,
            precision=dict(jax_x64=True, controller='float64',
                           flow_bc_internal='float32'),
            solver=cbf.to_dict(), environment=plant.to_dict(),
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            software=dict(jax=jax.__version__, numpy=np.__version__,
                          scipy=scipy.__version__, python=platform.python_version(),
                          backend=jax.default_backend()),
            source_hashes={name:sha(ROOT/name) for name in sources}),
        gradient=dict(status=gradient_status, norm=gradient_norm,
                      values=gradient_host.tolist(),
                      directions={name:value.tolist()
                                  for name, value in directions.items()}),
        numerical_resolution=dict(
            maximum_within_input_spread=spread, scale=scale,
            machine_epsilon=machine_epsilon, function_value_floor=e_F,
            slope_uncertainty_formula='2*e_F/h',
            resolved_rule='uncertainty <= 0.1*max(abs(slope),tau_s)',
            stability_rule=('three smallest resolved h; spread <= '
                            '0.05*max(abs(median slope),tau_s)')),
        baseline=dict(F_repeats=baseline['values'], F_median=baseline['median'],
                      repeat_summaries=baseline['repeat_summaries'],
                      trace=baseline['detail']['summary']),
        rows=public_rows, stability=stability,
        previous_outcome_evidence=dict(
            source='results/c1_vi_r_cert_probe_v1/analysis.json',
            verdict='FAIL',
            note=('previous independent paired intervention showed no outcome '
                  'improvement; not rerun here')),
        anomalies=dict(nonfinite_gradient=not np.isfinite(gradient_host).all(),
                       solver_failures=sum(row['trace']['solver'][
                           'production_solver_failures'] for row in public_rows),
                       diagnostic_kkt_failures=sum(row['trace']['solver'][
                           'diagnostic_failures'] for row in public_rows)))
    report = json_ready(report)
    atomic_save(args.out_dir/'report.json', report)
    np.savez_compressed(args.out_dir/'traces.npz', **trace_store)
    atomic_save(args.out_dir/'complete.json', dict(
        verdict=verdict, report_sha256=sha(args.out_dir/'report.json'),
        traces_sha256=sha(args.out_dir/'traces.npz'),
        production_code_modified=False, outcome_study_run=False,
        training_started=False))
    print(json.dumps(dict(
        out=str(args.out_dir), gradient_norm=gradient_norm,
        e_F=e_F, stability=stability, verdict=verdict), indent=2))


if __name__ == '__main__':
    main()
