"""Versioned, post-hoc C1 risk and correction diagnostics for executed traces."""
import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.risk.risk_function import RiskConfig, risk_diagnostics, trajectory_risk
from single_integrator.c1.risk.soft_activity import SoftRiskConfig, soft_risk_diagnostics
from single_integrator.c1.risk.risk_v1 import RiskV1Config, trajectory_risk_v1
from single_integrator.c1.risk.risk_v2 import RiskV2Config, trajectory_risk_v2
from single_integrator.c1.termination import EVENTS
from single_integrator.environment import GiveWayEnv


def risk_from_metadata(metadata):
    version = metadata.get('risk_version', 'R_risk_v0')
    config = dict(metadata['risk'])
    if version=='R_risk_v2':
        return RiskV2Config(**config)
    if version in ('R_risk_v1', 'R_risk_v1_1'):
        expected = 'joint_sum' if version == 'R_risk_v1' else 'per_agent'
        if config.get('unfinished_mode', expected) != expected:
            raise ValueError('risk version and unfinished-task semantics disagree')
        config['unfinished_mode'] = expected
        return RiskV1Config(**config)
    if version == 'R_risk_v0_1':
        return SoftRiskConfig(**config)
    if version == 'R_risk_v0':
        return RiskConfig(**config)
    raise ValueError(f'unsupported risk version: {version}')


def audit_trace(trace, plant, cbf, risk):
    """Uses the actual candidate and executed control, without another rollout.

    Warmup steps without a complete v1 window are explicitly invalid; a short
    completed episode has no measured trajectory risk, rather than zero risk.
    """
    env = GiveWayEnv(plant)
    positions = jnp.asarray(trace['positions_before'], jnp.float64)
    applied = jnp.asarray(trace['executed_velocity'], jnp.float64).reshape(-1, 4)
    candidate = jnp.asarray(trace['c1_candidate'], jnp.float64).reshape(-1, 4)
    A, b, h = jax.jit(jax.vmap(lambda p: barrier_constraints(
        p, jnp.asarray(env.walls), plant.to_dict(), cbf)))(positions)
    slack = jnp.einsum('tij,tj->ti', A, applied)-b
    blocks = A.reshape(-1, 17, 2, 2)
    if isinstance(risk, SoftRiskConfig):
        diagnostic = jax.jit(jax.vmap(lambda f, g, h, s: soft_risk_diagnostics(f, g, h, s, risk)))(
            candidate.reshape(-1, 2, 2), blocks, h, slack)
    else:
        mask = (h <= risk.rho) & (jnp.abs(slack) <= risk.active_tol)
        diagnostic = jax.jit(jax.vmap(lambda f, g, m: risk_diagnostics(f, g, m, risk)))(
            candidate.reshape(-1, 2, 2), blocks, mask)
    fields = {'c1_'+name: np.asarray(value) for name, value in diagnostic.items()}
    cone = diagnostic['risk']
    total, valid = np.asarray(cone), np.ones(len(cone), bool)
    trajectory_value = float(trajectory_risk(cone, risk))
    if isinstance(risk,RiskV2Config):
        all_positions=jnp.concatenate((positions[:1],jnp.asarray(trace['positions'])),axis=0)
        code=0
        for key,label in [('deadlock','safe_deadlock'),('task_success','success'),
                          ('wall_collision','wall_collision'),('agent_collision','agent_collision')]:
            if key in trace and bool(np.asarray(trace[key])[-1]):code=EVENTS[label]
        if code==0 and len(cone)>=plant.max_steps:code=EVENTS['other_timeout']
        result=trajectory_risk_v2(cone[None],all_positions[None],jnp.asarray(env.goals),risk,
                                  jnp.ones((1,len(cone)),bool),jnp.array([code]),plant.dt)
        for name,value in result.items():
            if name!='trajectory_risk':fields['c1_'+name]=np.asarray(value)[0]
        total,valid=fields['c1_total_risk_t'],fields['c1_valid_mask']
        trajectory_value=float(result['trajectory_risk'][0]) if code else None
    elif isinstance(risk, RiskV1Config):
        W = risk.window_steps(plant.dt)
        if len(cone) > W:
            all_positions = jnp.concatenate((positions[:1], jnp.asarray(trace['positions'])), axis=0)
            result = trajectory_risk_v1(cone, all_positions, jnp.asarray(env.goals), risk, W)
            for name, value in result.items():
                if name != 'trajectory_risk':
                    fields['c1_'+name] = np.asarray(value)[0]
            total, valid = fields['c1_total_risk_t'], fields['c1_valid_mask']
            trajectory_value = float(result['trajectory_risk'][0])
        else:
            total, valid, trajectory_value = np.full(len(cone), np.nan), np.zeros(len(cone), bool), None
            fields['c1_stall_risk_t'] = total.copy()
    fields.update(c1_cone_risk_t=np.asarray(cone), c1_total_risk_t=total,
                  c1_valid_mask=valid, c1_h=np.asarray(h), c1_cbf_residual=np.asarray(slack))
    raw = np.asarray(trace['c1_correction']).reshape(-1, 4)
    applied_delta = np.asarray(applied)-np.asarray(trace['c1_safe']).reshape(-1, 4)
    projected_delta = np.asarray(applied)-np.asarray(candidate)
    fields.update(c1_applied_correction=applied_delta, c1_final_projection_delta=projected_delta)
    summary = dict(trajectory_risk=trajectory_value, risk_valid_steps=int(valid.sum()),
                   mean_instantaneous_risk=float(total[valid].mean()) if valid.any() else None,
                   max_instantaneous_risk=float(total[valid].max()) if valid.any() else None,
                   mean_cone_risk=float(np.asarray(cone)[valid].mean()) if valid.any() else None,
                   J_def=float(np.mean(np.sum(applied_delta**2, axis=-1))),
                   hard_active_fraction=float(np.mean(np.any(fields['c1_active_mask'], axis=-1))),
                   min_cbf_residual=float(np.asarray(slack).min()))
    if 'c1_stall_risk_t' in fields:
        summary['mean_stall_risk'] = float(fields['c1_stall_risk_t'][valid].mean()) if valid.any() else None
    if isinstance(risk,RiskV2Config):
        summary.update(exposure_risk=float(fields['c1_exposure_risk']),
                       terminal_risk=float(fields['c1_terminal_risk']),
                       terminal_goal_risk=float(fields['c1_terminal_goal_risk']),
                       terminal_stall_risk=float(fields['c1_terminal_stall_risk']),
                       censored=bool(fields['c1_censored']))
    for name, values in [('raw_correction', raw), ('applied_correction', applied_delta),
                         ('final_projection', projected_delta)]:
        magnitude = np.linalg.norm(values, axis=-1)
        summary['mean_'+name+'_norm'] = float(magnitude.mean())
        summary['max_'+name+'_norm'] = float(magnitude.max())
    summary['final_projection_fraction'] = float(np.mean(np.linalg.norm(projected_delta, axis=-1) > cbf.intervention_tol))
    if 'min_swept_agent_distance' in trace:
        summary['min_agent_surface_distance'] = float(np.min(trace['min_swept_agent_distance']))
        summary['min_interagent_distance'] = summary['min_agent_surface_distance']+2*plant.agent_radius
    if 'min_swept_wall_distance' in trace:
        summary['min_wall_clearance'] = float(np.min(trace['min_swept_wall_distance']))
    return summary, fields


def aggregate_risk(summaries):
    measured = [s['c1'] for s in summaries if s['c1']['trajectory_risk'] is not None]
    return dict(risk_measured_episodes=len(measured),
                risk_unmeasured_episodes=len(summaries)-len(measured),
                mean_trajectory_risk=float(np.mean([s['trajectory_risk'] for s in measured])) if measured else None,
                max_trajectory_risk=max((s['trajectory_risk'] for s in measured), default=None))
