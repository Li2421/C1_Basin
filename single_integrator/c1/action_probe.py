"""Diagnostic action perturbations, before the unchanged second hard projection.

Offsets are fixed from prediction noise only. They never select paths or change
the shared policy, and are not used by training or deployment.
"""
import numpy as np
import jax.numpy as jnp


def calibrate_offsets(direction, samples, cbf, *, target_rms=.0025, component_cap=.025):
    """Match the *projected* same-state perturbation, without outcome/risk search.

    samples contains prediction-only (candidate, A, b) arrays for the prefix.
    Projection is continuous, so a sign-bracketed scalar root can calibrate
    magnitude. This does not require a monotonicity claim or path selection.
    """
    from single_integrator.cbf import project_velocity
    direction=np.asarray(direction,dtype=np.float64)
    if direction.ndim!=2 or direction.shape[1]!=4 or not np.isfinite(direction).all():
        raise ValueError('finite [steps,4] direction required')
    if not samples or target_rms<=0 or component_cap<=0:
        raise ValueError('prediction samples and positive bounds required')
    peak=float(np.max(np.abs(direction)))
    if peak==0:
        return np.zeros_like(direction),dict(zero_direction=True,matched=False,rms=0.,scale=0.)
    ray=direction/peak
    prepared=[]
    for candidate,A,b in samples:
        candidate,A,b=map(np.asarray,(candidate,A,b))
        if candidate.shape!=direction.shape or len(A)!=len(direction) or len(b)!=len(direction):
            raise ValueError('prediction prefix shape mismatch')
        base=np.stack([project_velocity(v,a,c,.5,cbf)[0].reshape(4) for v,a,c in zip(candidate,A,b)])
        prepared.append((candidate,A,b,base))
    def rms(scale):
        differences=[]
        for candidate,A,b,base in prepared:
            changed=np.stack([project_velocity(v+scale*d,a,c,.5,cbf)[0].reshape(4)
                              for v,d,a,c in zip(candidate,ray,A,b)])
            differences.append(changed-base)
        return float(np.sqrt(np.mean(np.asarray(differences)**2)))
    upper=component_cap;achieved=rms(upper)
    if achieved>=target_rms:
        lower=0.
        for _ in range(16):
            middle=(lower+upper)/2
            if rms(middle)<target_rms:lower=middle
            else:upper=middle
        achieved=rms(upper)
    return upper*ray,dict(zero_direction=False,matched=abs(achieved/target_rms-1)<=.01,
                          rms=achieved,scale=upper,peak_offset=upper,target_rms=target_rms)


class ActionOffsetField:
    def __init__(self, field, offsets, start_step=0):
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[1] != 4 or not np.isfinite(offsets).all():
            raise ValueError('finite [steps,4] offsets required')
        if not isinstance(start_step, int) or start_step < 0:
            raise ValueError('start_step must be a nonnegative integer')
        self.field, self.offsets, self.start_step, self.calls = (
            field, offsets, start_step, 0)
        self.physical_changes = []

    def prepare(self, params, obs, draws, A, b, max_speed, project):
        result = dict(self.field.prepare(params, obs, draws, A, b, max_speed, project))
        offset_index = self.calls - self.start_step
        if 0 <= offset_index < len(self.offsets):
            candidate = result['safe'] + result['correction']
            offset = jnp.asarray(self.offsets[offset_index][None])
            # Compare perturbation on this arm's same state/noise, not divergent
            # trajectories: measure only the direct effect of the intervention.
            before = project(candidate, A, b, max_speed)
            after = project(candidate + offset, A, b, max_speed)
            self.physical_changes.append(np.asarray(after-before)[0])
            result['correction'] = result['correction'] + offset
        self.calls += 1
        return result


def execute_offsets(params, field, initial, draws, plant, cbf, offsets,
                    *, start_step=0):
    from .evaluate_deadlock_union import execute
    wrapped = ActionOffsetField(field, offsets, start_step=start_step)
    row, trace = execute(params, wrapped, initial, draws, plant, cbf)
    changes = np.asarray(wrapped.physical_changes)
    row['direct_action_rms'] = float(np.sqrt(np.mean(changes**2))) if changes.size else 0.
    trace['direct_action_changes'] = changes
    return row, trace
