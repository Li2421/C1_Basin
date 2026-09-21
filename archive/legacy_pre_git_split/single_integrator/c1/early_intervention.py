"""Diagnostic-only prefix intervention using the unchanged authoritative executor.

This is not a deployable scene rule or a training policy. A fresh wrapper is
created for each replay. After a fixed number of control calls it restores the
reference parameters, evaluated on the *current* state and original noise.
"""
from numbers import Integral


class PrefixInterventionField:
    def __init__(self, field, reference_params, intervention_params, prefix_steps):
        if isinstance(prefix_steps, bool) or not isinstance(prefix_steps, Integral):
            raise ValueError('prefix_steps must be a nonnegative integer')
        if prefix_steps < 0:
            raise ValueError('prefix_steps must be a nonnegative integer')
        self.field = field
        self.reference_params = reference_params
        self.intervention_params = intervention_params
        self.prefix_steps = int(prefix_steps)
        self.calls = 0

    def prepare(self, ignored_params, *args, **kwargs):
        params = (self.intervention_params if self.calls < self.prefix_steps
                  else self.reference_params)
        result = self.field.prepare(params, *args, **kwargs)
        self.calls += 1
        return result


def execute_prefix(reference_params, intervention_params, field, initial,
                   draws, plant, cbf, *, prefix_steps):
    """Keep environment, both projections, first-event termination and audit.

The return adds diagnostic metadata only. Timeout subclassification remains
the caller's responsibility, exactly as for evaluate_deadlock_union.execute.
Zero steps is reference-only; max_steps is a full-episode intervention.
    """
    wrapped = PrefixInterventionField(field, reference_params,
                                     intervention_params, prefix_steps)
    if prefix_steps > plant.max_steps:
        raise ValueError('prefix exceeds original evaluation horizon')
    from .evaluate_deadlock_union import execute
    row, trace = execute(reference_params, wrapped, initial, draws, plant, cbf)
    row = dict(row, intervention=dict(
        kind='fixed_prefix_parameter_intervention',
        requested_steps=int(prefix_steps),
        executed_steps=min(wrapped.calls, int(prefix_steps)),
        prefix_seconds=float(prefix_steps * plant.dt),
        reference_restored_at_current_state=True,
        efficacy_scope='diagnostic_only_not_shared_policy_generalization'))
    return row, trace
