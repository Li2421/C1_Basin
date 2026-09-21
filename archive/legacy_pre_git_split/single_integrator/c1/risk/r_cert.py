"""Generic R_CERT aggregation, without task-specific certificates.

This module implements only the aggregation fixed by the R_CERT proposal.
Callers must separately establish, for every event e and certificate j,

    {tau: c[e,j,l](tau) > 0 for every l} subset complement(D_e).

No certificate margins are inferred from a trajectory here.
"""
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


LOG_TWO = math.log(2.0)


def atom_penalty(margin, scale):
    """Return h(-c/sigma), with h(z)=softplus(z)/log(2)."""
    return jax.nn.softplus(-jnp.asarray(margin) / jnp.asarray(scale)) / LOG_TWO


def aggregate(event_margins: Sequence, event_scales: Sequence, *, kappa: float):
    """Evaluate the exact sum / normalized soft-min / outer-max proposal.

    Each event entry has shape [M_e, L_e]. Rows index certificates and columns
    index the atoms of that certificate. Variable M_e or L_e across events is
    represented by separate sequence entries; empty events and certificates
    are rejected rather than assigned a convention.

    The scales and kappa are fixed positive constants. Gradients with respect
    to margins are preserved. The outer maximum is exact, including its
    ordinary nonsmooth tie semantics in JAX.
    """
    if not isinstance(event_margins, Sequence) or not isinstance(event_scales, Sequence):
        raise TypeError('event_margins and event_scales must be sequences indexed by event')
    if len(event_margins) == 0:
        raise ValueError('R_CERT is undefined for an empty event index set')
    if len(event_margins) != len(event_scales):
        raise ValueError('every event margin tensor requires a matching scale tensor')
    if not isinstance(kappa, (int, float)) or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError('kappa must be a fixed finite positive scalar')

    certificate_costs = []
    event_risks = []
    for event_index, (raw_margins, raw_scales) in enumerate(zip(event_margins, event_scales)):
        margins = jnp.asarray(raw_margins)
        scales_host = np.asarray(raw_scales)
        if margins.ndim != 2:
            raise ValueError(f'event {event_index} margins must have shape [M_e,L_e]')
        if margins.shape[0] == 0:
            raise ValueError(f'event {event_index} has an empty certificate index set')
        if margins.shape[1] == 0:
            raise ValueError(f'event {event_index} has certificates with no atoms')
        if scales_host.shape != margins.shape:
            raise ValueError(f'event {event_index} scales must match its margin tensor')
        if not np.isfinite(scales_host).all() or np.any(scales_host <= 0):
            raise ValueError(f'event {event_index} scales must be finite and positive')

        scales = jnp.asarray(scales_host, dtype=margins.dtype)
        costs = jnp.sum(atom_penalty(margins, scales), axis=1)
        risk = -kappa * (logsumexp(-costs / kappa) - math.log(margins.shape[0]))
        certificate_costs.append(costs)
        event_risks.append(risk)

    event_risks = jnp.stack(event_risks)
    return dict(R_CERT=jnp.max(event_risks), R_e=event_risks,
                S_ej=tuple(certificate_costs))
