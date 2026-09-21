"""Conditional Gaussian event integration with explicit moving-boundary gradients.

Research primitive only: callers must supply a complete, certified interval
description. This module does not infer robot event boundaries from samples.
"""
import numpy as np
from scipy.special import ndtr


def interval_probability(intervals, boundary_derivatives=None):
    """Integrate a disjoint union of intervals for an independent standard normal.

    boundary_derivatives has shape [interval, lower/upper, parameter]. The
    returned derivative is of the integral, not of a sampled binary label.
    Infinite endpoints must have zero supplied derivative. Completeness and
    correspondence to an actual deadlock event are the caller's obligations.
    """
    intervals=np.asarray(intervals,dtype=np.float64)
    if intervals.ndim!=2 or intervals.shape[1]!=2 or np.isnan(intervals).any():
        raise ValueError('expected non-NaN interval endpoints [N,2]')
    if np.any(intervals[:,0]>=intervals[:,1]):raise ValueError('strictly ordered interval endpoints required')
    if len(intervals)>1 and np.any(intervals[1:,0]<intervals[:-1,1]):
        raise ValueError('intervals must be sorted and nonoverlapping')
    lo,hi=intervals[:,0],intervals[:,1]
    # Use survival probabilities on positive tails, avoiding cancellation near1.
    mass=np.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))
    probability=float(np.sum(mass))
    if boundary_derivatives is None:return probability,None
    derivative=np.asarray(boundary_derivatives,dtype=np.float64)
    if derivative.ndim!=3 or derivative.shape[:2]!=intervals.shape or not np.isfinite(derivative).all():
        raise ValueError('finite derivatives [N,2,P] required')
    if np.any(derivative[~np.isfinite(intervals)]!=0):
        raise ValueError('infinite fixed endpoints cannot move')
    density=np.exp(-.5*intervals**2)/np.sqrt(2*np.pi)
    gradient=np.sum(density[:,1,None]*derivative[:,1]-density[:,0,None]*derivative[:,0],axis=0)
    return probability,gradient


def implicit_boundary_derivative(margin_parameter_derivative, margin_noise_derivative,
                                 *, minimum_slope=1e-10):
    """Derivative of a simple root rho(phi,z)=threshold at fixed other noise.

    This rejects tangent/ill-conditioned roots instead of inventing a gradient.
    Root completeness, local differentiability and topology stability are not
    established by this calculation.
    """
    g=np.asarray(margin_parameter_derivative,dtype=np.float64)
    dz=float(margin_noise_derivative)
    if not np.isfinite(g).all() or not np.isfinite(dz):raise ValueError('finite derivatives required')
    if not np.isfinite(minimum_slope) or minimum_slope<=0:raise ValueError('positive slope threshold required')
    if abs(dz)<minimum_slope:raise ValueError('non-simple or numerically unstable root')
    return -g/dz
