"""Frozen inference and step-calibration formulas for Direction A.

These functions do not choose budgets, directions, or steps.  They require the
pre-registered values as arguments and are usable only after the corresponding
independent samples or proved bounds exist.
"""
import math

import numpy as np


def directional_second_moment_bound(A_v, M_v, C_v, sigma_min, *, horizon=850):
    values = (A_v, M_v, C_v, sigma_min)
    if not all(np.isfinite(values)) or min(values) < 0 or sigma_min <= 0:
        raise ValueError("finite nonnegative bounds and positive sigma_min required")
    return float(horizon*(A_v*A_v/4.0 + M_v*M_v/(sigma_min*sigma_min)
                          + 8.0*C_v*C_v))


def gradient_direction_interval(z, V_v, *, alpha_g=0.025):
    """Specified clipped confidence interval for independent B scenarios."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("z must contain at least two finite scenario values")
    if not np.isfinite(V_v) or V_v <= 0:
        raise ValueError("a proved positive V_v is required")
    if not 0 < alpha_g < 1:
        raise ValueError("alpha_g must lie in (0,1)")
    n = len(z)
    L_g = math.log(4.0/alpha_g)
    a_g = math.sqrt(3.0*(n-1)*V_v/(56.0*L_g))
    clipped = np.clip(z, -a_g, a_g)
    variance = float(np.var(clipped, ddof=1))
    radius = (math.sqrt(2.0*variance*L_g/n)
              + 14.0*a_g*L_g/(3.0*(n-1)) + V_v/(4.0*a_g))
    mean = float(np.mean(clipped))
    return dict(n=n, alpha_g=alpha_g, L_g=L_g, a_g=a_g,
                raw=z, clipped=clipped, mean=mean, sample_variance=variance,
                radius=radius, interval=(mean-radius, mean+radius))


def outcome_interval(values, lower, upper, *, alpha_x=0.025/(8*3)):
    """Specified empirical-Bernstein interval for independent scenario means."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("values must contain at least two finite scenario means")
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError("valid finite global range required")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("observations exceed declared global range")
    n = len(values)
    log_term = math.log(4.0/alpha_x)
    variance = float(np.var(values, ddof=1))
    radius = (math.sqrt(2.0*variance*log_term/n)
              + 7.0*(upper-lower)*log_term/(3.0*(n-1)))
    mean = float(np.mean(values))
    return dict(n=n, mean=mean, sample_variance=variance, radius=radius,
                interval=(max(lower, mean-radius), min(upper, mean+radius)))


def path_kl_terms(p0, sigma0, mu0, p1, sigma1, mu1):
    """Per-step KL decomposition for old-policy histories."""
    p0, sigma0, p1, sigma1 = map(np.asarray, (p0, sigma0, p1, sigma1))
    mu0, mu1 = np.asarray(mu0), np.asarray(mu1)
    if mu0.shape != mu1.shape or mu0.shape[-1] != 4:
        raise ValueError("mu arrays must match with final dimension four")
    if p0.shape != sigma0.shape or p0.shape != p1.shape or p0.shape != sigma1.shape:
        raise ValueError("p and sigma arrays must share leading shape")
    if (np.any((p0 <= 0) | (p0 >= 1) | (p1 <= 0) | (p1 >= 1))
            or np.any(sigma0 <= 0) or np.any(sigma1 <= 0)):
        raise ValueError("probabilities must be interior and scales positive")
    bernoulli = (p0*np.log(p0/p1)+(1-p0)*np.log((1-p0)/(1-p1)))
    mean_square = np.sum((mu0-mu1)**2, axis=-1)
    gaussian = p0*(4*np.log(sigma1/sigma0)
                   +(4*sigma0*sigma0+mean_square)/(2*sigma1*sigma1)-2)
    return bernoulli, gaussian


def fisher_path_quadratic(p, sigma, alpha_direction, mu_direction,
                          log_sigma_direction):
    """Sum the frozen directional Fisher approximation over a path."""
    p, sigma = np.asarray(p), np.asarray(sigma)
    alpha_direction = np.asarray(alpha_direction)
    mu_direction = np.asarray(mu_direction)
    log_sigma_direction = np.asarray(log_sigma_direction)
    if (p.shape != sigma.shape or p.shape != alpha_direction.shape
            or p.shape != log_sigma_direction.shape
            or mu_direction.shape != p.shape+(4,)):
        raise ValueError("inconsistent path-direction shapes")
    terms = (p*(1-p)*alpha_direction**2
             + p/(sigma*sigma)*np.sum(mu_direction**2, axis=-1)
             + 8*p*log_sigma_direction**2)
    return float(np.sum(terms))
