"""Pointwise Gaussian integration-by-parts weight, NOT a robot risk certificate.

Callers must establish global regularity, boundary terms and integrability.
Singular noise gradients are rejected; adding a denominator epsilon would
change the mathematical identity. This helper never modifies a controller.
"""
import numpy as np


def scalar_parameter_weight(noise, margin_noise_gradient, margin_noise_hessian,
                            margin_parameter_derivative, mixed_noise_parameter,
                            *, minimum_gradient_norm=1e-8):
    """For v=rho_theta*grad_xi(rho)/||grad_xi(rho)||², return xi.v-div(v).

    This is a sensitivity weight, which may be negative, not a probability
    or nonnegative risk upper bound. One parameter direction is handled.
    """
    z,g,H,mixed=map(lambda x:np.asarray(x,dtype=float),
                  (noise,margin_noise_gradient,margin_noise_hessian,mixed_noise_parameter))
    a=float(margin_parameter_derivative)
    if z.ndim!=1 or not len(z) or g.shape!=z.shape or mixed.shape!=z.shape or H.shape!=(len(z),len(z)):
        raise ValueError('invalid derivative dimensions')
    if not all(np.isfinite(x).all() for x in (z,g,H,mixed)) or not np.isfinite(a):
        raise ValueError('finite derivatives required')
    if not np.isfinite(minimum_gradient_norm) or minimum_gradient_norm<=0:
        raise ValueError('positive minimum gradient norm required')
    if not np.allclose(H,H.T,atol=1e-10,rtol=1e-10):
        raise ValueError('margin Hessian must be symmetric')
    s=float(g@g)
    if s<minimum_gradient_norm**2:
        raise ValueError('noise gradient degenerate: cannot certify transport weight')
    velocity=a*g/s
    divergence=(mixed@g+a*np.trace(H))/s-2*a*(g@H@g)/(s*s)
    return float(z@velocity-divergence)
