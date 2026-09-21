"""Sign-preserving temporal risk with an additional genuine gradient path.

Let rho be the exact event margin and s the original smooth sum bound.
R = relu(1 + rho * exp(sign(rho) * s)), with sign(0)=+1.
The exponential is positive: the event separation at R=1 is unchanged.
On either open sign branch, R before ReLU increases in rho and in s:
dF/drho=exp(sign(rho)*s), dF/ds=abs(rho)*exp(sign(rho)*s).
All derivatives are of the actual forward function, never a straight-through
label or an independently substituted gradient. The kink at rho=0 remains.
"""
import jax
import jax.numpy as jnp
from .exact_margin import trajectory as exact_trajectory
from .deadlock_union import trajectory as smooth_trajectory


def ordered_score(rho,s):
    factor=jnp.exp(jnp.where(rho>=0,s,-s))
    return jax.nn.relu(1+rho*factor)


def trajectory(before,after,applied,goals,alive,*,terminal_timeout,**kwargs):
    exact=exact_trajectory(before,after,applied,goals,alive,terminal_timeout=terminal_timeout,**kwargs)
    smooth=smooth_trajectory(before,after,applied,goals,alive,terminal_timeout=terminal_timeout,**kwargs)
    rho=exact['robustness_hard'];s=smooth['J_live']
    return dict(J_live=jnp.where(exact['eligible'],ordered_score(rho,s),0.),
                robustness_hard=rho,eligible=exact['eligible'],guidance=s)
