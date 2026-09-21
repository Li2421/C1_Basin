"""Analytic generic event checks; no archived risk or robot rollout is rerun."""
import json
import math
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/c1_probability_gradient_limits_v1.json'
    if out.exists():raise FileExistsError(out)
    # For Bernoulli loss, optimize the exact CVaR variational expression over
    # its piecewise-linear breakpoints z=0,1. Here beta is upper-tail mass.
    rows=[]
    for beta in (.05,.2,.5,1.):
        for p in (0.,.01,.04,.1,.25,.75,1.):
            objective=lambda z:z+(p*max(1-z,0)+(1-p)*max(-z,0))/beta
            direct=min(objective(0.),objective(1.))
            formula=min(p/beta,1.)
            assert abs(direct-formula)<1e-14
            rows.append(dict(probability=p,tail_mass=beta,cvar=direct,
                dp_derivative=1/beta if p<beta else 0. if p>beta else None))
    # D_theta={theta+Z>0}, Z standard Gaussian. The true probability derivative
    # at theta=0 is nonzero, but an empirical indicator is locally constant a.s.
    # A one-sided continuous upper ramp can receive gradient only in [-h,0].
    h=.05
    band_probability=.5*math.erf(h/math.sqrt(2))
    report=dict(scope='analytic counterexamples and necessary checks, not a new validated robot risk',
        bernoulli_cvar=rows,
        gaussian_event=dict(theta=0.,deadlock_probability=.5,
            true_probability_derivative=1/math.sqrt(2*math.pi),
            fixed_finite_sample_indicator_pathwise_derivative=0.,
            upper_ramp_width=h,gradient_band_probability=band_probability,
            no_sample_in_gradient_band={str(k):(1-band_probability)**k for k in (2,4,50)}),
        implications=['CVaR of a binary event does not create a new gradient direction',
            'Accurate event scoring and nonzero population derivative do not guarantee useful few-sample pathwise derivatives',
            'This example does not quantify bias or variance in the real C1 environment'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['gaussian_event'],indent=2))


if __name__=='__main__':main()
