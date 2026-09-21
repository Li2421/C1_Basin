"""Analytic conditioning audit and cached trace inspection; no robot reruns."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'results/c1_conditional_transversality_v1.json'
    if out.exists():
        raise FileExistsError(out)
    # Independent Z,E ~ N(0,1). D_theta={theta+a*Z+b*E>0}, a²+b²=1.
    # Integrating Z gives q=Phi((theta+b*E)/a), when a>0.
    # At theta=0: E[dq/dtheta]=phi(0); second moment below is exact.
    rows = []
    target = 1 / math.sqrt(2 * math.pi)
    for a in (1., .5, .1, .05, .01):
        b = math.sqrt(1-a*a)
        second = 1/(2*math.pi*a*math.sqrt(a*a+2*b*b))
        # Rescale E=a*t/b to resolve narrow gradient peaks in quadrature.
        if b:
            numerical, _ = quad(lambda t: math.exp(-.5*t*t-.5*(a*t/b)**2)/(2*math.pi*b),
                                -np.inf, np.inf, epsabs=1e-12)
            band = float(2*ndtr(a/b)-1)
        else:
            numerical, band = target, 1.
        assert abs(numerical-target) < 1e-10
        rows.append(dict(a=a,b=b,mean_gradient=target,second_moment=second,
                         variance=max(0.,second-target**2),
                         relative_standard_error_50=math.sqrt(max(0.,second-target**2)/50)/target,
                         probability_in_one_sigma_gradient_band=band,
                         probability_50_samples_miss_band=(1-band)**50))
    # a=0: exact conditional integration leaves q=1{theta+E>0}.
    # Pathwise derivative is zero a.s., whereas derivative of its mean is phi(0).
    source = ROOT/'results/c1_conditional_event_sections_v1'
    observed=[]
    for rid in range(5):
        traces=[]
        for k in range(10):
            with np.load(source/f'trace_{rid}_{k}.npz') as z:
                traces.append({name:z[name].copy() for name in ('applied','positions_after')})
        common = min(len(t['applied']) for t in traces)
        actions=np.stack([t['applied'][:common] for t in traces])
        states=np.stack([t['positions_after'][:common] for t in traces])
        observed.append(dict(rid=rid,common_steps=common,
            max_action_coordinate_range=float(np.ptp(actions,axis=0).max()),
            first_action_coordinate_range=float(np.ptp(actions[:,0],axis=0).max()),
            max_position_coordinate_range=float(np.ptp(states,axis=0).max())))
    result=dict(scope='analytic counterexample and cached diagnostics, not robot gradient validation',
        gaussian_example=rows,
        orthogonal_boundary=dict(a=0.,mean_probability_derivative=target,
                                 conditional_pathwise_derivative_almost_surely=0.),
        cached_sections=observed,
        source_complete_sha256=hashlib.sha256((source/'complete.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_robot_replays=0,training_launched=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
