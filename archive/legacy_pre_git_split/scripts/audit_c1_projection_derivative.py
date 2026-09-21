"""Check the existing KKT VJP on smooth branches; retain weak-active counterexample.

This is a solver derivative audit, not a controller change or efficacy test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from single_integrator.cbf import CBFConfig, project_velocity
from single_integrator.c1.risk.joint_frozen import _adjoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/c1_projection_derivative_audit_v2.json')
    parser.add_argument('--speed-tol', type=float, default=1e-10)
    parser.add_argument('--solver-ftol', type=float, default=1e-12)
    parser.add_argument('--max-speed-cuts', type=int, default=32)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    rng = np.random.default_rng(2026091846)
    config = CBFConfig(speed_tol=args.speed_tol, solver_ftol=args.solver_ftol,
                       max_speed_cuts=args.max_speed_cuts)
    begun = time.monotonic()
    A = np.array([[1., 0., 0., 0.]])
    b = np.zeros(1)
    cot = np.array([-.2, .3, -.4, .1])
    cases = dict(interior=[.1, .2, .1, .1],
                 strict_linear=[-.2, .1, .1, .1],
                 strict_ball=[.8, .3, .1, .1])
    solve = lambda y, a, c: project_velocity(y, a, c, .5, config)[0].reshape(-1)
    rows = []
    for name, target in cases.items():
        y = np.array(target)
        p = solve(y, A, b)
        gy, ga, gb = _adjoint(y, A, b, p, cot, True)
        # These targets have either an inactive ball, one active halfspace,
        # or an active ball with an inactive halfspace, also for small h.
        def analytic(v, a, c):
            if name == 'strict_linear':
                return v + a[0]*(c[0]-a[0]@v)/(a[0]@a[0])
            q = v.reshape(2, 2)
            return (q / np.maximum(1., np.linalg.norm(q, axis=1)/.5)[:, None]).reshape(-1)
        exact_p = analytic(y, A, b)
        egy, ega, egb = _adjoint(y, A, b, exact_p, cot, True)
        for index in range(12):
            dy, da, db = rng.normal(size=4), rng.normal(size=A.shape), rng.normal(size=1)
            predicted = float(gy @ dy + np.sum(ga * da) + gb @ db)
            exact_predicted = float(egy @ dy + np.sum(ega * da) + egb @ db)
            measurements = []
            for h in (1e-3, 1e-4, 1e-5):
                plus = solve(y+h*dy, A+h*da, b+h*db)
                minus = solve(y-h*dy, A-h*da, b-h*db)
                actual = float(cot @ (plus-minus)/(2*h))
                ep, em = analytic(y+h*dy,A+h*da,b+h*db), analytic(y-h*dy,A-h*da,b-h*db)
                exact_fd = float(cot @ (ep-em)/(2*h))
                measurements.append(dict(h=h, fd=actual, absolute_error=abs(actual-predicted),
                    analytic_fd=exact_fd, analytic_absolute_error=abs(exact_fd-exact_predicted),
                    forward_solution_error=max(float(np.max(np.abs(plus-ep))),float(np.max(np.abs(minus-em))))))
            rows.append(dict(branch=name, direction=index, adjoint=predicted,
                analytic_adjoint=exact_predicted, base_solution_error=float(np.max(np.abs(p-exact_p))), measurements=measurements))
    # At weak activity this map has directional derivatives, not one linear derivative.
    y = np.zeros(4)
    p = solve(y, A, b)
    gy, _, _ = _adjoint(y, A, b, p, cot, True)
    e = np.array([1., 0., 0., 0.]); h = 1e-4
    weak = dict(adjoint=float(gy @ e),
                forward=float(cot @ (solve(y+h*e, A, b)-p)/h),
                backward=float(cot @ (p-solve(y-h*e, A, b))/h))
    worst = max(r['measurements'][-1]['absolute_error'] for r in rows)
    exact_worst = max(r['measurements'][-1]['analytic_absolute_error'] for r in rows)
    paths = ['scripts/audit_c1_projection_derivative.py', 'single_integrator/cbf.py',
             'single_integrator/c1/risk/joint_frozen.py']
    result = dict(scope='generic smooth-branch local derivative checks, not all rollout gradients',
                  seed=2026091846, rows=rows, smallest_step_max_absolute_error=worst,
                  tolerance=1e-5, smooth_branch_passed=worst < 1e-5, weak_active=weak,
                  analytic_smallest_step_max_error=exact_worst,analytic_branch_passed=exact_worst<1e-5,
                  solver_config=config.to_dict(),elapsed=time.monotonic()-begun,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}))
    if not result['smooth_branch_passed']:
        raise RuntimeError('smooth branch derivative check failed; inspect saved evidence')


if __name__ == '__main__':
    main()
