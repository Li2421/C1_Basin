"""Capture an unchanged solver failure; never substitute or accept its output."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import numpy as np
import single_integrator.cbf as cbf_module
from single_integrator.c1.train_deadlock_union import setup, noise, digest
from single_integrator.c1.evaluate_deadlock_union import execute


def main():
    if jax.default_backend() != 'cpu':
        raise RuntimeError('CPU only')
    folder = ROOT/'results/c1_conditional_risk_cpu_v1'
    output = folder/'solver_failure_capture'
    output.mkdir(exist_ok=False)
    protocol = json.loads((folder/'protocol.json').read_text())
    records = json.loads((folder/'records.json').read_text())
    seeds = protocol['prediction_seeds']+protocol['outcome_seeds']
    rid, index = divmod(len(records), len(seeds))
    seed = seeds[index]
    phi, field, plant, cbf, _ = setup()
    original = cbf_module.minimize
    calls = 0
    failures = []

    def capture(fun, x0, **kwargs):
        nonlocal calls
        calls += 1
        result = original(fun, x0, **kwargs)
        if not result.success:
            constraints = kwargs['constraints']
            matrix = constraints['jac'](np.zeros(4))
            lower = -constraints['fun'](np.zeros(4))
            target = -kwargs['jac'](np.zeros(4))
            np.savez(output/f'qp_{calls}.npz', matrix=matrix, lower=lower,
                     target=target, x0=x0, candidate=result.x)
            failures.append(dict(call=calls, message=str(result.message), status=int(result.status),
                                 min_residual=float(np.min(matrix@result.x-lower))))
        return result

    status = 'not_reproduced'
    error = None
    with patch.object(cbf_module, 'minimize', capture):
        try:
            execute(phi, field, protocol['starts'][rid], noise(seed, 20000+rid), plant, cbf)
        except cbf_module.CBFSolverError as exc:
            status, error = 'reproduced', str(exc)
    report = dict(status=status, error=error, rid=rid, noise_seed=seed, qp_calls=calls,
                  failed_qps=failures, completed_rollouts=len(records),
                  source_sha256=digest(Path(__file__)), original_solver_unchanged=True,
                  experiment_complete=False, sample_skipped=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
