"""Compare old/new hard projections on fixed samples of saved development traces."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv


def main():
    folder = ROOT/'results/c1_conditional_risk_cpu_v1'
    output = folder/'solver_failure_capture/projection_comparison.json'
    if output.exists():
        raise FileExistsError('do not overwrite audit')
    oldpath = folder/'solver_failure_capture/cbf_before.py'
    spec = importlib.util.spec_from_file_location('cbf_before_status_fix', oldpath)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    protocol = json.loads((folder/'protocol.json').read_text())
    env = GiveWayEnv(Config(**protocol['environment']))
    config = CBFConfig(**protocol['cbf'])
    comparisons, failures = [], []
    for record in json.loads((folder/'records.json').read_text()):
        name = f'trace_{record["rid"]}_{record["seed"]}.npz'
        with np.load(folder/name) as trace:
            for step in np.unique(np.linspace(0, len(trace['applied'])-1, 5, dtype=int)):
                env.positions = trace['positions_before'][step].copy()
                A, b, _ = barrier_constraints(env.snapshot(), config)
                target = trace['candidate'][step]
                current, status = project_velocity(target, A, b, env.config.max_speed, config)
                try:
                    previous, _ = old.project_velocity(target, A, b, env.config.max_speed, config)
                except old.CBFSolverError as exc:
                    failures.append(dict(trace=name, step=int(step), old_error=str(exc), new_status=status))
                    continue
                error = float(np.max(np.abs(current-previous)))
                if error > 1e-12:
                    raise AssertionError(f'changed previously accepted projection: {name}, {step}, {error}')
                comparisons.append(error)
    result = dict(scope='fixed five time samples per saved completed CPU development trajectory',
                  compared_previously_accepted=len(comparisons),
                  max_absolute_difference=max(comparisons, default=0.),
                  old_failures=failures, passed=True,
                  old_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
                  new_sha256=hashlib.sha256((ROOT/'single_integrator/cbf.py').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation='Sampled regression audit, not exhaustive solver correctness or policy efficacy')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
