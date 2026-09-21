"""Run the CPU-only Direction A semantic contract and write a JSON record."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = (
    "tests.test_c1_direction_a_policy",
    "tests.test_c1_direction_a_monitor_safety",
    "tests.test_c1_direction_a_mutations",
    "tests.test_c1_direction_a_frozen_integration",
)
SOURCES = (
    "single_integrator/c1/direction_a/__init__.py",
    "single_integrator/c1/direction_a/policy.py",
    "single_integrator/c1/direction_a/randomness.py",
    "single_integrator/c1/direction_a/estimators.py",
    "single_integrator/c1/direction_a/rollout.py",
    "single_integrator/c1/direction_a/statistics.py",
    "single_integrator/environment.py",
    "single_integrator/cbf.py",
    "single_integrator/diagnostics/stalled_outcomes.py",
    "single_integrator/c1/differentiable_rollout.py",
    *tuple(module.replace(".", "/")+".py" for module in MODULES),
    "scripts/audit_c1_direction_a_contract.py",
)


class RecordingResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self._current_id = test.id()

    def addSuccess(self, test):
        super().addSuccess(test)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_suite(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten_suite(item)
        else:
            yield item


def category(test_id):
    if "MomentAndOracle" in test_id:
        return "analytic_oracle"
    if "SamplingAndPointScore" in test_id:
        return "parameterization_logprob_score"
    if "ControllerAndMonitor" in test_id:
        return "controller_monitor"
    if "SafetyFailure" in test_id:
        return "safety_failure_handling"
    if "RequiredMutation" in test_id:
        return "negative_mutation"
    if "FrozenStatistics" in test_id:
        return "frozen_statistics_formula"
    if "FrozenControllerIntegration" in test_id:
        return "frozen_integration"
    return "uncategorized"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
        default=ROOT/"results/c1_direction_a_minimal_v1/contract_results.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to replace {args.out}")
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromNames(MODULES)
    ids = [test.id() for test in flatten_suite(suite)]
    stream = io.StringIO()
    runner = unittest.TextTestRunner(
        stream=stream, verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    transcript = stream.getvalue()
    print(transcript, end="")
    failed = {test.id(): detail for test, detail in result.failures}
    errors = {test.id(): detail for test, detail in result.errors}
    skipped = {test.id(): reason for test, reason in result.skipped}
    rows = []
    for test_id in ids:
        status = ("FAIL" if test_id in failed else "ERROR" if test_id in errors
                  else "SKIP" if test_id in skipped else "PASS")
        rows.append(dict(id=test_id, category=category(test_id), status=status,
                         detail=failed.get(test_id, errors.get(test_id,
                                             skipped.get(test_id)))))
    stage0_path = ROOT/"results/c1_direction_a_minimal_v1/stage0_manifest.json"
    stage0 = json.loads(stage0_path.read_text())
    source_hashes = {name:digest(ROOT/name) for name in SOURCES}
    checkpoint = ROOT/"baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    counts = Counter(row["category"] for row in rows if row["status"] == "PASS")
    payload = dict(
        schema="c1_direction_a_contract_v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        command=("JAX_PLATFORMS=cpu .venv-c1/bin/python "
                 "scripts/audit_c1_direction_a_contract.py "
                 "--out /tmp/c1_direction_a_contract_repro.json"),
        cpu_only=True, backend=jax.default_backend(), precision="jax_enable_x64=True",
        versions=dict(python=sys.version, jax=jax.__version__, numpy=np.__version__),
        result="CONTRACT PASS" if result.wasSuccessful() else "CONTRACT FAIL",
        tests_run=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped),
        passed_by_category=dict(sorted(counts.items())), tests=rows,
        declared_test_tolerances=dict(
            point_score_absolute=2e-12, moment_absolute=3e-12,
            analytic_derivative_relative=2e-12,
            exact_sampling="array equality where specified",
            safety="CBFConfig feasibility_tol/speed_tol"),
        integration_fixture_note=(
            "seed0 Flow-BC and toy Direction A values are contract fixtures only; "
            "they are not an approved phi_0 or pilot"),
        source_hashes=source_hashes,
        checkpoint_sha256=digest(checkpoint), checkpoint=str(checkpoint.resolve()),
        stage0_manifest_sha256=digest(stage0_path),
        stage3_to_5_status="NOT RUN",
        stage3_to_5_reason=stage0["blocked_reason"],
        missing_required_values=stage0["missing_required_values"],
        transcript=transcript,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key:payload[key] for key in
                      ("result", "tests_run", "failures", "errors")},
                     sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
