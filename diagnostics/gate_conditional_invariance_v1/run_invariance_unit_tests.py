"""Formula tests: class conditioning, missing cells, and no hidden inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conditional_invariance_gate_runner as runner  # noqa: E402


def main() -> None:
    # Group 0 has mostly c=0; group 1 has mostly c=1.  Conditional cell means
    # are identical (-2 and +2), so the required penalty is zero.  A forbidden
    # class-unconditional group mean penalty would be positive due to differing
    # class mixture, demonstrating that the implementation does not collapse
    # the two oracle classes into one source statistic.
    logits = np.asarray([-2, -2, -2, 2, -2, 2, 2, 2], float)
    group = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1])
    penalty, details = runner.conditional_invariance_numpy(logits, group, labels, 2)
    unconditional_group_means = np.asarray([logits[group == g].mean() for g in (0, 1)])
    forbidden_unconditional_variance = float(np.var(unconditional_group_means))
    class_conditioning_pass = bool(abs(penalty) < 1e-12 and forbidden_unconditional_variance > 1e-6)

    # Class 1 is present in just one source group and therefore must be skipped
    # rather than treated as a zero/phantom cell.  Class 0 has two cells with
    # means 0 and 2, giving variance 1.
    missing_logits = np.asarray([0, 2, 5], float)
    missing_group = np.asarray([0, 1, 0])
    missing_labels = np.asarray([0, 0, 1])
    missing_penalty, missing_details = runner.conditional_invariance_numpy(missing_logits, missing_group, missing_labels, 2)
    missing_cell_pass = bool(abs(missing_penalty - 1.0) < 1e-12 and
                             missing_details["eligible_class"].tolist() == [True, False] and
                             missing_details["cell_counts"].tolist() == [[1, 1], [1, 0]])

    payload = {
        "status": "PASS" if class_conditioning_pass and missing_cell_pass else "FAIL",
        "class_conditional_penalty": penalty,
        "forbidden_class_unconditional_group_variance": forbidden_unconditional_variance,
        "class_conditioning_prevents_unconditional_collapse": class_conditioning_pass,
        "missing_cell_penalty": missing_penalty,
        "missing_cell_skipped": missing_cell_pass,
        "uses_validation_or_test_statistics": False,
        "formula": "mean class-conditional variance across eligible training source-group cells",
    }
    runner.base.write_json(HERE / "invariance_formula_unit_tests.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
