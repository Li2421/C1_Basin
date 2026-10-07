#!/usr/bin/env python3
"""Prove that the isolated rollout wrapper is numerically canonical P0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta_representation_capacity.tools.representations import REPRESENTATIONS
from shared_control.diagnostic_corrector import DiagnosticCorrector


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"


def main() -> int:
    rng = np.random.default_rng(431902)
    maximum = 0.0
    inherited_maximum = 0.0
    comparisons = 0
    for _ in range(256):
        positions = rng.uniform((-3.8, -0.8), (3.8, 0.8), size=(4, 2))
        goals = rng.uniform((-3.8, -0.8), (3.8, 0.8), size=(4, 2))
        u_safe = rng.uniform(-0.5, 0.5, size=(4, 2))
        theta = rng.uniform((0.5, -0.5, 0.0), (1.25, 0.5, 0.75))
        canonical = DiagnosticCorrector(theta)(positions, goals, u_safe, 0.5)[0]
        wrapper = REPRESENTATIONS["P0-3D"].correction(theta, positions, goals, u_safe, 0.5)
        inherited = REPRESENTATIONS["P1-Agent6"].correction(REPRESENTATIONS["P1-Agent6"].embed_p0(theta), positions, goals, u_safe, 0.5)
        maximum = max(maximum, float(np.max(np.abs(canonical - wrapper))))
        inherited_maximum = max(inherited_maximum, float(np.max(np.abs(canonical - inherited))))
        comparisons += canonical.size
    output = {
        "schema": "double_bottleneck_eta3_p0_equivalence_v1",
        "rng_seed": 431902,
        "states": 256,
        "scalar_comparisons": comparisons,
        "maximum_absolute_difference": maximum,
        "agent6_p0_embedding_maximum_absolute_difference": inherited_maximum,
        "exact_equal": maximum == 0.0 and inherited_maximum == 0.0,
    }
    (STUDY / "p0_implementation_equivalence.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    return 0 if output["exact_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
