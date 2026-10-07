"""Shared read-only helpers for qualifying the frozen CL-FHCB method."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.cl_fhcb.certificate import (
    BIN_NAMES,
    EmpiricalContinuationCertificate,
)
from diagnostics.cl_fhcb.closed_loop import DiagnosticPhi


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SPEC = REPO_ROOT / "diagnostics/cl_fhcb/CL_FHCB_FROZEN_SPEC.md"
FROZEN_SPEC_SHA256 = "b748882cbce9c87ccc9d7bd2362a89151efac700f12d75e3c3287c9ce93cd64a"
FROZEN_METHOD_HASHES = {
    "diagnostics/cl_fhcb/closed_loop.py": "aeca60cc1968733ca2dde4535c0de11a5431e2adcbac01027005a2ffc6694fb0",
    "diagnostics/cl_fhcb/certificate.py": "1c09b3997ce41461dffab5674bab9a69d88c239a0e253761b4ca36ee85be9529",
    "diagnostics/cl_fhcb/data_v1/certificate.json": "eb334ca92a8aa1f07291f6a8df926384d6b948bcb3650cfb46270e8a3c708be7",
}

PHIS = (
    DiagnosticPhi(0.0, 0.0, 0.0, "zero"),
    DiagnosticPhi(0.25, 0.0, 0.0, "goal025"),
    DiagnosticPhi(0.0, -0.35, 0.0, "damp035"),
    DiagnosticPhi(0.0, 0.0, 0.25, "relative025"),
)
PHI_BY_NAME = {phi.name: phi for phi in PHIS}
K_VALUES = (20, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_method() -> None:
    if sha256(FROZEN_SPEC) != FROZEN_SPEC_SHA256:
        raise RuntimeError("frozen specification changed before qualification")
    for relative, expected in FROZEN_METHOD_HASHES.items():
        path = REPO_ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen method hash mismatch: {relative}: {actual}")


def locate_assets(explicit: Path | None = None) -> Path:
    candidates = [] if explicit is None else [explicit]
    candidates.extend((REPO_ROOT, REPO_ROOT.parent / "02_C1_Toy_GiveWay"))
    for root in candidates:
        if (
            root / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
        ).is_file() and (root / "datasets/give_way_si_short_v1/environment.json").is_file():
            return root.resolve()
    raise FileNotFoundError("frozen checkpoint/dataset assets not found")


def load_certificates() -> dict[str, EmpiricalContinuationCertificate]:
    path = REPO_ROOT / "diagnostics/cl_fhcb/data_v1/certificate.json"
    payload = json.loads(path.read_text())["certificates"]
    result = {}
    for name, record in payload.items():
        phi = PHI_BY_NAME[name]
        if tuple(record["phi"]) != phi.vector or tuple(record["bin_names"]) != BIN_NAMES:
            raise ValueError("frozen certificate metadata mismatch")
        result[name] = EmpiricalContinuationCertificate(
            phi=phi,
            horizon=int(record["horizon"]),
            transition_counts=np.asarray(record["transition_counts"], dtype=np.int64),
            transition_probabilities=np.asarray(
                record["transition_probabilities"], dtype=np.float64
            ),
            table=np.asarray(record["table"], dtype=np.float64),
        )
    return result


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mean_ci_bootstrap(
    values: np.ndarray, rng: np.random.Generator, replicates: int = 2000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float(center - radius), float(center + radius)
