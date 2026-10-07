"""Frozen provenance and durable-I/O helpers for startup-complete Dataset V1.

This module deliberately asserts source bytes before importing the controller
stack.  Multiple files with the same module names exist in the wider
workspace; accepting whichever happens to be first on ``sys.path`` would make
the new oracle labels scientifically uninterpretable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
BASE = SYSROOT / "results/risk_audit_seed0_baselines/mac_cbf"
BASE_CONFIG = BASE.parent / "config.json"

FROZEN_FILES = {
    SYSROOT / "single_integrator/environment.py": "427b1c0db1d68698e095a95e50a4404bae6c806a0f87351bfac431f6eeefd49b",
    SYSROOT / "single_integrator/cbf.py": "841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48",
    ROOT / "diagnostics/cl_fhcb/closed_loop.py": "aeca60cc1968733ca2dde4535c0de11a5431e2adcbac01027005a2ffc6694fb0",
    ROOT / "diagnostics/success_basin_multimodality/exact_projector.py": "e29d510dc1752f138bdfcc491f8f5008dbcd215282301396852bdbfd754ac544",
    SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl": "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in FROZEN_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen source missing: {path}")
        actual = sha256(path)
        observed[str(path)] = actual
        if actual != expected:
            raise RuntimeError(
                f"frozen source mismatch: {path}; expected {expected}, got {actual}"
            )
    return observed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def eta_key(values: Iterable[float]) -> tuple[float, float, float]:
    answer = tuple(round(float(value), 12) for value in values)
    if len(answer) != 3:
        raise ValueError(answer)
    return answer  # type: ignore[return-value]


def tuple_key(row: dict[str, Any]) -> tuple[str, tuple[float, float, float], int]:
    return str(row["state_id"]), eta_key(row["eta"]), int(row["seed"])


def _same_cached_record(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare every scientifically relevant value in two cached rollouts.

    CPU/GPU conic solves can differ at the last few decimal places, so use the
    same small tolerances as the established V2 cache audit.  In particular,
    never accept duplicates merely because their terminal outcomes agree: the
    first-step action and J_def determine the eventual oracle and label.
    """
    exact_fields = (
        "arm_id", "state_id", "seed", "seed_index", "outcome",
        "execution_error", "steps", "terminal_step", "rng_namespace",
        "state_sha256",
    )
    if any(old.get(field) != new.get(field) for field in exact_fields):
        return False
    if eta_key(old["eta"]) != eta_key(new["eta"]):
        return False
    try:
        if not math.isclose(
            float(old["J_def"]), float(new["J_def"]), rel_tol=0.0, abs_tol=1e-7
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    first_old, first_new = old.get("first_step"), new.get("first_step")
    if first_old is None or first_new is None:
        return first_old is first_new
    exact_first_fields = (
        "first_projection_status", "second_projection_status",
        "first_retry", "second_retry",
    )
    if any(first_old.get(field) != first_new.get(field) for field in exact_first_fields):
        return False
    for field in ("u_flow", "u_safe", "g_raw", "u_exec"):
        try:
            a = np.asarray(first_old[field], dtype=np.float64)
            b = np.asarray(first_new[field], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        if a.shape != (4,) or b.shape != (4,) or not np.allclose(a, b, atol=1e-10, rtol=0.0):
            return False
    for field in ("first_min_linear_residual", "second_min_linear_residual"):
        try:
            if not math.isclose(
                float(first_old[field]), float(first_new[field]),
                rel_tol=0.0, abs_tol=1e-9,
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _validated_cache_contract() -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    """Return current protocol data/hash and hash-verified startup state rows."""
    protocol_path = HERE / "protocol.json"
    state_manifest_path = HERE / "startup_state_manifest.jsonl"
    if not protocol_path.is_file() or not state_manifest_path.is_file():
        raise FileNotFoundError("startup cache contract is incomplete")
    states: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(state_manifest_path):
        state_id = str(row["state_id"])
        if state_id in states:
            raise RuntimeError(f"duplicate startup state_id: {state_id}")
        state_path = HERE / str(row["state_file"])
        if not state_path.is_file() or sha256(state_path) != row.get("state_sha256"):
            raise RuntimeError(f"startup snapshot hash mismatch: {state_id}")
        states[state_id] = row
    if not states:
        raise RuntimeError("startup state manifest is empty")
    protocol = json.loads(protocol_path.read_text())
    return sha256(protocol_path), protocol, states


def _stage_allowed_rows(
    stage_dir: Path, protocol_hash: str, protocol: dict[str, Any],
    states: dict[str, dict[str, Any]],
) -> dict[tuple[str, tuple[float, float, float], int], tuple[str, int]]:
    """Authenticate one stage definition and enumerate its permitted tuples."""
    definition_path = stage_dir / "stage_definition.json"
    if not definition_path.is_file():
        raise RuntimeError(f"cache stage lacks stage_definition.json: {stage_dir}")
    definition = json.loads(definition_path.read_text())
    if definition.get("stage") != stage_dir.name:
        raise RuntimeError(f"cache stage/name mismatch: {stage_dir}")
    if definition.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"cache stage protocol mismatch: {stage_dir}")
    arms_name = definition.get("arms_file")
    if not isinstance(arms_name, str) or Path(arms_name).name != arms_name:
        raise RuntimeError(f"invalid arms filename in cache stage: {stage_dir}")
    arms_path = HERE / arms_name
    if not arms_path.is_file() or sha256(arms_path) != definition.get("arms_sha256"):
        raise RuntimeError(f"cache stage arms hash mismatch: {stage_dir}")
    arms = json.loads(arms_path.read_text())
    if not isinstance(arms, list):
        raise RuntimeError(f"cache stage arms are not a list: {stage_dir}")
    allowed: dict[tuple[str, tuple[float, float, float], int], tuple[str, int]] = {}
    arm_ids: set[str] = set()
    state_eta: set[tuple[str, tuple[float, float, float]]] = set()
    frozen_seeds = [int(seed) for seed in protocol["oracle_seed_default"]]
    allowed_etas = {eta_key((0.0, 0.0, 0.0))}
    allowed_etas.update(eta_key(eta) for eta in protocol["candidate_etas"])
    for arm in arms:
        arm_id = str(arm["arm_id"])
        state_id = str(arm["state_id"])
        eta = eta_key(arm["eta"])
        seeds = [int(seed) for seed in arm["seeds"]]
        if arm_id in arm_ids or (state_id, eta) in state_eta:
            raise RuntimeError(f"duplicate arm in {arms_path}: {arm_id}")
        if state_id not in states or eta not in allowed_etas or seeds != frozen_seeds:
            raise RuntimeError(f"invalid arm contract in {arms_path}: {arm_id}")
        arm_ids.add(arm_id)
        state_eta.add((state_id, eta))
        for seed_index, seed in enumerate(seeds):
            key = (state_id, eta, seed)
            if key in allowed:
                raise RuntimeError(f"duplicate cache tuple in arms file {arms_path}: {key}")
            allowed[key] = (arm_id, seed_index)
    return allowed


def _validate_cached_row(
    row: dict[str, Any], path: Path,
    allowed: dict[tuple[str, tuple[float, float, float], int], tuple[str, int]],
    states: dict[str, dict[str, Any]],
) -> tuple[str, tuple[float, float, float], int]:
    """Fail closed unless a record belongs to its frozen arm and state bytes."""
    try:
        key = tuple_key(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"malformed cached tuple in {path}") from exc
    if key not in allowed:
        raise RuntimeError(f"cached tuple is absent from frozen stage arms: {key} in {path}")
    expected_arm_id, expected_seed_index = allowed[key]
    state = states[key[0]]
    if str(row.get("arm_id")) != expected_arm_id or int(row.get("seed_index", -1)) != expected_seed_index:
        raise RuntimeError(f"cached tuple arm/seed-index mismatch: {key} in {path}")
    if row.get("state_sha256") != state.get("state_sha256"):
        raise RuntimeError(f"cached tuple state hash mismatch: {key} in {path}")
    if int(row.get("rng_namespace", -1)) != int(state["rng_namespace"]):
        raise RuntimeError(f"cached tuple RNG namespace mismatch: {key} in {path}")
    outcome = row.get("outcome")
    if outcome not in {"success", "deadlock", "timeout", "collision", "execution_error"}:
        raise RuntimeError(f"invalid cached outcome: {key} in {path}")
    if (outcome == "execution_error") != (row.get("execution_error") is not None):
        raise RuntimeError(f"cached execution-error payload mismatch: {key} in {path}")
    try:
        steps = int(row["steps"])
        terminal_step = int(row["terminal_step"])
        j_def = float(row["J_def"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"malformed cached rollout metrics: {key} in {path}") from exc
    if steps < 0 or terminal_step != int(state["step"]) + steps or not math.isfinite(j_def) or j_def < 0.0:
        raise RuntimeError(f"invalid cached rollout metrics: {key} in {path}")
    first = row.get("first_step")
    if outcome != "execution_error" and first is None:
        raise RuntimeError(f"cached completed rollout lacks first step: {key} in {path}")
    if first is not None:
        for field in ("u_flow", "u_safe", "g_raw", "u_exec"):
            try:
                value = np.asarray(first[field], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"malformed cached first step: {key} in {path}") from exc
            if value.shape != (4,) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid cached first step: {key} in {path}")
    return key


def load_effective_records(raw_root: Path | None = None) -> tuple[dict, list[dict]]:
    """Load only stage-bound, state-bound records; reject conflicting duplicates."""
    root = HERE / "raw" if raw_root is None else raw_root
    protocol_hash, protocol, states = _validated_cache_contract()
    effective: dict[tuple[str, tuple[float, float, float], int], dict] = {}
    duplicates: list[dict] = []
    for path in sorted(root.glob("*/records.jsonl")) if root.exists() else []:
        allowed = _stage_allowed_rows(path.parent, protocol_hash, protocol, states)
        for row in read_jsonl(path):
            key = _validate_cached_row(row, path, allowed, states)
            if key in effective:
                old = effective[key]
                if not _same_cached_record(old, row):
                    raise RuntimeError(f"conflicting cached tuple {key} in {path}")
                duplicates.append({"key": repr(key), "path": str(path)})
            else:
                effective[key] = row
    return effective, duplicates
