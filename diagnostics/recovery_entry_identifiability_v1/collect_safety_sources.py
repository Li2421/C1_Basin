"""Collect frozen Safety roots and exact predeclared queried states.

This is a narrow adapter around the tested single-segment source collector.
It replaces only artifact paths, development-source enumeration, and adds a
hard deployment-feature dimension check.  It does not run branches or train a
probe.  At most two processes are accepted; each process is capped at three
CPU threads so the declared aggregate ceiling is six.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_entry_identifiability_v1"
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1/collect_safety_sources.py"


def cap_process_threads(limit: int = 3) -> None:
    """Cap this collector process before NumPy/JAX are imported by the base."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "JAX_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        current = os.environ.get(name)
        try:
            value = min(int(current), limit) if current is not None else limit
        except ValueError:
            value = limit
        os.environ[name] = str(max(1, value))
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" not in flags:
        flags = f"{flags} --xla_cpu_multi_thread_eigen=false".strip()
    if "intra_op_parallelism_threads" not in flags:
        flags = f"{flags} intra_op_parallelism_threads={limit}".strip()
    os.environ["XLA_FLAGS"] = flags


def load_base():
    spec = importlib.util.spec_from_file_location("single_segment_source_collector", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tested single-segment source collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_for_collection(source: dict[str, Any]) -> list[dict[str, Any]]:
    if source.get("development_only") is not True or source.get("final_test_generated_or_used") is not False:
        raise RuntimeError("source manifest is not development-only")
    if set(source.get("sources", {})) != {"development"}:
        raise RuntimeError("unexpected source group or final-test leakage")
    rows = list(source["sources"]["development"])
    if len(rows) != 120:
        raise RuntimeError(("expected 120 development roots", len(rows)))
    if len({row["source_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate development root source")
    for row in rows:
        if row.get("split") != "development" or row.get("development_only") is not True:
            raise RuntimeError((row.get("source_id"), "invalid cohort"))
        requested = row.get("requested_anchor_steps", [])
        if len(requested) != 2 or len(set(requested)) != 2:
            raise RuntimeError((row.get("source_id"), "requires two distinct anchor requests"))
        if any(not 0 <= int(step) < 850 for step in requested):
            raise RuntimeError((row.get("source_id"), "anchor outside official horizon"))
    return rows


def main() -> None:
    cap_process_threads()
    base = load_base()
    base.HERE = HERE
    base.SOURCE_MANIFEST = HERE / "development_source_manifest.json"
    base.PROTOCOL = HERE / "collection_protocol.json"
    base.HASHES = ROOT / "diagnostics/semantic_unified_controller_v1/frozen_assets.json"
    base.rows_for_collection = rows_for_collection

    original_save_anchor = base.save_anchor

    def checked_save_anchor(**kwargs):
        feature = kwargs["feature"]
        if getattr(feature, "shape", None) != (214,):
            raise RuntimeError(("deployment feature shape mismatch", getattr(feature, "shape", None)))
        row = original_save_anchor(**kwargs)
        row.update({
            "schema": "recovery_entry_generic_safety_state_v1",
            "development_cohort": "development",
            "selection_terminal_relative_independent": True,
            "selection_failure_type_independent": True,
            "selection_recovery_outcome_independent": True,
        })
        return row

    base.save_anchor = checked_save_anchor
    base.main()


if __name__ == "__main__":
    main()
