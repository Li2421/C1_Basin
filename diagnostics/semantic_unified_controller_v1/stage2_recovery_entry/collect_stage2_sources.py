"""Run frozen Safety sources and materialize Stage-2 generic anchors.

The implementation delegates the already-tested exact state/history/Flow
serialization to the single-segment collector after replacing only its frozen
paths and source-row enumeration.  No branch labels or training occur here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage2_recovery_entry"
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1/collect_safety_sources.py"


def load_base():
    spec = importlib.util.spec_from_file_location("single_segment_source_collector", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tested source collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_for_collection(source: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = ("train", "validation", "calibration")
    rows = [row for split in allowed for row in source["sources"][split]]
    if len(rows) != 160:
        raise RuntimeError(("expected 160 Stage-2 sources", len(rows)))
    if any(row["split"] not in allowed for row in rows):
        raise RuntimeError("unexpected split or final-test leakage")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate root source")
    return rows


def main() -> None:
    base = load_base()
    base.HERE = HERE
    base.SOURCE_MANIFEST = HERE / "source_split_manifest.json"
    base.PROTOCOL = HERE / "collection_protocol.json"
    base.HASHES = ROOT / "diagnostics/semantic_unified_controller_v1/frozen_assets.json"
    base.rows_for_collection = rows_for_collection
    base.main()


if __name__ == "__main__":
    main()
