"""Fail-closed SHA-256 verification for frozen semantic-controller assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_assets.json"
ASSET_KEYS = (
    "flowbc",
    "environment",
    "projection_constraints",
    "projection_retry",
    "event_priority_and_monitor",
    "startup_feature_builder",
    "direct_g_feature_normalization",
    "eta_basis_and_corrector",
    "local_direct_g",
    "structured_eta",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    checked = []
    for key in ASSET_KEYS:
        item = payload[key]
        path = Path(item["path"])
        observed = sha256(path)
        if observed != item["sha256"]:
            raise RuntimeError((key, str(path), observed, item["sha256"]))
        checked.append({"asset": key, "path": str(path), "sha256": observed})
    print(json.dumps({"status": "PASS", "checked": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
