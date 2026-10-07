"""Copy only frozen metadata and write implementation audits; never mutate inputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import group_dro_gate_runner as dro


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    source_manifest = ROOT / "diagnostics" / "gate_source_balancing_v1" / "frozen_fold_manifest.json"
    target_manifest = HERE / "frozen_fold_manifest.json"
    if not target_manifest.exists():
        shutil.copy2(source_manifest, target_manifest)
    elif target_manifest.read_bytes() != source_manifest.read_bytes():
        raise RuntimeError("existing Group-DRO frozen manifest differs from the source-balanced frozen manifest")
    dro.write_config_template(HERE / "config.json")
    dro.write_preserved_loss_audit(HERE / "preserved_loss_code_audit.md")
    dro.base.write_json(HERE / "implementation_manifest.json", {
        "frozen_manifest_source": str(source_manifest),
        "frozen_manifest_byte_identical": target_manifest.read_bytes() == source_manifest.read_bytes(),
        "implementation": "group_dro_gate_runner.py",
        "loss_modes": list(dro.LOSS_MODES),
        "pre_registered_eta_q": list(dro.DEFAULT_ETA_Q),
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "feature_schema_changed": False,
        "oracle_labels_changed": False,
    })


if __name__ == "__main__":
    main()
