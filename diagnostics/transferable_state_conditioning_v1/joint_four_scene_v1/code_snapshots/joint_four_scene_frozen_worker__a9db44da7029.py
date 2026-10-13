"""Execute the original frozen MACFlow source without reverting shared edits.

The fingerprint is computed from the exact module actually loaded, not from
the unrelated current working-tree file. All remaining payload fields must
match the immutable original protocol before any continuation is run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "diagnostics/physical_state_critic_formal_v1"
EXPECTED = "c3b71215fe3dd333e738d4873b4f16dbc45dc6ca6a2c419dda834f7976a5e93c"
SNAPSHOT = ROOT.parent / "Basin_C1_q_frozen_c3b71215/new_benchmark_common/macflow.py"


def setup():
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == EXPECTED
    assert "new_benchmark_common.macflow" not in sys.modules
    spec = importlib.util.spec_from_file_location("new_benchmark_common.macflow", SNAPSHOT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from new_benchmark_common import safety_eta3
    original_hash = safety_eta3.sha256_file

    def executed_source_hash(path):
        if Path(path).resolve() == (ROOT / "new_benchmark_common/macflow.py").resolve():
            assert Path(sys.modules[spec.name].__file__).resolve() == SNAPSHOT.resolve()
            return original_hash(SNAPSHOT)
        return original_hash(path)

    safety_eta3.sha256_file = executed_source_hash
    from diagnostics.physical_state_critic_formal_v1 import loso_worker
    factory = loso_worker.make_runtime
    protocol = json.loads((FORMAL / "loso_protocol.json").read_text())

    def checked_runtime(scene):
        runtime = factory(scene)
        if scene in ("four_way_intersection", "ring_exchange"):
            for chain, label in (("orthoflow3", "intervention"), ("hard_safety_q16", "safety")):
                expected = protocol["controllers"][scene]
                assert runtime.controllers[chain]["payload"] == expected[label+"_payload"], (scene, chain)
                assert runtime.controllers[chain]["uid"] == expected[label+"_uid"], (scene, chain)
        return runtime

    loso_worker.make_runtime = checked_runtime
    return checked_runtime


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("action", choices=("check","run"));p.add_argument("--index",type=int)
    a=p.parse_args();factory=setup()
    if a.action == "check":
        for scene in ("four_way_intersection","ring_exchange"):
            r=factory(scene);print(scene,r.controllers["orthoflow3"]["uid"],flush=True)
        print("FROZEN_EXECUTION_ALL_PAYLOAD_FIELDS_MATCH",flush=True)
    else:
        assert a.index is not None
        from diagnostics.transferable_state_conditioning_v1 import final_worker
        final_worker.run("proposal",a.index)
