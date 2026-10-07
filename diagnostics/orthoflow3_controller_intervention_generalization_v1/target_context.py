"""Frozen K16 target context replay. Cannot run before source model freeze."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_controller_context_loso_v1.evaluate import TARGET
from .train import FOLDS, OUT, read, write
from .rich_context import RichRuntime, cached


def build(fold):
    assert (OUT / "models_frozen.json").exists(), "source-only model freeze required"
    manifest = read(TARGET / fold / "manifest.json")
    scene = FOLDS[fold]
    if fold != "ring":
        physical = read(TARGET / fold / "physical.json")
    else:
        # The newer physical_inputs() helper can resample the stochastic
        # current Flow reference. Reuse the previously frozen physical replay
        # that was already verified byte-close to this exact K16 entity pool.
        previous = read(OUT.parent / "orthoflow3_controller_context_loso_v1" /
                        "target_context_ring.json")
        physical = previous["physical"]
        assert previous["input_replay_matches_frozen"] and len(physical) == len(manifest)
    x = rep.batch([rep.entities(p) for p in physical])
    frozen = dict(np.load(TARGET / fold / "entities.npz"))
    for k in frozen:
        np.testing.assert_allclose(x[k], frozen[k], atol=2e-6, rtol=2e-6)
    runtime = RichRuntime(scene)
    profile = read(OUT / "protocol.json")["profiles"].get(scene)
    if profile is None:
        from .intervention import ROOT
        if scene == "toy_giveway":
            toy = read(ROOT / "diagnostics/orthoflow3_controller_conditioning_probe_v1/protocol.json")
            alternate_path = toy["flow_paths"]["1"]
        else:
            raise RuntimeError(f"missing alternative controller for {scene}")
    else:
        alternate_path = profile["alternate_path"]
    wrong_runtime = RichRuntime(scene, alternate_path)
    contexts = []
    wrong_contexts = []
    errors = []
    for item, p in zip(manifest, physical):
        result = cached(runtime, {"state_uid": item["state_uid"], "physical": p}, [0., 0., 0.])
        if result["valid"]:
            contexts.append(result["features"]["mean"][:10])
        else:
            contexts.append([0.] * 10)
            errors.append({"state_uid": item["state_uid"], "error": result["error"]})
        wrong = cached(wrong_runtime, {"state_uid": item["state_uid"], "physical": p}, [0., 0., 0.])
        if wrong["valid"]:
            wrong_contexts.append(wrong["features"]["mean"][:10])
        else:
            wrong_contexts.append([0.] * 10)
            errors.append({"state_uid": item["state_uid"], "wrong_controller_error": wrong["error"]})
    np.savez_compressed(OUT / f"target_nominal_{fold}.npz",
                        context=np.asarray(contexts, np.float32),
                        wrong_context=np.asarray(wrong_contexts, np.float32))
    write(OUT / f"target_nominal_{fold}.json",
          {"scene": scene, "state_uids": [m["state_uid"] for m in manifest],
           "input_replay_matches_frozen": True, "target_labels_used": False,
           "source_models_frozen": True, "invalid_contexts": errors,
           "context_protocol": "H8 paired nominal Flow+safety response, two fixed probe streams"})
    print(json.dumps({"fold": fold, "states": len(contexts), "invalid": len(errors)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=FOLDS, required=True)
    args = parser.parse_args()
    build(args.fold)
