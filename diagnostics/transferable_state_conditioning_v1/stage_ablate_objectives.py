"""DEVELOPMENT-only concat/residual comparisons on cached held-out Q4 labels."""
from __future__ import annotations

import copy
import json

from diagnostics.physical_state_critic_formal_v1.train_unified import select_key
from diagnostics.transferable_state_conditioning_v1.assemble_source import SCENES, read, sha
from diagnostics.transferable_state_conditioning_v1.stage_source import STAGE
from diagnostics.transferable_state_conditioning_v1.stage_evaluate_loso import audit
from diagnostics.transferable_state_conditioning_v1.train_baselines import SEEDS, NEAR_TIE


def best_run(panel, kind):
    rows = [read(STAGE / "fits" / panel / kind / f"seed{s}" / "complete.json")
            for s in SEEDS]
    return min(rows, key=lambda r: (*select_key(r["selected"], NEAR_TIE), r["seed"]))


def main():
    frozen = read(STAGE / "final_models_frozen.json")
    for kind in ("state_concat", "state_residual"):
        config = copy.deepcopy(frozen)
        for scene in SCENES:
            panel = f"loso_{scene}"
            run = best_run(panel, kind)
            ck = STAGE / "fits" / panel / kind / f"seed{run['seed']}" / "checkpoint.msgpack"
            assert sha(ck) == run["checkpoint_sha256"]
            info = config["panels"][panel]
            info.update(state_kind=kind, state_seed=run["seed"],
                        state_checkpoint=str(ck),
                        state_checkpoint_sha256=run["checkpoint_sha256"])
            for bank in ("proposal", "anchor"):
                result = audit(scene, bank, config, variant=kind)
                print(json.dumps({"kind":kind,"scene":scene,"bank":bank,
                                  "mean":result["mean_success"],
                                  "state_minus_mismatch":result["paired_state_bootstrap"]["matched_minus_mismatch"]}),
                      flush=True)


if __name__ == "__main__":
    main()
