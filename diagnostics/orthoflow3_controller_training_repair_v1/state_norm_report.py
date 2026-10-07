"""Apply the unchanged source gate to the input-scale control only."""
import numpy as np
from .data import OUT, read, write
from .train import SEEDS, STATE_NORMALIZED


def checks(kind):
    full = [read(OUT / "models" / kind / f"seed{s}/summary.json") for s in SEEDS]
    baseline = {(k, s): read(OUT / "models" / k / f"seed{s}/summary.json") for k in ("eta_only", "additive") for s in SEEDS}
    acc = lambda r, key: r[key]["accuracy"] if r[key]["accuracy"] is not None else 0.
    values = {
        "controller_reversal_evidence": full[0]["correct"]["controller_reversals"]["total"] >= 10 and full[0]["correct"]["controller_reversals"]["families"] >= 3,
        "controller_reversal_accuracy": np.median([acc(r["correct"], "controller_reversals") for r in full]) >= .60,
        "beats_eta_only_two_seeds": sum(r["correct"]["selected_B15"] > baseline["eta_only", s]["correct"]["selected_B15"] for r, s in zip(full, SEEDS)) >= 2,
        "beats_additive_two_seeds": sum(r["correct"]["selected_B15"] > baseline["additive", s]["correct"]["selected_B15"] for r, s in zip(full, SEEDS)) >= 2,
        "context_used": np.median([r["wrong_controller"]["NLL"]-r["correct"]["NLL"] for r in full]) >= .02 or np.median([r["correct"]["selected_B15"]-r["wrong_controller"]["selected_B15"] for r in full]) > 0,
        "state_interaction_accuracy": np.median([acc(r["correct"], "state_reversals") for r in full]) >= .60,
        "state_response_shuffle_degrades": np.median([acc(r["correct"], "state_reversals")-acc(r["state_response_shuffle"], "state_reversals") for r in full]) > 0,
    }
    return {k: bool(v) for k, v in values.items()}, full


def main():
    original, _ = checks("physical_context")
    assert original == read(OUT / "source_gate.json")["checks"], "Gate thresholds must remain unchanged"
    tested, models = checks(STATE_NORMALIZED)
    result = {"checks": tested, "passed": all(tested.values()), "target_labels_opened": False,
              "original_gate_unmodified": True, "new_continuations": 0,
              "seeds": [{"seed": r["seed"], "best_step": r["best_step"],
                         **{v: {k: r[v][k] for k in ("NLL", "MAE", "eligible", "selected_B15",
                                                    "controller_reversals", "state_reversals")}
                            for v in ("correct", "wrong_controller", "state_shuffle", "state_response_shuffle")}}
                        for r in models]}
    write(OUT / "state_normalized_gate.json", result)
    print(result)


if __name__ == "__main__":
    main()
