"""Finalize the repair checkpoint honestly; the research acceptance gate failed."""
import json
from . import data
from .train import SEEDS


def snapshot():
    with data.connect(True) as db:
        rows = [dict(r) for r in db.execute("SELECT * FROM rollout WHERE experiment_uid=? ORDER BY rollout_uid", (data.EXPERIMENT,))]
    return {"records": len(rows), "content_digest": data.digest(rows)}


def main():
    before = snapshot()
    original = data.read(data.OUT / "merge_audit.json")
    data.merge()
    after = snapshot()
    assert before == after
    data.write(data.OUT / "merger_idempotency_audit.json", {"before": before, "after": after,
        "unchanged": True, "prior_merge_audit": original, "new_continuations": 0})
    audit = data.read(data.OUT / "postflight_audit.json")
    gate = data.read(data.OUT / "source_gate.json")
    norm_gate = data.read(data.OUT / "state_normalized_gate.json")
    assert audit["all_attempted_records_in_db"] and not audit["not_attempted"]
    assert not gate["passed"] and not norm_gate["passed"]
    model = {str(s): data.read(data.OUT / "models/physical_context" / f"seed{s}/summary.json") for s in SEEDS}
    provenance = data.read(data.OUT / "dataset_provenance.json")
    included = {u for r in provenance for u in r["rollout_uids"]}
    result = {
        "scope": "Data/execution/training repair and source-family validation; not a passed unseen-controller or cross-scene experiment",
        "verdict": "DATA_REPAIR_COMPLETE_SOURCE_INTERACTION_GATE_NOT_PASSED",
        "global_db": {"new_native_records": after["records"], "requested": audit["requested"],
            "planned_historical_reuse": 108, "planned_non_numerical_reusable": audit["exact_reusable"],
            "numerical_records_preserved_not_imputed": audit["numerical_not_imputed"],
            "not_attempted": 0, "collision": audit["collision"], "native_conflicts": 0,
            "incorrect_precision_records_preserved_and_quarantined": 4825,
            "dataset_all_canonical_seed_records_including_extra_history": len(included),
            "training_numeric_exclusions": 12, "merger_repeat_idempotent": True},
        "repairs_verified": ["Native float32 Flow sampler isolated from process-global context import side effects",
            "Every native journal validated against exact physical state, exact eta, seed, checkpoint, basis, safety and RNG fingerprints",
            "Complete true-t0 30-state x 16-eta x 3-controller matrix; 24 TRAIN and 6 VAL source families, no overlap",
            "Equal total controller likelihood mass 1/3 with matched minibatch pairs",
            "Observed successes/failures only; numerical failures excluded and Q16 bounds retained",
            "All 1440 H20 contexts valid; 48 action path checks agree exactly",
            "Three seeds, matched 1500 steps; source VAL NLL only for checkpoint selection",
            "CPU/GPU logit replay passed; all dataset and normalization versions recorded"],
        "source_validation": {"independent_families": 6, "state_controller_cases": 18,
            "oracle_eligible": 17, "coverage_failure": 1,
            "eta_only_B15_by_seed": [13, 13, 13], "additive_B15_by_seed": [13, 13, 13],
            "physical_context_B15_by_seed": [model[str(s)]["correct"]["selected_B15"] for s in SEEDS],
            "wrong_context_B15_by_seed": [model[str(s)]["wrong_controller"]["selected_B15"] for s in SEEDS],
            "controller_reversals_correct_of_36": [model[str(s)]["correct"]["controller_reversals"]["correct"] for s in SEEDS],
            "state_reversals_correct_of_48": [model[str(s)]["correct"]["state_reversals"]["correct"] for s in SEEDS],
            "controller_context_used": True, "state_interaction_learned_reliably": False,
            "selection_gain_statistically_established": False},
        "zero_rollout_followups": {"local_baselines": "source_failure_diagnosis.json",
            "finite_count_noise": "source_count_noise_audit.json",
            "source_train_heterogeneity_columns_BY005": 17,
            "known_controller_eta_prior_B15": "16/17; diagnostic only, not zero-shot and not state-conditioned",
            "input_scale_control": "state_normalized_gate.json",
            "input_scale_control_adopted": False},
        "scientific_status": {"controller_information_in_weights_and_selection": "SUPPORTED_ON_SOURCE_HELD_OUT_FAMILIES",
            "state_variation_is_only_Q4_noise": "NOT_SUPPORTED; first-four-seed common-Q null rejected in 17/48 columns with BY correction",
            "current_h_C_information_sufficiency": "UNDERRESOLVED",
            "state_learning_failure": "REMAINS; simple local methods and neural model do not reliably resolve reversals",
            "unseen_controller_or_LOSO_success": "NOT_ESTABLISHED_BY_THIS_REPAIR"},
        "frozen_target_opened": False, "generator_modified": False,
        "budget": {"actual_new_continuations_including_quarantined": 13933,
            "declared_envelope": 15360, "remaining": 1427,
            "full_unopened_24_state_confirmation_cost_if_uncached": 6144,
            "confirmation_not_authorized_by_current_envelope_and_source_gate_not_passed": True},
        "next_required_evidence": "Source-side test separating predictable state-dependent residuals from controller-specific eta preference; independent target only after qualification. No automatic extra basin search or test-label opening.",
        "reproduce": {"alignment": "python -m diagnostics.orthoflow3_controller_training_repair_v1.alignment_audit --complete all",
            "training": "python -m diagnostics.orthoflow3_controller_training_repair_v1.train train --kind <eta_only|additive|physical_context> --seed <17|23|41>",
            "diagnosis": "python -m diagnostics.orthoflow3_controller_training_repair_v1.source_diagnosis"}}
    data.write(data.OUT / "repair_decision.json", result)
    state = data.read(data.OUT / "working_state.json")
    state.update(phase="data_repair_complete_source_gate_failed_diagnostics_complete",
                 new_native_records_in_db=9108, native_numerical_records_in_db=12,
                 source_gate_passed=False, state_scale_control_complete=True,
                 state_scale_control_adopted=False, future_stages=[],
                 independent_held_controller_test_started=False,
                 required_next_step=result["next_required_evidence"], decision_file="repair_decision.json")
    data.write(data.OUT / "working_state.json", state)
    hypotheses = data.read(data.OUT / "hypothesis_status.json")
    hypotheses["training_support_mismatch"]["status"] = "TRUE_T0_SUPPORT_REPAIRED; NOT_A_PROOF_OF_TRANSFER"
    hypotheses["sparse_state_eta_controller_crossing"]["status"] = "COMPLETE_CROSS_MATRIX_BUILT_AND_AUDITED"
    hypotheses["controller_loss_imbalance"]["status"] = "CORRECTED_AND_WEIGHT_AUDITED; CONTEXT_NOW_AFFECTS_SELECTION"
    hypotheses["H20_context_sufficiency"]["status"] = "UNDERRESOLVED; SOURCE_INTERACTION_GATE_FAILED"
    hypotheses["state_input_scale_as_sufficient_fix"] = {"status": "NOT_SUPPORTED", "evidence": "state_normalized_gate.json"}
    hypotheses["finite_count_noise_as_entire_state_variation"] = {"status": "NOT_SUPPORTED", "evidence": "source_count_noise_audit.json"}
    data.write(data.OUT / "hypothesis_status.json", hypotheses)
    with data.connect() as db, data.transaction(db):
        metadata = json.loads(db.execute("SELECT metadata_json FROM experiment WHERE experiment_uid=?",
                                        (data.EXPERIMENT,)).fetchone()[0])
        metadata.update(repair_stage_status=result["verdict"],
                        final_decision_path=str(data.OUT / "repair_decision.json"),
                        native_precision="float32", target_labels_opened=False,
                        excluded_precision_experiment=data.EXPERIMENT + "_precision_quarantine",
                        dataset_sha256=data.old.digest(data.OUT / "dataset.npz"))
        db.execute("UPDATE experiment SET metadata_json=?,end_time=CURRENT_TIMESTAMP WHERE experiment_uid=?",
                   (data.canonical(metadata), data.EXPERIMENT))
    print({"verdict": result["verdict"], "db": result["global_db"], "source": result["source_validation"]})


if __name__ == "__main__":
    main()
