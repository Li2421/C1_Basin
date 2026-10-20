"""Write the mandatory stop report after the preregistered G1 evidence failure."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: dict) -> None:
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    prereg = json.loads((HERE / "PREREGISTRATION.receipt.json").read_text())
    assert prereg["sha256"] == digest(HERE / "PREREGISTRATION.md")
    audit = json.loads((HERE / "G1_EVIDENCE_AUDIT.json").read_text())
    assert audit["status"] == "G1_FAILED_UNVERIFIABLE_HISTORICAL_FIRST_ACTION"
    assert [r["db_records"] for r in audit["runs"]] == [1360, 1024]
    assert all(r["source_files_missing"] == r["source_hash_mismatch"] == 0 for r in audit["runs"])
    assert all(r["record_first_action_present"] == r["record_first_action_bytes_present"] == r["record_trace_path_present"] == 0 for r in audit["runs"])
    historical_g0 = ROOT / "diagnostics/four_controller_census_v1/CONSISTENCY_GATE.json"
    old = json.loads(historical_g0.read_text())
    assert old["passed"] and old["required"] == len(old["checks"]) == 8
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history = {
        "utc": now,
        "status": "AUDITED_BEFORE_ROUND19_EXECUTION",
        "round14b_report": "diagnostics/four_controller_census_v1/REPORT.md",
        "round14b_g0_path": str(historical_g0.relative_to(ROOT)),
        "round14b_g0_sha256": digest(historical_g0),
        "round14b_g0_exact_eight_passed": True,
        "round14b_census": "384 old-state Q2 cases for v13 best/final; exact execution identity and state semantics retained",
        "e8_runtime": "diagnostics/controller_ff_factorial_repair_v1/",
        "round13_eta_switch": "diagnostics/cross_scene_eta_schedule_v1/",
        "intervention_v12_records": 1360,
        "intervention_v10_wide_records": 1024,
        "intervention_conditioning": "canonical v13 Flow at t0, alternate checkpoint Flow for future steps",
        "reusable": "historical code and outcomes only for their original execution identity",
        "missing_for_g1": "recorded historical first-action bytes (and any per-step trace)",
        "new_label_need_if_gates_passed": "25 eligible checkpoints on 32 fresh Four states and fixed eta/schedule arms; not executed",
    }
    write("HISTORICAL_EVIDENCE_AUDIT.json", history)
    gates = {
        "utc": now,
        "status": "STOP_STAGE0_G1_NOT_PASSED",
        "G0": {
            "status": "NOT_RUN_AFTER_G1_STOP",
            "historical_round14b_same_v13_gate": "PASS_8_OF_8",
            "historical_gate_sha256": digest(historical_g0),
            "round19_adapter_byte_equivalence": "UNVERIFIED",
        },
        "G1": {
            "status": "NOT_PASSED_UNVERIFIABLE",
            "required": "8 v12 and 8 v10-wide records each compare outcome, episode length and recorded first-action bytes",
            "v12_records": 1360,
            "v10_wide_records": 1024,
            "historical_first_action_records": 0,
            "reason": "All source journals omit first_action and first_action_bytes; DB schema omits first_action; no trace paths. Replayed values cannot stand in for independent historical bytes.",
            "evidence_audit_sha256": digest(HERE / "G1_EVIDENCE_AUDIT.json"),
        },
        "G2": {
            "status": "NOT_RUN_AFTER_G1_STOP",
            "read_only_candidate_audit": "13 same-environment best plus 12 parameter-distinct final; config and forward gate not executed",
        },
        "no_round19_gates_or_probes_executed": True,
        "no_new_success_labels_generated": True,
    }
    write("GATE_REPORT.json", gates)
    result = {
        "utc": now,
        "status": "INCONCLUSIVE_STOPPED_AT_G1",
        "hypotheses": {
            "stage0_equivalence": "INCONCLUSIVE",
            "four_controller_eta_regime_difference": "INCONCLUSIVE",
            "H_compress_Four": "INCONCLUSIVE",
            "probe_to_E_learnability": "INCONCLUSIVE",
        },
        "preregistration_sha256": digest(HERE / "PREREGISTRATION.md"),
        "g1_evidence_audit_sha256": digest(HERE / "G1_EVIDENCE_AUDIT.json"),
        "gate_report_sha256": digest(HERE / "GATE_REPORT.json"),
        "new_probes": 0,
        "new_success_labels": 0,
        "new_rollouts": 0,
        "db_preflight_postflight": "NOT_APPLICABLE_NO_ROLLOUTS_SUBMITTED",
        "video_delivery": "INCOMPLETE_NOT_ATTEMPTED_AFTER_STAGE0_STOP; Four deadlock roles unavailable",
    }
    write("results.json", result)
    lead = ("阶段 0 等价门未通过：G1 的历史记录没有首动作字节，无法完成规定的逐位比较；"
            "G0/G2 在此硬停止后未运行。Four 是否存在跨 controller 的 η 泛化问题为 INCONCLUSIVE；"
            "H_compress_Four 与由无标签探测学习 E 均为 INCONCLUSIVE（未进入相应阶段）。")
    report = [
        "# Round19 Four controller axis", "", lead, "",
        "## 事实与门结果", "",
        "预注册已在任何 Round19 等价门、探测或新成功标签前写入并留 UTC/SHA256 收据。",
        "旧 round14B 的 v13 等价门在其原有适配器上 8/8 通过；这不是本轮新适配器的 G0。",
        "只读扫描共享 DB 对应的原始 journal：v12 best 1,360 条，v10-wide best 1,024 条；源文件都存在，原始记录哈希均与 DB 匹配。两组记录的 `first_action`、`first_action_bytes` 和 `trace_path` 均为 0 条；DB rollout 表也无首动作列。详见 `G1_EVIDENCE_AUDIT.json`。",
        "G1 要求每个 checkpoint 8 条历史记录的终止原因、步数和首动作字节逐位复现。没有历史首动作，关键比较无法成立；按预注册判为 NOT_PASSED_UNVERIFIABLE，并执行阶段 0 硬停止。G0/G2 的本轮执行状态均为 NOT_RUN，不将历史 v13 结果冒充新适配器验证。",
        "只读配置审计找到 13 个同环境 best 和 12 个参数不同的 final；这只是候选清单，G2 的 config/前向门尚未运行。",
        "", "## 解释与范围", "",
        "这一停止说明当前历史证据不足以验证逐 checkpoint 适配层，并不证明 Four controller 共享 η regime，也不否定 Ring 的机制。阶段 1–4 均未开始，故没有新探测、state、rollout、标签或模型。没有触及共享 DB 写入。",
        "本轮未形成 Four 评价 batch 的视频；无合格 deadlock 判据或同初态死锁基线，不能把 timeout 用作 deadlock_show/deadlock_resolution。视频交付标记为 INCOMPLETE，不声称完成。",
        "", "## 可追溯文件", "",
        "`PREREGISTRATION.md`、`PREREGISTRATION.receipt.json`、`HISTORICAL_EVIDENCE_AUDIT.json`、`G1_EVIDENCE_AUDIT.json`、`GATE_REPORT.json`、`results.json`、`STOP.receipt.json`。",
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n")
    handoff = [
        "# Round19 handoff", "", lead, "",
        "工作目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。本轮文件仅在 `diagnostics/four_controller_axis_v1/`。",
        "先读预注册及收据，再读 `G1_EVIDENCE_AUDIT.json` 和 `GATE_REPORT.json`。`audit_g1_evidence.py` 只读共享 DB 与历史 journal；`finalize_stop.py` 生成停止报告。",
        "G1 不能被同代码重新计算的首动作补足；需要独立保存的历史首动作字节或全程轨迹，或者另行预注册不同的验证门。当前授权下不得越过此门启动阶段 1–4。",
        "旧 v12/v10-wide intervention 采用 canonical v13 t0 再切到替代 Flow 的 conditioning，不能直接当作本轮纯替代 Flow 的成功标签。旧 round14B 的 v13 G0 8/8 通过，但 Round19 新适配器未建立。",
        "视频交付未完成；Four 不具备可用的正式 deadlock 判据，timeout 不可代替死锁证据。",
    ]
    (HERE / "HANDOFF.md").write_text("\n".join(handoff) + "\n")
    files = ["PREREGISTRATION.md", "PREREGISTRATION.receipt.json", "audit_g1_evidence.py",
             "G1_EVIDENCE_AUDIT.json", "HISTORICAL_EVIDENCE_AUDIT.json", "GATE_REPORT.json",
             "results.json", "REPORT.md", "HANDOFF.md", "finalize_stop.py"]
    write("STOP.receipt.json", {"status": "STOPPED_AS_PREREGISTERED", "utc": now,
                                "reason": "G1 historical first-action bytes absent",
                                "files": {name: digest(HERE / name) for name in files}})
    print(lead)


if __name__ == "__main__":
    main()
