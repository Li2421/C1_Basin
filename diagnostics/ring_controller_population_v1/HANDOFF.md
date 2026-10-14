# Round15 handoff

UTC: 2026-10-13T22:30:00.478466+00:00
Complete waves: 4/4; controllers: 56/56
Hypotheses: {'H_cov_L': 'REJECTED', 'H_cov_E': 'REJECTED', 'H_switch': 'REJECTED'}

## Design and evidence

Four independent GPU training seeds per each of 14 Ring Flow recipes are original seed +30001..+30004. All 56 controllers share 32 new physical Ring states. Q4 uses future seeds 0..3 and F10/F13/F2/S200/S300/S400/S500/S600. R5 and R6 are outcome-blind probes frozen before the first new full-rollout outcome. Core physics, safety, controller and success code was not changed; the only action intervention changes eta at the selected step.

PREREGISTRATION.md, ANALYSIS_DETAIL.md and their receipts fix analysis definitions. HISTORICAL_EVIDENCE_AUDIT.json, HISTORICAL_CODE_AUDIT.json, CHECKPOINT_NOVELTY_AUDIT.json, COMPATIBILITY_AUDIT.json and STATE_DEDUP_AUDIT.json document reuse and compatibility. CONSISTENCY_GATE.json records 32 exact historical replays. The rollout cache index is shared_rollout_db/rollout.sqlite; completed_POSTFLIGHT.json is the standard reuse-oriented postflight, and POSTFLIGHT_UNKNOWN_AUDIT.json confirms all standard missing cases are present numerical failures retained as unknown.

## Data and results

dataset.jsonl has one row per controller × state × action × future seed in complete waves, including unknown flags and raw-record links. Raw traces and per-case journal shards are retained under wave*/ and journals/. results.json has lower/upper rates, labels, split-half reliability, ICC intervals, grouped CV predictions, regrets and hypotheses. REPORT.md starts with the three required answers. All learning curves are DEVELOPMENT. The pure-Q video exemption was frozen before results.

## Reproduce

Python: /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

From the repository root, run `python -B diagnostics/ring_controller_population_v1/analyze.py` only after complete-wave verification and DB postflight. Do not regenerate unknown rollout labels.

Hard local-time deadlines were 07:30 for a new wave, 10:00 for all rollouts/probes, and 10:30 for REPORT/HANDOFF and compute release. Review WAVE_PLAN.json, wave*/START.receipt.json and RESOURCE_USAGE.jsonl for timing and resource evidence.
