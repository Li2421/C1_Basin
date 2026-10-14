# Round13b handoff

Current state: STOPPED_AT_TRAINING_REPRODUCTION_GATE. Read REPORT.md, PREREGISTRATION.md, TRAINING_PLAN.json and TRAINING_REPRODUCTION_GATE.json. The required <=1% best/final dev-loss reproduction gate failed (6.693%, 3.974%). Do not start the 14 new-seed trainings, R5 probes, labels, or main rollouts under this preregistration. No target success labels exist. Reproduction files are under reproduction_gate/base_u_v2_macflow/. All writes were confined to this experiment directory; no historical asset or DB was changed.

Historical reference: diagnostics/ring_exchange_stage1/base_u_v2_macflow/{config.json,metrics.jsonl,training_summary.json,best.pkl,final.pkl}; source protocol and FP32 repairs: diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/. The numerical source evidence remains valid for its original semantics. A new attempt requires explanation of the divergence and a recorded gate pass before any target outcomes.
