# Round28 handoff

Scope: docs/prompt_round28_decentralized_pipeline_20261015.md. All new files live in this directory; historical code unchanged, no commit. Python: /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python. Start by reading REPORT.md, results.json and PREREGISTRATION.md plus receipts.

## Execution and evidence

- design.py/setup and main_design already executed; do not recreate frozen files or states.
- pipeline.py and run.py are frozen new decentralized implementation and rollout runner. gate.py aggregates the 16 historical TT consistency replays; UNIT_TESTS.json covers local components. IMPLEMENTATION_FROZEN.receipt.json hashes immutable action code.
- Main Slurm21301: 18 CPU workers, no GPU, 12544 cases, full traces. Expert21319 uses one freed CPU after7 main shards finished:32 states, one privileged plan each. Worker errors are unknown, never rerun.
- Main submission preceded preflight sealing by roughly two seconds; dependency guaranteed sealing before execution. Read SUBMISSION_TIMING_AUDIT.json and original/current receipt; do not erase this deviation or claim preflight preceded submission.
- merge.py is the only SQLite merger; wraps official Ingestor with exact identity/hash validation. main/expert_CACHE_POSTFLIGHT.json + MERGE_RECEIPT.json reconcile all12576 new task rows. Gate replays are not inserted as new labels. Journals become .merged.
- Fresh-state audit checked DB identity and recoverable geometry, but some historical rows have unresolved physical state. STATE_DEDUP_AUDIT.json records the limitation, no fabricated complete physical dedup claim.

## Reproduce analysis from existing labels only

Run from repository root, one CPU, with OMP/MKL/OPENBLAS threads1:

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/decentralized_pipeline_v1/analysis.py
/home/zhihan/research/Basin_C1/.venv-video/bin/python -B diagnostics/decentralized_pipeline_v1/videos.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/decentralized_pipeline_v1/report.py
```

videos.py returns2 because formal deadlock roles are unavailable; run report.py regardless of that expected rejection. Never use `&&` to suppress report after this video gap. r28_finalize.sbatch captures the status, generates report, then propagates nonzero video acceptance. Existing supplementary videos are truthful diagnostics, not formally accepted deadlock videos. Manifest complete=false is intentional and required. No extra simulations to find illustrative successes.

## Statistical details and boundaries

14 controllers,32 states,2 future seeds,14 arms. Hierarchical bootstrap5000 controller thenstate clusters, all arms/seeds paired, oracles reselected per draw. Unknown preserved; success lower0/upper1 only for bounds, not a known failure. DC1/DC4 decision uses range of LOWER CI endpoints across compatible unknown assignments, not full CI overlap. DC2 exact regret interval excludes S500 self comparison. The four-arm oracle is descriptive in-sample. DC3/DC5 safety and stalls are descriptive. Ring has no formal deadlock criterion.

Policy inference still shares observations/public key; each robot executes only its own queried row. New local safety/correction does not establish independent-observation policies. RHR comparison covers only preregistered fixed heuristic. Geometry and native continuous safety/event checks unchanged, x64False restored after all imports; local QP geometry remains float64. No tuning after outcomes. Plot rendering uses the existing .venv-c1-plots Python3.12; videos use existing .venv-video Python3.12. Numerical analysis and all rollouts remain on prescribed .venv-c1 Python3.11. No environments were modified. POSTPROCESS_REPAIR.json preserves the missing-matplotlib failure and source hashes.

## Status

Numerical analysis complete. Formal batch completion remains FALSE due to unavailable formal Ring deadlock evidence. Read videos/manifest.json, VIDEO_ACCEPTANCE.json and official_video_entry.log. Keep all raw data, traces, checkpoints and DB evidence; no cleanup authorized.
