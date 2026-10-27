# Active user task (resume 2026-10-23)

Execute `docs/prompt_baseline_literature_scenes_resume_and_supplement_20261023.md` fully, without asking again. Finish part 1 and physically write its COMPLETE.json before beginning part 2. Part 2 has NOT started; its directory must not be used yet.

Latest resource steering: use all 24 CPU shares now (23 workers + 1 management), release 6 at Singapore 2026-10-23 09:00 = UTC 2026-10-23 01:00. Thereafter 18 total (17+1). GPU cap 2; currently CPU only. Persistent `resume_resources.py` guard and `dispatch_resume.py` enforce cap. All new files only repair directory and later supplement directory. No Git writes, no package installs, no original-source mutation, no runtime monkeypatch.

## Historical incident and accepted resume

Old copied CBS binary wrote 34440 files in sealed v2 pure-query cache. User explicitly accepted ONLY that old exception. Original READONLY_BOUNDARY_STOP/IMPACT, STOPPED.json and failed INTEGRITY_END.json stay unchanged. No cache deletion/rollback. Original one indirect import's missing opening SHA is still disclosed.

Resume verified v2 outside old cache unchanged. INTEGRITY_RESUME_START_20261023.json has1196 protected hashes. READONLY_RESUME_BASELINE.json is737MB metadata inventory, ~3.18million v1/v2 files. `resume_audit.py` and `other_readonly_guard.py` scan batch mtimes (other guard ~5.5million entries). Any NEW protected write -> RESUME_FATAL_STOP and stop all work. New accepted-old-cache baseline still detects further writes there.

Safe static import graph is `resume_backend/`; first project import must be `resume_bootstrap`. Rebuilt `native_resume/cbs_kernel_resume.so` caches only `runtime_cbs_query_cache_resume/`. No old unsafe native module may be imported. WRITE_PATH_AUDIT.json PASSED: 32 bitwise saved replays (8/scenario), static7-library audit, strace persistent writes private. WRITE_PATH_READY.receipt.json seals it. Original353 initial-circle journals now officially merged. Remaining7 initial-circle cases permanently skipped, not unknown.

## Part 1 science state

Clarified unit gates PASSED. Narrow:7501 premise-valid,6629 feasible,872 infeasible,1 old unknown preserved; crowd8795 valid,6065 feasible,2729 infeasible,1 old unknown. Both0 safety/permutation failures, max permutation <4e-9. All old snapshots/failed old gate preserved. gamma .3/.2 unchanged. Circle discrete fallback gamma .3 and10000 strict interval snapshots already passed.

All pilots passed observed zero collisions: narrow352 (3 unknown preserved), crowd352 (0 unknown), circle discrete360 (0 unknown). Initial circle353 with2 collisions retained as failed initial-form pilot. `PART1_STAGE_SELECTIONS.json` selects narrow_initial, crowded_initial, circle_fallback.

`part1_driver.py` is running persistently (PID in PART1_DRIVER_PID.json) with lock, progress PART1_PROGRESS.json, logs driver_actions/. It completed pilot job28261 and launched full job28284 with23 workers. Expensive groups scheduled first. Current full planned22528+39424+5760=67712; pilot rows/old numerical unknowns reused, deterministic seed1 aliases explicit. Full phase queue RESUME_QUEUES/full. Default official journals only. Driver recovers unfinished infrastructure groups without rerunning existing raw or unknown; CPU deadline handled. Driver runs post-batch readonly guards and official merger. It writes PART1_COMPUTATION_READY.json after actual full data/merge, NOT COMPLETE.

Scientific worker entry `resume_worker.py`, script resume_worker.sbatch, source proof RESUME_IMPLEMENTATION.receipt.json. Do NOT edit frozen science modules. New coupled diagnostic reference `stage2_joint_resume.py` correctly sums b_i+b_j (old2*b_i invalid for lookahead rows); source observation-only. All control/physics/criteria unchanged.

## Remaining part 1 steps

1. Monitor PART1_PROGRESS, squeue, worker logs. Share concise commentary at least every60s; don't claim completion while jobs active. Read driver error if any. Don't start a second driver. User status questions steer continuing task, don't cancel it.
2. When PART1_COMPUTATION_READY exists, verify diagnostics completeness (every raw DIAGNOSTICS/<uid>.json; load saved npz and `resume_backend/stage2_diagnostics_resume.summary` to backfill if needed, no controller rollout). No extra training.
3. Run `part1_statistics.py <scene>` for each selected full scene, preferably three1-CPU Slurm jobs within cap after readonly precheck. It uses original v2 bootstrap/contrast + AST-copied final classify/weighted_rates/controller_drops. 5000 bootstrap per distinct instance subset, type→instance→state, paired seed means; old-v2 and first-stage comparisons. Generates STATISTICS_<scene>.json, COMPACT_FIXED_<scene>.json, sources.
4. Run `postflight_resume.py` after official merge. Checks every actual new raw including old353 against SQLite exact identities/outcomes/canonical raw hash; door23040 exact reuse; no pending journals. Unknown rows represented, never retried just because official preflight excludes unknown.
5. readonly postchecks. Run `finalize_part1.py` only when science+statistics jobs finished. It checks protected source+Git+old failed audit retained and baseline SHA; writes new INTEGRITY_RESUME_END, full DYNAMICS_STAGE tables/CSV, REPORT/GAPS/HANDOFF, clarifications in STAGE2_FIX, stops old first-part resource guard, then COMPLETE.json. Old failed INTEGRITY_END never overwritten. Some finalizer/report code has not executed yet: inspect errors normally, use new helper/source if frozen dependency issue. No COMPLETE before real required work.

## Part 2 only after first COMPLETE

Create `diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/`, experiment `exp_baseline_literature_scenes_supplement_eta_swap8_v1`. Read total prompt again for specifics. Independent opening audit/history/write-path audit. HJB-GNN paper/repo parameter sources must be fetched then (clone only into supplement/external/hjbgnn, do NOT run/install external repo). Must browse referenced paper/video/officialrepo primary sources; HJB-GNN RSS2026 arXiv2506.22117v2.

Part2:16 fixed eta from controller_loco_real_flow_v1/protocol.json eta[0..15];5 static scenes (3 old +2 new swap8). 8 controllers (first qualified in each v2 training type),16 training states,seed0; reuse exact old labels. Rank type-equal lower success, index ties; top3 ->6 ordered two-eta schedules (.968 or6/7); freeze best fixed and pair before evaluation. Eval old v2 eval/new last16 states × all qualified ×seed0; compare eta13/human and report best differences/type descriptive. Moving swap8 uses free selected actions, no own screening.

New swap8 free/4L static/moving versions:32 states each, first16train,last16eval, moving same robot states as free, radius/params by paper>code>figure>declared defaults. 8 robots opposite circle; agent radius.05; sensing.5;dt.05; frozen perturb seeds2026102311/12/13. Moving disk.3,speed.25, parentstate-determined path, not nominal-visible; safety uses current obstacle position plus.0125 inflation, full responsibility. 8types48 actors unchanged algorithms/params, static geometry visible, sensing safety only. Separate geometry +48-actor single-agent capability each version, then32states×seed0/1 baseline48 +3fixed qualified; diagnostics, taxonomy,5000 paired bootstrap. Own safe backend/cache rebuild/audits, no first-part mutation after completion. Total prompt has exact fallback assumptions/layout/horizon. Final supplement COMPLETE only after all authorized work or explicit protocol gaps, not just background launch.
