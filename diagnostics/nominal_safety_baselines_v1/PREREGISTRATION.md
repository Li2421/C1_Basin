Round29 preregistration — frozen at 2026-10-15T04:36:18.075821+00:00 UTC

Authority: docs/prompt_round29_nominal_safety_baselines_20261015.md.
Read AGENTS.md, shared_rollout_db/README.md, round27 REPORT/HANDOFF and frozen code.
16 frozen controllers (M0–M5, P0–P5, four Flow TT), accepted round27 32 states,
seed 0–1 and identical exact seed keys; 2 chains: no_safety and hard_safety.
2048 complete rollouts, only genuinely missing; retain unknown without rerunning.
Controller source loaded directly from round27; parameters/checkpoints hashes
verified against V2 receipt and main protocol. FROZEN_INPUTS is the input receipt.
No controller tuning, physics/projection/event/success changes or training.
After imports AND each lazy runtime construction restore jax_enable_x64=False,
as round27 protocol requires; assert at each physical step and record status.
no_safety executes native nominal directly; hard_safety executes exactly one
native safety.project(nominal). These follow safety_eta3 chain dispatch with
round27 RNG key_for; neither computes a correction. Orthoflow3 gate calls the
frozen Observed Runtime.action(old), with unchanged schedule and RNG.

Before baseline execution replay 8 known round27 main records: M0/M5, P0/P5,
all four Flow references; deterministic first eligible known record, alternating
F10/S500 when possible. Compare native termination, episode_length, and first
action dtype/shape/bytes against stored traces. Also compare full positions.
Stop on any mismatch. These 8 explicitly required validation repeats are listed
separately from missing baseline execution; never rerun unknown.

Metrics: category and controller × eight methods: nominal only, safety baseline,
F10, F13, best fixed oracle (max of aggregate F10/F13/F2 per controller), frozen
A3_full, S500, full-menu oracle (max aggregate of six arms per controller).
Items 3–8 reuse LABELS and choices; verify point estimates against round27.
Unknown is neither known failure nor success: report identified success interval
[success/total,(success+unknown)/total], known-only rate and count. To reproduce
round27 exactly, table point values and CIs describe confirmed-success fraction
(lower bound), explicitly not a failure classification. Also give upper bounds
and bootstrap envelopes allowing unknown outcomes. Numerical failure rate is
unknown incidence. Collision/timeout/stagnation separately, swept simulator
clearances from every step, success requires all agents collision-free arrival.
Stagnation descriptive: no success, >=100 steps, final 100-step reduction in
sum goal distances <0.05; never call it deadlock.
5000 paired hierarchical bootstrap replicates: resample controllers first within
category, then 32 state clusters within sampled controller, keep both seeds and
all methods paired. Per-controller CIs resample states only. Bootstrap seed
202610152901 + group index. Recompute aggregate oracle maxima each replicate.
Key contrasts A3_full minus each baseline, percentile 95% CI plus unknown envelope.

Pure Q evaluation data: no video entry, per AGENTS.md explicit exception.
No git commit; no historical asset changes. All files (including local cache DB,
journals, Slurm logs/caches) written only here. The shared authoritative DB is
read-only snapshotted, since writing it would violate the file scope; official
planner and single-process merger operate on this local snapshot. Global index
ingestion is deferred; this limitation must appear in REPORT/HANDOFF.
CPU shard cap across ALL user's workstation jobs 18, GPU shard cap 2; use CPU
only, wait for available resources and count existing jobs before execution.
Design → preflight → gate → missing-only → journal → single merge → postflight.
