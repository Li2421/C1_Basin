# Round19b Four controller axis: supplemental preregistration

This is the before-gate supplement to `diagnostics/four_controller_axis_v1/PREREGISTRATION.md`. The Round19 design, 13 best plus parameter-distinct final controller cohort, state counts, fixed/schedule arms, thresholds, unknown policy, feature freeze, DB lifecycle, and Four video requirement are inherited unchanged. This supplement supersedes only G1 and the output directory. All new study files belong to `diagnostics/four_controller_axis_v2/`; v1 and every other historical asset remain unchanged. No git commit.

## Why G1 changed

Round19 v1 stopped at Stage 0 without executing G0/G2 or generating any new state, probe, success label, rollout, or model. Its read-only audit found 1,360 v12 and 1,024 v10-wide historical intervention records, all without first actions or trajectories. The old G1 byte criterion was therefore unverifiable. Round19b replaces G1 with the four-field historical fingerprint, keeping G0 and G2 unchanged. This is a design correction before any v2 gate outcome.

## Historical evidence reuse

The v1 preregistration and `G1_EVIDENCE_AUDIT.json` are read-only evidence. E8 v13 raw records and old intervention DB/journals are read-only references. The old intervention condition is `training_dev_intermediate_state_episode_eta_v1+committed_base_t0_then_alt_future_v1`: v13 supplies the committed t0 action; the alternate checkpoint supplies future Flow. Those labels have a distinct controller/conditioning identity and cannot be reused as the new pure-checkpoint census labels. Round14B Q2 labels are on old states and do not answer this new-state study. The new adapter may reuse old code by importing it but writes only to this directory.

## Fixed gate record selection, before reproduction

For each of v12 best and v10-wide best, join shared DB `rollout` to `controller_config` by exact checkpoint file SHA256. Keep nonquarantined records with `outcome` exactly `success` or `timeout`, `numerical_failure=0`, and all four fingerprint fields available both in DB and their original source journal. Verify DB source hash, source record state/eta/controller/seed identity and fingerprint equality before selection. Within each checkpoint and outcome, sort ascending by `rollout_uid`. For a stratum of size N, select eight zero-based positions `floor((j+0.5)*N/8)`, j=0,…,7; require distinct indices. Thus each checkpoint contributes eight successes and eight timeouts. Freeze the 32 selected IDs, source paths and expected exact float64 bytes before executing any v2 gate. No selection from reproduction outcomes. Numerical failures are excluded only from this equivalence sampling, never silently recoded in later study labels.

Historical `min_agent_distance` and `min_wall_distance` correspond to source journal `minimum_agent_clearance` and `minimum_wall_clearance`; before a gate, verify the DB decimal round trip retains identical IEEE-754 float64 bytes. The old intervention CPU Slurm scripts set `JAX_PLATFORMS=cpu` and do not enable x64; old Python modules do not update `jax_enable_x64`. Match their default `jax_enable_x64=False`, CPU backend, single worker thread, checkpoint-native normalization/actor width, identical state, committed base t0 action, `future_index`, `future_root_seed`, and `rng_namespace`. If precision provenance or any seed/conditioning component cannot be established, G1a is unverifiable and blocks later work.

## Stage 0 hard gates

G0 is the original Round19 exact eight-case v13 best comparison against E8, now executed with the **new v2 adapter**: terminal outcome, episode length, and float64 first-action bytes all exactly equal. The old Round14B gate cannot substitute.

G1a is mandatory: replay the 32 frozen records under the historical intervention condition. Each record must match historical `outcome`, `episode_length`, `min_agent_distance`, and `min_wall_distance` exactly, with both distances compared by float64 bytes and no tolerance. Any mismatch or unavailable evidence fails G1a and stops Stage 0.

G1b is conditional and nonblocking only when G1a passes: select four of the 32 G1a IDs, exactly the first selected success and timeout for each checkpoint. If the old intervention runtime can run locally with instrumented full step trace, run old and v2 implementations on these four and compare first-action and every simulator position/velocity/action byte. A mismatch is reported; G1b is not one of the three hard blockers. If historical runtime cannot be run or cannot expose comparable per-step trace without changing its execution, mark `NOT_AVAILABLE` with the concrete reason; do not fabricate a trace.

G2 is the unchanged original gate over the candidate cohort: all non-normalizer config and structure must match v13, except v10-wide actor width; repeated same-key forward outputs must be byte-identical and finite, with FP32 explicitly restored after checkpoint import. Any G0/G1a/G2 failure or unverifiability stops before Stage 1 and any new success label. Gate replay outputs stay only in this v2 directory, marked `gate`, with no insertion into or replacement of historical DB records.

## Subsequent stages and delivery

Only after G0, G1a and G2 pass, execute inherited Stage 1 unlabeled 16-state × 3-eta × 200-step probe for each eligible checkpoint and freeze all features/PCA-2 with UTC/SHA256 receipt before any new success label. Then execute inherited Stage 2 on 32 fresh DB-deduplicated Four states, F10/F13/F15, seeds 0/1. Enter Stage 3 only if ≥3 controllers show a nonmajority best eta ≥10pp ahead of the majority arm or ≥3 controllers have best fixed-arm success ≤85%; Stage 3 schedules, E/L/N, H_compress criterion and Stage 4 grouped leave-one-run-out ridge alpha=1 criterion are exactly as v1 preregistered. Exact requests go through design, official preflight, only genuinely missing seeds, shard journals, single-process merge and official postflight. Before each Slurm submission check live usage; total user load ≤18 CPU /2 GPU shards. Four video roles require full trace and the uniform entry; no timeout is called deadlock. If legitimate deadlock roles are absent, mark video delivery incomplete.

The first paragraph of `REPORT.md` states G0/G1a/G1b/G2 status and, if all hard gates pass, answers the Three Four questions. Each hypothesis is SUPPORTED, REJECTED or INCONCLUSIVE, with facts separated from interpretation.
