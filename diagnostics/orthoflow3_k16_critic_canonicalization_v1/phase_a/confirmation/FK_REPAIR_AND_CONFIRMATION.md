# Phase-A frozen confirmation: FK repair and completed run

## Scope and invariant

This is the previously frozen Phase-A confirmation, not a new cohort or a
model-selection iteration. The proposal manifest, state manifest, generator,
critic, K=16 sampling rule, MACFlow seeds, controller chain, and CPU backend
were unchanged. The 36-shard CPU job was resumed as Slurm array `1554`.

| Frozen artifact | SHA-256 |
|---|---|
| `proposals.json` | `da8bb707cbe62859e074f49a641bbf8130b02b4d3f7b9fcc8391eedd8543524b` |
| `state_manifest.json` | `a7d745f7f0552c0c0f7179e3214712f69ff560493809fbc63017cf056f454649` |
| `../critic_frozen.json` | `6c1b76f1ce1a117018fa9b7dd448ea41d850ff0be16c4020b2b24fa4ca2fe393` |
| Frozen critic checkpoint | `c656f0131a1578f7218276ea911831f6e1d60d4694e91a903cf7ef050c6efafc` |

## Root cause and repair

The shared DB already contained an `experiment` row at the confirmation-root
path, registered by a historical importer under a different UID. The
confirmation runner's `INSERT OR IGNORE` at the same UNIQUE `experiment.path`
therefore silently did not register its own UID. Each subsequent
`source_file.experiment_uid` insert failed its foreign key. All 36 original
shards stopped after writing one flushed raw JSONL rollout each.

The runner now uses the dedicated, deterministic
`phase_a/confirmation/run_provenance` experiment path and verifies that its
experiment row exists. The sink supports DB-only replay of a previously
flushed raw row. The 36 original rows were recovered without recomputation or
duplicate JSONL append: one during the pilot and 35 at shard startup.
Neither change affects rollout actions, states, seeds, or outcomes.

## Completeness and DB checks

There are 108 states (24 Double-Bottleneck, 24 Four-Way, 60 Ring), each with
17 frozen generator candidates plus B0, under 16 fixed MACFlow seeds:
`108 × 18 × 16 = 31,104` exact seed slots. All 36 shards completed; the
collector reported **zero missing slots**. The new confirmation experiment
owns 31,104 exact rollout rows, including the 36 recovered raw rows. The
remaining 31,068 slots were executed by this resumed array. There were
31,659 physical attempts due to the canonical identical retry protocol.
Exactly 197 seed slots remained numerical-uncertified after three retries
(110 Four-Way, 87 Ring); they were not counted as task failures. Valid
rollouts had zero collisions. `PRAGMA foreign_key_check` returned no rows and
`PRAGMA quick_check` returned `ok`. All 108 scenario/shard `source_file`
hashes match the physical JSONL files, and every `rows_seen` count matches
its file's line count (31,695 total raw lines: 36 recovered + 31,659 new
attempts).

## Fresh confirmation (empirical B15 = at least 15/16 valid successes)

| Scenario | B0 | K16 oracle | Old critic | New critic | New rescue | New break | New oracle miss | New exploitation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Double-Bottleneck | 4/24 | 24/24 | 24/24 | 24/24 | 20 | 0 | 0 | 0 |
| Four-Way | 1/24 | 24/24 | 21/24 + 3 unresolved | 19/24 + 5 unresolved | 18 | 0 | 0 certified | 0 |
| Ring | 42/60 | 60/60 | 48/60 | 56/60 | 16/18 B0 non-robust | 2/42 B0 robust | 4 | 1 |

Ring's fresh-cohort critic gap decreased from 12 to 4 states, and break
decreased from 9 to 2; this is independent confirmation of a material ranking
improvement, not proof of perfection. Double-Bottleneck did not regress.
Four-Way has no certified non-robust selected eta under either critic, but the
new critic has five numerical-uncertified selections versus three under the
old critic. In three states, old selection was certified robust while new
selection remained unresolved. Thus the frozen result does **not** certify
the predeclared no->5-percentage-point Four-Way regression gate. The numerical
unknowns must not be imputed as successes or failures, and no adaptation to
this confirmation cohort was made.

The reproducible per-state and seed-level outputs are `results.json` and
`seed_evidence.json`. The appropriate Phase-A decision is
`PHASE_A_CRITIC_PARTIAL`: Ring ranking improved substantially on a fresh
cohort, while Four-Way numerical uncertainty prevents a clean joint pass.

No Phase-B canonicalization or later experiment was started by this repair.
