# Basin_C1 global rollout database

`rollout.sqlite` is the authoritative index for expensive closed-loop evidence. Experiment artifacts remain immutable provenance; the database is a deduplicated cache/index, not their replacement.

## Mandatory workflow

1. Serialize every planned `scenario × state × exact eta × controller × seed` request.
2. Run preflight before Slurm submission:

   ```bash
   cd /home/zhihan/research/Basin_C1
   python -m shared_rollout_db.plan \
     --manifest diagnostics/my_experiment/planned_rollouts.json \
     --output diagnostics/my_experiment/cache_preflight.json
   ```

3. Submit only `genuinely_missing` seeds. `EXACT_REUSE` requires all semantic fingerprints to match. `AMBIGUOUS` is never auto-reused.
4. Workers write experiment output as usual and append shard journals through `src.cache_writer.append_journal`. Workers never contend on SQLite.
5. A single process runs `python -m shared_rollout_db.src.merger --experiment EXPERIMENT_UID` after the batch.

The standard manifest contains `requests`, each with exact `state_uid`, `eta_uid`, `controller_uid`, and `seed_keys`. Human aliases are accepted only when they resolve uniquely. Exact eta identity uses the three IEEE-754 float64 byte strings; nearby eta is analysis-only.

## Default evidence policy

- Routine robustness is **B15 = at least 15/16** under `seed_policy_v1.json`.
- Reuse all compatible historical Q16/Q32/Q64 evidence.
- Q64 is stronger historical evidence and is not recertified by default.
- An exact state–eta–controller with all 16 compatible seeds requires zero new rollouts.
- Only genuinely missing compatible seeds may be executed.
- Aggregate 63/64, 31/32, and 15/16 certify B15 because the entire aggregate contains at most one failure; general 60/64 does not identify the fixed standard-16 outcome.
- Special final-paper Q64 certification must be requested explicitly.

## Compatibility levels

- `EXACT_REUSE`: state, eta, controller and seed semantics are exact and conflict-free.
- `PARTIAL_SEED_REUSE`: exact semantics but some requested seeds are absent.
- `AGGREGATE_REUSE`: exact counts exist but seed identities are unavailable; only mathematically permitted aggregate conclusions apply.
- `INCOMPATIBLE`: semantic fingerprint differs or no compatible record exists.
- `AMBIGUOUS`: provenance is insufficient; automatic reuse is forbidden.

## Concurrency

SQLite uses WAL, full synchronous commits, foreign keys and retry/busy timeout. GPU workers write append-only journals. A single merger atomically commits them after completion.

