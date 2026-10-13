# Formal physical-only Q experiment: paused after cache audit

On 2026-10-08 the user prioritized a smaller cache-first pilot. Slurm arrays
6455, 6463, 6513, 6554 and their dependent finalizer 6559 were cancelled;
their completed rollout journals were merged into the shared database and all
raw evidence, state manifests and frozen Ring FF checkpoints were retained.
The full-scale experiment has no final verdict. Exact coverage for all its
610,304 requested seed slots is recorded in `minimal_reuse_db_audit.json`.
The requested smaller cache-first pilot and 48-fresh-state Toy confirmation
are complete; see `../physical_state_critic_minimal_v1/FINDINGS.md`. They do
not replace the paused full four-scene formal verdict.

The former active-job list below is historical, not currently running.

The experiment is running. No full within-scene or LOSO verdict is available
until all 10 panels pass Q16 TEST collection and paired evaluation.

Frozen now:

- Four physical-state files: 512 TRAIN + 32 VAL + 128 fresh TEST per scene.
- Historical matched Toy/Ring TT/FF Sobol K16 protocol and separate stochastic
  K16 four-scene FF LOSO protocol, candidate proposals and seed identities.
- Ring FF physical23 (Q(s,η)) and fair eta-only checkpoints, selected only on
  the same TRAIN/VAL labels; 128 fresh Ring FF TEST choices frozen before labels.
- Unified remaining-fit training/selection rule in `unified_train_plan.json`.

Active Slurm jobs (2026-10-08):

- `6455`: matched Toy/Ring TT/FF missing TRAIN/VAL rollouts.
- `6463`: Ring FF fresh 128-state Q16 TEST rollouts, already-frozen selectors.
- `6554`: corrected Toy/DB/Four source TRAIN/VAL LOSO shards 0–999.
- `6513`: LOSO source shards 1000–1295, unaffected by Toy key correction.
- `6559`: dependency-gated source merge, tensor assembly, 54 model fits and
  further test-stage scheduling; requires `6455`, `6554`, `6513` to succeed.

All workers save per-seed raw JSON, and the single-writer merger indexes them
into `shared_rollout_db/rollout.sqlite`. The dependency-gated stages refuse
incomplete outcome tensors and keep TEST labels out of model selection.
`COMPLETE.txt`, `verdict.json` and `FINAL_REPORT.md` are written only after all
10 paired evaluation panels finish. The A–E verdict follows the predeclared
paired-B15 rule in `unified_train_plan.json`; the full tables and uncertainty
bounds should still be read before using the result.

The Toy seed-key correction is documented in
`loso/toy_seed_identity_correction_audit.json`; superseded pilot evidence is
retained and excluded by corrected rollout identities. This is a pure-Q batch,
so the repository's video-delivery requirement does not apply.
