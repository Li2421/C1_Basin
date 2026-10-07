# Forced initial structured eta with learned exit audit

- Development-only, 200 new unconditioned authoritative WIDE episodes.
- IC root seed: `2026120101`; Flow root seed: `2026120102`.
- Freeze and overlap-audit the manifest before any rollout.
- Matched controllers: Safety; Forced-Eta+Exit; frozen learned-entry+exit reference; frozen Direct-g H8 reference.
- Forced condition has no learned entry decision: structured eta is predicted once at step 0, one structured transition is mandatory, exit is first queried at step 1, and Safety controls permanently after exit.
- No training, eta search, oracle query, H/L in the tested architecture, or controller modification.
- Horizon 850 physical steps, dt 0.05 s, authoritative event priorities and two hard projections.
- Resource cap for this execution: 2 GPU shards, 6 CPU threads, 36 GiB host-memory allocation.

