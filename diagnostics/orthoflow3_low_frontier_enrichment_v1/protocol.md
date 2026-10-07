# OrthoFlow3 low-frontier enrichment audit

This confirmation audit reuses exactly the ten states designated `ACTIVE_REQUIRED` by `orthoflow3_analytic_mindef_stability_v1`. It freezes a shared scrambled-Sobol extension (seed 810031; indices 256–287 in the authoritative eta domain) before any new outcomes. Initial screening evaluates indices 256–271 with the same first eight matched Flow futures for every state. Promotion does not consult true rollout J: A is the lowest raw eta norm among 8/8 candidates, B the lowest start-Gram proxy among remaining candidates, with the predeclared farthest-normalized-distance fallback. Candidate success is confirmed only at B63 (>=63/64); true J is the prior authoritative mean successful full-continuation J_def.

## Updated resource policy

When it is late night/early morning and scheduler/GPU inspection confirms the
machine is essentially idle, up to six GPU shards are permitted. Otherwise the
normal shared-server limit remains two shards. This run remained on two shards:
its tasks were already partitioned and expanding mid-run would risk duplicate
state/eta/seed continuations.
