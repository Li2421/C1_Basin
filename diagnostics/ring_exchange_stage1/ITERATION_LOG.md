# Ring Exchange Stage-I iteration log

All policy development below used development rollouts only.  No frozen-test
archive was opened during these iterations.

| Candidate | Representation / data change | Training sampler | Development outcome (30 nominal) | Decision |
|---|---|---|---:|---|
| v7 | First local radial/tangential physical chart; Base-U | Early 50%, nominal-only | 0/30 success; 4 outer collisions; 26 timeouts | Coordinate chart removed obstacle collisions but exposed exit/goal drift. |
| v8 | v7 plus full-development-policy uniform recovery (848 successful continuations) and safe-chord near-goal expert | Early 50%, nominal-only | 0/30 success; 1 outer collision; 29 timeouts | Extra recovery data alone did not place recovery prefixes in the early sampler. |
| v9 | Same v8 data | Early 50% across **all** nominal and recovery trajectories | 18/30 success; 0 collisions; 12 timeouts | The key correction: recovery's first 15 actions are no longer drowned out by its long continuation. |
| v10 | v8 plus v9-timeout safe terminal-window and goal-near re-query: 687 successful continuations | Same early-all sampler | 26/30 success; 0 collisions; 4 timeouts | Frozen Stage-I candidate; no further Ring development before frozen evaluation. |

The master then ran the one-time frozen test without any intervening Ring data
collection, training, or parameter selection: 47/60 success (78.33%), zero
obstacle/outer-wall collision, one agent collision (1.67%), and 12 timeouts
(20%).  This outcome is descriptive only; no post-test adaptation is allowed
or has been performed.

The v10 collector sourced exactly 12 v9 development timeouts.  It used only
valid terminal-window states and valid goal-near/non-jointly-converged states,
then locally perturbed and re-queried the centralized expert.  It recorded
source rollout, source time, perturbation seed, and recovery result.  It added
372 terminal-window and 315 goal-near successful continuations; 12 invalid
perturbations and no expert failures were skipped.  These are classified as
`uniform_recovery`, not collision-targeted recovery.
