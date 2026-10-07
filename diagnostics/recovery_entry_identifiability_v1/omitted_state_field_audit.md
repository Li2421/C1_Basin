# Omitted deployment-state field audit

The primary probe uses only the frozen 214-D deployment feature. No future or terminal-relative field is included.

| Saved field | Representation status |
|---|---|
| `positions` | encoded (positions and observation) |
| `velocities` | encoded |
| `step` | encoded by four timing coordinates |
| `history_start_step` | encoded |
| `candidate_since` | encoded |
| `stuck_timer` | encoded |
| `max_stuck_timer` | encoded |
| `ever_candidate_deadlock` | encoded |
| `error_history` | last 41 samples encoded; earlier real history omitted |
| `first_success_step` | omitted but invariant -1 at nonterminal queries |
| `first_deadlock_step` | omitted but invariant -1 at nonterminal queries |
| `first_wall_collision_step` | omitted but invariant -1 at nonterminal queries |
| `first_agent_collision_step` | omitted but invariant -1 at nonterminal queries |
| `done` | omitted but invariant false |
| `complete_real_history` | omitted but invariant true |

The only nonconstant deployment-available information omitted from `h_t` is the portion of real goal-error history older than the 41-sample tail. Physical and monitor state used by control are otherwise encoded, often redundantly.

Because the primary grouped probes were weak, one secondary source-held-out ridge probe added 12 causal summaries of pre-tail real history (initial, mean, standard deviation, min, max, and total progress for two agents). It did not use future information.

Secondary metrics: `{"count": 184, "mae": 0.16603738453204847, "pearson": 0.02018865719078369, "rmse": 0.2760874366042356, "spearman": 0.034229934488569334}`
