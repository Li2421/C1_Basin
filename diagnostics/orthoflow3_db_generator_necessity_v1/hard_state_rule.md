# Outcome-blind hard Double-Bottleneck state rule

The candidate inventory is every source-unique true-t0 family in the pre-existing DB train/validation/untouched-test pools after excluding all source groups used in the prior transform-development, 64/16/16 selector split, or 32-state fresh replication. No controller or eta outcome is read.

For each remaining state, only initial positions, initial velocities, frozen goals, bottleneck geometry, agent radius, and wall segments are inspected. Eight difficulty components are computed: smaller left/right mean arrival-time gap to the first bottleneck; smaller four-agent arrival-time spread; smaller mean distance to the first bottleneck; smaller opposite-direction straight-route lateral gap at x=0; smaller initial opposite-direction lateral gap; smaller same-side pair distance; smaller initial pair clearance; and smaller initial wall clearance. Each is converted to an empirical hard-percentile rank over the eligible inventory. The hard score is their unweighted mean. Ties use SHA256(`db_continuous_generator_hard_v1|source_group`). The top 48 are frozen.

The ranking and manifest are frozen before any new Safety, mode, selector, codebook, or eta-search outcome. The rule intentionally does not use prior success, deadlock, timeout, collision, controller score, or eta evidence.
