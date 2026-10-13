# Additional prespecified factor holdouts (zero additional rollout)

Freeze before factor critic fitting and before inspecting any final-state factor
outcomes. The8 cost-probe cases all used state0; all factor test states8..15 remain
unread. Add COUNT: train onlyN2, evaluateN4; TIME: train onlyH400, evaluateH850,
with all other width/topology/controller factors covered and new test states.
These directly answer independent agent-count and time-budget extrapolation,
which factor mean effects alone cannot establish. Use the same four arms,
three seeds, fixed optimization; no model/feature/hyperparameter selection.
Added24fits, no physical rollout. Total factor90fits including two ablations.
GPU budget estimate<30minutes with4shards,20GB,4CPUs; job hard limit1hour.
