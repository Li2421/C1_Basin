# Part 3 Round 3 archived-label analysis plan

Frozen before running the new ablation. P1 Q4 labels and Round 2 nearest-source results are already opened development evidence, so this is a source-only outer-LOCO reanalysis, **not** a newly independent population test. No new rollout, static probe, controller modification, or critic training beyond the small matching model below.

## Data and split

Use all 14 real Ring FF Flow checkpoints, the same 24 opened states, K16 physical η and all four P1 seeds. Outer fold holds out one real checkpoint. Each inner fold holds out one of the remaining 13 checkpoints. Source features/labels alone set every normalization, metric temperature, shrinkage and model choice. Held Q labels enter only final scoring. Numerical unknown is an interval `[success/4,(success+unknown)/4]`; use the interval midpoint for source fitting and hyperparameter selection, and report unknown-safe intervals on final metrics. Do not remove checkpoints, states, candidates or unknown cells.

## Models and inner choice

R0 is the per-state, per-η source mean. The common model for R3, R5, R3+near, R3+R4, R5+R4 is a source-conditioned kernel match: for held state `s`, take a softmax of negative static-controller descriptor distances to the source controllers **at that same physical state**, predict every η by weighted source `Q(source,s,η)`, and shrink toward R0. State identity/physical state selects the source row, η selects the K16 head, and controller features determine source weights; hence the model can express state×controller×η patterns. Its limitation is that it tests checkpoint transfer on the same 24 states, not transfer to new states.

R3 is archived compact fingerprint; R5 is archived R3 plus pose/time indexed response; R3+near adds only the near-goal moving 3 pose×3 time×2×2 Jacobian block per agent (144 values, versus R5's 696 added values); R4 is archived executed-action probe. All features are deployment-time static and outcome-free.

Compare coordinate standardization (`std` floor 0.05) and semantic-block RMS standardization. Distances are mean squared and divided by the median nonzero source-source distance within the training fold. Softmax inverse temperatures `{0.5,1,2,4}` and source-match weights `{0.25,0.5,0.75,1}` are the fixed grid. Select one setting per representation by mean inner held-checkpoint Q-midpoint MSE, with deterministic lexicographic ties. R0 has no hyperparameters. This protocol specifically tests whether the previous nearest-source result was a metric/reader failure.

## Evaluation and gate

Primary endpoints: outer held Q4 MSE interval, K16 top-1 selected Q interval, regret versus target empirical Q4 oracle interval, selected-minus-R0 and selected-minus-source-global-fixed paired intervals, per-checkpoint values. Secondary: mean within-state K16 rank Spearman, non-tied η-pair accuracy, selected η distribution, candidate headroom, target-seed half-crossfit oracle-minus-source-fixed, complete-cell split-half Q4 noise proxy, and error decomposition by controller level, η curve and state×η residual.

Source-global-fixed η maximizes the source checkpoints' all-state Q midpoint; ties use lower K16 index. No target labels enter its choice. The empirical target oracle is a noisy upper benchmark, not population optimum. Compare aggregate and every checkpoint; report 14-checkpoint paired bootstrap intervals as descriptive because this checkpoint inventory is selected/opened. Call transfer gate passed only if one controller-informed arm has (i) lower mean Q MSE than R0 and R3, (ii) positive mean unknown-safe selection gain versus both R0 and source-fixed, (iii) nonnegative lower-bound gain versus both on at least 10/14 held checkpoints, and (iv) positive 95% checkpoint-bootstrap lower endpoint of both paired gains. Otherwise output `TRANSFER_GATE_FAILED`. No new Q16 experiment is authorized by a failed gate.
