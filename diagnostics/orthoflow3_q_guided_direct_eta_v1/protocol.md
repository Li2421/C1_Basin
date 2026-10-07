# OrthoFlow3 frozen-Q-guided Direct-eta controlled ablation v1

Only the actor `G(h)->eta` is trained. The OrthoFlow3 implementation, source
split, 214-D features, target file, normalizations, actor architecture,
optimizer, batching, and early stopping are inherited exactly from the frozen
MSE-only baseline. Q is the frozen Q-v2 seed-41 checkpoint and has no optimizer
state or mutable statistics. Its validation-only frozen feasibility threshold
is `tau=0.771`.

The sole intervention is `L_total=L_eta+lambda*relu(tau-Q(h,G(h)))^2` for
`lambda in {0.01,0.03,0.10,0.30,1.00}` and seeds `{17,23,41}`. Checkpoints are
early-stopped and selected by validation eta-MSE, matching the baseline.

All 15 checkpoints receive matched 8-seed closed-loop evaluation on the same
20 validation states. The top three are promoted to 32 seeds total under the
predeclared lexicographic rules. One actor is frozen before any TEST rollout.
TEST uses the same 20 states and 64 future identities as the baseline. After
TEST is frozen, an outcome-blind fresh 400-episode WIDE cohort is frozen and
evaluated under matched randomness for Safety, baseline, and guided actors.
Q is never used in deployment. There is no J, gate, search, snapping, cadence,
mode switch, or change to either safety projection.
