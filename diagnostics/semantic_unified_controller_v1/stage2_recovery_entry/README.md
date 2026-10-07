# Stage 2: state-driven recovery entry infrastructure

This directory implements the frozen three-way NORMAL decision experiment:

- `S`: Safety for the current transition, then the frozen Stage-1 S/L policy.
- `L`: one Direct-g correction for the current transition, then the same frozen
  Stage-1 S/L policy.
- `R`: predict structured eta once at the current state and run that fixed eta
  densely to terminal.  Exit is deliberately disabled in this stage.

There is no H/L clock, future-failure feature, online oracle, eta search, or
final-test access.

`build_mode_branch_manifest.py` accepts any complete, content-hashed generic
NORMAL-anchor manifest through `--state-manifest`.  It requires outcome-blind
selection, at most two anchors per root source, source-disjoint train and
validation groups, and freezes 32 common future streams for all three actions.
The default old manifest is for compatibility/testing only; a larger frozen
generic cohort can be supplied without changing code.

`finalize_mode_results.py` assigns a three-way target only if one action beats
both alternatives beyond the predeclared 0.02 practical margin under the
conservative paired 90% intervals.  Proven three-way equivalence defaults to
Safety.  Every other input remains unresolved and is excluded from class
training.

`train_mode_head_grid.py` fits only resolved TRAIN targets with a
`214 -> 64 -> 64 -> 3` SiLU MLP.  It uses seeds 17/23/41, AdamW, and the frozen
break-penalty family 0/1/3; model selection uses validation evidence only.
All checkpoints remain provisional until full-loop validation.

No rollout or training was launched while creating this infrastructure.
