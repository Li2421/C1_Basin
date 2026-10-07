# Startup-complete learned G_phi pilot

Status: **NOT RUN — OFFLINE FAIL-STOP**

The startup-aware feature implementation, startup oracle dataset, merged
dataset, from-scratch training, and projection replay checks completed.
However, the selected startup-complete model's state-grouped mean L2 error on
the exact unchanged warm-history V3 test set was `0.047905945`, versus
`0.015412729` for the previous V3 checkpoint (a `3.108x` increase).

The task explicitly required stopping before closed-loop evaluation if
warm-history performance was catastrophically degraded. Therefore no fresh
production evaluation seed list was frozen and no Safety or learned-G_phi
full episode was run. This avoids spending evaluation seeds or presenting a
controller whose offline retention gate failed.

Final classification:

`STARTUP_COMPLETE_TRAINING_NEEDS_REVISION`

The smallest justified next experiment is to repeat the same from-scratch
architecture, loss, data, seeds, and split while saving epoch checkpoints and
selecting on the validation-only startup/warm Pareto frontier under a
pre-registered warm-history retention guard. No closed-loop test should occur
unless a checkpoint satisfies both cohort gates.
