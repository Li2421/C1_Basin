# Execution record

Audit before new work: reviewed repository AGENTS.md, P1 protocol/raw runner/raw example, Round 2 R5 files, Round 3 nested matching code/results. Existing P1 labels and R5 are compatible; no closed-loop trajectory arrays are saved by P1. Other inspected mediation diagnostics concern static action operators or different panels and cannot substitute for this real-checkpoint trace evidence. The remaining executable gap is a learned physical-state critic with both target controller and target states excluded; no new rollout is necessary for that test.

Saved PROTOCOL.md and run.py, then SHA256 receipt `frozen_inputs.sha256` before executing this new comparison. Source-only model choice was fixed before its results, but original P1 data have already been opened, so no independent-confirmation claim is made.

Commands from repository root:

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_three_questions_20261012/run.py > diagnostics/part3_three_questions_20261012/run.log 2>&1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_three_questions_20261012/verify.py > diagnostics/part3_three_questions_20261012/verification.log 2>&1
```

CPU only, model comparison elapsed 3.92 seconds. 56 outer folds, 2 fixed model arms, 3 inner controller/state splits; no further capacity or feature search. All models retain training-row identities, coefficients, intercepts, bandwidth and regularization in `models/`; deterministic transforms can be rebuilt from the hashed input files and script.

Reused 21,504 P1 seed records, including 272 numerical-unknown records. New rollouts 0, new behavior queries/features 0, modified historical labels 0. Cache DB not mutated. No batch video requirement for this pure cached Q experiment.

Verification checks the input hashes and all 56 controller/state disjoint splits and full evaluation coverage. It also changes every nontraining label in a representative fold and verifies unchanged chosen hyperparameters and predictions. An independent direct linear solve checks the spectral kernel solution. `verification.json` records the results.
