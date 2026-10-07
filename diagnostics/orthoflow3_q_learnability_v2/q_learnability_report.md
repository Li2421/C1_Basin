# OrthoFlow3 Q learnability and directional fidelity v2

## Decision

**Q_PREDICTIVE_AND_DIRECTIONALLY_FAITHFUL**

The simple Q surrogate generalizes to held-out source groups and its local eta
gradient passed the predeclared true closed-loop direction test. This supports
testing, but does not yet validate, a future Direct-eta actor trained through the
frozen Q. Q is diagnostic/training-time only; no production controller, J, or G
was trained here.

## Exact conditioning and leakage

The estimated quantity is `Q(h(z, xi_0), eta)`: augmented state and the exact
query-step Flow draw represented in h are fixed; eta is fixed for the full
continuation; only Flow draws after the first transition are resampled. Archived
state-level Q counts that changed `xi_0` were rejected as labels. The ambiguity
classification was **not triggered**. Query-step feature replay error stayed
below 1e-10 (smoke maximum 2.22e-16).

The 424-state source pool had 413 states with enough RNG provenance. The frozen
split is 80/20/20 unique states and 80/20/20 distinct leakage groups for
TRAIN/VAL/TEST. Group overlap is zero. Normalization used TRAIN only; VAL alone
selected checkpoints and the zero threshold; TEST was not used for tuning. True
gradient outcomes were generated only after checkpoint hash
`8cf8605d1aa6c053faddb72ffe1bc4e0f77c0f0975ab752f48b05c0b6a4eb7cd` was frozen. No J_def label was used.

## Data and model

The common eta cloud is explicit zero plus the first 23 frozen authoritative
Sobol points. The base table has 2,880 candidates and 13,440 continuations.
TRAIN candidate histogram is:

- 0/4: 416
- 1/4: 41
- 2/4: 30
- 3/4: 33
- 4/4: 1400

TRAIN success prevalence is 0.7552;
61/80 states contain both
successful and unsuccessful eta. The informativeness preflight passed.

Q is a `217->256->256->1` SiLU MLP with 121,857
parameters and candidate-level binomial NLL. Seeds 17, 23, 41 were trained;
VAL selected seed 41 at epoch 271 with NLL
0.209661.

## Held-out TEST prediction

- TEST binomial NLL: **0.292811**
- Constant TRAIN-prevalence NLL: 0.541578
- Logistic-linear TEST NLL: 0.424427
- Brier score against candidate Q8: **0.040042**
- Statewise Spearman mean/median: **0.499 / 0.623**
- Statewise Kendall mean/median: **0.416 / 0.516**
- Among the 14/20 states with nonconstant Q8: Spearman mean/median
  0.713/0.762,
  Kendall 0.594/0.620.

Across every TEST state, Q's top-ranked member of the 24-point cloud achieved
true Q8=1.0 (mean and median), equal to the best observed candidate; mean random
candidate expectation was 0.768.

## Zero eta

The first eight TEST states were audited to 64 trials. Six were B63 zero-feasible
and two were not. Using the VAL-only threshold 0.771, false-feasible =
**0** and false-infeasible =
**1**. The lone false-infeasible state had
predicted Q=0.236 and observed 63/64; the remaining robust counts were five 64/64,
while both nonrobust cases were 0/64. The same Q handled zero; no gate was used.

## True directional fidelity

Six frozen TEST cases used normalized alpha=0.05 with no clipping. Their
`Q32_minus` values were [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], bases [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], and plus values [1.0, 1.0, 1.0, 1.0, 1.0, 1.0].

- Q-plus > Q-minus: **4/6**
- Q-plus >= Q-base: **6/6**
- Q-base >= Q-minus: **6/6**
- mean / median Q-plus minus Q-minus: **0.667 / 1.000**
- predicted/true directional sign agreement: **4/6**
- descriptive predicted-change/true-change Pearson: 0.892
- catastrophic directional failures: **0**

Four cases changed from true Q32=0 along minus to Q32=1 along plus. The two sign
"disagreements" were flat all-success landscapes (1/1/1), not reversed
directions. No failure-analysis case met Q-plus < Q-minus or the catastrophic
criterion. The optional random-direction control was skipped to prioritize the
predeclared primary evidence within budget.

## Answers and next step

Yes: Q_hat generalizes as a state-conditioned OrthoFlow3 success-basin surrogate
on held-out source groups. Yes: grad_eta Q_hat is locally control-relevant enough
to justify **testing** a future G(h)->eta trained through frozen Q. This does not
establish that Q is necessary or that such a G will work in closed loop.

The single smallest justified next experiment is a diagnostic, source-group-held-
out small G(h)->eta trained through this frozen Q, with a predeclared closed-loop
comparison against a simple Direct-eta baseline and no online Q search. Do not
start it automatically.

## Runtime and resources

Scientific continuations: 14,416; physical steps: 4,563,152.
Recorded stage critical paths were 16.3
min base-completion, 1.6
min zero, and 2.1
min gradient; Q training took 23.6 s. End-to-end audit
wall time was 35.9 min including semantic
inspection, an integration correction, and reporting.

Maximum allocation was 6 GPU shards, 12 CPU threads, and 84 GB requested RAM.
The completion batch capped each JAX process at 0.14 of a 97,887 MiB GPU
(~13,704 MiB/process). Slurm peak accounting was disabled, so actual peak GPU,
CPU, and RAM are unavailable; the post-run GPU reading was 2 MiB and 0%.
