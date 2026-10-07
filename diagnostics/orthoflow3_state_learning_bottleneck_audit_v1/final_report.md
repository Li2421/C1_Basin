# State learning bottleneck adjudication

## Conclusion

The evidence does **not** support “state has not been learned.” It supports a narrower conclusion: some state-dependent probability and eta-ranking signal is learned, mainly through the state-conditioned response context, but robust state-adaptive selection has not been demonstrated stably beyond a strong controller–eta prior. This is a source-validation finding, not a new held-controller or cross-scene result.

The clearest immediate obstacle is an underdiscriminative B15 benchmark. Insufficient independent-family support and weak/stochastic residual fitting also matter. Current evidence does not uniquely assign the remaining error to the input representation, architecture, or loss.

## Scope and data integrity

- Source: `orthoflow3_controller_state_residual_factorial_v1`, large H20, with matched H80 and prior controls.
- Ring source-side interventions under three known frozen controllers; 46 TRAIN families, 7 source-VAL families, 16 common exact eta per family/controller.
- 2,208 TRAIN controller-state-eta cells and 336 VAL cells. TRAIN observed-count distribution: 10 Q3, 2,173 Q4, 25 Q16. VAL: 329 complete Q16 cells; eight unresolved numerical/unobserved outcomes across the remaining seven cells.
- Canonical standard-seed aggregation was reconstructed read-only from the global DB and exactly matches the frozen dataset. Source group overlap is zero.
- Main model/checkpoints, generator, labels and splits were not changed. Held-controller and LOSO target labels were not opened. NEW ROLLOUT = 0.
- This is post-hoc source-VAL analysis after checkpoint selection. It must not be described as a fresh independent confirmation.

## 1. Why the earlier “state unused” interpretation is too strong

The actual context is `C = response(h, eta, controller)`: two short physical probes, nominal and candidate-conditioned, started from the actual state. Its features include progress, clearance, safety response and correction response. A context-only model therefore still receives state information. Removing the explicit physical encoder is not a state-free ablation.

For the three large-H20 full models:

| Initialization seed | Correct NLL | Mean NLL over all 5,040 whole-family reassignments | Correct state reversals / 76 | Reassignment mean |
|---|---:|---:|---:|---:|
| 17 | 0.2479 | 0.2833 | 25 | 9.24 |
| 23 | 0.2537 | 0.2761 | 21 | 7.81 |
| 41 | 0.2375 | 0.3116 | 25 | 9.62 |

The reassignment moves the complete, physically consistent h/context prediction profile to another family's labels, preserving controller and eta identities. Unlike an isolated h shuffle, this does not create a mismatched h/C input. This is descriptive evidence of useful state matching, not a confirmatory permutation p-value because VAL already influenced checkpoints.

Context-only models also predict 23–25/76 reversals; eta-only predicts none. Full versus context-only probability improvements are small and not consistent across seeds. Thus the explicit h branch's *incremental* benefit is not established, but state information is not absent.

After subtracting each controller/eta column mean, full-model predicted versus empirical state residual correlations are 0.343, 0.365 and 0.549. Predicted residual variance is only 10–27% of empirical residual variance. This indicates partial, attenuated state response—not a successful complete fit. Empirical Q16 variance includes sampling noise, so attenuation alone is not proof of underfitting.

## 2. The B15 benchmark has only one state-adaptive case of headroom

There are 21 controller-state cases but only seven independent state families. Oracle has certified B15 candidates in 17 cases; four are coverage failures.

| Method | Certified B15 in eligible cases | Numerically unresolved selections |
|---|---:|---:|
| Source-trained eta-only, each seed | 15/17 | 1 |
| TRAIN-estimated fixed eta per known controller | 16/17 | 0 |
| Full H20, seed 17 | 15/17 | 1 |
| Full H20, seed 23 | 17/17 | 0 |
| Full H20, seed 41 | 15/17 | 1 |
| Oracle | 17/17 | 0 |

The descriptive best fixed choice per controller also reaches 16/17. Its per-controller scores are 6/6, 7/7, 3/4. Therefore only one additional certified success is possible through state adaptation beyond that controller-fixed baseline. The TRAIN-estimated prior already attains that 16/17 without looking at VAL to choose eta.

Consequently, a model could learn meaningful state-dependent Q while showing almost no B15 top-1 gain. Many ranking reversals concern alternatives that do not beat the common strong choice. Of 76 empirical reversal quadruples, only ten reverse robust versus non-B15 ordering, and only two reverse robust versus Q-upper-bound <= 0.5 ordering. Even those do not imply all other common candidates fail.

**Correction to interpretation:** 15 certified successes plus one numerical unknown is not 15 successes plus two observed failures. The corresponding certified interval is [15,16]/17, with one known failure. Historical artifacts are retained; this audit makes the bounds explicit.

## 3. What the loss audit does and does not show

The implemented loss uses actual success and failure counts, controller-balanced Bernoulli NLL. Each minibatch uses the same 32 state–eta indices for all three controllers. Failed numerical continuations are not invented as successes/failures. TRAIN-only eta/context normalization exactly reproduces the frozen checkpoints' saved normalization. The audited CPU/training-backend logit discrepancy is below 2e-6; physical and context encoder weights both changed and have nonzero gradients. Existing controller-path checks show correct base-first/future-intervention actions. These checks do not establish all possible semantics bugs are impossible, but no relevant mismatch was found in the audited path.

On the full H20 training histories, Spearman correlations of VAL NLL with reversal-correct count are -0.409, -0.329, -0.697: lower NLL generally accompanies *better*, not worse, ordering. Checkpoint and seed fluctuations remain substantial. Some ranking-best checkpoints differ from NLL-best checkpoints; this does not prove the objective is wrong, and no alternate checkpoint was promoted using these diagnostics.

For a correctly specified conditional Bernoulli probability, minimizing population NLL recovers p. If candidate probabilities are learned accurately, ordering by p is correct for expected continuation success. What is **not** guaranteed is that a finite-data reduction in average NLL improves every near-tie, every fixed-seed B15 threshold, or every top-1 decision. This is a distinction between finite-sample fitting/decision sensitivity and an invalid likelihood.

Full TRAIN NLL is 0.211–0.249 versus VAL 0.237–0.254. The empirical per-cell entropy lower bound is 0.094, but that bound includes fitting Q4 noise; it cannot be used to conclude the network simply needs more capacity.

## 4. Dataset signal is real; predictability is incomplete

TRAIN's first two versus next two compatible seed outcomes give residual correlation 0.652 after removing controller–eta means, and 0.600 after additionally removing state main effects. Cross-half interaction covariance is positive, with a family-bootstrap 95% interval [0.0246, 0.0335]. Hence “the state interaction is all Q4 label noise” is not supported. Approximately 75% of observed Q4 interaction variance is reproducible under this split-half estimator; this estimates repeatability, not predictability from the provided inputs.

Q16 reversal counts must also retain finite-seed uncertainty. Of 76 empirical strong reversals, 31 pass the conservative nominal two-direction paired test, and none survives adjustment across all 7,560 searched quadruples. Split-eight-seed discovery finds 99 and 123 reversals respectively, of which 67 appear in both halves with the same directions/gap. There is aggregate reproducible signal, but not 76 independent, precisely established facts.

Only 46 independent TRAIN families and three known controllers support this residual learning. Repeating 16 eta and seeds provides useful crossing but does not turn those into thousands of independent physical situations.

## 5. New simple-model diagnostic

Reused the previously defined fixed-controller/exact-eta kernel probe on the expanded data. It receives the same actual count labels. Alpha grid [0.03, 0.3, 3, 30] is unchanged and selected by six-fold TRAIN-family CV; VAL is not used for tuning.

| Input | Source VAL NLL | B15 | State reversals |
|---|---:|---:|---:|
| Controller–eta TRAIN prior | 0.26115 | 16/17 | 0/76 |
| Unified h | 0.26095 | 16/17 | 1/76 |
| H20 context | 0.25812 | 16/17 | 0/76 |
| h + context | 0.25931 | 16/17 | 0/76 |

All three probes select the strongest tested regularization. The flexible low-regularization versions perform worse in TRAIN-family cross-validation. This does not prove that neural architecture is optimal; it does rule out the easy explanation that an obvious local fixed-eta predictor already solves the residual and only this neural network fails. These known-controller/eta diagnostic heads are not proposed as a transferable final model.

## 6. Bottleneck adjudication

| Candidate bottleneck | Current finding |
|---|---|
| Benchmark | Directly demonstrated limited state-adaptive B15 headroom and only seven independent VAL families. Main obstacle to a strong top-1 claim. |
| Dataset | Real repeated state signal exists, but few independent families/controllers and mostly Q4 supervision limit precise residual learning. Contribution not isolated causally. |
| Input sufficiency | Baseline h alone was previously disproved sufficient by controller swaps. H20 adds useful information, but its completeness is not established. Removing explicit h is confounded by h information retained in C. |
| Architecture/optimization | Learns some state dependence; residual is attenuated and seed-sensitive. No simple kernel breakthrough. Cannot uniquely attribute error to architecture. |
| Loss | No audited formula/count bug; NLL and ordering usually improve together. No evidence that replacing NLL or repeating ranking loss is the justified first action. |
| Implementation | Counts, normalization, controller path and backend checks pass. Numerical unresolved selections must remain bounds. |

H80 being worse is not proof that more information cannot help: the same 24 summary channels are recomputed over four seconds, replacing the one-second summaries. H80 is not a nested superset retaining early dynamics. This comparison does not settle information sufficiency.

## Next minimal steps

1. Improve **discriminative independent evidence**, not the architecture first. On source TRAIN only, identify natural physical regimes and a common candidate rule with reproducible state-dependent robust alternatives and genuine headroom beyond controller-fixed choices. Freeze independent confirmation families before outcomes; retain all cases, including coverage failures and easy cases. If headroom is again small, label the benchmark underdiscriminative rather than cherry-pick successes. Existing target TEST must remain untouched during this design.
2. For reproducible reversal contrasts, separate TRAIN fit from family-held-out predictability using the frozen inputs/pure-NLL baseline and controlled diagnostics. Reuse stronger seed evidence first; add only targeted TRAIN verification where uncertainty changes the causal conclusion. If TRAIN contrasts fit but held-out fail, prioritize support/conditioning. If TRAIN contrasts cannot fit despite distinct adequate inputs, inspect learning/representation. Do not infer either solely from one B15 count.

No new loss, enlarged network, generator modification or LOSO promotion is justified by this audit alone.

## Reproduction

From `/home/zhihan/research/Basin_C1`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_state_learning_bottleneck_audit_v1.audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_state_learning_bottleneck_audit_v1.fixed_eta_probe
```

Detailed results: `data_integrity.json`, `input_and_state_residual_audit.json`, `state_assignment_permutations.json`, `benchmark_headroom.json`, `TRAIN_label_reliability.json`, `reversal_uncertainty.json`, `loss_and_fit_audit.json`, `expanded_fixed_eta_probe.json`.
