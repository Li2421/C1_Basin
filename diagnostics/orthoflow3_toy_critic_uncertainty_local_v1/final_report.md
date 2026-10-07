# Toy K=16 critic uncertainty and local-basin audit

The frozen 200-state WIDE-IC hard cohort has 194 states containing a B15 generator proposal, while the original three-seed W1 critic chooses B15 on 181. This experiment changes neither the generator nor the 16 proposals, controller, Success Basin definition, critic architecture, or pure-NLL objective. It adds two independently initialized/minibatched W1 critics (seeds 47 and 59) to the existing seeds 17, 23, 41, using the same 2,876 structured-plus-wide TRAIN pairs and 144-pair VAL panel. No hard-state outcome enters training or selection. **NEW ROLLOUT = 0.**

The pre-registered conservative grid is lambda = {0, 0.25, 0.5, 1, 2}; its VAL criterion is mean selected empirical Q, regret, then smaller lambda. All five lambda values selected the same VAL eta and had mean selected Q 0.9792, so lambda **0** was frozen before hard TEST. Most VAL labels are 8-seed observations, not B15 certifications; this panel has weak power to select a conservative penalty.

| Selection rule on frozen K=16 proposals | B15 / 200 | Oracle gap | Mean selected Q16 | Rescue / break versus original critic |
|:--|--:|--:|--:|--:|
| Original three-seed mean-logit critic | 181 | 13 | 0.9588 | — |
| Five-seed mean probability, VAL-selected lambda=0 | **183** | 11 | 0.9622 | 3 / 1 |
| LCB lambda=0.25 (TEST diagnostic) | 183 | 11 | 0.9622 | 3 / 1 |
| LCB lambda=0.5 (TEST diagnostic only) | 184 | 10 | 0.9625 | 4 / 1 |
| LCB lambda=1 (TEST diagnostic only) | 183 | 11 | 0.9584 | 3 / 1 |
| LCB lambda=2 (TEST diagnostic only) | 182 | 12 | 0.9541 | 3 / 2 |
| Six-point virtual local mean, radius 0.05 | 183 | 11 | 0.9606 | 3 / 1 |

The actual VAL-selected policy gains only 2/200 states (1 percentage point) over the original critic; state-paired bootstrap 95% interval for this difference is −1 to +3 percentage points. Three original errors are rescued (episodes 70, 151, 177) and one previously correct state is broken (episode 191). The gain is **not caused by uncertainty penalization** because frozen lambda is zero. As an additional attribution check, the original three seeds averaged as probabilities select 182/200, while adding two seeds gives 183/200; one gain comes from score aggregation and one from added members. Lambda=0.5's 184/200 adds episode 152 relative to five-seed mean, but is a TEST-only diagnostic and must not be adopted without independent selection/replication.

## Does member disagreement identify false positives?

Among the 13 original false top-1 choices, ensemble probability standard deviation has median 0.0274, versus 0.00000190 for the 181 correct original choices, 0.00000958 for all B15 proposals, and 0.00105 for all 3,200 proposals. Its discrimination AUROC for false versus correct original top-1 among oracle-coverable states is 0.808; 11/13 false choices exceed the correct-choice median uncertainty. Therefore uncertainty contains a real **diagnostic signal**. But 8/13 false choices are assigned >0.9 probability by **all five** members, including states with severe true-Q errors (e.g., episode 102: Q16=0, five probabilities 0.924–0.991; episode 115: Q16=3/16, all five essentially 1). That is systematic shared bias, not epistemic variance that an LCB can reliably suppress. The 13 original chosen failures have average true Q16 0.567 despite high critic scores; the existing proposal audit identified overestimation and misranking but did not prove a single cause.

## Local true-basin evidence and virtual stability

For each of the 13 errors, we audited the bad top-1, the nearest B15 proposal, and, when distinct, the highest-Q oracle proposal: 34 unique centers. Under the exact state/controller fingerprint and standard 16 seeds, **none** has another complete-Q16 eta within normalized radius 0.02, 0.05, or 0.10. The nearest observed B15 proposal to a bad center is 0.138–0.845 away (median 0.442), so the existing points cannot resolve a local success boundary.

At radius 0.05, six virtual axis perturbations scored by the five-member critic produce no isolated predicted-spike proxy. For bad centers, the mean predicted Q changes by only 0.0004 on average, with median local predictive variance 0.000012. The critic is making a **smooth high-prediction plateau**, not an obvious single-point numerical spike. The virtual-local selection score chooses the same B15 states as the five-member pointwise mean: 183/200, zero paired B15 rescue or break. These are model-only perturbations; without true neighbor rollouts they cannot certify a robust basin interior or a sharp true boundary.

A fixed six-axis local-probe plan at the three radii contains 612 unique state–eta pairs / 9,792 standard-seed continuations. Global database preflight: exact 0, partial 0, aggregate 0, genuinely missing 9,792. The planner's `incompatible` counter is its generic no-record label here; the controller profile itself is `EXACT_PROFILE` with authoritative OrthoFlow3 hash. Because the missing count exceeds 500, **no local rollout was run**.

## Verdict

- `ENSEMBLE_UNCERTAINTY_IDENTIFIES_FALSE_PEAKS` **partially as a diagnostic**, but many high-confidence false positives are shared by all five critics.
- `LCB_SELECTION_NO_GAIN` under the only VAL-frozen lambda (0). The TEST-only lambda=0.5 result cannot be treated as a selected policy.
- Local geometry: **UNDERRESOLVED**, with no evidence for isolated predicted spikes. Do not assert `TRUE_BASIN_BOUNDARIES_ARE_SHARP` from these centers alone.
- Main supported explanation: **systematic high-confidence bias plus partial epistemic uncertainty**; proposal distribution shift is plausible, but only 4/13 old errors had the strong distance-shift proxy, and true local boundary topology remains unmeasured.

The validated change is **181/200 → 183/200**, still below the 194/200 proposal oracle. There is no evidence here to modify the final pipeline on the basis of LCB or local stability alone.

Artifacts: `protocol.json`, `val_lambda_grid.json`, `frozen_selection.json`, `selection_summary.json`, `selection_per_state.csv`, `original_13_false_top1_ensemble.csv`, `local_neighborhood_audit.csv`, `local_probe_preflight.json`.
