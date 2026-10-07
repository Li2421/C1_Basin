# Controller state-residual factorial: source-only result

**Verdict: source state-residual gate not passed.** The longer H80 response did not help, and doubling distinct TRAIN families improved probability prediction but did not yield a stable state-specific B15 selection gain. The independent held-controller confirmation and strict LOSO target labels remain unopened. This is **not** evidence that a controller-conditioned feasibility law is impossible; the present source validation is underdiscriminative for robust top-1 selection and the TRAIN matrix has only four observed seeds per pair.

## Frozen design and integrity

Ring true-t0, three compatible frozen future Flow controllers, the same 16 exact eta probes under every controller/state. The H20/H80 arms use identical task labels, architecture, optimizer, 1,500 training steps, and seeds 17/23/41. Outcome-blind source-family sampling selected 24 original plus 24 additional TRAIN families and eight new VAL families. Short controller-response probe safety-projection numerical errors occurred for three complete families (indices 33, 37, 53); they were excluded identically from **all** models before training, without looking at task success. Effective sizes: 24 versus 46 TRAIN families, seven VAL families. Original task rollout records for excluded families remain in the global DB.

Global cache preflight: 15,360 requested, 4,712 exact-reusable requests, 15 partial-reusable requests, zero aggregate, and 10,633 planner-missing seeds. Five were previously executed numerical failures and were not rerun. Six GPU shards executed the remaining **10,628** continuations. All 10,628 journal records were atomically merged and independently aligned with canonical state/eta/controller/seed keys. Postflight: **15,341 exact reusable seeds, 19 numerical attempts retained separately, 0 unattempted, 0 ambiguous, 0 incompatible, 0 conflicts, 0 collisions**. Fourteen numerical attempts were new; none was imputed into Q16.

## Independent source-family VAL

Seventeen controller-state cases had at least one B15 candidate. Numbers below are medians over three independent training seeds, except the non-neural known-controller prior.

| Model | VAL NLL | B15 selected / 17 | Strong state eta reversals / 76 |
|---|---:|---:|---:|
| Eta-only, 46 TRAIN families | 0.341 | 15 | 0 |
| Known-controller × eta TRAIN-count prior | 0.261 | 16 | 0 |
| Additive A(h,C)+B(eta), H20, 46 | 0.297 | 15 | 0 |
| Full h+C+eta, H20, 24 | 0.271 | 16 | 13 |
| Full h+C+eta, H20, 46 | **0.248** | 15 (seed range 15–17) | 25 |
| C+eta, physical h masked, H20, 46 | 0.249 | 15 | 25 |
| Full h+C+eta, H80, 46 | 0.289 | 15 | 10 |

The H20 full model's NLL is lower than eta-only and additive, and its controller-context replacement raises median NLL by 0.380. Thus the context is genuinely used for probability prediction. But explicit physical **h** adds essentially nothing beyond the eta-conditioned response context: the h-masked control has matching NLL and reversal performance. Shuffling h changes NLL in two seeds but does not change B15 selected in any seed. H80 is worse than H20 on NLL in all three seeds and never improves B15; adding more horizon is not a supported fix.

The 46-family H20 model improves source VAL NLL over 24-family H20 by 0.018–0.033 across matched seeds, so source diversity helps probability learning. It reaches 17/17 B15 in one seed but only 15/17 in the other two. Eta-only already reaches 15/17; the stronger, non-deployable known-controller × eta prior reaches 16/17. There is too little robust-selection headroom here to claim a stable state-specific gain. On the naturally occurring strong state ranking reversals the full model gets 21–25/76, while its h-masked context control gets 23–25/76. A simple fixed-k state-local baseline occasionally selects 17/17, but has poor reversal accuracy and was not selected by an independent tuning set; this does not establish transferable state learning.

## Interpretation and decision

This experiment separates two effects: controller-response features do contain useful predictive information, but the observed source success can mostly be explained without separately learning an h-dependent eta preference. The current source TRAIN pair labels average roughly four seeds and the seven-family VAL is nearly saturated for B15. Thus a failure to beat eta-only on top-1 does **not** establish information insufficiency; nor does lower NLL establish robust-selection transfer. Longer H80 response is directly disfavored by the matched-label comparison. More source families improved NLL but not stable top-1. The remaining bottleneck is underresolved between sparse/noisy state×eta supervision, a weakly discriminative candidate panel, and model/representation limits.

Per the frozen [decision rules](decision_rules.md), the held-controller target is not opened. The pre-registered cap of 10,752 new continuations has only 124 remaining after this phase, so a meaningful further targeted source matrix cannot be added under this phase's budget. The minimal follow-on is a separately authorized, outcome-blind source-family candidate panel with demonstrable eta-ranking reversals and lower global-eta saturation, plus stronger TRAIN count evidence; only then should an independently frozen held-controller robust-selection test be opened. No generator or safety semantics were changed.

## Resource use

The node has one ~98 GB RTX PRO 6000 GPU exposed as eight Slurm shards. Six shards and 12 CPU cores were used at peak, reserving two shards and at least six CPU cores for others. During six-way simulation, sampled GPU SM utilization was 55–74% with ~3.5 GB memory allocated. Low memory is expected for these small per-step simulations; it is not unused parallel shard capacity. Interrupting/rebuilding the nearly complete final batch would have increased latency and introduced executor-validation risk. Training used six-way GPU parallelism; all controls and reports are complete.

Reproduction: `protocol.json`, `planned_rollouts.json`, `cache_preflight.json`, `cache_postflight.json`, `alignment_audit.json`, `context_failure_audit.json`, `source_validation_metrics.csv`, `paired_comparisons.json`, `known_controller_eta_prior.json`, `local_baseline_results.json`, and the checkpoint trees under `variants/` and `additive_variants/`.
