# Controller-information diagnostic (source families only)

No controller context is yet validated for **cross-scene robust eta selection**. Matched intervention data show controller-dependent eta preference. A fixed 24D controller fingerprint learns known controllers but fails on a held controller. An eta-conditioned H20 response improved Q prediction on one freshly held-out Flow with no B15 candidates. A subsequent independent Flow with an oracle-eligible K16 pool showed **no B15 selection gain** over eta-only or wrong-controller context; see the held-controller follow-up below.

## Protocol

- Existing `triplet_dataset_h20.npz`; 12 source TRAIN states and four independent source-family VAL states per scene, four matched eta candidates per state. Toy has two controllers; DB, Four-Way and Ring have three.
- Same critic encoder/trunk and observed-count Bernoulli NLL, three initialization seeds (17, 23, 41). Partial records contribute only actual observed successes/failures; no Q16 imputation. Checkpoints use source VAL NLL.
- Compared eta-only; eta plus explicit controller ID (privileged diagnostic, not deployable zero-shot); unified state plus H20 physical controller response; that response plus ID; and a new 24D physical Flow/safety response fingerprint from eight outcome-blind source TRAIN states.
- For a stricter test, withheld the third frozen Flow's *labels and normalization statistics* during fingerprint-model training and checkpoint selection, then evaluated that Flow on the independent source VAL families. The fingerprint itself is computed by querying the frozen controller; no outcome labels or full task rollout enter it.
- No frozen cross-scene target K16 outcomes were used for model selection. The initial diagnostic used zero new task continuations; the subsequent source-only matrix and fresh-controller checks added **595**, all journaled and merged into the global rollout DB.

## Direct evidence

| Source-family VAL | Eta-only | Explicit controller ID + eta | H20 physical context + state | Response-bank fingerprint + eta |
|---|---:|---:|---:|---:|
| Four-Way: certified controller-induced eta-order reversals | 0/7 (all seeds) | 6/7 (all seeds) | 0–1/7 | 6/7 (all seeds) |
| Ring: certified reversals | 0/14 | 0–1/14 | 0/14 | 0–8/14, unstable |

In Four-Way, known-controller B15 selection was 9/9 for explicit ID and response-bank fingerprint; H20 physical context ranged 7–9/9. Replacing the correct fingerprint with another controller's fingerprint changed the selected eta in 3–5/9 available-B15 slots and reduced B15 selection from 9/9 to 6–7/9. By contrast, replacing H20 changed selection in only 0–2/9 slots and did not change B15 selection. The fingerprint is thus *used* by the model, but this alone says little about extrapolation.

The decisive held-controller check is worse:

| Third Flow's labels withheld during training | Four-Way | Ring |
|---|---:|---:|
| Fingerprint + eta B15 (seeds 17/23/41) | 3/4, 3/4, 3/4 | 0/2, 0/2, 0/2 |
| Fingerprint + state + eta B15 | 3/4, 3/4, 3/4 | 1/2, 1/2, 0/2 |
| Fingerprint + eta, certified reversals involving held Flow | 1/3 all seeds | 0/8 all seeds |

With third-Flow labels included, the corresponding Four-Way fingerprint+eta model gave 4/4 and 2/3 held-Flow reversals; Ring results were seed-unstable even with labels. DB gave 4/4 for all methods, but had **zero certified VAL reversals** and is not discriminative here.

The matched labels do contain interaction evidence: Ring TRAIN has 34 certified controller-induced eta-order reversals across 9/12 states; Four-Way TRAIN has 33 across 11/12. Their supervision geometry differs: Ring's 48 TRAIN pairs span 28 exact eta (typically one state per eta), whereas Four-Way's 45 pairs span 11 exact eta. Ring therefore tests a much harder joint state–eta generalization; abundant reversals alone do not guarantee shared eta support. Existing independent unseen-controller H20/H100 audit also found Ring poor (nominal H20 selected B15 on 1/2, 1/2, 0/2 available states across seeds; rich H20 0/2), so the new fingerprint does not reverse that conclusion.

## Adjudication

1. **Controller information is genuinely necessary for exact feasibility.** The archived controller-swap counterexample keeps current (h,\eta\), first action and seeds equal while future Flow changes, with 8/64 pairs flipping from 16/16 to 0/16.
2. **The tested compact H20 context is not sufficient as a proven eta-selection signal.** It is only moderately controller-decodable (Four-Way and Ring ~56% versus 33% chance), and correct/wrong replacement seldom changes B15 choice on the older intervention panel. The fresh v8 test below reveals a narrower positive probability signal.
3. **A richer controller fingerprint mainly recovers known-controller preference, not robust unseen-controller generalization.** In Four-Way the model can use it, but held-controller selection and held-controller reversals drop; in Ring the held controller is essentially not solved. The fingerprint is built from source-scene states and cannot be passed off as a cross-scene invariant representation.
4. **State–eta–controller learning remains underresolved.** Ring has interaction labels, but sparse reuse of each exact eta across states plus unstable three-seed fits prevent isolating representation insufficiency from support/optimization. A full controller function plus complete Markov physical state would in principle carry the missing information; none of the tested finite summaries is shown sufficient.

## Follow-up 1: source TRAIN cross-matrix

A four-eta panel was chosen by exact-eta frequency among Ring source TRAIN records, not VAL outcomes. The same four eta were evaluated across 12 TRAIN states under the base and first alternative Flow. Cache preflight: 768 requested, 384 exact reuse, 45 partial reuse, 339 truly missing. The 339 new continuations were journaled and merged with zero conflicts. Postflight: 752 exact reuse; 14 records partial because two seed-level runs had genuine numerical failures, which remain excluded from exact Q evidence. The 29 new state–eta cells contributed 462 valid observed continuations. An initial v1 manifest had mismatched controller UID/checkpoint and was stopped by an assertion **before any rollout**; v2 is the valid protocol.

This matrix does contain state dependence: across the four shared eta, two show both Q8 ≤0.5 and Q8 ≥0.875 on different source states, with 15 and 32 strong within-controller state-order reversals under the two Flow variants. It is therefore not merely more universal-positive data. Yet retraining the same fingerprint critics with the same steps and seeds, while keeping the third Flow's labels out, did **not** reliably solve that held controller: on its four VAL states with B15 candidates in two, fingerprint+eta stayed 0/2 in all seeds; fingerprint+state+eta changed from 1/2, 1/2, 0/2 to 1/2, 1/2, 1/2. NLL improved more often than selection. Crossed data alone are not a sufficient fix.

## Follow-up 2: fresh fourth Flow v8

Before seeing its outcomes, a compatible frozen v8 local Flow was selected by checkpoint chronology. Its 24D fingerprint was computed on the same outcome-blind eight-state source TRAIN bank. On the frozen four source-VAL states × four existing eta, preflight found 256 genuinely missing seeds; all 256 were run, journaled, merged, and returned EXACT_REUSE on postflight. Source-only model checkpoints were fixed before opening the v8 labels.

The v8 candidate pool is **underdiscriminative for B15 selection**: oracle = 0/4. All 16 candidates have full Q16; mean empirical Q16 is 0.082, versus base controller's mean observed Q lower bound 0.629 on the same state–eta pairs. Eight base candidates were B15; seven of those became ≤8/16 under v8. No selected-B15 number from this pool should be interpreted as critic success or failure.

It is nevertheless a sharp probability/representation test. The fixed-bank v8 fingerprint is only 0.0159 raw L2 from base, closer than the existing v7 fingerprint; eta-only NLL across seeds is 0.575/0.592/0.564, while bank-fingerprint critics are generally worse (NLL 0.779–1.924). At the *evaluated state and eta*, the H20 physical response changes substantially: median normalized distance from base is 0.948; all ten pairs with Q contrast ≥0.5 have H20 distance >0.25. The existing H20 critic, without v8 training labels, attains NLL 0.533/0.542/0.447. Replacing the correct v8 H20 with matched base H20 worsens NLL to 0.618/0.637/0.481. Replacing only eta-conditioned response channels accounts for most of this degradation; replacing only nominal channels has a much smaller effect.

Thus the question is not simply “controller information missing from every finite input.” A **state/eta-local, physically computed short response** carries transferable information across these Ring Flow checkpoints, and an existing model uses it in probability prediction. The fixed controller-level fingerprint does not. But v8 has no B15 proposal, the older v7 selector test remains weak, and archived strict cross-scene LOSO with H20 still fails. Neither a deployable minimal context nor cross-scene state–eta generalization is established.

## Follow-up 3: independent Flow v11 with oracle-eligible K16 pool

An independently trained, unseen compatible Ring Flow (seed 88123) was frozen before task outcomes. Twelve new source-VAL families absent from earlier controller probes each received the same score-blind 16-eta source-TRAIN panel. Source-trained eta-only, correct H20 and wrong-base-H20 probabilities were frozen before rollout. Preflight found 3,072 genuinely missing continuations; all 3,072 were journaled and merged, with zero conflicts. Nine numerical failures remain explicitly unresolved at the seed level. All 12 states had both B15 and confirmed non-B15 candidates, and oracle coverage was 12/12.

Across seeds 17/23/41, **all three methods selected B15 on 11/12 states**. Correct H20 rescued zero eta-only failures and zero wrong-context failures. Correct context changed top-1 choices on 3–6 states, but did not improve B15 selection. Its observed-count NLL (0.693/0.740/0.709) was worse than both eta-only (0.544/0.533/0.556) and wrong-base H20 (0.584/0.615/0.628). The one persistent failed family has only 2/16 B15 candidates; one false 1/16 top eta received probability 0.999 from correct H20. The pool is partly easy (134/192 pairs B15), but 107 strong state–eta ranking reversals show nontrivial interaction; wrong context predicts these reversals about as well as correct context. This is **`HELD_CONTROLLER_SELECTION_NOT_SUPPORTED`**, not proof that controller-conditioned feasibility is impossible. Full audit: `held_controller_robust_selection_v1/final_report.md`.

The next independent test, if needed, must freeze new hard true-t0 families and a source-only candidate rule before labels are seen. Do not retune on the now-opened v7/v8/v11 panels, modify the generator, or promote H20 into the cross-scene system yet.

Reproduction: `probe.py`, `crossmatrix_ring.py`, `fresh_controller.py`, `held_robust_benchmark.py`, and companion audit/evaluation scripts in this directory. Machine-readable follow-up results are in `crossmatrix_ring_v2/`, `fresh_controller_v8_v1/` and `held_controller_robust_selection_v1/`; per-run checkpoints and normalizers are under `models/`.
