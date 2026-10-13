# Part 3 Round 2 preregistration — frozen before new outcomes

## Fixed execution

- Ring Exchange, native true-t0 generator, FF, H700, dt 0.05, unchanged OrthoFlow3 and hard projection.
- Real Flow pair: `base_u_v2_macflow`, `base_u_v10_local_macflow` from P1. The protocol stores exact controller UIDs, checkpoint and adapter SHA256, observation frame, normalization and evaluator profile.
- Candidate bank: original P1 K16, but only η indices **1 and 2**. The choice uses **P1 seed indices 0–1 only**. For each pair of K16 candidates, calculate two controllers' 24-state average directional preferences. Rank by the positive opposite-preference score `max(0,min(Δ_v2,−Δ_v10),min(−Δ_v2,Δ_v10))`; ties use higher mean statewise two-choice V, then smaller indices. Unknown old outcomes count as zero in this discovery score. The top pair is (1,2), score 0.35417, discovery V 0.14583. P1 seed indices 2–3 and Round 2 outcomes do not choose candidates.
- State generator: existing `controller_q_transfer_round3_v1.state_generator.generate`, `split=part3_round2_confirm`, sequential draw seeds **2026101200–2026101204**, all five retained. No controller outcome criterion. Before freezing, the five exact state UIDs and physical content hashes were absent from the shared database; they are recorded in `protocol.json`.
- Common random numbers: existing `execution.key_for`, root 2026100403, fold state UID token, seed index 0–15, physical step. Same seed index/key schedule for both controllers and η. Numerical failure stays **unknown**.
- Maximum new physical continuations: 320. Only `genuinely_missing` exact requests after official cache preflight may run. No video batch is required for pure Q label confirmation.

## Frozen analysis

The unit for cross-state uncertainty is the physical state (n=5). Q16 for each of 20 cells is `[success/16,(success+unknown)/16]`; report binomial Wilson 95% intervals for observed valid trials, together with the unknown bounds. Keep collision, timeout and numerical failure separate. Report every state and both candidates, including failures.

Define the predeclared controller interaction using η2 minus η1:

`I_s = [Q(v2,s,η2)-Q(v2,s,η1)] - [Q(v10,s,η2)-Q(v10,s,η1)]`.

Positive `I` supports differing preferences; report both controller differences separately, seed-paired uncertainty, unknown-safe lower/upper bounds, and the five state signs. Do not infer optimal choice conflict from `I>0` alone.

For each state, report the empirical two-candidate oracle information value under a uniform controller prior:

`V_s=½[max_η Q(v2,s,η)+max_η Q(v10,s,η)]−max_η½[Q(v2,s,η)+Q(v10,s,η)]`.

The Q16 plug-in V is upward biased by selecting on the same seeds. The **primary selection estimate** is two-fold seed crossfit: choose controller-specific and blind η on seeds 0–7, evaluate their paired difference on seeds 8–15; reverse halves and average. Unknown outcomes give conservative paired lower/upper bounds, with shared chosen cells canceling exactly. Report state-grouped bootstrap 95% intervals (fixed RNG seed 2026101217, 10,000 resamples) and an exact five-state sign count. This interval is descriptive for five generator draws, not a population certificate. Also report Q16 candidate coverage and whether any controller strictly prefers opposite η in the unknown-safe intervals.

**Independent empirical critic confirmation** requires the new-state paired interaction mean to stay positive under unknown-safe assignment and its state bootstrap 95% lower limit to exceed zero. **Independent empirical selector confirmation** requires primary crossfit mean positive under unknown-safe assignment and its state bootstrap 95% lower limit to exceed zero, with at least three of five states having positive paired crossfit value. If either fails, classify evidence as `INCONCLUSIVE` or `REJECTED_FOR_THIS_PANEL`, distinguish candidate saturation/coverage, numerical unknown, and state heterogeneity, and retain the original pair. No alternative pair, η or state will be screened after outcomes. Even if both pass, five states do not establish a population necessity theorem or deployable held-checkpoint selector benefit.

## Representation diagnostics (outcome-free feature construction)

Reuse P1's 7 pose × 3 pseudo-time × 5 pseudo-action field probes and four nominal draws. R3 and R4_action remain exactly as archived. The repaired R5 descriptor retains R3 plus the **pose/time-indexed 2×2 finite-difference response Jacobians** and the raw five responses for near-goal moving poses 4–6. It uses no η or outcomes during construction. `R5` versus raw sampled behavior, controller pair distances, and outcome association on old P1 labels are exploratory. Leave-checkpoint-out nearest-source Q transfer may use source checkpoints' labels and all static features, but never held checkpoint labels for feature scaling, neighbor choice or fitting. Held labels are evaluation only. This is not a certified transfer bound.
