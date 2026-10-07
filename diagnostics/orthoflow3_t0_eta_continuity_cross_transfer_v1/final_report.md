# OrthoFlow3 true-t0 robust eta continuity and cross-transfer audit

## Result

**TARGET_SELECTION_MULTIMODALITY_DOMINANT**

`BEGIN_BASIN_REPRESENTATION_SEARCH = YES`. No model was trained and no basin representation was fitted.

## State and target geometry

The 40 source-isolated true-t0 states have 1-NN normalized h0 distances: min 3.888, median 7.217, max 12.027. Across all 780 pairs, Spearman(h-distance, target-distance) is 0.063; among rank-1/2/3/5 neighbors it is 0.221. There are 2 predeclared target-jump candidates. The 40 labels use only 9 distinct eta targets; the largest repeated target mode occurs in 16/40 states.

## Cross-transfer

The frozen targeted audit contains 73 symmetric state pairs (146 directed Q64 tests): 53 mutual, 15 one-way, and 5 no robust transfer. The closest-distance bin has mutual-or-one-way rate 93.8% and mutual rate 78.1%. 2/2 large target jumps preserve mutual B63 transfer. Complete k=3 neighborhood audits cover 28/40 states with mean overlap 89.3%; complete k=5 audits cover 9/40 with mean overlap 86.7%. Across cached plus targeted evidence, 257 symmetric exact-Q64 pairs are available; their bin statistics are reported separately and explicitly marked ascertainment-biased.

The two predeclared near-state/large-target-jump cases both transferred mutually at B63. Thus their distant labels do not represent a demonstrated basin discontinuity: they are different robust choices inside strongly overlapping local success sets. Transfer weakens moderately with continuous h-distance (Spearman below), but the small binned controls do not show a clean monotonic decay. There is no repeated very-close no-transfer evidence sufficient to claim a genuine local discontinuity.

## State metric and coverage

Full normalized 214-D h0 distance has Spearman -0.454 with mean cross-transfer Q64 (negative means transfer weakens with distance). The strongest documented subset was `current_control` at -0.459. This negligible difference does not establish a substantially superior replacement metric; the exact comparison is in `state_metric_comparison.csv`. State coverage is sufficient for the local ambiguity diagnosis, although the 1-NN distances show that 40 states remain sparse for learning a detailed global basin map.

Midpoint-state testing was skipped because interpolating positions or h0 cannot preserve exact Flow xi0/RNG conditioning and full true-t0 physical semantics. No genuine basin discontinuity is therefore claimed.

## Answers

- Did point regression fail mainly because the target is set-valued/multimodal? **Yes, this audit supports that as the dominant structural cause:** nearby states can have far-apart selected labels while both labels remain mutually B63.
- Is current true-t0 state coverage sufficient? **Sufficient for this local cross-transfer diagnosis, but not dense enough for an unconstrained global basin reconstruction.**
- Should analytic basin representation search begin? **YES.** It should model state-conditioned overlapping feasible sets, not regress one arbitrary eta.

## Next step

Run one constrained analytic representation comparison obeying `basin_representation_requirements.md`, using existing exact Q64 positives and negatives with state-held-out validation before any new controller training.
