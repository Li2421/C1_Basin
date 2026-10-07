# OrthoFlow3 conservative basin-ball audit v1

## Decision

**BASIN_BALL_AUDIT_UNDERRESOLVED**

The 24-state population and 18 directions were frozen before outcomes. No new rollout was launched because the mandatory core cannot fit the 15,000-continuation automatic cap without weakening the protocol.

- Selected states: 24 ({'train': 18, 'val': 3, 'test': 3}).
- Exact common-cloud cache reuse: 2756/4608; 1852 new trials would be needed merely to finish 8-seed common screening.
- Minimum post-center work for a nontrivial ball: 20544 new continuations.
- Projected minimum including common completion and one center promotion/state: 23740 new continuations, approximately 6780337 physical steps.
- Projected worst-case with five coarse radii and three bisections: 46780 continuations, approximately 13360748 steps.

The fixed post-center validation alone contains six boundary B63 extensions (8064), inside screening/promotions (6336), and outside shell (1536); it cannot be removed without changing the requested science. Local-pair and ellipsoid stages were not included in these estimates.

No evidence about ball validity, center robustness, anisotropy, or Q-exploitation rejection was inferred from this stop. Training `H(h)->[c,r]` is **not yet supported**.

The single smallest justified next experiment is a protocol-preserving pilot on the first 6 states in the frozen permutation. It keeps all 18 directions and validation rules, with a deterministic estimate of 5832-11592 new continuations (approximately 1665667-3310769 physical steps).
