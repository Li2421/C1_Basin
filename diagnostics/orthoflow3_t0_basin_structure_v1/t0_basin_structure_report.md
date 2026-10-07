# OrthoFlow3 true-t0 basin structure audit v1

## Decision

**T0_BASIN_AUDIT_UNDERRESOLVED**

The audit was stopped when aggregate valid execution reached **17,904 new continuations and 8,569,264 physical steps**, exceeding the frozen 7,000,000-step automatic cap by 1,569,264. No robustness criterion was weakened and no neural network was trained. Earlier feature-replay attempts were quarantined: they executed zero physical steps and are excluded from every scientific result.

## Frozen cohort and completion

- Attempted: 8 genuine t=0 states from 8 independent source groups.
- Robust centers: 8/8; every resolved center was 64/64.
- Ball geometry constructed: 6/8.
- Full independent inside validation complete: 5/8.
- Margin-usable (`r>=0.20`): 2/5 completed balls.
- Internal false inclusions: 0/20 mandatory promoted points; all 60/60 independent screening points in the five completed balls were 8/8.

## Completed-ball geometry

| state | zero screen | zero B63 | center eta | center Q64 | radius | validation | margin usable | zero in ball |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0_WIDE_perm00_ep0082 | 6/8 | unresolved | (0.898438, -0.156250, 0.445312) | 64/64 | 0.021250 | complete | no | False |
| T0_WIDE_perm01_ep0217 | 8/8 | unresolved | (0.625000, 0.000000, 0.375000) | 64/64 | 0.249688 | complete | yes | False |
| T0_WIDE_perm02_ep0182 | 0/8 | unresolved | (0.625000, 0.000000, 0.375000) | 64/64 | 0.201875 | complete | yes | False |
| T0_WIDE_perm03_ep0205 | 6/8 | unresolved | (0.625000, 0.000000, 0.375000) | 64/64 | 0.085000 | complete | no | False |
| T0_WIDE_perm04_ep0195 | 8/8 | unresolved | (0.898438, -0.156250, 0.445312) | 64/64 | 0.021250 | incomplete | underresolved | False |
| T0_WIDE_perm05_ep0179 | 7/8 | unresolved | (0.898438, -0.156250, 0.445312) | 64/64 | 0.170000 | complete | no | False |
| T0_WIDE_perm06_ep0074 | 6/8 | unresolved | (0.566406, -0.140625, 0.316406) | 64/64 | unresolved | not constructed | underresolved | BALL_UNRESOLVED |
| T0_WIDE_perm07_ep0139 | 6/8 | unresolved | (0.898438, -0.156250, 0.445312) | 64/64 | unresolved | not constructed | underresolved | BALL_UNRESOLVED |


Completed radii: mean 0.145563, median 0.170000, min 0.021250, max 0.249688. Three of five complete balls are smaller than 0.20; the partial evidence therefore leans toward t0 spheres often being too small for the frozen margin-learning criterion, despite perfect observed internal reliability.

Pairwise normalized center distance: mean 0.261719, median 0.407583, min 0.000000, max 0.475159. Relative to mean radius, the ratio has mean 2.167 and median 2.067.

The five completed full balls have all-intersection `False` and maximum overlap 3/5. Their 0.8-retained regions have all-intersection `False` and maximum overlap 3/5. This is evidence against the universal-region degeneracy seen in the intermediate training balls, but it is not the preregistered 8-state answer.

## Zero feasibility and frozen controllers

`eta=0` received only 8-seed screening before the hard stop. Screening counts were [6, 8, 0, 6, 8, 7, 6, 6], but none were promoted to 64; therefore ZERO_B63 frequency is unresolved. Zero lies outside every constructed t0 ball.

The existing fresh-WIDE records provide the exact frozen controller eta predictions for these episode-start h values. Relative to the five complete balls, inside-ball counts are G_LOWJ 0/5, G_CENTER 0/5, and G_MARGIN 0/5. Those old records contain one episode outcome each, not Q64; no new controller rollouts were launched after the budget stop.

## Interpretation

- Are true t0 balls sufficiently large for margin learning? **Not established.** Only 2/5 completed balls satisfy `r>=0.20`; three are small (including two at 0.02125 and 0.085).
- State-dependent or universal? **Partial evidence favors state dependence:** no common point exists across the five full or retained balls, and maximum retained overlap is 3/5.
- Major t0/intermediate phase mismatch? **Suggestive but underresolved.** Intermediate labels all share center `(0.625,0,0.375)` and radii 0.23375–0.35362; two of five completed t0 centers shift by about 0.408 normalized units and several t0 radii collapse. The complete eight-state/controller-Q64 comparison was not affordable.
- Does this explain CENTER/MARGIN falling below Safety? **Plausible but not demonstrated**, because eta-zero B63 and matched controller Q64 were the post-ball stages omitted by the hard stop.

## Smallest justified next step

With explicit approval for the additional cost, resume only the three incomplete state validations plus eta-zero B63 and frozen-controller Q64 checks; do not collect new states or train a model. Recompute a physical-step-aware cost ceiling from the observed 478.6 steps/continuation before resuming.
