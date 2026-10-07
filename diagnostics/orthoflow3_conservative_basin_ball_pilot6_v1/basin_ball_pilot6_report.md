# OrthoFlow3 conservative basin-ball pilot6 v1

## Decision

**BALL_PILOT_FAILS_FEASIBILITY**

- States: 6; B63 centers: 3; resolved balls: 0.
- Explicit-zero special centers that cannot support a positive-radius active-domain ball: 3.
- New continuations: 424 total (360 valid-protocol; 64 quarantined pre-patch domain-gap queries); new physical steps: 117229.
- Inside screening failures: 0; B63 false inclusions among promoted inside points: 0.

## Frozen-state outcomes

- `N_r104_s125`: zero=8/8; max common-cloud result=8/8; 8/8 candidates=9.
- `ZR_P_r052_m080_s95400004_p253`: zero=5/8; max common-cloud result=6/8; 8/8 candidates=0.
- `S_r043_p02`: zero=8/8; max common-cloud result=8/8; 8/8 candidates=17.
- `N_r076_s119`: zero=4/8; max common-cloud result=6/8; 8/8 candidates=0.
- `R_D2_s95101009_p112`: zero=8/8; max common-cloud result=8/8; 8/8 candidates=22.
- `N_r004_m120`: zero=5/8; max common-cloud result=6/8; 8/8 candidates=0.

## Interpretation

Three states resolved to `c0=(0,0,0)` with Q64=64/64.  The frozen eta domain treats zero as an explicit special point outside the continuous ACTIVE box `[0.5,1.25] x [-0.5,0.5] x [0,0.75]`; therefore any positive-radius Euclidean ball centered at zero necessarily includes unauthorized eta values.  These are correctly recorded as `BALL_DEGENERATE_ZERO_SPECIAL`, not treated as positive-radius balls.

The other three states had no 8/8 common-cloud candidate (their maximum screening result was 6/8), hence the frozen center rule provided no candidate eligible for B63 promotion.  Consequently no valid positive-radius ball exists in this pilot, and independent inside/shell validation and anisotropy are not applicable rather than zero-error results.

The two archived critic-exploitation cases have no exact state match in this six-state prefix, so their ball membership is NOT_APPLICABLE.  The 64 pre-patch zero-center ray records are preserved in a quarantine CSV and excluded from all geometry calculations.

Sparse directional probing did not construct a useful generic continuous 3-D inner ball under the exact frozen domain.  A robust zero center is a plausible controller value for its individual zero-sufficient states, but it is not a generic positive-radius ball-center label; these results do not justify training `H(h)->[c,r]`.

No network was trained and no controller semantics were modified.
