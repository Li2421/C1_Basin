# OrthoFlow3 continuous-domain inner-ball pilot6 v1

## Decision

**CONTINUOUS_INNER_BALL_PROMISING_BUT_CONSERVATIVE**

After replacing the historical disconnected search domain by the exact continuous bridge domain, a B63 center and a contained empirical inner ball were resolved for all six frozen states. Independent validation found no false inclusion: all 72 inside points screened 8/8 and all 18 predeclared 64-seed validation points were 64/64. The sphere is nevertheless strongly conservative: 47/48 points at approximately 1.15 times the ball radius still screened 8/8, four states were strongly anisotropic, and one state produced a nearly degenerate sphere.

No network was trained. Q, J_def, canonical eta, and eta norm were not used to construct centers or radii.

## Frozen population and geometry

The exact states were:

1. `N_r104_s125`
2. `ZR_P_r052_m080_s95400004_p253`
3. `S_r043_p02`
4. `N_r076_s119`
5. `R_D2_s95101009_p112`
6. `N_r004_m120`

The domain was the exact archived convex hull

`E_bridge = conv({(0,0,0)} union ([0.5,1.25] x [-0.5,0.5] x [0,0.75]))`.

Geometry used

`eta_tilde = (eta - [0.875,0,0.375]) / [0.75,1.0,0.75]`.

The normalized bridge volume was `1.2222222222222223`. The machine-readable domain contains nine vertices and 14 triangulated half-space rows with convention `n dot eta_tilde + b <= 0`.

## Cache and common cloud

The preflight inventory found 2,200 compatible exact cached tuples. During the staged run, 12,736 exact tuple requests were served from the combined pre-existing and accumulating run cache. The run generated 12,688 unique new continuations; exact duplicates were not rerun.

All states were resolved using the first 32 accepted points of the common frozen E_bridge Sobol sequence plus explicit zero and compatible archived evidence. No state required the predeclared extension to points 33--64.

## Robust centers and balls

| State | center c0 (physical eta) | center Q64 | margin | r_domain | r_success | r_ball | V_ball / V_Ebridge | anisotropy ratio | CV | category | shell 8/8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| N_r104_s125 | (0.898438, -0.156250, 0.445312) | 64/64 | 0.280816 | 0.343750 | 0.018750 | 0.015938 | 0.0014% | 32.820 | 0.782 | strong | 8/8 |
| ZR_P_r052_m080_s95400004_p253 | (0.625000, 0, 0.375000) | 64/64 | 0.416025 | 0.416025 | 0.424065 | 0.353621 | 15.1549% | 2.186 | 0.251 | moderate | 8/8 |
| S_r043_p02 | (0.625000, 0, 0.375000) | 64/64 | 0.404376 | 0.416025 | 0.275000 | 0.233750 | 4.3772% | 3.229 | 0.339 | strong | 8/8 |
| N_r076_s119 | (0.625000, 0, 0.375000) | 64/64 | 0.404376 | 0.416025 | 0.293750 | 0.249688 | 5.3349% | 3.156 | 0.365 | strong | 7/8 |
| R_D2_s95101009_p112 | (0.625000, 0, 0.375000) | 64/64 | 0.416025 | 0.416025 | 0.312500 | 0.265625 | 6.4231% | 2.967 | 0.309 | strong | 8/8 |
| N_r004_m120 | (0.625000, 0, 0.375000) | 64/64 | 0.416025 | 0.416025 | 0.424065 | 0.353621 | 15.1549% | 2.186 | 0.251 | moderate | 8/8 |

All final balls passed the half-space containment check. Radius statistics were: mean `0.245374`, median `0.257656`, minimum `0.015938`, maximum `0.353621`. Volume-fraction statistics were: mean `7.7411%`, median `5.8790%`, minimum `0.00139%`, maximum `15.1549%`.

The very small `N_r104_s125` ball was data-driven, not a domain artifact: the `+eta2` ray failed screening at normalized radius 0.025 (2/8), while its inward promoted point at radius 0.01875 was 64/64. After the fixed shrinkage its final radius was 0.0159375.

## Independent validation

- Inside screening: 72/72 eta were 8/8.
- Predeclared inside robust promotion: 18/18 eta were 64/64.
- Additional promotions caused by inside screening failure: 0.
- Confirmed false inclusions: 0/18 robustly promoted points (0%).
- Outside shell: 47/48 eta were 8/8; the remaining point was 7/8.

The inside results support empirical reliability of the shrunk balls. The nearly universal outside-shell success shows that the balls leave substantial successful space unused and should not be interpreted as close reconstructions of the full success sets.

## Anisotropy and sphere adequacy

No state was mildly anisotropic, two were moderately anisotropic, and four were strongly anisotropic under the frozen bins. A sphere therefore gives a reliable inner set, but it is not an efficient geometric description. The evidence is not a failure of set-valued feasibility geometry: five states retained nontrivial spherical volumes, and every independently promoted interior point was robust. It is evidence that an ellipsoid comparator is warranted before scaling spherical labels.

## ZERO and ACTIVE states

The exact zero evidence classified three states as ZERO-sufficient (`N_r104_s125`, `S_r043_p02`, `R_D2_s95101009_p112`) and three as ACTIVE-required. The max-margin procedure did not force zero. All six centers were displaced from zero and zero lay outside every final ball. For ZERO-sufficient states, normalized center-to-zero distances were 0.972--1.346 and signed zero-to-ball distances were 0.706--1.330. Thus center feasibility is strong, but center deployment would not be a minimum-deformation policy; that question was intentionally outside this audit.

## Q-exploitation cases

Neither archived false-high-Q state (`S_r016_p16` or `S_r000_p00`, including the Q_hat approximately 0.915 / true Q64=0 case) matched any of the six exact state/h/Flow conditionings. The inside/outside check is therefore `NOT_APPLICABLE`; no claim is made that these balls would reject those cases.

## Answers

1. After removing the historical domain artifact, useful conservative 3-D inner balls exist for 5/6 states, while a valid but nearly degenerate ball exists for the sixth.
2. A free robust-margin center is a plausible feasibility target: centers were resolved for 6/6 states and every center was 64/64.
3. Sphere geometry is reliable but materially conservative. Strong anisotropy in 4/6 and outside-shell success of 47/48 make an ellipsoid comparator scientifically warranted.
4. Set-valued supervision is supported in principle, but it is premature to scale a spherical oracle directly. The smallest next experiment is an axis-aligned conservative ellipsoid comparator on these same six states, using the existing axis rays and a small independent interior validation. If that passes, scale the selected geometry and compare center-MSE Direct-eta with basin margin-set loss.

## Runtime and integrity

- New continuations: 12,688
- Physical steps: 3,498,278
- Completed batched-run wall time: 4,719.44 s (78.66 min)
- Maximum GPU shards: 1
- Observed GPU memory: 602 MiB
- CPU allocation: 8 threads
- RAM allocation: 24 GiB; post-completion peak accounting unavailable
- Outcomes: 11,845 success, 615 safe deadlock, 228 timeout
- Collision, invalid action, projection failure, NaN/Inf, numerical failure: 0
- Feature replay maximum error: 0
- Budget: compliant (`12,688 < 15,000`; `3,498,278 < 5,000,000`)

No H, Q, J, G, classifier, gate, flow, or mixture model was trained, and no follow-up experiment was started.
