# OrthoFlow3 zero/ACTIVE gap connectivity audit v1

## Decision

**HISTORICAL_DOMAIN_GAP_IS_ARTIFACT.** The historical `eta_1 >= 0.5` bound is an oracle-search bound, not a controller or safety constraint. Every finite bridge eta was accepted without clipping or reinterpretation by the unchanged OrthoFlow3 and second-projection stack.

## Frozen population and endpoints

The first 12 states in the frozen 32-state migration order were used without outcome stratification: R_D2_s95101013_p215, P_r198_m080, Q_pair226_m020, N_r042_s109, N_r071_s519, S_r017_p17, N_r032_s334, N_r061_s347, S_r050_p09, N_r131_s119, R_D1_s95106007_p70, ZR_P_r052_m080_s95400004_p253. Post-selection, 8 were ZERO-sufficient and 4 ACTIVE-required. Each state had at least one cached nonzero B63 endpoint; 3 states had a second frozen far endpoint.

## Radial connectivity

Among ZERO-sufficient states, 8 / 8 frozen endpoint paths were `RADIAL_CONNECTED`: both endpoints were B63, all 11 alpha probes were 8/8, and no confirmed interior failure interval existed. There were 0 confirmed failure-gap paths and 0 unresolved paths. Consequently the conditional 64-point bridge graph was not triggered.

The zero-neighborhood probe produced 284/288 8/8 points (98.61%). The four non-8/8 representatives were promoted; 0/4 were B63. Across all frozen queries, 2 distinct state/eta pairs inside `E_bridge` but outside the historical domain were directly B63-confirmed, and 378 such pairs screened 8/8.

For ACTIVE-required states, 1/7 endpoint paths had their first directly confirmed B63 point at physical `eta_1 < 0.5`; the detailed transition widths and nonmonotone cases are in `radial_path_classification.csv`. Thus the old 0.5 threshold does not define a universal physical onset of feasibility.

## Topology interpretation

State classifications: {'CONTINUOUS_ZERO_TO_ACTIVE_SUPPORTED': 8, 'ACTIVE_REQUIRED_FEASIBILITY_BOUNDARY': 4}. The data support a single continuous bridge domain for future basin work. They do not support empirically disconnected zero and active components at this resolution. Zero is special in the historical search parameterization, but is not empirically isolated in closed-loop success geometry.

This finite audit does not prove global connectedness. `RADIAL_CONNECTED` means B63 endpoints plus uniformly successful frozen screening probes and no confirmed failure interval under the predeclared promotion rule; it does not relabel 8-seed points as B63.

## Validity and safety

All 408 distinct intermediate state/eta queries were implementation-valid. No eta was clipped/reinterpreted. New rollout integrity: {'execution_errors': 0, 'collisions': 0, 'nan_or_inf_reported': 0}.

## Implication

Future basin learning should use one continuous `E_bridge = conv({0} union ACTIVE_BOX)` domain, not `{eta=0} union ACTIVE_BOX`. A sphere/ellipsoid inner approximation remains worth testing after correcting that domain issue. The smallest next experiment is to repeat the 6-state conservative inner-geometry pilot on `E_bridge`; no model is trained here.

## Runtime

Reused continuations: 504. New continuations: 5360 (3456 screening + 1904 promotion). Physical steps: 1396671. Rollout critical-path wall time: 7.52 minutes. Maximum: 4 GPU shards, 594 MiB/shard observed, 8 CPU threads, 56 GiB scheduled RAM.
