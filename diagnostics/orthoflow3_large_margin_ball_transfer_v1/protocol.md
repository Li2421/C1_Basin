# OrthoFlow3 large-margin ball transfer pilot v1

This model-free audit freezes six validated continuous-domain source balls, with eligibility thresholds `r_anchor>=0.20`, `r_transfer>=0.20`, and retained radius `>=0.75`. Five anchors are margin eligible. Exact same-trajectory augmented states are replayed at offsets -4, -1, +1, +4 where available; no h perturbation is used. The transferred center is fixed to the source center. Candidate shrink factors are 1.00, 0.85, 0.75 in that order.

Each neighbor center is B63-tested with 64 Q-v2-conditioned future streams. Each candidate ball has 12 independent frozen Sobol-to-ball validation points; four outcome-blind points at radial fractions .25, .60, .90, .97 receive 64 seeds, all others receive eight screening seeds and every screening failure is promoted to 64. The largest passing margin-eligible radius is retained. Accepted balls receive six 1.10-radius axis shell probes with eight seeds. No network is trained.
