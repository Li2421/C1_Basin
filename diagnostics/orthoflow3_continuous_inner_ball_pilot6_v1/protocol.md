# OrthoFlow3 continuous-domain inner-ball pilot6 v1

The state population is the exact six-state controlled rerun specified by the protocol. Geometry uses the archived normalization `eta_tilde=(eta-[0.875,0,0.375])/[0.75,1,0.75]` and the exact convex hull of zero plus all eight ACTIVE-box vertices. All candidate/ray/inside/shell coordinates are frozen algorithmically before their respective outcomes.

- Common cloud: one deterministic unscrambled Sobol rejection sequence over the E_bridge bounding box; first 32 accepted points, with the next 32 as the sole predeclared extension.
- Center: maximize `min(d_fail,d_domain)` over 8/8 or exact B63 candidates; no J/Q/eta-norm criterion.
- Rays: the archived 18 directions, radii (0.025, 0.05, 0.1, 0.2, 0.35, 0.5), 0.95 domain-boundary probe, and at most three fixed bisections.
- Robustness: B63 is >=63/64; 8/8 is screening only. Eight smallest directions are promoted, with inward fallback and at most ten directions.
- Ball: `r_ball=.85*min(r_success,r_domain)` and must be wholly contained in E_bridge.
- Validation: 12 independent Sobol-to-ball points, three predeclared robust promotions at approximately .25/.60/.90 radius, all screening failures promoted, plus eight independent 1.15-radius shell points.
- No neural model is trained; learned Q and J_def are not used for construction.
