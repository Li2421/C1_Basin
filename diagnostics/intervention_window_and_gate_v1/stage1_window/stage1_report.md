# Stage 1 intervention-window audit

- Frozen states: **277** across **117** source groups.
- Coarse delays: `[0, 4, 8, 16, 32, 64]`; adaptive extension: `128`.
- Transition inventory: `{'BOUNDED': 91, 'RIGHT_CENSORED_OR_UNRESOLVED': 186}`.
- Frozen Stage-2 candidate horizons: `[4, 8, 16]`.
- Candidate selection was completed before any gate training and used no model/test performance.

`y_H=1` is available only where paired `Q0-QH` has a bootstrap 95% CI
strictly above zero. Exact no-effect or supported improvement is resolved
negative; every nonzero effect whose CI crosses zero remains `H_AMBIGUOUS`.

The downstream controller remains the previously validated frozen-`eta_best`
approximation, not arbitrary-state receding oracle re-query. Q and J_def are
reported separately and are never scalarized.
