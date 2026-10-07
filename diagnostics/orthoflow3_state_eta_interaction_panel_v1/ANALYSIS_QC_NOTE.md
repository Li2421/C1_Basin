# Analysis QC note

The frozen state cohort, 24-eta panel, model scores, and rollout outcomes were not changed after evaluation began. During analysis, it became clear that 76 of 18,432 seed slots remained numerical after the initial execution plus three identical retries. The first analysis implementation incorrectly required a whole 24-eta state row to be complete for rank similarity and emitted NaN regrets; this was caught before accepting the report.

The final analysis therefore follows the task's numerical rule:

- state-pair rank similarity uses only eta columns with exact Q16 on both states;
- ranking reversals are counted only when all four Q16 cells are exact;
- per-state observed best/top-3 are reported with a flag if an unresolved eta's upper bound could change the best;
- the global eta is selected by mean Q lower bound over all 24 frozen candidates, and its mean-Q/B15 uncertainty is reported as an interval;
- oracle and regret estimates retain lower/upper bounds from unresolved cells;
- additive decomposition uses all exact observed Q cells via descriptive least-squares additive fit and is explicitly not treated as causal evidence.

No model, state, eta, rollout, or threshold was changed. This is an analysis correction necessitated by the pre-registered numerical-failure semantics, not a new experiment or outcome-driven model decision.
