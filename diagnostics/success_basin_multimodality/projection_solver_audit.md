# SBMA projection-solver audit

## Verdict

All 754 indexed UNKNOWN trajectories from the preceding SBGA are classified as **NUMERICAL_SOLVER_FAILURE**. There is no observed TRUE_PROJECTION_INFEASIBILITY and no evidence that an UNKNOWN cell was a physical outcome hole.

The projection problem was not changed:

`min_u 0.5 ||u-target||^2`, subject to the same 17 linear CBF/wall inequalities `A u >= b` and the same two per-agent Euclidean speed balls `||u_i||_2 <= 0.5`.

No slack, clipping, reduced speed cap, safety relaxation, or changed physical constraint was introduced.

## Previous 754 UNKNOWN records

The archived failures comprise 549 `speed_cut_limit` and 205 `qp_failed` events from the prior SLSQP plus outer-support-plane implementation. Every stored executed prefix was replayed exactly; maximum position error was zero. At each terminal pre-failure state, the equivalent exact SOCP returned a certified feasible solution. Across all 754 reconstructions, the minimum linear residual was `-1.98e-14` and maximum per-agent speed was `0.5000000000029`.

For 518 records the old exception was reproduced at the reconstructed second projection call. For 236, the failed call itself could not be reproduced: the archived trace omitted the rejected target/action, and SLSQP did not repeat its old numerical exception. Those cases remain transparent in `projection_solver_audit.json` with `failed_projection=null`; their exact SOCPs at the same terminal pre-failure state were nevertheless feasible. Thus `all_legacy_failures_reproduced=false` is not hidden or converted into an episode label.

Independent pre-existing backend regression covered 43,952 identical projection inputs. The exact SOCP had zero failures, a maximum action difference of `1.88e-6` from accepted old outputs, and no difference above the predeclared `5e-6` numerical-equivalence tolerance.

## SBMA numerical events

The primary exact backend rejected four unique calls under its strict external certificate during the final scientific sample. The invalid attempts were retained with `outcome=null`. Their same `(state, eta, seed)` continuations were rerun from the original starting state using one of two numerically independent, mathematically identical fallbacks:

- tighter Clarabel SOCP settings, with the same cones and objective;
- SLSQP using the exact nonlinear Euclidean-ball constraints, followed by active-constraint KKT certification.

One accepted action used the tighter SOCP and three used nonlinear polishing. All four stored actions were re-solved independently and their original backend certificates reproduced with zero action difference. Worst certified values over these actions were well within the frozen external acceptance thresholds: minimum linear residual at least `-6.1e-18`, maximum speed excess at most `3.4e-16`, conic/SLSQP stationarity at most `1.03e-9`, and complementarity at most `1.78e-8` (threshold `2e-6`).

Five invalid raw attempts remain preserved: two Phase-A failures, one unsuccessful first repair of one of them, and two interpolation failures. They are never called collision, success, strict deadlock, or timeout. Final tables substitute only a successful same-input full rerun for each invalid scientific trial. Four accepted fallback actions occurred over the 6,144 final outcome-bearing continuations.

## Conclusion

The prior topological ambiguity caused by projection UNKNOWN points is resolved for this experiment. Hard projection remains active twice per physical step and preserves the exact mathematical feasible set. Statistical UNKNOWN cells in the final map reflect finite outcome uncertainty around the fixed confidence threshold, not unresolved solver exceptions.

Machine-readable details: `projection_solver_audit.json` and `verification.json`.
