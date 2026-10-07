# One-step intervention-necessity audit

## Result

**LONG_HORIZON_LABEL_TOO_CONSERVATIVE_FOR_STEP_GATE**

This audit compares two matched branches. Branch I activates the state's pre-existing minimum-deformation `eta_best` at the current step; Branch N executes exactly `u_safe` for the current step. From `t+1`, both branches use the same frozen `eta_best` DiagnosticCorrector, FlowBC randomness, hard projections, monitor, and terminal semantics. Future intervention is therefore allowed in both branches and only the current action differs.

This is the closest existing oracle-consistent continuation, not a receding-horizon oracle re-query. The validated oracle defines one eta at the audited state and keeps it fixed for the continuation; a new eta search at every reached next state does not exist in the current oracle pipeline.

## Fixed states

- Difficult oracle-stable boundary states: **13** (6 long-label zero, 7 long-label one).
- Pre-registered easy controls: **8**.
- Paired continuations per state: **64–256**.
- Difficult one-step classifications: {'ONE_STEP_EFFECT_AMBIGUOUS': 1, 'NOW_INTERVENTION_NOT_NECESSARY': 12}.
- Difficult `y_long=1` states for which delaying intervention one physical step produced no supported success loss: **6/7**.
- Difficult `y_long=1` states classified current-step necessary: **0/7**; ambiguous: **1/7**.
- Reverse mismatches (`y_long=0` but current intervention necessary): **0**.
- Pre-registered easy `y_long=1` controls with no observed one-step success loss: **4/4**.

Six of the seven difficult `y_long=1` states had `Q_N=Q_I=1` over 64 matched seeds. The sole adaptive state, `RBV_Q_pair228_m080_s95401001_p018`, reached 256 pairs: `Q_N=253/256=0.98828125`, `Q_I=256/256=1`, and paired `Delta_Q=0.01171875` with bootstrap 95% interval `[0, 0.02734375]`; it therefore remains ambiguous rather than being declared current-step critical.

## Statistical rule

`Delta_Q = Q_I - Q_N` uses matched continuation seeds. `NOW_INTERVENTION_NECESSARY` requires the paired bootstrap 95% interval to lie above zero. `NOW_INTERVENTION_NOT_NECESSARY` is assigned without an arbitrary effect threshold only when every observed matched success indicator has exactly zero effect (or intervention is significantly harmful). Other cases remain `ONE_STEP_EFFECT_AMBIGUOUS` and were adaptively extended when applicable.

Future minimum deformation is reported descriptively in `qn_qi_comparison.csv`; it is not used to redefine success or the one-step label.

The preregistered one-step contrast (`delay=0` versus `delay=1`, i.e. 0.05 s) was completed. Optional delays 2 and 4 were not run: adaptive sampling of the sole ambiguous one-step state was prioritized, and the primary temporal-alignment question was already resolved without changing the downstream policy.

## Integrity

The pre-bulk smoke reproduced an existing oracle first action to `7.81e-17`, exactly reproduced the one-step augmented state, verified Branch N's executed action equals `u_safe`, and verified the eta-zero I/N branches are identical. No learned gate/G_phi was trained or evaluated closed loop.
