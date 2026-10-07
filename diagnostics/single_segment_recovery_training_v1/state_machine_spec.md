# Frozen single-segment controller semantics

The only permitted mode sequence is:

`SAFETY_BEFORE -> RECOVERY -> SAFETY_AFTER`.

The controller memory consists of `mode`, `recovery_used`, `entry_step`,
`exit_step`, `recovery_transitions`, and (for System ETA only) the immutable
`eta_latched` value.  No episode can re-enter recovery after reaching
`SAFETY_AFTER`.

At each nonterminal step the Flow realization, first hard-safety projection,
and startup-aware feature are computed once and shared by the decision and
action path.  In `SAFETY_BEFORE`, ENTER changes mode and executes the first
recovery action on the same physical step.  In `RECOVERY`, the exit head is
not eligible until at least one recovery transition has executed.  EXIT
changes mode and executes Safety on that same step.  `SAFETY_AFTER` executes
Safety to the original absolute horizon and cannot query either head again.

System G re-queries the frozen Direct-g checkpoint on every active recovery
step.  System ETA predicts eta exactly once on entry, applies the frozen
de-normalization/clipping convention, and retains the same eta while the
state-dependent basis fields are recomputed each active step.  Both systems
apply the second authoritative projection after each recovery correction.

Physical event priority and monitor history belong to the environment and are
never reset by a mode transition.  An absorbing terminal event is never
resumed.  Startup left padding exists only in the network feature view; it
does not alter real monitor history.

The executable specification is `state_machine.py`; the frozen integrity
tests are in `test_state_machine.py`.
