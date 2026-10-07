# Semantic controller state-machine specification

## Memory

The deterministic controller memory is:

```
mode             in {NORMAL, RECOVERY, SAFETY_AFTER}
eta_latched      in R^3 or null
recovery_used    boolean
entry_step       integer or null
exit_step        integer or null
recovery_steps   nonnegative integer
local_events     nonnegative integer
```

The only legal mode path is:

```
NORMAL -> RECOVERY -> SAFETY_AFTER
```

`LOCAL` does not change mode.  Recovery can be entered at most once.  Once
`SAFETY_AFTER` is reached, the controller can never re-enter recovery.

## Step semantics

Before a decision, the environment applies frozen terminal-event priority.  A
terminal state is absorbing and must not request another control action.

In `NORMAL`, compute exactly one current Flow realization, its first Safety
projection, and the 214-D startup-aware feature.  The mode head selects:

- `SAFETY`: execute `u_safe`.
- `LOCAL`: query frozen Direct-g exactly once; execute
  `Pi_safe(u_safe + g_local)`; stay in `NORMAL`.
- `ENTER_RECOVERY`: query frozen eta exactly once, apply its frozen
  de-normalization/clipping, latch it, and execute
  `Pi_safe(u_safe + C(z, eta_latched))` on this same step.

In `RECOVERY`, eta is never re-predicted.  Compute the 217-D exit input
`[h_t, eta_latched]`.  Select:

- `CONTINUE`: execute the current structured correction after the second
  projection.
- `EXIT`: switch to `SAFETY_AFTER` and execute `u_safe` on this same step.

In `SAFETY_AFTER`, execute Safety forever.  No learned mode decision is made.

## Invariants

- No periodic phase, cadence, fixed burst length, cooldown, or H/L feature.
- Entry executes at least one recovery transition, so zero-duration recovery
  is impossible.
- Eta has three finite coordinates and is immutable after latching.
- Every correction uses the frozen second hard projection.
- Direct-g is queried once per selected local event and never held.
- Flow is sampled once per physical step and shared by that step's decision
  and action construction.
- Mode transitions do not reset absolute time, RNG, history, monitor memory,
  timers, or latches.
- Startup padding affects only the network feature; it never mutates the real
  monitor history.

The executable pure transition skeleton is `stage0_integrity/semantic_state_machine.py`.
It intentionally contains no environment runner, learned head, training code,
or timing schedule.
