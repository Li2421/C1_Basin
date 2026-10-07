# C1 true full-horizon Q geometry: frozen system and policy semantics

Date: 2026-09-22.  This directory is a diagnostic study of the true mapping

\[
(z_t,\phi)\longmapsto Q_D(z_t;\phi)
\]

only.  It does not define an `R_risk`, certificate, learned critic, or trained
corrector.

## Frozen episode

The plant, monitor, collision tests, event precedence, and projections are
unchanged from `single_integrator/environment.py` and
`single_integrator/cbf.py`.  Two planar single integrators use `dt=0.05 s`,
four joint velocity components, per-agent speed bound `0.5 m/s`, and a maximum
of 850 transitions.  The exact augmented pre-action state is

```text
positions[2,2]
last executed velocity[2,2]
absolute physical step
the ordered last 41 two-agent goal-error samples
candidate_since (or inactive)
first-event terminal latch
```

The error history and candidate latch are retained when a cached intermediate
state is restored.  Resetting the monitor at a checkpoint is forbidden.

Strict deadlock is the existing `window_progress_v2` event: after the 2.0 s
history is available, success must be false, maximum absolute two-agent window
progress must be strictly below `0.01 m`, and maximum current agent speed must
be strictly below `0.025 m/s`; the candidate must remain continuously true for
5.0 s.  The old stalled-timeout outcome is not used.  Post-transition event
precedence is collision, success, strict deadlock, timeout.  Collision and
timeout are never counted as deadlock.

## Frozen stochastic controller

At every physical step a fresh 4D standard Gaussian is supplied to the frozen
Flow-BC checkpoint.  Flow integrates ten learned vector-field Euler steps,
de-normalizes and component-clips its output, followed by the per-agent speed
bound.  The exact closed loop is

\[
u_{Flow}=\pi_{Flow}(z,\xi),\quad
u_{safe}=\Pi_{\cal U}(z)u_{Flow},\quad
g=G_\phi(z,u_{safe}),\quad
w=u_{safe}+g,\quad
u_{exec}=\Pi_{\cal U}(z)w.
\]

Both projections are the hard Euclidean projection onto all 17 CBF
halfspaces and the two speed balls.  The same `phi` is used through the entire
remaining episode, while Flow, the first projection, `G_phi`, and the second
projection are recomputed every physical step.  Every reported probability
integrates the Flow randomness.

## `G_phi` interface

The repository's final-network *architecture* exists as
`ResidualCorrection`, but no usable trained final parameter set is present.
That interface is deterministic: normalized flattened observation dimension
20 plus current first-projected action dimension 4 and a fixed scalar zero go
to a 4D joint-velocity residual.  There is no mean/gate/scale output and no
residual sampling distribution.

For geometry diagnosis only, this study reuses the already frozen small
state-dependent family (it is not trained or refit):

\[
G_\phi(o,u_{safe})=
\phi_g b_{goal}(o)+\phi_su_{safe}+\phi_rb_{rel}(o),
\qquad \phi\in\mathbb R^3.
\]

For each agent, `b_goal` is the current goal displacement bounded to the
radius-0.5 Euclidean ball, and `b_rel` is current self-minus-other displacement
bounded to the same ball.  The three gains are shared across agents, so the
basis is exchange-equivariant and contains no agent priority, passing side, or
yield/pass mode.  Output dimension is four and deterministic.  It is
recomputed from current state and `u_safe` at every step.

The symmetric probe grid, states, seeds, uncertainty rules, and stopping gates
were fixed in `predeclared_protocol.json` before any new geometry continuation
was run.
