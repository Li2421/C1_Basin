# CL-FHCB frozen-system reconstruction

Date reconstructed: 2026-09-22 (Asia/Singapore).  This document is the
prerequisite reconstruction for the new Closed-Loop Full-Horizon Continuation
Barrier.  It describes the active simulator and controller interfaces; it does
not adopt any of the old cone, one-step, `R_CERT`, or stalled-timeout methods.

## Frozen plant and episode

There are two planar single-integrator agents.  The physical bookkeeping state
at physical index (k) is

\[
  x_k=(p_{0,k},p_{1,k},v_{0,k}^{\rm last},v_{1,k}^{\rm last})\in\mathbb R^8,
  \qquad p_{i,k},v_{i,k}^{\rm last}\in\mathbb R^2 .
\]

The last velocity is not inertial--the transition is

\[
  p_{k+1}=p_k+0.05u_k,\qquad v_{k+1}^{\rm last}=u_k--
\]

but it is part of the physical controller state because Flow and the current
residual interface observe it.  Joint actions have shape `[2,2]` (four scalar
components), each agent has the Euclidean speed bound (\|u_i\|_2\le0.5),
and the deadline is 850 physical transitions (42.5 s).  Goals and finite wall
segments are fixed episode constants.  The frozen short-scene configuration is
the `evaluation_environment` in the approved seed-0 Flow checkpoint and has
`corridor_half_length=1.3`, `corridor_width=0.4`, `bay_top=0.68`, agent radius
0.16, goal tolerance 0.08, and the remaining constants serialized in the
freeze specification.

The observation is `[2,10]`.  For agent (i), with (j=1-i), its row is

\[
  [p_i,\ v_i^{\rm last},\ q_i-p_i,\ p_j-p_i,\
    v_j^{\rm last}-v_i^{\rm last}].
\]

Flow-BC is stochastic at every physical step.  Given the current observation,
it samples a fresh four-dimensional standard Gaussian, integrates the learned
conditional vector field by ten Euler steps, de-normalizes, clips each raw
component to `[-1,1]`, and then applies the per-agent 0.5 speed bound.  The
checkpoint is the approved seed-0 25k checkpoint with SHA256
`8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`.
Thus all probabilities below include the independent per-physical-step Flow
Gaussian draws.  The plant itself is deterministic conditional on the executed
action.

## Two hard projections and current residual interface

The hard map (\Pi_{\mathcal U}(p)) is the Euclidean projection of a joint
velocity onto all 17 CBF halfspaces (one pairwise row and eight finite-wall rows
per agent) and both per-agent Euclidean speed balls.  The frozen CBF constants
are `gamma=1`, `gamma_wall=1`, and `separation_buffer=1e-4`; projection solver
tolerances do not change the mathematical set.

The exact control order reconstructed from the active C1 code is

\[
\begin{aligned}
 r_k &\sim N(0,I_4),\\
 u_{\rm Flow,k} &= \operatorname{FlowBC}(o(z_k),r_k),\\
 u_{\rm safe,k} &= \Pi_{\mathcal U}(p_k)(u_{\rm Flow,k}),\\
 g_k &= G_\phi(o(z_k),u_{\rm safe,k}),\\
 w_k &= u_{\rm safe,k}+g_k,\\
 u_{\rm exec,k} &= \Pi_{\mathcal U}(p_k)(w_k),\\
 z_{k+1} &= F_{\rm monitor}(z_k,u_{\rm exec,k}).
\end{aligned}
\]

Both projections are mandatory.  The current `ResidualCorrection` interface is
deterministic: it consumes the flattened 20-dimensional current observation,
the 4-dimensional current first-projected action, a fixed scalar zero, and no
environment-conditioning dimensions, then emits exactly four deterministic
joint-velocity residual components.  It does **not** emit a mean, gate/logit,
scale, or distribution.  Deployment randomness therefore comes from Flow,
not from sampling `G_phi`.  No trained final residual parameters are present in
this checkout, so CL-FHCB construction will use a small deterministic,
state-dependent diagnostic family with this same 4D residual contract.  It
will not replace deployment by a fixed action vector.

The same (\phi) is held throughout an episode, while the observation, first
projection, and (G_\phi) output are recomputed at every physical timestep.
There is no intervention cutoff at the risk-observation horizon (K).

## Exact strict-deadlock monitor and augmented Markov state

At the post-action state (z_{k+1}), let (e_{i,k+1}=\|q_i-p_{i,k+1}\|_2).
After the 2.0 s (40-transition) history window is available, the interpolated
window progress is the historical goal error minus the new goal error.  A
candidate is true exactly when:

* success is false;
* `max(abs(window_progress)) < 0.01` m; and
* `max(agent speed) < 0.5 * 0.05 = 0.025` m/s.

When the candidate first becomes true, `candidate_since` is set to that
post-action step and its timer is zero.  Strict deadlock occurs after the
candidate remains continuously true for 5.0 s.  At `dt=0.05`, this is 101
candidate samples spanning 100 intervals.  Any false candidate resets
`candidate_since` and the timer.  The old separately defined stalled-timeout
classification is not part of DEADLOCK here.

An exact pre-action augmented Markov state is

\[
 z_k=(p_k,v_k^{\rm last},k,E_k,c_k,\ell_k),
\]

where (E_k) is the ordered/ring-buffer representation of the goal-error
history needed for the next 2.0 s interpolation (41 two-agent samples suffice
for the frozen integer 40-step window), (c_k\) is `candidate_since` (or the
equivalent candidate age plus an inactive flag), and (\ell_k) is the terminal
first-event latch.  Only these monitor fields affect future event
probabilities.  Reporting-only `ever_candidate_deadlock`, `max_stuck_timer`,
rewards, and collision counters are excluded.  The fixed goals, walls, plant
constants, CBF constants, Flow weights, and (\phi) are policy/environment
parameters rather than per-state variables.

## Events, precedence, and horizon

Each action interval uses swept segment geometry as well as endpoint/outside
tests.  The four CL-FHCB outcomes are:

1. `collision`: wall/outside or inter-agent collision;
2. `success`: both goal errors at most 0.08 m;
3. `strict_deadlock`: the monitor above triggers;
4. `timeout`: no earlier-priority event by the deadline.

The authoritative post-transition precedence is collision, then success, then
strict deadlock, then timeout.  Consequently a collision simultaneous with
success/deadlock is collision; success precedes deadlock; and a deadlock on the
last allowed transition precedes timeout.  For stopping-time notation, (H)
denotes the timeout boundary *after* the event checks of transition 850.  This
lexicographic event ordering makes a deadlock found on the last transition
strictly earlier than the timeout boundary, consistent with

\[
 Q_D(z_t;\phi)=\Pr_\phi[\tau_D<\min(\tau_S,H)\mid z_t].
\]

Collision is an absorbing, separately reported outcome with deadlock value
zero.  Success has deadlock value zero.  Timeout without earlier strict
deadlock has deadlock value zero.  All continuation probabilities use the same
closed-loop (G_\phi) through whichever first event terminates the episode.

## Source basis for reconstruction

The authoritative active files are `single_integrator/environment.py`
(plant, monitor, collision, outcome precedence), `single_integrator/cbf.py`
(hard projection), `flowbc/giveway_flowbc_agent.py` (Flow distribution),
`single_integrator/c1/differentiable_rollout.py` (double-projection residual
placement), `single_integrator/c1/models/residual.py` (deterministic 4D
interface), and `single_integrator/outcomes.py` (first-event taxonomy).  Exact
source/config hashes are recorded in `CL_FHCB_FROZEN_SPEC.md` after the isolated
construction implementation is finalized.
