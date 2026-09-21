# C1 research specification

> Current experiment index (2026-09-17): see [C1_CURRENT_SPEC.md](C1_CURRENT_SPEC.md).
> The dated candidates below are preserved as historical specifications. Their
> risk definitions, relative epsilon calibration and dataset protocols must not
> be substituted for the currently frozen deadlock-union experiment.

## Dataset-initial-state experiment protocol (2026-09-13)

The explicit `--start-distribution dataset` experiment samples original
Flow-BC train pair initial states (pairs 0–199) uniformly with replacement.
The two expert modes share each pair's initial state and are counted once.
There is no synthetic risk pool in this mode. Flow-BC remains frozen; C1 uses
new on-policy noise and residual rollouts, not expert-action regression.

Before training, calibrate epsilon as epsilon-fraction times zero-residual
full-episode mean risk over all 200 train starts with frozen noise, including
zero-risk episodes. Use the original 25 val-pair starts (200–224), with separate
frozen noise, for checkpoint selection. The trainer checks exact membership
and multiplicity. Calibration and training thus estimate the same empirical
initial-state distribution; their noise realizations differ.

This selection set is not the untouched final test. Final comparisons retain
the original wide 200-start test suite, both frozen model seeds, Flow seed 42,
850 steps, and the baseline six-class offline outcome protocol. Extra C1
interaction/update budgets must be reported. Generalization is an empirical
question, not guaranteed by reusing the Flow-BC training initial states.

The controller, risk definition, primal-dual objective and safety constraints
below are unchanged. Legacy mixture experiments must remain identifiable.

## Active candidate: R_risk_v2, full first-event episodes

This is a versioned research correction (`balanced_terminal_v3`), not a
validated liveness theorem or a geometric deadlock probability. The frozen
sampler, two hard projections, physical residual and primal-dual objective
are unchanged. Only the modular risk and trajectory support are revised.

Training and validation now include exactly the actions through the first
authoritative environment event (success, collision, detected deadlock or
timeout). The event callback calls `GiveWayEnv.step` with reconstructed
window/timer state; it does not invent another detector. Discrete event
decisions are detached, while physical states and controls retain BPTT
derivatives within the selected event branch. No claim of an unbiased
gradient through jumps of the terminal event distribution is made.
Inactive batch lanes are masked and their terminal states frozen. Both
J_def and risk use per-episode normalization over actually executed actions.
The v2 training entry requires horizon >= the original 850-step time limit;
short prefixes cannot be reported as complete training episodes.

The geometric risk c_t is unchanged. A complete progress window at action
interval t uses V(x_{t+1-W}) - V(x_{t+1}), so the final action contributes.
Before a complete window is available, stall risk is unavailable/zero but
geometric exposure is retained. s_t is the v1_1 smooth unfinished-task stall
indicator. E is the original power-mean-plus-mean of c_t+s_t-c_t*s_t, using
the actual episode mask. Stable normalization avoids tiny-risk underflow.

For terminal failure, let e = max_a(max(d_a-goal_tolerance,0)) using smooth
distance, and define q = f+(1-f)*e/(e+L). For success q=0. Let z be the last
complete-window stall risk for failures, and zero for success or an
unavailable window. Defaults are f=0.5, L=0.1 m, w=0.75, g=0.75:

    terminal = g*q + (1-g)*z
    R_risk = (1-w)*E + w*terminal

Collisions are assigned risk 1 and remain separate safety failures. The
weighted combination preserves goal derivatives when geometric risk
saturates; nested OR compositions were rejected because they can erase
those derivatives. Configuration requires w*g*f > 1-w, so every terminal
failure ranks above every success by construction (default failure lower
bound 0.28125, success upper bound 0.25). This separation is part of the
definition and is NOT evidence that the risk predicts unseen failures.
The failure floor creates a discrete outcome jump; inside a failure branch,
the remaining-distance and stall terms retain learning signals.

Safety risk comparisons use the zero-residual C1 candidate u_safe, while
C1 comparisons use its actual candidate u_safe+G. Old nominal-input Safety
audits are historical diagnostics and must be labelled separately.

Historical v0_1/v1/v1_1 training remains explicitly selectable and uses its
old segment protocol. Values across revisions are not interchangeable.

## Implementation correction: R_risk_v1_1

The historical provisional `R_risk_v1_1` retains the
v1 cone/activity, normalized joint task potential, physical progress window,
union of cone and stall risks, and power-mean aggregation. Only the smooth
unfinished-task indicator changes to follow the environment's requirement
that **both agents** must be within their individual goal tolerances:

`m_a = sigmoid((d_a - goal_tolerance)/task_mask_temperature)` and
`unfinished = m_0 + m_1 - m_0*m_1`.

Here `d_a` is the existing smooth per-agent distance. Thus an agent already
at its goal cannot cancel another agent's remaining distance. This is a
smooth surrogate, not the binary environment success detector. Historical
`R_risk_v1` replays explicitly retain the joint-sum indicator; do not mix
v1 and v1_1 checkpoints or risk values in one experiment.

The union is evaluated as `c+s-c*s`, and the shared power-mean implementation
uses a finite zero-risk subgradient. These numerical changes preserve the
stated risk formula. The controller, hard projection, frozen baseline, and
primal-dual objective are unchanged.

Historical v1/v1_1 training uses fixed-horizon, nonterminating differentiable rollout
segments. Its `J_live` is a finite-horizon proxy, not an estimate certified
over full, first-terminal-event task episodes. Full-task evaluation must use
the environment's original success/collision/deadlock/timeout termination.
Short evaluated episodes without a complete risk window report missing risk,
not zero risk. Neither completing optimizer updates nor lowering a component
alone establishes task-level liveness or constraint satisfaction.

## Provisional training risk revision: R_risk_v1

The accepted C1 V0 control/projection/primal-dual/BPTT infrastructure is fixed.
The v1 geometric component uses continuous constraint activity
a_i = sigmoid((rho-h_i)/tau_h) * exp(-(s_i/tau_s)^2), where
s_i = A_i u_HS - b_i. tau_h and tau_s are configurable and positive.
The exact hard mask h<=rho and abs(s)<=active_tol remains diagnostic only.

Local geometric candidates satisfy h<=rho+candidate_sigmas*tau_h (default
candidate_sigmas=6), independently of binding. This local outer shell avoids
a membership switch at h=rho; distant constraints are excluded. At the outer
cutoff/topology changes the geometry remains piecewise; global smoothness
at those transitions is not claimed. Zero per-agent vectors are excluded.

The validated analytic cone yields c_a using the V0 signed margin, with
full-plane c_a=1 and empty c_a=0. A_a=max_i a_i over relevant local candidates,
r_a=A_a*c_a. The resulting system geometric component is logged as
`cone_risk_t`. active_tol has no influence on this differentiable component.

R_risk_v1 adds a modular GiveWay task potential V_task: normalized joint
smooth goal distance. Over a physical window W it measures net potential
decrease Delta, constructs a smooth unfinished-task stall risk
`stall_risk_t`, and combines it with `cone_risk_t` by
`r_t=1-(1-cone_risk_t)(1-stall_risk_t)`. Only timesteps with a complete
window are aggregated by the unchanged alpha-weighted power mean plus mean.
Thus reducing slack/cone activity cannot alone make an unfinished stalled
trajectory low risk. window_seconds, delta_prog, tau_prog, task smoothing,
and the repository goal tolerance are configurable. Hard TRO categories and
environment success/deadlock/timeout remain separate diagnostics. This is
provisional R_risk_v1; R_risk_v0_1 remains for regression and old audits.

## Active version: C1 V0

The only active learned control path is frozen Flow-BC sampling with noise xi,
baseline nominal speed bounding, baseline hard projection, an additive residual
in physical control coordinates, and a final hard projection. No residual is
injected into the Flow Euler vector field. Both reference and corrected control
use the same sample xi. The final candidate is not radially clipped before its
projection. Baseline parameters are fixed while their state derivatives remain.

For R_risk_v0, the current nominal task input is the candidate before the final
projection (u_safe + G); binding is measured using the final projected control.
Per-agent zero safety-force blocks are removed before constructing each cone.
Nonzero directions are normalized, circular gaps include the wrap-around gap,
and rank/coverage distinguish empty, ray, line, wedge, half-plane, full-plane.
Empty agent cones have risk zero; full-plane cones have risk one. Ray/line
distances are nonnegative. Full-dimensional proper cones have negative interior
margin, zero boundary margin, and positive exterior distance.

rho, active_tol, D0, kappa, alpha and p are configurable. R_risk_v0 is provisional,
not a frozen paper definition. Discrete active-set/cone classification is
piecewise; A(x), b(x), baseline output and selected distances retain derivatives.

## Objective and safety invariant

C1 adds a learned correction to the hard-safe baseline control:

\[
u_\phi^{\rm HS}=\Pi_{\mathcal A}(u_{\rm safe}+G_\phi).
\]

Hard safety is supplied only by the unchanged projection/filter. The learned
correction must never bypass or replace it.

## Active training objective

The active method is a single-policy constrained optimization problem:

\[
G^\star=\arg\min_G J_{\rm def}(G)
\quad\text{subject to}\quad J_{\rm live}(G)\leq\epsilon,
\]

\[
J_{\rm def}=\mathbb E\left[\frac1T\sum_t
\|u_{\phi,t}^{\rm HS}-u_{{\rm safe},t}\|_M^2\right],
\qquad
J_{\rm live}=\mathbb E_{\tau\sim P_\phi}[R_{\rm risk}(\tau)].
\]

For a nonnegative dual variable \(\lambda\), optimize

\[
\mathcal L_\phi=J_{\rm def}+\lambda(J_{\rm live}-\epsilon),
\qquad
\lambda\leftarrow[\lambda+\eta_\lambda(J_{\rm live}-\epsilon)]_+.
\]

The trainer must log \(J_{\rm def}\), \(J_{\rm live}\), the constraint
residual, and \(\lambda\) for every update.

The numerical trainer uses fixed-batch, fixed-noise, fixed-dual backtracking
on an Adam proposal (successive factors of 1/2, at most eight backtracks).
Accept only a finite, nonincreasing primal objective on that batch; otherwise
retain both parameters and Adam state. A zero proposal also retains Adam state.
An accepted scaled proposal commits the moments for that gradient observation.
The dual update continues to use the pre-step risk, as in the simultaneous
primal-dual scheme above. Record both pre-step and post-step metrics, proposal
trials, and acceptance. This is a numerical safeguard, not an additional C1
constraint, feasibility certificate, or population-descent guarantee.

## Provisional deadlock-risk function

At timestep \(t\), compute

\[
D_t=d_{\rm signed}(-F_V(x_t),\operatorname{Cone}(F_{H_\rho}(x_t))).
\]

Here \(F_V\) is the task force and \(F_{H_\rho}\) contains the safety-force
vectors of active or look-ahead-active constraints. Convert it to instantaneous
risk and trajectory risk using

\[
r_t=\sigma\left(\kappa\frac{D_0-D_t}{D_0}\right),
\qquad
R_{\rm risk}(\tau)=
\alpha\left(\frac1T\sum_t r_t^p\right)^{1/p}
+(1-\alpha)\frac1T\sum_t r_t.
\]

Risk remains modular. Its implementation must include diagnostics for
\(D_t\), \(r_t\), active constraints, cone geometry, and degenerate cases.

## Required experiment behavior

Episodes terminate with mutually exclusive success, agent collision, wall or
obstacle collision, timeout, or deadlock outcomes. Do not treat timeout as
deadlock without an explicit detector. Log outcome rates, minimum safety
distance, episode length, mean/max risk, mean/max instantaneous risk, raw and
applied correction magnitudes, and the fraction and magnitude changed by the
projection.

The original baseline and hard-safe baseline settings remain frozen. Evaluation
compares them with C1 on the same initial states and random-number protocol.
