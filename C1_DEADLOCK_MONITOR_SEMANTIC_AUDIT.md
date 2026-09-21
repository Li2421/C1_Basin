# C1 operational deadlock monitor and projection: semantic audit

Date: 2026-09-18  
Scope: read-only reconstruction of the implemented C1 controller, operational strict-deadlock monitor, terminal stalled-timeout classifier, dynamics, and the two hard projections. This document does not define a new risk, certificate, witness, loss, or experiment.

## 0. Authoritative path and notation

The current deadlock-union setup loads the frozen evaluation environment, forces `terminate_on_deadlock=True`, constructs the residual controller, and uses the default CBF configuration (`single_integrator/c1/train_deadlock_union.py:48-62`). Its rollout rejects any plant/configuration other than `dt=.05`, `max_speed=.5`, `goal_tolerance=.08`, all three first-event termination flags true, and `max_steps=850` (`single_integrator/c1/rollout_deadlock_union.py:15-21`). The operational evaluator computes both projections through `project_velocity`, executes the second result in `GiveWayEnv.step`, and then applies the stalled-timeout classifier only after the episode (`single_integrator/c1/evaluate_deadlock_union.py:23-57, 84-118`). The combined endpoint used by C1 statistics is exactly

```text
strict_or_stalled(row)
    = bool(row['any_deadlock'])
      OR (row['six_class_outcome'] == 'stalled_deadlock').
```

(`single_integrator/c1/paired_statistics.py:10-14, 24-37`.)

Notation:

- \(x_n=(x_{0,n},x_{1,n})\in\mathbb R^4\) is the joint position after \(n\) executed actions, \(n=0,\ldots,850\).
- \(u_k=(u_{0,k},u_{1,k})\in\mathbb R^4\) is the final-projection output executed from \(x_k\) to \(x_{k+1}\), \(k=0,\ldots,849\).
- \(g_i\) is agent \(i\)'s goal and \(d_{i,n}=\lVert g_i-x_{i,n}\rVert_2\) in metres.
- \(s_k=\max_i\lVert u_{i,k}\rVert_2\) in m/s.
- \(N=850\), \(dt=0.05\) s, so the horizon is \(42.5\) s. These values are frozen in the checkpoint evaluation environment (`baseline_309_314/checkpoints/seed0/benchmark.json:6-31`) and asserted by the C1 rollout (`single_integrator/c1/rollout_deadlock_union.py:15-21`).

All formulas below use zero-based action indices and state-after-action indices as defined above. “Raw strict predicate” means the Boolean `info['deadlock']` computed on a step. “Recorded strict event” means the first-event outcome `safe_deadlock`/the C1 `any_deadlock` endpoint.

## 1. Exact reconstruction of \(D_{strict}\), \(D_{stalled}\), and \(D\)

### 1.1 Strict candidate and strict trigger

The active monitor parameters are:

| Quantity | Exact value | Units | Source |
|---|---:|---|---|
| progress window | `2.0` | s | `single_integrator/environment.py:31-35`; frozen value `baseline_309_314/checkpoints/seed0/benchmark.json:24-28` |
| hold duration | `5.0` | s | same sources |
| progress threshold | `0.01` | m | same sources |
| speed threshold | `max_speed * speed_epsilon_fraction = 0.5 * 0.05 = 0.025` | m/s | `single_integrator/environment.py:15, 35, 206-209` |
| goal tolerance | `0.08` | m | `single_integrator/environment.py:26, 192-194` |

The fields `slow_speed=.05`, `congestion_goal_error=.10`, and `deadlock_seconds=1.0` are explicitly deprecated metadata and are not used by the active `window_progress_v2` predicate (`single_integrator/environment.py:27-35`).

After action (k), put (n=k+1). Because (2.0/0.05=40), the literal candidate predicate is

\[
C_k \equiv
\underbrace{(n\ge 40)}_{W_k}
\;\land\;
\underbrace{\bigl(\exists i\in\{0,1\}:d_{i,n}>0.08\bigr)}_{G_k:\ \text{not joint success}}
\;\land\;
\underbrace{\bigl(\forall i:\ |d_{i,n-40}-d_{i,n}|<0.01\bigr)}_{P_k}
\;\land\;
\underbrace{\bigl(\forall i:\ \lVert u_{i,k}\rVert_2<0.025\bigr)}_{V_k}.
\]

Equivalently in action indices, (P_k) compares (d_{i,k-39}) with (d_{i,k+1}). `not success` is the negation of `all(errors <= .08)`, hence it is **any-agent** strict `> .08`, while both the progress and speed predicates are **all-agent** strict `<` predicates. The code computes the current post-action goal errors, `max(abs(window_progress))`, and the maximum executed speed exactly in this order (`single_integrator/environment.py:178-209`).

The progress statistic is neither cumulative path length nor best-so-far improvement. It is the signed endpoint difference

\[
\Delta d_{i,k}=d_{i,k-39}-d_{i,k+1},
\]

followed by absolute value per agent and a maximum across the two agents. There is no averaging, normalization, clipping, or tolerance other than the strict (0.01\) m comparison (`single_integrator/environment.py:195-209`). A regression larger than (0.01\) m and an improvement larger than (0.01\) m both make (P_k) false because of the absolute value; the regression behavior is exercised explicitly in `single_integrator/tests/test_deadlock.py:19-24`.

Let (a_k) be the first action index in the current uninterrupted run of true candidates. The implementation stores the corresponding post-step count `candidate_since`; on the first true sample its timer is zero, and on later true samples it is the number of intervening step intervals times `dt` (`single_integrator/environment.py:210-218`). The raw strict trigger is

\[
R_k \equiv C_k\ \land\ ((k-a_k)\,0.05 \ge 5.0-10^{-10}).
\]

Since the indices are integral, this is equivalently

\[
R_k \equiv \bigwedge_{j=k-100}^{k} C_j,
\qquad k\ge139.
\]

Thus the hold requires **101 consecutive candidate samples spanning 100 intervals = 5.0 s**, not 100 samples. The first possible trigger is after action (k=139), at step count (140) and time (7.0) s: 2.0 s to fill the first progress window plus the 5.0 s hold. The code comparison is non-strict `>= 5.0 - 1e-10` (`single_integrator/environment.py:219`), and the expected first trigger at step 140/time 7.0 s is checked in `single_integrator/tests/test_environment.py:49-55`.

For the first-event operational protocol, define

\[
D_{strict}=1
\quad\Longleftrightarrow\quad
\text{the recorded first terminal event is `safe_deadlock`}.
\]

This means some (R_k) is reached while the episode is still running and no collision wins on that same step. Success cannot coexist with (C_k), because (C_k) contains `not success`. Collision can coexist with the raw (R_k), but collision has higher terminal/outcome precedence; see Section 4. The environment's raw predicate is `deadlock = candidate and timer >= ...`, while C1's endpoint stores `any_deadlock = any(trace['deadlock'])` (`single_integrator/environment.py:219-229`; `single_integrator/c1/evaluate_deadlock_union.py:47-57`). The active C1 setup terminates on the first raw strict trigger, so in this protocol `any_deadlock` and recorded strict deadlock coincide unless a same-step higher-priority collision occurs; C1 evaluation treats any collision as a fatal audit error rather than publishing an outcome (`single_integrator/c1/train_deadlock_union.py:48-62`; `single_integrator/c1/evaluate_deadlock_union.py:37-53`).

### 1.2 Terminal stalled-timeout predicate

The stalled classifier is not part of `GiveWayEnv.step`. It is an offline split applied only when the already-recorded outcome is exactly `other_timeout` (`single_integrator/diagnostics/stalled_outcomes.py:1-5, 17-21`). Its default parameters are:

| Quantity | Exact value | Units | Comparison | Source |
|---|---:|---|---|---|
| terminal sample window | `2.0` | s requested | `steps = round(window_seconds/dt)` | `single_integrator/diagnostics/stalled_outcomes.py:17-23` |
| terminal speed threshold | `0.05` | m/s | strict `<` | `single_integrator/diagnostics/stalled_outcomes.py:17-18, 30-33` |
| terminal goal-error-change threshold | `0.02` | m | strict `<` | same source |

For the actual (dt=.05), `steps=round(2/.05)=40`. The C1 evaluator supplies `max_speed` reconstructed from `trace['applied']`, i.e. the final-projection control, and supplies goal errors reconstructed from `trace['positions_after']` (`single_integrator/c1/evaluate_deadlock_union.py:113-118`). A timeout has all 850 samples, so the exact predicate is

\[
\begin{aligned}
D_{stalled}\equiv{}&
(\text{original recorded outcome} = \texttt{other_timeout})\\
&\land\ \bigl(\forall k\in\{810,\ldots,849\}:\ \max_i\lVert u_{i,k}\rVert_2<0.05\bigr)\\
&\land\ \bigl(\forall i\in\{0,1\}:\ |d_{i,850}-d_{i,811}|<0.02\bigr).
\end{aligned}
\]

The speed predicate is an AND over all 40 time samples; each sample's `max_speed < .05` is itself an AND over both agents. The progress predicate is an AND over the two agents. The code uses `np.all` for both (`single_integrator/diagnostics/stalled_outcomes.py:22-37`).

The indexing is literal: the 40 speed samples are actions (810{:}849), but the stored error array contains post-action states (d_1,\ldots,d_{850}). Therefore `errors[-40]` is (d_{811}), and the endpoint error change spans 39 integration intervals, (1.95) s, not 2.0 s. The risk adapter documents and preserves this historical `(samples-1)*dt` behavior (`single_integrator/c1/risk/deadlock_union.py:14-21, 28-36`). There is no additional confirmation timer for stalled deadlock.

If a generic trace contains fewer than 40 samples, the function returns the input outcome unchanged with `enough_window=False`; it does not classify stalled (`single_integrator/diagnostics/stalled_outcomes.py:25-29`). This partial-window branch cannot occur for an actual first-event `other_timeout`, because timeout is reached only at `step_count >= 850` (`single_integrator/environment.py:223-228`).

### 1.3 Union

The operational C1 endpoint is exactly

\[
D = D_{strict}\lor D_{stalled}.
\]

In stored rows, this is `any_deadlock OR six_class_outcome == 'stalled_deadlock'` (`single_integrator/c1/paired_statistics.py:24-37`). The branches are mutually exclusive under the first-event protocol: `classify_timeout_trace` leaves every non-`other_timeout` outcome unchanged, so a recorded strict deadlock can never also be reclassified as stalled (`single_integrator/diagnostics/stalled_outcomes.py:17-21`).

### 1.4 Conditions that do **not** exist in the event

Neither strict nor stalled deadlock tests `w`, `u_safe`, raw Flow-BC output, CBF activity, CBF slack, pair distance, wall distance, a blocking/contact geometry, relative orientation, a selected route, an escape direction, cumulative displacement, cumulative progress, best progress, or a fraction/count of low-speed samples. Collision/contact is a separate higher-priority terminal event, not a conjunct inside `candidate_deadlock` (`single_integrator/environment.py:183-229`). The strict candidate is only window-ready AND not-success AND endpoint-distance stagnation AND joint low executed speed (`single_integrator/environment.py:195-219`). The stalled classifier is only original-timeout eligibility AND tail low executed speed AND tail endpoint goal-error change (`single_integrator/diagnostics/stalled_outcomes.py:17-37`).

In particular, there is no local/one-agent deadlock atom: one agent at zero while the other has speed at least the threshold makes the joint `max_speed` test false. This follows from `speeds.max()` in the strict detector (`single_integrator/environment.py:193, 206-209`) and the evaluator's per-time maximum in the stalled trace (`single_integrator/c1/evaluate_deadlock_union.py:115-117`).

## 2. Recursive helper expansion

There are no unexpanded `is_stalled`, `progress`, `low_speed`, `unfinished`, or `blocked` helpers in the authoritative monitor path. Their primitive operations are:

1. `errors = norm(goals - positions_after, axis=-1)` (`single_integrator/environment.py:178-192`).
2. `success = all(errors <= 0.08)` (`single_integrator/environment.py:192-194`).
3. `past = (1-alpha)*distance_history[lo] + alpha*distance_history[hi]`, with `lo=floor(window_start)`, `hi=ceil(window_start)`, and `alpha=window_start-lo` (`single_integrator/environment.py:195-205`). At the frozen integer window, `alpha=0` and this is exactly (d_{n-40}).
4. `window_progress = past - errors`; then `max(abs(window_progress)) < .01` (`single_integrator/environment.py:205-209`).
5. `speeds = norm(executed_velocity, axis=-1)`; then `max(speeds) < .5*.05` (`single_integrator/environment.py:173-177, 193, 206-209`).
6. The candidate-run start is stored once, timer is `(step_count-candidate_since)*dt`, and any false candidate sets both `candidate_since=None` and timer to zero (`single_integrator/environment.py:210-218`).
7. Stalled classification computes `steps=round(2/dt)`, slices the last `steps` scalar maximum-speed samples, computes `abs(errors[-1]-errors[-steps])`, and applies `all(<.05) AND all(<.02)` (`single_integrator/diagnostics/stalled_outcomes.py:22-37`).

No helper performs normalization, clipping, mean/min aggregation, a best-so-far comparison, or path integration in either operational event.

## 3. Temporal, history, and latch semantics

### 3.1 Windows and endpoints

| Item | Exact semantics | Source |
|---|---|---|
| simulator time step | (dt=0.05\) s | `single_integrator/environment.py:10-15`; frozen value `baseline_309_314/checkpoints/seed0/benchmark.json:6-10` |
| horizon | 850 actions, states (x_0\ldots x_{850}), 42.5 s | `single_integrator/environment.py:13-15, 223-228` |
| strict progress window | 40 action intervals; at action (k), compare (x_{k-39}) and (x_{k+1}) | `single_integrator/environment.py:195-205` |
| strict confirmation | 101 consecutive candidate samples, from actions (k-100\ldots k), spanning 5.0 s | `single_integrator/environment.py:210-219`; confirmed at step 140 by `single_integrator/tests/test_environment.py:49-55` |
| stalled speed tail | last 40 executed actions, (u_{810}\ldots u_{849}) | `single_integrator/diagnostics/stalled_outcomes.py:22-33` |
| stalled progress endpoints | post-action errors (d_{811}) and (d_{850}), 39 intervals/1.95 s apart | same source; explicit historical note `single_integrator/c1/risk/deadlock_union.py:19-21` |

The strict monitor combines (u_k), (x_{k+1}), and a historical state sample (x_{k-39}). It does not evaluate the candidate at (x_k) before the action. The stalled classifier combines 40 final executed controls with two post-action position-derived error samples (`single_integrator/environment.py:178-209`; `single_integrator/diagnostics/stalled_outcomes.py:22-33`).

Before the strict progress history is full, `window_ready=False`, `window_progress=[nan,nan]`, and the leading `window_ready and ...` makes the candidate false (`single_integrator/environment.py:195-209`). For a non-integer configured window, the code linearly interpolates the two adjacent historical distance samples; the frozen 2.0/.05 window is integral (`single_integrator/environment.py:196-205`).

Strict persistence means continuous satisfaction at every sampled step. It is not a count, fraction, or accumulated amount of candidate time. One false candidate resets the run start and timer immediately; continuous true candidates do not reset them (`single_integrator/environment.py:210-218`). A speed excursion, sufficient positive progress, sufficient negative progress, success, or an unready window therefore resets the timer because each makes `candidate=False`. The separate historical latches described below do not reset when the current candidate becomes false.

### 3.2 Historical state

On `reset`, the implementation initializes:

```text
candidate_since = None
stuck_timer = max_stuck_timer = 0
ever_candidate_deadlock = False
first_success_step = first_deadlock_step = None
first_wall_collision_step = first_agent_collision_step = None
done = False
```

(`single_integrator/environment.py:147-164`.)

Let (H_n) mean that a raw strict trigger has occurred in the first (n) actions. The summary latch is exactly

\[
H_0=false,\qquad H_{n}=H_{n-1}\lor R_{n-1}.
\]

Operationally, `first_deadlock_step` is assigned only when the current `deadlock` flag is true and the field is still `None`; later false `deadlock` values cannot clear it (`single_integrator/environment.py:219-222`). `summary()['deadlock']` is `first_deadlock_step is not None`, whereas per-step `info['deadlock']` is the current raw predicate (`single_integrator/environment.py:229, 235-236`). A nonterminating diagnostic can therefore recover dynamically while retaining a permanent recorded deadlock; this behavior and reset clearing are tested in `single_integrator/tests/test_deadlock.py:52-65`.

`ever_candidate_deadlock` is a different latch: it records that **one candidate sample** occurred, not that the 5 s hold completed (`single_integrator/environment.py:158-161, 210-213, 235-236`). It is not part of (D_{strict}\) or (D_{stalled}\).

In the active operational C1 protocol, `terminate_on_deadlock=True`, so a strict trigger immediately ends the episode and later recovery cannot occur (`single_integrator/c1/train_deadlock_union.py:48-62`; `single_integrator/environment.py:223-228`). The historical latch remains semantically important for nonterminating diagnostics and for any certificate evaluated on a continued model trajectory: a terminal-only current predicate cannot erase an earlier (R_k).

## 4. First-event ordering

`GiveWayEnv.step` computes all current flags, writes every previously unset first-occurrence field, and then chooses termination in this order:

```text
(wall_collision OR agent_collision)  -> collision
else success                          -> success
else strict deadlock                  -> deadlock
else step_count >= max_steps          -> timeout
else                                  -> running
```

(`single_integrator/environment.py:183-228`.)

The terminal labeler refines collision precedence as:

```text
agent_collision > wall_collision > success > safe_deadlock > other_timeout
```

(`single_integrator/outcomes.py:5-21`.) Thus:

- simultaneous agent and wall collision records `agent_collision`;
- either collision beats same-step success, strict deadlock, and timeout;
- success beats strict deadlock and timeout (although raw strict candidate already excludes success);
- strict deadlock on step 850 beats timeout;
- timeout is recorded only if none of the earlier conditions wins.

All same-step first-occurrence fields whose raw flags are true may still be latched before termination, even when another event wins the recorded outcome, because the latch loop precedes the termination chain (`single_integrator/environment.py:219-227`). Once `done=True`, another `step` raises and no subsequent event can be observed (`single_integrator/environment.py:170-172, 228`). Solver failure is not an episode outcome; `CBFSolverError` is explicitly an experiment error and has no fallback label (`single_integrator/cbf.py:44-48`).

The six-class stalled split preserves every non-timeout first event and only subdivides `other_timeout` (`single_integrator/diagnostics/stalled_outcomes.py:1-5, 12-21`). C1 acceptance additionally requires success, strict deadlock, and timeout to be mutually exclusive and rejects collision-bearing results from the safety-certified endpoint (`single_integrator/c1/acceptance.py:17-33`).

## 5. Executed dynamics and signal semantics

### 5.1 Controller transformations

For a fixed state and Flow noise, the implemented controller is:

1. Flow-BC integrates its four-dimensional flow by Euler steps in model space; if normalization is enabled, it de-normalizes by `act_scale` and `act_mean`, then componentwise clips the result to `[-1,1]` (`single_integrator/c1/differentiable_rollout.py:36-47`). The frozen model uses normalization and ten Flow steps (`baseline_309_314/checkpoints/seed0/config.json:20-43`).
2. `bounded_nominal` reshapes into two 2-D agent blocks and radially scales each block to norm at most `max_speed=.5` (`single_integrator/c1/differentiable_rollout.py:6-9`).
3. The first hard projection produces `safe = projection(nominal,A,b,speed)` (`single_integrator/c1/differentiable_rollout.py:23-29`).
4. The residual receives that physical `safe` control and returns a four-dimensional additive correction; there is no output activation or scale on the zero-initialized final dense layer (`single_integrator/c1/models/residual.py:8-20, 27-50`).
5. `candidate = safe + correction`; there is no radial or component clipping of this candidate (`single_integrator/c1/differentiable_rollout.py:23-34`).
6. The final hard projection produces `applied = projection(candidate,A,b,speed)` (`single_integrator/c1/differentiable_rollout.py:31-34`). In the operational evaluator the same sequence is visible explicitly, and the second result is passed directly to `env.step` (`single_integrator/c1/evaluate_deadlock_union.py:28-44`).

Thus, with (u_{FlowBC,k}) denoting the already de-normalized, component-clipped, radially speed-bounded nominal,

\[
u_{safe,k}=\Pi_{U(x_k)}(u_{FlowBC,k}),\qquad
w_k=u_{safe,k}+G_\phi(\cdot),\qquad
u_{\phi,k}=\Pi_{U(x_k)}(w_k).
\]

Both projections use the same (A(x_k),b(x_k)) computed before the action (`single_integrator/c1/evaluate_deadlock_union.py:28-37`). The projections remain distinct because their targets differ.

### 5.2 Plant

The exact plant is

\[
x_{k+1}=x_k+0.05\,u_{\phi,k},\qquad
\texttt{env.velocities}_{k+1}=u_{\phi,k}.
\]

The environment copies the supplied final-projection control, rejects nonfinite/wrong-shape input, rejects a block norm above `max_speed+1e-9`, integrates by forward Euler, and then copies that same value into `self.velocities` (`single_integrator/environment.py:170-182`). It computes collisions only after this integration and never changes the position or velocity in response (`single_integrator/environment.py:183-233`); the no-response behavior is tested in `single_integrator/tests/test_environment.py:27-40`.

There is no action gain after the final projection, no plant clipping, velocity filter, acceleration state, actuator lag, collision correction, goal-region zeroing, or other post-projection modification. Overspeed causes an exception rather than clipping (`single_integrator/environment.py:170-181`); success/collision/deadlock only selects termination after the full update (`single_integrator/environment.py:183-228`).

Therefore **physical velocity equals (u_{\phi,k}) exactly for every accepted step**, not merely conditionally on a tracking model. `tracking_error` is identically the difference between the just-assigned velocity and `u`, and `integration_residual` checks the Euler equality (`single_integrator/environment.py:181, 229`). The exact equality and instantaneous stopping are tested in `single_integrator/tests/test_environment.py:9-18`.

The strict speed atom reads `u` directly (`single_integrator/environment.py:173-177, 193, 206-209`). The stalled speed atom is rebuilt from the stored `applied` final-projection controls (`single_integrator/c1/evaluate_deadlock_union.py:113-117`). Neither event reads `w` or `u_safe`.

## 6. Final hard projection and first projection

### 6.1 Feasible set

At joint position (x=(p_0,p_1)), the implemented set is

\[
U(x)=\{u\in\mathbb R^4:A(x)u\ge b(x),\ \lVert u_0\rVert_2\le0.5,\ \lVert u_1\rVert_2\le0.5\}.
\]

There are 17 linear CBF rows: one pair row and eight finite-wall rows for each of two agents. The environment constructs exactly eight wall segments (`single_integrator/environment.py:111-124`), and the CBF builder creates the pair row followed by one block-sparse row per agent/wall pair (`single_integrator/cbf.py:82-93`).

For pair relative position (r=p_0-p_1),

\[
h_p=\lVert r\rVert_2^2-d_{safe}^2,\quad
d_{safe}=2(0.16)+0+10^{-4}=0.3201\text{ m},
\]

and the constraint is

\[
[2r^\top,-2r^\top]u\ge-\gamma h_p,
\qquad \gamma=1.
\]

For wall segment \([a_\ell,b_\ell]\), let \(c_{i\ell}\) be the clipped closest point on the finite segment, \(\delta_{i\ell}=p_i-c_{i\ell}\), \(r_{i\ell}=\lVert\delta_{i\ell}\rVert_2\), and \(n_{i\ell}=\delta_{i\ell}/r_{i\ell}\). Then

\[
h^w_{i\ell}=r_{i\ell}-0.16-\frac4{600}-0.005-10^{-4},\qquad
n_{i\ell}^\top u_i\ge-\gamma_w h^w_{i\ell},\quad\gamma_w=1.
\]

These primitive closest-point, barrier, gradient, and row operations are in `single_integrator/cbf.py:51-93`; all numerical CBF defaults are in `single_integrator/cbf.py:16-29`; the plant radii/margins are in `single_integrator/environment.py:15-25`.

There are no goal constraints, modes, priorities, auxiliary decision variables, relaxation/slack variables, slack penalties, or recovery terms. The module states this explicitly (`single_integrator/cbf.py:1-7`), and the optimizer receives only the four-vector decision plus hard linear inequalities and two speed balls (`single_integrator/cbf.py:96-127`).

### 6.2 Optimization and metric

For target \(y\in\mathbb R^4\), both projections solve the same joint four-dimensional problem

\[
\Pi_{U(x)}(y)=\arg\min_{u\in U(x)}\frac12\lVert u-y\rVert_2^2.
\]

The metric is the identity: there is no nonidentity (Q). The operational implementation's objective and gradient are exactly `.5*dot(u-target,u-target)` and `u-target` (`single_integrator/cbf.py:96-105, 121-127`). The alternative CVXPYLayer path declares the same `.5*sum_squares(u-target)` objective, 17 linear inequalities, and two 2-norm speed constraints (`single_integrator/c1/socp.py:17-33`).

The first projection uses (y=u_{FlowBC,k}) and returns (u_{safe,k}); the second uses (y=w_k=u_{safe,k}+G_\phi) and returns the executed (u_{\phi,k}) (`single_integrator/c1/differentiable_rollout.py:23-34`). No constraint or solver setting differs between the two projections at a given state.

### 6.3 Operational solver, tolerances, failure behavior

The actual deadlock-union operational/evaluation path uses SciPy SLSQP through `project_velocity` for both projections (`single_integrator/c1/evaluate_deadlock_union.py:28-37`). The current differentiable deadlock-union rollout's `controller_projection` also calls that same routine in its forward pass (`single_integrator/c1/risk/joint_frozen.py:16-18, 37-57`; `single_integrator/c1/rollout_deadlock_union.py:41-47`). SciPy is pinned to 1.15.3 (`requirements.txt:65-75`).

The speed balls are handled by adaptive outer supporting planes. The first polyhedron contains the CBF halfspaces and component box `[-.5,.5]^4`; a tangent halfspace is added for every block that exceeds the Euclidean speed ball, and the candidate is re-solved. There is no post-solve radial clipping (`single_integrator/cbf.py:96-105, 116-163`). If an exact minimizer over the current outer polyhedron lies inside both true balls, it is also the minimizer over (U(x)), because (U(x)) is a subset of that outer polyhedron.

The literal numerical settings are:

| Check/setting | Value | Source |
|---|---:|---|
| SLSQP `ftol` | `1e-12` | `single_integrator/cbf.py:23, 121-127` |
| SLSQP iterations per cut round | `200` | `single_integrator/cbf.py:28, 126-127` |
| maximum speed-cut rounds | initial solve plus at most 32 added-cut rounds | `single_integrator/cbf.py:29, 120-163` |
| linear feasibility tolerance | `1e-9` | `single_integrator/cbf.py:24, 136-137` |
| active-row threshold for KKT check | residual `<=1e-7` | `single_integrator/cbf.py:138-151` |
| stationarity/complementarity tolerance | `2e-6` | `single_integrator/cbf.py:25, 139-151` |
| speed tolerance | `1e-10` | `single_integrator/cbf.py:26, 152-154` |
| environment overspeed rejection | `1e-9` above max speed | `single_integrator/environment.py:173-177` |

An already feasible target is returned unchanged (`single_integrator/cbf.py:112-114`). Otherwise, a nonfinite/infeasible candidate, failed KKT check, or exhaustion of speed cuts raises `CBFSolverError`; solver status alone is not accepted, and there is no fallback action (`single_integrator/cbf.py:128-163`). The operational evaluator does not catch such an error to replace the action (`single_integrator/c1/evaluate_deadlock_union.py:23-57`). After success, it only reshapes/copies the returned vector before execution; no control modification follows the final solve (`single_integrator/c1/evaluate_deadlock_union.py:35-44`).

For completeness, the separate general C1 `ExactProjection` path uses CVXPYLayers/diffcp with `eps=1e-11`, `max_iters=100000`, and validates CBF feasibility/speed against the default CBF tolerances (`single_integrator/c1/socp.py:41-60`). Its packages are pinned (`single_integrator/c1/requirements.txt:1-11`), but no explicit `solve_method` is supplied in the repository call; the exact internal diffcp cone-solver selection is therefore **UNRESOLVED from repository code alone**. This does not affect the current deadlock-union operational path, which calls SciPy SLSQP directly.

## 7. Atom-by-atom VI eligibility

Eligibility convention: `SUPPORTED` means the implemented dynamics and monitor prove a nontrivial membership (u_{\phi,k}\in B(z_k)) (possibly using the minimal closed superset of a strict atom). `UNSUPPORTED` means only a tautological event-preimage or all of \(\mathbb R^4\) is available as a current-control set. `CONDITIONAL` names the exact missing assumption. A sound membership need not be an equivalence, but any enlargement is identified and proved.

| Event | Atom | Exact formula | Signal type | Temporal? | Can it imply (u_{\phi,k}\in B)? | Candidate (B) | Proof available from current dynamics? | Status |
|---|---|---|---|---|---|---|---|---|
| strict | window ready | (k+1\ge40) | time/history availability | yes | no nontrivial current-control restriction | none | It depends only on the index (`single_integrator/environment.py:196-200`). | UNSUPPORTED |
| strict | unfinished joint task | (\exists i:d_{i,k+1}>.08) | post-action goal distance | instantaneous post-state | yes | (B^{goal}_k), Section 8.2 | Exact Euler dynamics turns post-action goal distance into an exact function of (u_k) (`single_integrator/environment.py:178-194`). | SUPPORTED |
| strict | endpoint stagnation | (\forall i:|d_{i,k-39}-d_{i,k+1}|<.01) | historical and post-action goal distance | 40-step endpoint window | yes, but **not** as a low-control ball | (B^{prog}_k(.01,d_{k-39})), Section 8.3 | Exact Euler dynamics gives an annular preimage in each 2-D action block; no low-control inference is used (`single_integrator/environment.py:195-209`). | SUPPORTED |
| strict | low executed speed | (\forall i:\lVert u_{i,k}\rVert_2<.025) | final-projection executed control/physical velocity | current action; repeated by hold | yes | (B^{speed}(.025)), Section 8.1 | `speeds` is computed from executed `u`, and plant velocity equals `u` (`single_integrator/environment.py:173-181, 193, 206-209`). | SUPPORTED |
| strict | 5 s persistence | (\bigwedge_{j=k-100}^{k}C_j) | candidate history/timer | 101 samples | not as one set on only the terminal action; yes after onset-window expansion | the above supported sets for every (j\in[k-100,k]) | The timer recursion proves the expansion (`single_integrator/environment.py:210-219`), but discarding the onset/history indices would be unsound. | CONDITIONAL |
| strict | first-event/recording guard | episode running before (k), and no same-step collision wins | terminal-event history and collision flags | yes | no simple current-control atom is implemented | none | This is outcome precedence, not a deadlock numerical predicate (`single_integrator/environment.py:219-228`; `single_integrator/outcomes.py:9-21`). | UNSUPPORTED |
| strict | historical latch | (H_n=H_{n-1}\lor R_{n-1}) | stored history | entire episode | only at the historical trigger action(s), not necessarily at the current recovered action | retain the supported (B)'s at every possible onset branch | The latch survives later recovery (`single_integrator/environment.py:219-236`; `single_integrator/tests/test_deadlock.py:52-65`). | CONDITIONAL |
| stalled | original timeout eligibility | recorded outcome is exactly `other_timeout` | categorical first-event result | full episode | it implies final post-state is unfinished in the authoritative path, hence (u_{849}\in B^{goal}_{849}); it does not encode the whole timeout history as a simple set | (B^{goal}_{849}) is a proved necessary enlargement, not an exact timeout set | Timeout is below success/deadlock/collision in the terminal chain (`single_integrator/environment.py:223-228`), and the classifier is called with that recorded outcome (`single_integrator/diagnostics/stalled_outcomes.py:17-21`). | SUPPORTED |
| stalled | enough samples | trace length \(\ge40\) | trace length | yes | no nontrivial action restriction | none | The short-trace branch is purely a length check (`single_integrator/diagnostics/stalled_outcomes.py:25-29`). | UNSUPPORTED |
| stalled | tail low executed speed | (\forall k=810{:}849,\forall i:\lVert u_{i,k}\rVert_2<.05) | stored final-projection controls | 40 actions | yes, at each indexed action | (B^{speed}(.05)) | C1 constructs tail speed from `trace['applied']`; physical velocity equals applied control (`single_integrator/c1/evaluate_deadlock_union.py:113-117`; `single_integrator/environment.py:180-181`). | SUPPORTED |
| stalled | terminal endpoint stagnation | (\forall i:|d_{i,850}-d_{i,811}|<.02) | two post-action goal-distance samples | 39 intervals between endpoints | yes at the final action (k=849), but **not** as an instantaneous low-control ball | (B^{prog}_{849}(.02,d_{811})) | (d_{i,850}) is an exact function of (x_{i,849},u_{i,849}); the classifier supplies the fixed historical (d_{i,811}) (`single_integrator/diagnostics/stalled_outcomes.py:30-33`; dynamics `single_integrator/environment.py:178-192`). | SUPPORTED |
| union | OR and six-class preservation | (D_{strict}\lor D_{stalled}) | historical/categorical | full episode | not itself a new per-action set | branch-specific sets only | The union is explicit in endpoint statistics, and the stalled split preserves non-timeouts (`single_integrator/c1/paired_statistics.py:35-37`; `single_integrator/diagnostics/stalled_outcomes.py:17-21`). | UNSUPPORTED |

No row equates endpoint stagnation with low control. Its supported set is the exact Euler preimage derived next.

## 8. Minimal faithful bad-control sets

All control sets below live in joint velocity space \(\mathbb R^4\), with each 2-D block measured in m/s. Strict `<` monitor sets are open. For distance-based VI use, the sets below are their closures: this is the **minimal closed enlargement**. The enlargement is sound because every strict-event member is in its closure; only equality-boundary controls are added.

### 8.1 Joint low-speed set

For \(\rho>0\),

\[
B^{speed}(\rho)=\{q\in\mathbb R^4:\lVert q_0\rVert_2\le\rho,\ \lVert q_1\rVert_2\le\rho\}.
\]

- strict uses \(\rho=.025\) m/s at every candidate action in a triggering streak;
- stalled uses \(\rho=.05\) m/s for actions (810{:}849).

This set is state-independent, closed, convex, compact, four-dimensional as an ambient subset, and joint (the Cartesian product of two agent balls). Its exact Euclidean distance is

\[
\operatorname{dist}^2(q,B^{speed}(\rho))
=\sum_{i=0}^1\max(\lVert q_i\rVert_2-\rho,0)^2.
\]

The event-to-set inclusion follows directly from the strict `<` comparisons on the final applied control (`single_integrator/environment.py:193, 206-209`; `single_integrator/diagnostics/stalled_outcomes.py:30-33`; C1 signal source `single_integrator/c1/evaluate_deadlock_union.py:113-117`).

### 8.2 Post-action unfinished set

At pre-action state (x_k), define the velocity-space goal center

\[
a_{i,k}=\frac{g_i-x_{i,k}}{dt}\quad\text{(m/s)},\qquad R_g=\frac{0.08}{dt}=1.6\text{ m/s}.
\]

Exact Euler dynamics gives

\[
d_{i,k+1}=dt\,\lVert u_{i,k}-a_{i,k}\rVert_2.
\]

Therefore the minimal closed set containing the strict unfinished atom is

\[
B^{goal}_k
=\bigcup_{i=0}^1\{q\in\mathbb R^4:\lVert q_i-a_{i,k}\rVert_2\ge R_g\}.
\]

It is state-dependent, closed, generally nonconvex, unbounded, and joint as a union of two per-agent cylinders. Its exact distance is

\[
\operatorname{dist}(q,B^{goal}_k)
=\min_i\max(R_g-\lVert q_i-a_{i,k}\rVert_2,0).
\]

The strict atom uses `>` and hence lies in this `>=` closure. An authoritative `other_timeout` also implies no final success and therefore the same final-action membership; this is a necessary enlargement of the full timeout semantics, not an equivalence. The premises are the Euler update and post-action success test (`single_integrator/environment.py:178-194`) plus timeout precedence (`single_integrator/environment.py:223-228`).

### 8.3 Endpoint-stagnation set

Let (D_i\ge0) be the fixed historical goal distance used by an endpoint comparison, and let \(\epsilon>0\). Define

\[
L_i=\frac{\max(0,D_i-\epsilon)}{dt},\qquad
H_i=\frac{D_i+\epsilon}{dt}.
\]

The minimal closed Euler preimage of

\[
|D_i-d_{i,k+1}|<\epsilon\quad\text{for both agents}
\]

is

\[
B^{prog}_k(\epsilon,D)
=\prod_{i=0}^1\{q_i\in\mathbb R^2:L_i\le\lVert q_i-a_{i,k}\rVert_2\le H_i\}.
\]

For strict deadlock, (D_i=d_{i,k-39}) and \(\epsilon=.01\) m. For stalled deadlock at (k=849), (D_i=d_{i,811}) and \(\epsilon=.02\) m. This set is closed and bounded; it is generally nonconvex when any (L_i>0), and becomes convex only in the special case where every inner radius is zero. It is joint as a Cartesian product of two agent annuli. With (r_i=\lVert q_i-a_{i,k}\rVert_2), its exact Euclidean distance is

\[
\operatorname{dist}^2(q,B^{prog}_k)
=\sum_{i=0}^1
\begin{cases}
(L_i-r_i)^2,&r_i<L_i,\\
0,&L_i\le r_i\le H_i,\\
(r_i-H_i)^2,&r_i>H_i.
\end{cases}
\]

This is an exact dynamics-based action preimage, not a claim that low net goal-distance change implies low instantaneous control. The endpoint statistic and its strict comparisons are implemented at `single_integrator/environment.py:195-209` and `single_integrator/diagnostics/stalled_outcomes.py:30-33`; the required Euler equality is `single_integrator/environment.py:178-181`.

### 8.4 What has no minimal useful instantaneous set

Window readiness, trace length, first-event precedence, “no earlier event,” and the historical latch do not by themselves restrict a selected current (u_k) to a nontrivial geometry without embedding the whole event evaluator into the definition of (B). Such a tautological preimage is not useful projection semantics and is classified unsupported. Persistence and the latch are instead handled soundly by retaining all possible onset branches and applying Sections 8.1-8.3 at their actual historical action indices (`single_integrator/environment.py:210-222, 235-236`).

## 9. Projection optimality and the VI formula

### 9.1 Exact mathematical projection

The set (U(x)) is an intersection of linear halfspaces and two closed Euclidean balls, hence is closed and convex. Whenever it is nonempty, the exact minimizer of

\[
\min_{u\in U(x)}\tfrac12\lVert u-w\rVert_2^2
\]

is unique and satisfies

\[
(w-u)^\top(v-u)\le0\qquad\forall v\in U(x).
\]

The premises—identity-metric quadratic objective, linear CBF inequalities, and two Euclidean speed balls—are the literal implementation (`single_integrator/cbf.py:96-127`; independently `single_integrator/c1/socp.py:20-33`). No weighted-metric version is needed because the actual (Q) is (I_4).

For any nonempty set (B) containing the executed (u), convex or not, and any (v\in U(x)), put (m=(w+v)/2). Then

\[
\begin{aligned}
\operatorname{dist}^2(m,B)-\tfrac14\lVert w-v\rVert_2^2
&\le \lVert m-u\rVert_2^2-\tfrac14\lVert w-v\rVert_2^2\\
&=(w-u)^\top(v-u)\\
&\le0.
\end{aligned}
\]

Consequently the proposed Euclidean expression

\[
c_{VI}(w,v,B)=\operatorname{dist}^2((w+v)/2,B)-\tfrac14\lVert w-v\rVert_2^2
\]

is **EXACTLY VALID for the exact optimization problem implemented**. It applies to the closed speed, goal, and annular progress sets above; convexity of (B) is not required for the inequality, only membership (u\in B) and an exact distance.

### 9.2 Literal floating-point return

The operational solver does not furnish an exact real-arithmetic optimality proof. It accepts feasibility to (10^{-9}), speed to (10^{-10}), and stationarity/complementarity to (2\times10^{-6}) (`single_integrator/cbf.py:23-29, 128-154`). Therefore the unqualified exact inequality `<= 0` is not guaranteed bit-for-bit for every returned floating-point vector. The formula requires no metric change, but a machine-checkable certificate over literal solver outputs must either establish the VI residual directly or carry a rigorously derived numerical allowance. This audit does not invent that allowance.

Thus the requested classification is:

```text
Metric/formula classification:       EXACTLY VALID
Literal tolerance-level guarantee:   CONDITIONAL / not established by current checks
```

The remaining condition is numerical, not a weighted-(Q) correction. Solver failure must remain an experiment failure; assigning an arbitrary action or favorable event would break the premise (`single_integrator/cbf.py:44-48, 136-163`).

## 10. Final audit summary

### A. Exact event formulas

With (C_k=W_k\land G_k\land P_k\land V_k),

\[
\begin{aligned}
W_k &: k+1\ge40,\\
G_k &: \exists i:\ d_{i,k+1}>.08,\\
P_k &: \forall i:\ |d_{i,k-39}-d_{i,k+1}|<.01,\\
V_k &: \forall i:\ \lVert u_{i,k}\rVert_2<.025,\\
R_k &: \bigwedge_{j=k-100}^{k}C_j,\quad k\ge139,\\
D_{strict}&:\text{ recorded first event is a strict trigger }R_k,\\
D_{stalled}&:(\text{recorded original event is `other_timeout`})\\
&\quad\land\ \forall k=810{:}849,\forall i:\lVert u_{i,k}\rVert_2<.05\\
&\quad\land\ \forall i:|d_{i,850}-d_{i,811}|<.02,\\
D&=D_{strict}\lor D_{stalled}.
\end{aligned}
\]

Sources: strict primitives and timer `single_integrator/environment.py:178-229`; stalled primitives `single_integrator/diagnostics/stalled_outcomes.py:17-37`; union endpoint `single_integrator/c1/paired_statistics.py:24-37`.

### B. Complete atomic-condition list

Strict: window ready; any-agent unfinished goal condition; all-agent absolute endpoint-distance stagnation; all-agent current executed low speed; 101-sample continuous candidate persistence; first-event/collision precedence; historical first-trigger latch. Stalled: original `other_timeout` eligibility; at least 40 trace samples; all-agent low executed speed at every one of the final 40 actions; all-agent absolute goal-error endpoint change; preservation of the original first event. There is no blocking/interaction, contact, CBF-activity, `w`, `u_safe`, raw-policy, displacement, cumulative-progress, best-progress, or local-deadlock atom. Sources: `single_integrator/environment.py:195-229`; `single_integrator/diagnostics/stalled_outcomes.py:17-37`.

### C. Atoms admitting sound projection-compatible mappings

- strict low speed \(\to B^{speed}(.025)\);
- stalled tail low speed \(\to B^{speed}(.05)\);
- strict unfinished post-state \(\to B^{goal}_k\);
- strict endpoint stagnation \(\to B^{prog}_k(.01,d_{k-39})\);
- stalled endpoint stagnation at the final action \(\to B^{prog}_{849}(.02,d_{811})\);
- authoritative timeout eligibility has the proved necessary final unfinished consequence \(u_{849}\in B^{goal}_{849}\).

The last three mappings use the exact direct-Euler dynamics; none replaces progress by low control. Source premises: `single_integrator/environment.py:178-209, 223-228`; `single_integrator/diagnostics/stalled_outcomes.py:30-33`.

### D. Atoms not admitting a standalone useful current-control mapping

Window readiness, trace length, first-event ordering, no-earlier-event history, and the historical latch. Persistence is only sound after retaining its full 101-index onset window; the historical latch is only sound after retaining every possible trigger branch. Sources: `single_integrator/environment.py:195-222, 235-236`; `single_integrator/outcomes.py:9-21`.

### E. Is strict deadlock fully coverable by VI certificates?

**Yes for sound one-sided branch coverage, after explicit onset-window expansion:** every raw candidate in a triggering streak supplies exact control memberships, including the speed ball and exact goal/progress preimages. **No as a single instantaneous or logically equivalent VI condition:** first-event precedence, continuous persistence, and the historical latch must remain explicit, and (c_{VI}\le0) is necessary under the event but not sufficient to identify it. The timer/latch source is `single_integrator/environment.py:210-222, 235-236`.

### F. Is stalled deadlock fully coverable by VI certificates?

**Yes for sound one-sided branch coverage:** each of the 40 speed atoms and the final endpoint-progress atom has an exact set, and original timeout implies final unfinishedness. **No as a single instantaneous or logically equivalent VI condition:** the authoritative `other_timeout` first-event/history gate and the exact tail indexing must remain outside the VI. Sources: `single_integrator/diagnostics/stalled_outcomes.py:17-37`; caller `single_integrator/c1/evaluate_deadlock_union.py:113-118`.

### G. Does the final projection have the required optimality property?

The declared problem is the joint identity-metric Euclidean projection, so the exact optimizer satisfies \((w-u)^T(v-u)\le0\) for all \(v\in U(x)\), and the proposed Euclidean \(c_{VI}\) needs no metric modification (`single_integrator/cbf.py:96-127`). The literal SciPy return is only tolerance-certified, so an exact machine-level nonpositivity claim remains conditional on a separate numerical residual bound (`single_integrator/cbf.py:128-163`).

### H. Remaining theoretical blockers

1. A certificate over literal floating-point controls needs a rigorous treatment of the accepted feasibility, speed, stationarity, and complementarity tolerances; the repository currently checks tolerances but does not derive a VI error bound (`single_integrator/cbf.py:128-154`).
2. Historical strict deadlock cannot be evaluated from only the present recovered state/control; all possible onset windows must be retained (`single_integrator/environment.py:219-236`).
3. Stalled deadlock cannot drop or soften the categorical original-timeout gate, and its goal-error comparison must retain the implemented \(d_{811}\) versus \(d_{850}\) indexing (`single_integrator/diagnostics/stalled_outcomes.py:17-37`).
4. The goal and endpoint-progress bad-control sets are state/history dependent and generally nonconvex, although their Euclidean distances are analytic as given above. No source supports replacing them by fixed low-control balls.
5. First-event precedence and no-earlier-event history are not projection-optimality statements and require separate logical handling (`single_integrator/environment.py:219-228`; `single_integrator/outcomes.py:9-21`).

All operational event thresholds, indices, dynamics, and the active SLSQP projection path were resolved from the repository. The only explicitly **UNRESOLVED** implementation detail is the internal cone-solver selection of the separate CVXPYLayers/diffcp path because the call supplies tolerances but no `solve_method` (`single_integrator/c1/socp.py:41-49`); that path is not used by the current deadlock-union operational evaluator.
