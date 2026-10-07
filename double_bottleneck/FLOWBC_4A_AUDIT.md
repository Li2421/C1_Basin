# Joint 4-Agent Flow-BC Audit and Pilot Contract

Date: 2026-09-23

Status: **design/audit only**. No Double-Bottleneck Flow-BC code has been
implemented, no checkpoint has been created, and no training has been run.
Training remains blocked until the master explicitly confirms Gates A--C
(expert feasibility, physically genuine multimodality assessment, and pilot
dataset quality).

## 1. Decision

The minimal scientifically clean extension is a new, scenario-local Stage-I
conditional Flow-Matching policy that conditions on the complete four-agent
observation and generates all four agents' actions jointly:

\[
p_0(u_{A1},u_{A2},u_{B1},u_{B2}\mid o,\;\mathcal E_{\rm fixed}),
\qquad u\in\mathbb R^8.
\]

For the first fixed-geometry pilot, the concrete tensor contract is

```text
observations: [B, 4, 18] -> row-major flatten -> [B, 72]
flow noise:   [B, 8]
joint action: [B, 8]     -> reshape           -> [B, 4, 2]
```

The vector field is one fully connected joint network with an 8-D output. It
is not four independent per-agent policies. Although its base Gaussian has
independent coordinates, the conditioned nonlinear flow can create correlated
8-D action samples.

The current `FixedDyadFlowAdapter`, which invokes a frozen two-agent policy on
`(A1,B1)` and `(A2,B2)`, is explicitly rejected as the final four-agent
Flow-BC baseline. It is a useful plumbing/compatibility diagnostic only. Its
distribution factorizes at the two policy calls, omits the other four pairwise
interactions from the learned condition, and was trained on a different
geometry. Global hard projection after those calls does not turn it into a
learned joint four-agent behavior model.

## 2. Existing implementation audit

### 2.1 Frozen Toy Give-Way policy

`flowbc/giveway_flowbc_agent.py` hard-codes:

- `NUM_AGENTS=2`, `OBS_DIM=10`, `ACT_DIM=2`;
- an observation flattening `[B,2,10] -> [B,20]`;
- CFM base/data/interpolated variables of shape `[B,4]`;
- an `ActorVectorField` whose action/output dimension is 4;
- sampled output reshaping to `[B,2,2]`.

Its training objective is standard straight-line conditional flow matching:

\[
x_0\sim\mathcal N(0,I),\quad x_1=u_{\rm expert},\quad
x_t=(1-t)x_0+t x_1,\quad v^*=x_1-x_0,
\]

with MSE between the learned vector field and `v*`. Sampling integrates the
learned field with ten Euler steps. Critic, Q guidance, and distillation are
absent. This objective and network primitive can be retained; the old class,
constants, checkpoint, and loader cannot.

`flowbc/giveway_dataset.py` also assumes `[T,2,10]` observations and
`[T,2,2]` actions and uses Toy-specific pair IDs and splits. It must not be
silently reused for four-agent data. Likewise, `flowbc/train.py` constructs the
two-agent class and carries Toy-specific data defaults and checkpoint loading.

### 2.2 Double-Bottleneck plant interface

The scenario's authoritative order is defined in
`double_bottleneck/scenario.py`:

```text
global index:  0    1    2    3
agent name:    A1   A2   B1   B2
direction:     L->R L->R R->L R->L
```

`DoubleBottleneckEnv.observation()` returns one 18-D row per agent:

\[
o_i=[p_i,\;v_i,\;q_i-p_i,\;(p_j-p_i,v_j-v_i)_{j\ne i}],
\]

where the other-agent blocks use ascending **global** index, excluding `i`.
Thus the neural condition is `[4,18]`, or 72 dimensions after canonical
row-major flattening. The executed action is `[4,2]`, or 8 dimensions after
the same global-order flattening.

The physical position state has 8 scalars. The observation additionally uses
8 last-applied-velocity scalars; these velocities are policy memory/bookkeeping
for the single-integrator plant rather than inertial dynamics. Goals are fixed
for the present environment and appear through the four 2-D goal-error blocks.
The 72-D observation intentionally contains repeated relational information.
Keeping it is the smallest extension of the existing joint Flow-BC interface
and avoids inventing a second state encoder during the pilot.

At an `ActorVectorField` call, the MLP input is 81-D: 72 observation values,
the current 8-D flow state, and one flow-time scalar. Its output is 8-D.

### 2.3 Current rollout path is not the joint-policy pilot path

`double_bottleneck/controller.py` and `rollout.py` currently encode the
two-dyad compatibility protocol. In particular, the RNG includes `dyad_id`,
and rollout metadata hard-codes `flow_dyads=[[0,2],[1,3]]`. The controller also
performs Flow -> hard projection -> eta correction -> hard projection.

A true joint-Flow pilot must not pass through this path unchanged: doing so
would record false provenance and would evaluate the already projected/eta
pipeline instead of the raw speed-bounded Flow-BC. After Gates A--C, use a
scenario-local joint-policy runner (or a carefully generalized
Double-Bottleneck runner) with one joint 8-D flow key per physical step and no
`G_phi`. Raw Flow-BC validation and the later hard-safety baseline must remain
separately named evaluations.

## 3. Fixed geometry and environment encoding

The first pilot may omit an explicit geometry vector **only because every
training and evaluation episode uses exactly one fixed Double-Bottleneck
environment**. Absolute positions, velocities, goals, and inter-agent
relations are in the observation; the fixed wall layout can be absorbed by
network parameters. A constant geometry vector would add no within-dataset
information.

This is not geometry-general Flow-BC. A checkpoint must therefore contain and
enforce:

- the full serialized environment config;
- its environment fingerprint;
- observation schema/version and agent order;
- wall/goal construction source hashes (or equivalent dataset provenance);
- `fixed_geometry=true` and an explicit prohibition on arbitrary-geometry
  claims.

The loader/evaluator must reject a mismatched environment fingerprint rather
than warn and continue. It must also reject old 20-input/4-output checkpoints;
there is no shape-compatible or scientifically meaningful warm start from the
frozen two-agent head.

If later work varies geometry within the same five-rectangle topology, an
explicit, versioned geometry encoder becomes required. A possible starting
point is the independent layout/physics parameter vector (world/chamber/outer
dimensions, bottleneck length/width, agent and wall radii/margins, speed and
time scale), or a wall-segment encoder. That is a separate architecture and
dataset question and is outside this fixed-geometry pilot.

## 4. Ordering and permutation contract

The pilot uses exactly the canonical `[A1,A2,B1,B2]` order for observations,
actions, states, normalization statistics, files, inference outputs, and
metrics. Actions flatten as

```text
[u_A1_x,u_A1_y,u_A2_x,u_A2_y,u_B1_x,u_B1_y,u_B2_x,u_B2_y].
```

No runtime sorting by position, distance to bottleneck, or planned priority is
allowed. Such sorting would change output identity across time. The fixed order
does not prescribe passage priority; it only assigns tensor slots to physical
agents.

The initial MLP is not permutation-equivariant. Moreover, a correct agent
permutation cannot be implemented by merely permuting the four observation
rows because each row's three relative-agent blocks also have global-index
semantics. The pilot should therefore use no arbitrary permutation
augmentation. A future symmetry augmentation must transform positions, goals,
actions, agent identities, and every relative block consistently and must be
validated as an actual symmetry of the geometry and task.

Coordination-mode labels or passage-order metadata are for analysis only and
must not be policy inputs in this pilot. Conditioning on a supplied mode would
turn the problem into several commanded unimodal policies rather than the
desired unconditional joint multimodal baseline.

## 5. Dataset-to-training contract

Agent 2's artifact should remain trajectory-first even though Stage-I Flow-BC
trains on individual state-action rows. For every episode, the auditable raw
record should contain at least:

| Field | Required shape/meaning |
|---|---|
| `initial_state` | positions and last-applied velocities, `[4,4]` or an explicitly versioned equivalent |
| `positions` | full physical trajectory, `[T+1,4,2]` |
| `applied_velocities` | full velocity/bookkeeping trajectory, `[T+1,4,2]` |
| `observations` | pre-action observations, `[T,4,18]` |
| `actions` | **executed expert velocities**, `[T,4,2]` |
| `next_observations` | post-action observations, `[T,4,18]` |
| `dones` | `[T]`, true only at the recorded terminal transition |
| outcome fields | terminal reason, expert success, collision flags, timeout, episode length |
| clearance fields | episode minimum swept inter-agent surface distance and swept wall clearance |
| analysis metadata | initial-condition ID/regime, planner seed/initialization/hypothesis, and coordination-mode metadata if available |
| provenance | environment config/fingerprint, expert version/config, dataset schema, source hashes |

If the planner has a pre-filter proposal and a different executed safe action,
both may be logged, but `actions` used for behavior cloning must mean the
executed velocity that produced `positions[t+1]`. Audit arrays should preferably
remain float64; the training adapter can cast validated observation/action rows
to float32.

The adapter must accept only explicitly validated, collision-free successful
expert episodes for training. Failed attempts remain in the audit corpus with
their true labels but are not silently mixed into expert BC. Before exposing a
row, it must verify:

1. exact shapes, finite values, and canonical order;
2. one environment fingerprint across the selected split;
3. `T` consistency across every trajectory/action/terminal array;
4. radial action norm no greater than `max_speed` (within a declared tolerance);
5. `positions[t+1] = positions[t] + dt * actions[t]` within a tight tolerance;
6. stored observations agree with the environment observation construction;
7. no swept agent/wall collision is hidden in an episode labeled success;
8. all four goal errors satisfy the success rule at the successful terminal;
9. no exact duplicated trajectory is assigned extra sampling weight.

All alternative plans/modes from one identical initial condition must be kept
in the same train/validation/test group. Splitting individual trajectories
would leak the same start and nearly the same prefix across sets. Nearby
perturbation families should also be grouped where Agent 2 has a natural base
state identifier. Splits and per-regime/per-mode counts must be frozen in a
manifest before training.

For a small dataset, uniform sampling over concatenated transitions can let
long or deliberately waiting trajectories dominate. The recommended pilot
sampler is: choose an initial-condition group, choose one validated rollout
within that group, then choose a timestep. This preserves available modes
without multiplying the weight of a state merely because more optimizer seeds
were tried or an episode lasted longer. Any regime or mode reweighting must be
explicitly reported; it must not be used to manufacture multimodality.

The present environment resets initial velocities to zero. The observation
still includes last-applied velocity and it varies along each trajectory. The
dataset must not claim nonzero **initial**-velocity coverage unless the plant
interface is explicitly extended and validated elsewhere.

## 6. Minimal isolated implementation after gate approval

No shared Toy module needs modification. After approval, the clean file-level
extension is scenario-local, for example:

```text
double_bottleneck/flowbc_4a_agent.py    # 72-condition / 8-action CFM
double_bottleneck/flowbc_4a_dataset.py  # validated episode/group splits
double_bottleneck/train_flowbc_4a.py    # pilot-only training/provenance
double_bottleneck/evaluate_flowbc_4a.py # raw joint-policy rollouts
```

The agent may reuse `ActorVectorField`, `ModuleDict`, and `TrainState` from the
same external MACFlow implementation, but it must define new 4-agent constants
and a new checkpoint schema. The public inference contract should be

```python
sample_actions(observations: Array[B,4,18], seed) -> Array[B,4,2]
```

with strict shape/finite checks. The CFM construction becomes
`x_0,x_1,x_t,v_fwd: [B,8]`, while `t: [B,1]`. Observation and action
normalization statistics are fitted on the training split only, have lengths
72 and 8, and are stored in the checkpoint. The old `position_only` mask should
not be enabled: applied velocities are useful state/history features for
maintaining coordination decisions.

At physical inference, inverse-normalized samples are converted to
`[4,2]` and radially bounded per agent to `max_speed`. This action-space bound
is not the hard-safety baseline. The unfiltered pilot sends the radially bounded
joint action to the plant and reports all collisions; the separately labeled
hard-safety evaluation later applies the global four-agent projection once.

Use one joint random key per time step:

```text
episode_key = fold_in(PRNGKey(seed), rollout_id)
step_key    = fold_in(episode_key, absolute_step)
```

`step_key` generates one `[8]` base sample. There is no `dyad_id`. Common
random numbers may be used for paired controller comparisons, but outcome or
episode length must never enter the key.

## 7. Predeclared small pilot configuration

The pilot should preserve the existing Stage-I method rather than introduce a
new generative architecture:

| Item | Pilot value |
|---|---|
| objective | straight-line conditional flow-matching MSE only |
| condition/action | 72 / 8 |
| MLP | `(256,256,256)` GELU hidden layers, 8-D output |
| layer norm | off initially, matching the existing baseline |
| optimizer | Adam, learning rate `3e-4` |
| preprocessing | train-only observation and action standardization |
| flow integration | 10 Euler steps |
| batch | up to 256, reduced if the validated train set is smaller |
| seeds | one training seed for the first pilot |
| budget | short smoke/overfit test, then at most 20,000 pilot updates |
| selection | best fixed validation CFM loss; test split remains untouched |
| excluded | critic, Q guidance, distillation, `G_phi`, eta, hyperparameter sweep |

This is roughly the same small MLP family as the Toy policy (about 155k dense
parameters for the 81-D vector-field input). The 20k cap is a ceiling, not a
requirement to continue after validation has clearly failed or converged.
Checkpointing should retain the best validation model and record elapsed time,
device/shard count, and all provenance. Use at most one GPU shard for this
pilot.

No old checkpoint parameters should be loaded. In particular, expanding or
duplicating the old 4-D output head would falsely import the frozen dyad
factorization and would still leave the condition dimension incompatible.

## 8. Pilot validation plan

### 8.1 Unit and numerical checks

Before any full rollout:

- creation, loss, update, and sampling accept exactly `[B,4,18]` and
  `[B,4,2]` and reject old two-agent shapes;
- all losses, gradients, parameters, normalized values, and samples are finite;
- the network output, flow variables, and returned action have dimensions
  8, 8, and `[B,4,2]` respectively;
- normalization round-trips and checkpoint save/load reproduce samples for an
  identical key;
- identical observation/key gives identical action, while changed keys
  ordinarily change samples;
- a tiny batch can be deliberately overfit as a plumbing test;
- post-sampling radial bounds hold without invoking the hard safety filter;
- checkpoint/environment fingerprint mismatch is a hard error.

### 8.2 Distribution and expert comparisons

Validation CFM loss is necessary but not evidence of a useful policy. On
held-out expert states, draw multiple joint samples and report at least:

- per-agent and joint action-norm distributions;
- best-of-K and mean sample-to-expert action errors (clearly labeled as
  teacher-forced diagnostics, not trajectory success);
- cross-agent action covariance/correlation to confirm the model is not being
  evaluated as four independent policies;
- coverage of distinct expert action branches at states where Agent 2 has
  established genuine alternatives.

Mode labels must not be used to choose the closest sample at execution time.
They are post-hoc diagnostics only.

### 8.3 Closed-loop validation

Use the three initial-condition regimes and held-out perturbation groups, with
a small predeclared set of rollout noise seeds. For each rollout report:

- all-four-agent success, collision, deadlock, and timeout rates;
- completion time/episode length;
- minimum swept inter-agent surface distance and wall clearance;
- path length, excess path length/backtracking, waiting time, and action total
  variation (jitter);
- bottleneck entry/exit order and chamber occupancy signature;
- distance to the nearest successful expert trajectory using a declared
  time-aligned or dynamic-time-warped position metric.

The policy must be evaluated first as raw **speed-bounded joint Flow-BC**. Any
global hard-projection result is a separate safety baseline and must not be
used to hide raw-policy collisions in the Flow-BC report.

Plausible joint coordination means that a rollout completes safely without
rapidly switching incompatible passage decisions and that its order/occupancy
signature matches a physically successful expert mode. Pointwise action MSE
alone cannot establish this.

## 9. Multimodality caveat

An 8-D conditional flow can represent multimodal *instantaneous joint-action*
distributions, but the existing Stage-I protocol resamples base noise at every
physical time step. It has no episode-persistent latent mode. The observation's
positions and last-applied velocities may provide enough hysteresis after
trajectories diverge, but near a symmetric decision boundary independent
resampling can cause mode switching or jitter.

This must be measured, not hidden by fixing a planner mode at inference. If the
validated expert data is genuinely multimodal and the pilot repeatedly mixes
incompatible modes, the correct conclusion is that transition-level Flow-BC is
insufficient for this task. A trajectory model or an explicitly trained
persistent latent would then be a separate follow-up architecture. Reusing one
noise vector for an entire episode without training/validating that protocol
is not an automatic fix and must not be silently substituted.

Conversely, if Gate B finds only one robust physical coordination mode in the
relevant state distribution, the model may still be trained as joint BC, but
it must not be advertised as a demonstrated multimodal baseline.

## 10. Gate/status summary

- **Gate A (expert feasibility): not confirmed in this audit.** Agent 1 must
  establish high small-suite success with no systematic collision or wall
  violation.
- **Gate B (multimodality): not confirmed in this audit.** Agent 2 must show
  qualitatively distinct successful passage/occupancy orders, not numerical
  optimizer perturbations.
- **Gate C (pilot dataset quality): not confirmed in this audit.** The shapes,
  dynamics consistency, collision labels, group splits, coverage, duplicates,
  and provenance described above must pass.
- **Gate D (pilot joint Flow-BC): not started.** No code, training, checkpoint,
  or pilot rollout exists yet.

Therefore Agent 4's present status is **prepared but blocked by Gates A--C**.
It is not ready for a scientific eta/basin experiment, and the frozen two-dyad
adapter must not be cited as satisfying the missing joint four-agent baseline.
