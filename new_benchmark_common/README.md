# New benchmark shared Stage-I pipeline

This package is the only shared code intended for the Four-Way Intersection
and Ring Exchange Stage-I baseline. Scenario directories retain their own
environment, expert, sampling distribution, observation construction, safety,
visualization, and analysis-only coordination labels.

## Collection

Implement `ScenarioProtocol` and call:

```python
collect_nominal_and_uniform(
    scenario, dataset_root, scenario_config=asdict(config),
    nominal_counts={"train": 120, "dev": 30, "test": 60},
    uniform_anchors=8, seed=0,
)
```

This performs independent nominal sampling for all three splits, and uniform
anchor perturb-and-requery for train/dev only. Failed expert queries are counted
and successful continuations receive immutable audit fields. The test split
cannot be used as recovery source.

For plants whose recovery state includes more than the numeric trajectory state
(for example goals, velocity, or timers), implement the optional
`recovery_state_at(trajectory, time)` scenario hook. It returns that physical
state before the common perturbation call. It must not return a crossing or
circulation label.

After a train/dev collision investigation, obtain a `Trajectory` for a valid
pre-collision rollout and use `targeted_requery_candidates(...)`. Store accepted
wall/obstacle continuations with source `targeted_wall_obstacle`; each record
requires source rollout/time, collision identity/distance, seed, and expert
result. Build Base-U and Base-U+W with exactly the same nominal/uniform seeds;
the latter appends only these audited records. Agent collision records have the
separate `targeted_agent` source.

## Training and evaluation

`train_stage1(dataset_root, output_root, ...)` opens train/dev only and trains a
true four-agent 8-D action density using official `ActorVectorField`,
`ModuleDict`, `TrainState`, conditional flow matching, Adam, and exactly ten
Euler sampler steps. It has no mode feature, recurrent state, action chunk,
critic, eta, or `G_phi` path.

For frozen evaluation, scenario code supplies a `RolloutAdapter` and calls
`evaluate_closed_loop`. It returns full-task success, wall/obstacle collision,
agent collision, timeout, episode length, timestep-0 OOD, and rollout OOD.
Call `teacher_forced_rmse` and `k_step_stability` for teacher RMSE and K =
1/5/10/25/50/100 divergence plus K=100 collision. The generic code exposes
raw rows so scenario-local visualization can create geometry-specific heatmaps.
