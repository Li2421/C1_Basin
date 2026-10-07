# Ring Exchange

Ring Exchange is a four-agent, 2-D single-integrator benchmark for lateral
circulation consistency. Agents `A,B,C,D` begin around a central forbidden
disk and each receives a goal near the opposite side of a broad annulus. The
radial free width is 2.35 while a disk diameter is 0.32, so the central region
is not a one-agent passage.

The environment has no right-of-way or CW/CCW rule. A learned policy receives
canonical rows `A,B,C,D`, each with own state/goal, all other-agent relative
states, and central-obstacle/outer-boundary geometry (`[4,23]`). The final
five geometry entries are relative obstacle centre, obstacle radius, outer
radius, and signed obstacle clearance. Circulation signatures belong only to
analysis and centralized-expert dataset metadata.

```bash
PYTHONPATH=. uv run --with numpy --with pytest pytest -q ring_exchange/tests
PYTHONPATH=. uv run --with numpy python -m ring_exchange.pilot --count 30
```

`sample_initial_state(split, seed)` uses separate deterministic random streams
for `train`, `development`, and `test`. It varies global angle, angular/radial
starts, velocities, asymmetry, spacing, and goal offsets, while sampling only
safe nominal states. `CentralizedExpert` searches both `cw` and `ccw` joint
spatiotemporal route hypotheses and validates candidates via actual swept
environment replay. To deliberately include both valid modes in a dataset,
call `plan_hypothesis(env, CirculationHypothesis("cw"))` or `"ccw"`.

The expert-pilot command evaluates both alternatives on 30 independent
development states. On the implemented v1 configuration, this yielded 60/60
collision-free successes and successful CW plus CCW routes for every sampled
state. This validates expert reliability and geometric multimodality; it does
not train a policy or use test states.
