# Coupled Dual-Intersection eta stress test

This is a standalone diagnostic.  It does not import a learned policy, load a
checkpoint, create demonstrations, or train anything.  Its sole intervention
family is the repository's existing globally shared, three-dimensional eta:

```text
B_goal[i] = bound(goal[i] - position[i])
B_rel[i]  = mean_{j != i}(bound(position[i] - position[j]))
g[i]      = eta_goal*B_goal[i] + eta_safe*u_safe[i] + eta_relative*B_rel[i]
u_exec    = hard_project(u_safe + g)
```

The analytic nominal controller only follows fixed route waypoints.  It has no
priority, reservation, yield, phase, bridge, or deadlock-avoidance logic.

## Map and oracle

There are two vertical junctions connected by one narrow horizontal bridge.
Four horizontal agents exchange sides through both junctions; two vertical
agents cross each junction.  The `oracle.py` scheduler moves an explicitly
ordered sequence of agents.  It is a solvability certificate only and is never
called by the nominal/safety/eta controller.

The first command creates the immutable 96-IC corpus and verifies the oracle
gate plus the nominal and hard-safety comparisons:

```bash
"$C1_PYTHON" -m diagnostics.eta_complexity_stress_test.run --stage verify
```

Only after that gate succeeds, run the search study:

```bash
"$C1_PYTHON" -m diagnostics.eta_complexity_stress_test.run --stage search \
  --search-samples 512 --dense-samples 16384 --solver-budget 512
```

`data/benchmark_ics_v5.json` is written once, embeds the environment fingerprint
and its own digest, and refuses a different replacement.  Results go under
`results/`; they include raw trajectory arrays, machine-readable search data,
scatter/heatmap figures, oracle/safety/eta trajectories, and generated
`REPORT.md` with separate capacity and solver-difficulty verdicts.

The repository has no frozen eta bounds. The default domain
`[0,1.5] × [-1,1] × [-1,1]` is therefore an explicitly declared finite
experimental domain. Since every basis row is bounded at 0.5 m/s, the
triangle-bound on the pre-second-projection correction is 1.75 m/s; the
existing hard projection applies the final speed/CBF constraint. It is not a
claim of mathematical eta bounds.

An empty finite Sobol sample is reported as an empirical non-observation, never
as a proof that the continuous success set is empty.
