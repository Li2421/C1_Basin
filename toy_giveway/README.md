# Toy Give-Way scenario namespace

`toy_giveway` is the explicit scenario entry point for the two-agent Toy
Give-Way benchmark.  It is intentionally a thin compatibility layer over the
historical `single_integrator` implementation.

This arrangement is non-destructive:

- `single_integrator.*` remains the single source of truth, so historical C1
  scripts, serialized metadata, source hashes, and imports remain valid;
- the new namespace contains no copied physics, projection, Flow-BC, monitor,
  or outcome logic;
- large checkpoints, datasets, caches, and diagnostics are neither moved nor
  duplicated;
- importing `toy_giveway` does not load JAX or a checkpoint.  Flow-BC and the
  evaluator are available through opt-in modules `toy_giveway.flowbc` and
  `toy_giveway.rollout`.

New scenario-facing code may use:

```python
from toy_giveway import Config, GiveWayEnv
from toy_giveway.safety import CBFSafetyFilter
```

Historical code may continue using:

```python
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFSafetyFilter
```

The exported classes are the exact same Python objects, not reimplementations
or subclasses.  Run the checkpoint-free integration smoke test with:

```bash
python3 -m toy_giveway.smoke --steps 16
```

Run the complete maintained regression suite with:

```bash
bash scripts/test.sh
```
