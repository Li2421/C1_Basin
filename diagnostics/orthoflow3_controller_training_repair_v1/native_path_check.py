"""Action/representation replay only: zero task continuations."""
import jax
import numpy as np
from .data import OUT, read, write, bd, CONTROLLERS
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20


def main():
    jax.config.update("jax_enable_x64", False)
    states = read(OUT / "states.json")
    physical = read(OUT / "physical.json")
    native = bd.TrainingRuntime("ring_exchange", states, parent=True)
    ctx = h20.rc.RichRuntime("ring_exchange")
    maximum = 0.
    stored_error = 0.
    changes64 = []
    for row, p in zip(states[:4], physical[:4]):
        env = native.make_env()
        native.reset(env, row)
        other = ctx.core.reset(p["physical"])
        for seed in range(4):
            jax.config.update("jax_enable_x64", False)
            key = jax.random.fold_in(jax.random.PRNGKey(bd.FUTURE_ROOT), bd.state_token(row["uid"]))
            key = jax.random.fold_in(jax.random.fold_in(key, seed), 0)
            a = native.flow_world(env, key)
            b = ctx.base_flow(other, key)
            maximum = max(maximum, float(np.max(np.abs(a-b))))
            jax.config.update("jax_enable_x64", True)
            wrong = native.flow_world(env, key)
            changes64.append(float(np.linalg.norm(a-wrong)))
            jax.config.update("jax_enable_x64", False)
        key = jax.random.fold_in(jax.random.PRNGKey(bd.CONDITIONING_FLOW_ROOT), bd.state_token(row["uid"]))
        stored_error = max(stored_error, float(np.max(np.abs(native.flow_world(env, key)-p["physical"]["flow"]))))
    assert maximum < 1e-7 and stored_error < 1e-6
    result = {"task_continuations": 0, "action_checks": 16,
              "native_vs_context_base_action_max_error": maximum,
              "stored_current_flow_reference_max_error": stored_error,
              "accidental_x64_action_L2_median": float(np.median(changes64)),
              "flow_sampler_precision": "float32", "safety_physical_numpy_precision": "float64",
              "passed": True}
    write(OUT / "native_path_check.json", result)
    print(result)


if __name__ == "__main__":
    main()
