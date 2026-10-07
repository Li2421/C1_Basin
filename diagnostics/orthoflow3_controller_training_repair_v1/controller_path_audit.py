"""Zero continuation action replay: each actual checkpoint vs context runtime."""
import jax
import numpy as np
from .data import OUT, read, write, bd
from new_benchmark_common.macflow import load_checkpoint
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20


def main():
    jax.config.update("jax_enable_x64", False)
    states, physical = read(OUT / "states.json"), read(OUT / "physical.json")
    protocol = read(OUT / "protocol.json")
    native = bd.TrainingRuntime("ring_exchange", states, parent=True)
    base = native.agent
    result = {"task_continuations": 0, "phase": "action-only consistency check", "controllers": {}}
    for profile in protocol["profiles"]:
        ctx = h20.rc.RichRuntime("ring_exchange", profile["path"])
        alternate = base if not profile["path"] else load_checkpoint(
            profile["path"], expected_environment_fingerprint=protocol["environment_fingerprint"])[0]
        errors, first_errors, intervention_differences = [], [], []
        for s, p in zip(states[:4], physical[:4]):
            env = native.make_env()
            native.reset(env, s)
            other = ctx.core.reset(p["physical"])
            for seed in range(4):
                key = jax.random.fold_in(jax.random.PRNGKey(bd.FUTURE_ROOT), bd.state_token(s["uid"]))
                key = jax.random.fold_in(jax.random.fold_in(key, seed), 0)
                native.agent = base
                first = native.flow_world(env, key)
                first_errors.append(float(np.max(np.abs(first - ctx.base_flow(other, key)))))
                native.agent = alternate
                future = native.flow_world(env, key)
                actual_ctx = ctx.base_flow if ctx.alt_flow is None else ctx.alt_flow
                errors.append(float(np.max(np.abs(future - actual_ctx(other, key)))))
                intervention_differences.append(float(np.linalg.norm(future - first)))
        native.agent = base
        assert max(errors) < 1e-7 and max(first_errors) < 1e-7
        result["controllers"][profile["name"]] = {
            "checkpoint_sha256": profile["sha256"], "future_action_checks": len(errors),
            "native_vs_context_future_max_error": max(errors),
            "preserved_base_first_action_max_error": max(first_errors),
            "alternative_vs_base_action_L2_median": float(np.median(intervention_differences)),
            "jax_enable_x64": jax.config.x64_enabled}
        assert not jax.config.x64_enabled
    result["passed"] = True
    write(OUT / "controller_path_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
