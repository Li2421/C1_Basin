"""Train one CPU-only MACFlow-official joint 4-agent Stage-I pilot."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import argparse
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .flowbc_4a_agent import (
    ActorVectorField,
    DoubleBottleneckFlowBCAgent,
    MACFLOW_OFFICIAL,
    ModuleDict,
    TrainState,
    get_config,
    load_checkpoint,
    parameter_count,
    sample_bounded_actions,
    save_checkpoint,
)
from .flowbc_4a_dataset import FlowBC4ADataset, fit_train_normalization


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "diagnostics" / "double_bottleneck_expert_dataset"
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics" / "double_bottleneck_macflow_pilot_v2"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _source_hashes() -> dict[str, str]:
    sources = {
        "double_bottleneck_agent": REPO_ROOT / "double_bottleneck/flowbc_4a_agent.py",
        "double_bottleneck_dataset": REPO_ROOT / "double_bottleneck/flowbc_4a_dataset.py",
        "double_bottleneck_trainer": REPO_ROOT / "double_bottleneck/train_flowbc_4a.py",
        "toy_agent": REPO_ROOT / "flowbc/giveway_flowbc_agent.py",
        "macflow_networks": MACFLOW_OFFICIAL / "utils/networks.py",
        "macflow_flax_utils": MACFLOW_OFFICIAL / "utils/flax_utils.py",
    }
    return {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in sources.items()
    }


def _fixed_batches(dataset, batch_size: int, count: int):
    return tuple(dataset.sample(batch_size) for _ in range(count))


def _mean_loss(agent, batches, seed: int) -> float:
    root = jax.random.PRNGKey(seed)
    values = []
    for index, batch in enumerate(batches):
        loss, _ = agent.total_loss(
            batch, agent.network.params, rng=jax.random.fold_in(root, index)
        )
        values.append(float(loss))
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise FloatingPointError("non-finite fixed evaluation loss")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.steps, args.batch_size, args.log_interval, args.validation_batches) <= 0:
        raise ValueError("training arguments must be positive")
    if args.steps > 2000:
        raise ValueError("this pilot is capped at 2000 updates")
    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("the pilot must be CPU-only")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")

    train_data = FlowBC4ADataset(args.dataset, "train", seed=args.seed)
    val_data = FlowBC4ADataset(args.dataset, "val", seed=args.seed + 1)
    if train_data.environment_fingerprint != val_data.environment_fingerprint:
        raise ValueError("train/validation fixed-geometry fingerprints differ")

    config = get_config()
    config.update(fit_train_normalization(train_data))
    config["normalize"] = True
    config["environment_fingerprint"] = train_data.environment_fingerprint
    config["max_speed"] = float(train_data.config["max_speed"])
    example = train_data.sample(args.batch_size)
    agent = DoubleBottleneckFlowBCAgent.create(
        args.seed,
        jnp.asarray(example["observations"]),
        jnp.asarray(example["actions"]),
        config,
    )

    # Fail closed if a local stand-in is accidentally substituted.
    if not isinstance(agent.network, TrainState):
        raise TypeError("agent is not using MACFlow_Official TrainState")
    if not isinstance(agent.network.model_def, ModuleDict):
        raise TypeError("agent is not using MACFlow_Official ModuleDict")
    actor = agent.network.model_def.modules["actor_bc_flow"]
    if not isinstance(actor, ActorVectorField):
        raise TypeError("agent is not using MACFlow_Official ActorVectorField")

    fixed_train = _fixed_batches(train_data, args.batch_size, args.validation_batches)
    fixed_val = _fixed_batches(val_data, args.batch_size, args.validation_batches)
    run_config = {
        "schema": "double_bottleneck_macflow_stage_i_4a_pilot_run_v1",
        "arguments": _jsonable(vars(args)),
        "device": {
            "forced_platform": os.environ["JAX_PLATFORMS"],
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "dataset": {
            "root": str(train_data.root),
            "manifest_sha256": train_data.manifest_sha256,
            "environment_fingerprint": train_data.environment_fingerprint,
            "train_episodes": len(train_data.episodes),
            "train_families": len(train_data.family_names),
            "train_transitions": len(train_data),
            "val_episodes": len(val_data.episodes),
            "val_families": len(val_data.family_names),
            "val_transitions": len(val_data),
            "sampling": "uniform transitions, identical to Toy GiveWayDataset.sample",
        },
        "model": _jsonable(dict(agent.config)),
        "parameter_count": parameter_count(agent),
        "implementation": {
            "method": "macflow_official_stage_i_joint_flow_bc",
            "actor_class": f"{ActorVectorField.__module__}.{ActorVectorField.__name__}",
            "actor_source": inspect.getfile(ActorVectorField),
            "module_dict_class": f"{ModuleDict.__module__}.{ModuleDict.__name__}",
            "train_state_class": f"{TrainState.__module__}.{TrainState.__name__}",
            "toy_source_equivalent": True,
            "only_dimension_change": "joint observation 20->72; joint action 4->8",
        },
        "source_sha256": _source_hashes(),
        "scientific_scope": {
            "true_joint_4agent": True,
            "fixed_dyads": False,
            "fixed_geometry": True,
            "g_phi": False,
            "eta": False,
            "critic": False,
            "q_guidance": False,
            "distillation": False,
        },
    }
    (output / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

    initial_train = _mean_loss(agent, fixed_train, args.seed + 1000)
    initial_val = _mean_loss(agent, fixed_val, args.seed + 2000)
    best_val = initial_val
    best_step = 0
    best_agent = agent
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()

    def write_metrics(step, batch_loss, fixed_train_loss, fixed_val_loss, info=None):
        row = {
            "step": int(step),
            "batch_train_loss": float(batch_loss),
            "fixed_train_loss": float(fixed_train_loss),
            "fixed_val_loss": float(fixed_val_loss),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if info is not None:
            row.update(
                grad_max=float(info["grad/max"]),
                grad_min=float(info["grad/min"]),
                grad_norm=float(info["grad/norm"]),
            )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    write_metrics(0, initial_train, initial_train, initial_val)
    save_checkpoint(
        output / "best.pkl",
        agent,
        {"step": 0, "fixed_val_loss": initial_val, "run_config": run_config},
    )

    last_batch_loss = initial_train
    last_train = initial_train
    last_val = initial_val
    last_info = None
    for step in range(1, args.steps + 1):
        batch = train_data.sample(args.batch_size)
        agent, info = agent.update(batch, step)
        values = np.asarray([float(value) for value in info.values()])
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite training numerics at step {step}")
        last_info = info
        last_batch_loss = float(info["bc_flow_loss"])
        if step % args.log_interval == 0 or step == args.steps:
            last_train = _mean_loss(agent, fixed_train, args.seed + 1000)
            last_val = _mean_loss(agent, fixed_val, args.seed + 2000)
            write_metrics(step, last_batch_loss, last_train, last_val, info)
            if last_val < best_val:
                best_val, best_step, best_agent = last_val, step, agent
                save_checkpoint(
                    output / "best.pkl",
                    agent,
                    {"step": step, "fixed_val_loss": last_val, "run_config": run_config},
                )

    elapsed = time.perf_counter() - started
    save_checkpoint(
        output / "final.pkl",
        agent,
        {"step": args.steps, "fixed_val_loss": last_val, "run_config": run_config},
    )

    probe_observation = jnp.asarray(fixed_val[0]["observations"][:4])
    probe_key = jax.random.PRNGKey(9127)
    before_reload = np.asarray(best_agent.sample_actions(probe_observation, probe_key))
    restored, restored_metadata = load_checkpoint(
        output / "best.pkl", train_data.environment_fingerprint
    )
    after_reload = np.asarray(restored.sample_actions(probe_observation, probe_key))
    reload_max_abs = float(np.max(np.abs(before_reload - after_reload)))
    bounded = np.asarray(sample_bounded_actions(restored, probe_observation, probe_key))
    max_bounded_speed = float(np.linalg.norm(bounded, axis=-1).max())
    if reload_max_abs != 0.0 or not np.isfinite(bounded).all():
        raise RuntimeError("checkpoint reload/sample numeric validation failed")
    if max_bounded_speed > float(config["max_speed"]) + 1e-6:
        raise RuntimeError("speed-bounded inference exceeded max_speed")

    summary = {
        "status": "trained_macflow_joint_4agent_pilot_pending_rollout_evaluation",
        "steps": args.steps,
        "seed": args.seed,
        "backend": jax.default_backend(),
        "implementation": run_config["implementation"],
        "parameter_count": parameter_count(agent),
        "training_seconds": elapsed,
        "initial_fixed_train_loss": initial_train,
        "final_fixed_train_loss": last_train,
        "initial_fixed_val_loss": initial_val,
        "final_fixed_val_loss": last_val,
        "best_fixed_val_loss": best_val,
        "best_step": best_step,
        "train_loss_decreased": bool(last_train < initial_train),
        "val_loss_decreased": bool(best_val < initial_val),
        "checkpoint_reload_max_abs": reload_max_abs,
        "sample_shape": list(after_reload.shape),
        "max_bounded_sample_speed": max_bounded_speed,
        "last_gradient_finite": bool(
            last_info is not None
            and np.isfinite([float(value) for value in last_info.values()]).all()
        ),
        "best_checkpoint_metadata": _jsonable(restored_metadata),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
