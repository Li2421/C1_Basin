"""GiveWay-v1 Stage-I Flow-BC training script.

Usage (macflow_baseline conda env):
    conda run -n macflow_baseline python train.py --split train

Smoke-test mode (100 steps):
    conda run -n macflow_baseline python train.py --smoke_test
"""
import argparse
import json
from datetime import datetime
import os
import sys
import time
from pathlib import Path

import pickle

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent.parent
MAC_OFFICIAL = REPO_ROOT / "01_MACFlow_Baseline_Reproduction" / "MACFlow_Official"
DATA_DIR     = REPO_ROOT / "02_C1_Toy_GiveWay" / "datasets" / "give_way_v2" / "raw"
CKPT_DIR     = SCRIPT_DIR / "checkpoints"

sys.path.insert(0, str(MAC_OFFICIAL))

from giveway_dataset      import GiveWayDataset
from giveway_flowbc_agent import GiveWayFlowBCAgent, get_config
# ------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--position_only", action="store_true", help="Mask redundant previous-velocity inputs in SI; same network and loss")
    p.add_argument("--normalize", action="store_true", help="Train-only observation/action standardization; physical actions restored at inference")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--split",         type=str,   default="train",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--train_steps",   type=int,   default=100_000)
    p.add_argument("--log_interval",  type=int,   default=1_000)
    p.add_argument("--save_interval", type=int,   default=50_000)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    p.add_argument("--recovery_data_dir", type=Path)
    p.add_argument("--recovery_fraction", type=float, default=0.5)
    p.add_argument("--init_checkpoint", type=Path)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--smoke_test",    action="store_true",
                   help="Run only 100 steps to verify everything works")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.train_steps  = 100
        args.log_interval = 10

    if min(args.batch_size, args.train_steps, args.log_interval, args.save_interval, args.val_batches) <= 0:
        raise ValueError("Batch size, steps, intervals and val_batches must be positive")
    if not 0 < args.recovery_fraction < 1:
        raise ValueError("recovery_fraction must be strictly between 0 and 1")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    out_dir = args.out_dir or SCRIPT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out_dir}")
    np.random.seed(args.seed)

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = GiveWayDataset(
        data_dir=str(args.data_dir),
        split=args.split,
        rng=np.random.default_rng(args.seed),
    )

    validation = GiveWayDataset(str(args.data_dir), split="val", rng=np.random.default_rng(args.seed + 1)) if args.split == "train" else None
    recovery = GiveWayDataset(str(args.recovery_data_dir), split=args.split, rng=np.random.default_rng(args.seed + 10)) if args.recovery_data_dir else None
    recovery_val = GiveWayDataset(str(args.recovery_data_dir), split="val", rng=np.random.default_rng(args.seed + 11)) if recovery is not None and validation is not None else None
    recovery_val_batches = [recovery_val.sample(args.batch_size) for _ in range(args.val_batches)] if recovery_val is not None else []
    # Fixed validation samples and flow noise make checkpoint losses comparable.
    val_batches = [validation.sample(args.batch_size) for _ in range(args.val_batches)] if validation else []
    if not validation:
        print("Validation disabled: selected training split is not train; avoiding overlap.")

    # ── Startup summary ──────────────────────────────────────────────
    config = get_config()
    initial_state = None
    if args.init_checkpoint:
        with args.init_checkpoint.open("rb") as f:
            initial_state = pickle.load(f)
        if "config" in initial_state:
            config.update(initial_state["config"])
    if args.learning_rate is not None:
        config["lr"] = args.learning_rate
    if args.position_only:
        if initial_state is not None:
            raise ValueError("Position-only preprocessing must be trained from scratch")
        config['position_only'] = True
    if args.normalize:
        if initial_state is not None:
            raise ValueError("Normalization must be fitted from scratch, not changed on an existing checkpoint")
        if args.split != "train":
            raise ValueError("Normalization statistics require the training split")
        from normalization import fit_normalization
        config.update(fit_normalization(dataset, recovery, args.recovery_fraction))
    print("=" * 60)
    print(f"[Dataset]  split={args.split}  pairs={dataset.n_pairs}  "
          f"episodes={dataset.n_episodes}")
    print(f"           transitions={len(dataset)}  "
          f"(real T, not episodes×fixed_len)")
    print(f"           obs shape:  {dataset.observations.shape}")
    print(f"           act shape:  {dataset.actions.shape}")
    print(f"[Model]    use_critic={config['use_critic']}  "
          f"use_q_guidance={config['use_q_guidance']}  "
          f"use_distillation={config['use_distillation']}")
    print(f"           critic/Q/distill SKIPPED – Stage-I bc_flow_loss only")
    print(f"[Train]    steps={args.train_steps}  batch={args.batch_size}  "
          f"seed={args.seed}")
    print("=" * 60)

    # ── Example batch for model init ─────────────────────────────────
    ex_batch = dataset.sample(args.batch_size)
    ex_obs   = jnp.asarray(ex_batch["observations"])  # [B,2,10]
    ex_act   = jnp.asarray(ex_batch["actions"])       # [B,2,2]

    # ── Agent ────────────────────────────────────────────────────────
    agent = GiveWayFlowBCAgent.create(
        seed=args.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        config=config,
    )

    if initial_state is not None:
        agent = flax.serialization.from_state_dict(agent, initial_state["agent"])
        agent = agent.replace(rng=jax.random.PRNGKey(args.seed))

    (out_dir / "config.json").write_text(json.dumps(dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, model=config.to_dict()), indent=2))
    metrics_path = out_dir / "metrics.jsonl"
    best_val = float("inf")

    def save_checkpoint(name, step):
        path = out_dir / name
        with path.open("wb") as f:
            pickle.dump({"agent": flax.serialization.to_state_dict(agent), "config": config.to_dict(), "step": step, "split": args.split, "seed": args.seed, "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None, "recovery_data_dir": str(args.recovery_data_dir) if args.recovery_data_dir else None}, f)
        print(f"[Checkpoint] saved → {path}")
    t0   = time.time()
    last = t0
    last_log_step = 0

    # ── Training loop ────────────────────────────────────────────────
    for step in range(1, args.train_steps + 1):
        if recovery is None:
            batch = dataset.sample(args.batch_size)
        else:
            n_recovery = round(args.batch_size * args.recovery_fraction)
            original_batch = dataset.sample(args.batch_size - n_recovery)
            recovery_batch = recovery.sample(n_recovery)
            batch = {key: np.concatenate([original_batch[key], recovery_batch[key]], axis=0) for key in original_batch}
        agent, info = agent.update(batch, step)

        if step % args.log_interval == 0 or step == 1 or step == args.train_steps:
            elapsed  = time.time() - last
            loss_val = float(info["bc_flow_loss"])
            its      = step - last_log_step
            print(f"step {step:>7d} | bc_flow_loss={loss_val:.6f} | "
                  f"{its / elapsed:.1f} it/s")
            record = {"step": step, "train_bc_flow_loss": loss_val, "elapsed_seconds": time.time() - t0}
            if val_batches:
                losses = [float(agent.total_loss(b, agent.network.params, rng=jax.random.fold_in(jax.random.PRNGKey(args.seed + 2), i))[0]) for i, b in enumerate(val_batches)]
                val_loss = float(np.mean(losses))
                record["val_bc_flow_loss"] = val_loss
                print(f"             val_bc_flow_loss={val_loss:.6f}")
                selection_loss = val_loss
                if recovery_val_batches:
                    recovery_losses = [float(agent.total_loss(b, agent.network.params, rng=jax.random.fold_in(jax.random.PRNGKey(args.seed + 12), i))[0]) for i, b in enumerate(recovery_val_batches)]
                    recovery_loss = float(np.mean(recovery_losses))
                    record["recovery_val_bc_flow_loss"] = recovery_loss
                    selection_loss = (1 - args.recovery_fraction) * val_loss + args.recovery_fraction * recovery_loss
                    print(f"             recovery_val_bc_flow_loss={recovery_loss:.6f}")
                record["selection_val_loss"] = selection_loss
                if selection_loss < best_val:
                    best_val = selection_loss
                    save_checkpoint("best_val.pkl", step)
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            last = time.time()
            last_log_step = step

        if step % args.save_interval == 0 or step == args.train_steps:
            save_checkpoint(f"ckpt_{step:07d}.pkl", step)

    total = time.time() - t0
    print(f"\nDone. {args.train_steps} steps in {total:.1f}s  "
          f"({args.train_steps / total:.1f} it/s avg)")


if __name__ == "__main__":
    main()

