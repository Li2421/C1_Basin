"""Compare N=50 Flow with held-out, non-opposing expert queue actions early in episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions


def run(dataset: Path, checkpoints: list[Path], output: Path, *, split="dev",
        steps=(0, 10, 25, 50, 100), seed=931):
    dataset, output = Path(dataset), Path(output)
    if split not in ("train", "dev"):
        raise ValueError("onset audit only supports train/dev")
    manifest = json.loads((dataset / "manifest.json").read_text())
    examples = []
    for row in manifest["files"]:
        if row["split"] != split or row["source"] != "nominal":
            continue
        with np.load(dataset / row["file"], allow_pickle=False) as data:
            initial = json.loads(str(data["initial_state_json"].item()))
            if initial.get("mode") != "full":
                continue
            observations = data["observations"]
            actions = data["actions"]
            for step in steps:
                if step < len(actions):
                    examples.append((row["rollout_id"], initial["direction"], step,
                                     observations[step].copy(), actions[step].copy()))
    if not examples:
        raise ValueError("no held-out full-density expert frames")
    obs = np.stack([item[3] for item in examples])
    results = {}
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint)
        agent, _ = load_checkpoint(checkpoint,
                                   expected_environment_fingerprint=manifest["environment_fingerprint"])
        predictions = np.asarray(sample_bounded_actions(agent, obs, jax.random.PRNGKey(seed)))
        rows = []
        for item, prediction in zip(examples, predictions):
            rollout_id, direction, step, _observation, expert = item
            expert_speed = np.linalg.norm(expert, axis=1)
            model_speed = np.linalg.norm(prediction, axis=1)
            held = expert_speed < 0.05
            active = expert_speed > 0.10
            rows.append(dict(rollout_id=rollout_id, direction=direction, step=step,
                expert_active=int(np.sum(active)), model_active=int(np.sum(model_speed > 0.10)),
                held_agents=int(np.sum(held)),
                held_false_active=int(np.sum(held & (model_speed > 0.10))),
                held_model_mean_speed=float(np.mean(model_speed[held])) if np.any(held) else None,
                active_model_mean_speed=float(np.mean(model_speed[active])) if np.any(active) else None))
        grouped = {}
        for direction in ("LR", "RL"):
            for step in steps:
                subset = [r for r in rows if r["direction"] == direction and r["step"] == step]
                if subset:
                    grouped[f"{direction}_t{step}"] = dict(frames=len(subset),
                        expert_active_mean=float(np.mean([r["expert_active"] for r in subset])),
                        model_active_mean=float(np.mean([r["model_active"] for r in subset])),
                        held_false_active_mean=float(np.mean([r["held_false_active"] for r in subset])),
                        held_model_mean_speed=float(np.mean([r["held_model_mean_speed"] for r in subset])))
        results[checkpoint.name + "_" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:8]] = dict(
            checkpoint=str(checkpoint), grouped=grouped, rows=rows)
    report = dict(schema="gap1_n50_heldout_onset_action_audit_v1",
                  dataset_manifest_sha256=hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
                  split=split, modes=["full"], steps=list(steps), seed=seed,
                  checkpoint_results=results,
                  caveat="One-step held-out actions; full task success is assessed separately")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    args = parser.parse_args()
    report = run(args.dataset, args.checkpoint, args.output, split=args.split)
    print(json.dumps({key: value["grouped"] for key, value in report["checkpoint_results"].items()},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
