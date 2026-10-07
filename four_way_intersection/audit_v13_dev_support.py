"""Pre-training development timestep-0 support audit for v13.

This reads training and development trajectories only.  In particular it does
not instantiate ``JointTransitionDataset(..., 'test')`` or access test files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from new_benchmark_common.dataset import JointTransitionDataset
from new_benchmark_common.evaluation import _ood_distances


def run(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    train = JointTransitionDataset(dataset, "train", seed=0)
    dev = JointTransitionDataset(dataset, "dev", seed=1)
    # Nominal t=0 states are the deployment-relevant independent global draw;
    # dev recovery states are intentionally excluded from this initial-state
    # check.  Comparing them to a recovery-dominated transition bank would
    # confound *global initial coverage* with the intentionally local recovery
    # density, so the support bank is train nominal t=0 only.
    train_t0 = np.stack([trajectory.observations[0] for trajectory in train.trajectories if trajectory.source == "nominal"])
    dev_t0 = np.stack([trajectory.observations[0] for trajectory in dev.trajectories if trajectory.source == "nominal"])
    calibration = _ood_distances(train_t0[1::2], train_t0[::2])
    threshold = float(np.quantile(calibration, .99))
    distances = _ood_distances(dev_t0, train_t0)
    result = {"scope": "train plus development nominal t=0 only", "test_opened": False,
              "train_transitions": int(len(train)), "train_nominal_initial_states": int(len(train_t0)),
              "development_nominal_initial_states": int(len(dev_t0)),
              "threshold_train_nn_p99": threshold, "timestep_0_ood_rate": float(np.mean(distances > threshold)),
              "distance": {"min": float(distances.min()), "median": float(np.median(distances)),
                           "p95": float(np.quantile(distances,.95)), "max": float(distances.max())}}
    output.mkdir(parents=True)
    (output / "support.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("output")
    args = parser.parse_args(); print(json.dumps(run(args.dataset, args.output), indent=2))
