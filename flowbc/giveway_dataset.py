"""GiveWay-v1 offline dataset adapter for MAC-Flow Stage-I training.

Split strategy: pair-based so A_FIRST and B_FIRST of the same initial condition
always land in the same split.
  train: pair 0–199   (400 episodes)
  val:   pair 200–224 ( 50 episodes)
  test:  pair 225–249 ( 50 episodes)

Each file uses its real T (variable-length); transitions are concatenated.
Files may also contain explicitly marked independent one-step SI interventions;
this adapter samples rows and makes no continuity assumption. Do not use those
files as contiguous sequences in a future sequence/trajectory learner.
Only observations and actions are used for Stage-I training; rewards,
next_observations and dones are retained but not used.
"""
import glob
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

SPLIT_PAIRS: Dict[str, range] = {
    "train": range(0, 200),
    "val":   range(200, 225),
    "test":  range(225, 250),
}


class GiveWayDataset:
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val", "test", "all"] = "train",
        rng: Optional[np.random.Generator] = None,
    ):
        data_dir = Path(data_dir)
        all_files = sorted(data_dir.glob("*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        # Filter to requested pair_id range
        if split == "all":
            selected = all_files
        else:
            pair_range = SPLIT_PAIRS[split]
            selected = [
                f for f in all_files
                if int(np.load(f, allow_pickle=False)["pair_id"]) in pair_range
            ]

        obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []
        n_pairs_seen = set()

        for path in selected:
            ep = np.load(path)
            T = int(ep["observations"].shape[0])   # real episode length
            obs_list.append(ep["observations"].astype(np.float32))          # [T,2,10]
            act_list.append(ep["actions"].astype(np.float32))               # [T,2,2]
            rew_list.append(ep["rewards"].astype(np.float32))               # [T,2]
            next_obs_list.append(ep["next_observations"].astype(np.float32)) # [T,2,10]
            done_list.append(ep["dones"])                                    # [T]
            n_pairs_seen.add(int(ep["pair_id"]))

        self.observations      = np.concatenate(obs_list,      axis=0)  # [M,2,10]
        self.actions           = np.concatenate(act_list,      axis=0)  # [M,2,2]
        self.rewards           = np.concatenate(rew_list,      axis=0)  # [M,2]
        self.next_observations = np.concatenate(next_obs_list, axis=0)  # [M,2,10]
        self.dones             = np.concatenate(done_list,     axis=0)  # [M]

        self._size      = len(self.observations)
        self._rng       = rng if rng is not None else np.random.default_rng(0)
        self.n_episodes = len(selected)
        self.n_pairs    = len(n_pairs_seen)
        self.split      = split

        print(
            f"[GiveWayDataset] split={split}  pairs={self.n_pairs}  "
            f"episodes={self.n_episodes}  transitions={self._size}  "
            f"obs{self.observations.shape}  act{self.actions.shape}"
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations":      self.observations[idx],        # [B,2,10]
            "actions":           self.actions[idx],             # [B,2,2]
            "rewards":           self.rewards[idx],             # [B,2]   (unused Stage-I)
            "next_observations": self.next_observations[idx],   # [B,2,10](unused Stage-I)
            "dones":             self.dones[idx],               # [B]     (unused Stage-I)
        }

    def __len__(self) -> int:
        return self._size

