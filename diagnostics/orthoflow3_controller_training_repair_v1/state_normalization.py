"""Source-TRAIN-only shared-channel scaling; never agent-slot-specific."""
import numpy as np


def normalize_entities(x, training_states):
    xx = {k: np.asarray(v).copy() for k, v in x.items()}
    active = np.asarray(training_states, int)
    am = x["agent_mask"] > 0
    n = am.shape[1]
    masks = {"agents": am,
             "pairs": am[:, :, None] & am[:, None, :] & ~np.eye(n, dtype=bool)[None],
             "obstacles": am[:, :, None] & (x["obstacle_mask"][:, None, :] > 0),
             "globals": np.ones(len(am), bool)}
    norms = {}
    for key, mask in masks.items():
        values = np.asarray(x[key][active][mask[active]], np.float64)
        center = values.mean(0)
        scale = np.maximum(values.std(0), .05)
        xx[key] = np.where(mask[..., None], (x[key]-center)/scale, 0).astype(np.float32)
        norms[key] = {"center": center.tolist(), "scale": scale.tolist(), "active_training_entities": len(values)}
    norms["protocol"] = {"training_state_indices": active.tolist(), "std_floor": .05,
                         "channel_statistics_shared_across_entity_slots": True,
                         "padding_and_diagonal_pairs_excluded": True, "labels_used": False}
    return xx, norms
