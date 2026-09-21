"""Initial-position distribution shared by demonstrations and evaluation."""
import numpy as np

INIT_X_NOISE = 0.03
INIT_Y_NOISE = 0.002


def sample_initial_positions(rng):
    positions = np.array([[-0.85, 0.0], [0.85, 0.0]], dtype=np.float32)
    for i in range(2):
        positions[i] += np.array([
            rng.uniform(-INIT_X_NOISE, INIT_X_NOISE),
            rng.uniform(-INIT_Y_NOISE, INIT_Y_NOISE),
        ], dtype=np.float32)
    return positions
