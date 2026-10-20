import math
import numpy as np
from ring_exchange.environment import Config,RingInstance,_angle_vector

def sample_instance(seed: int, split: str = "train", config: Config | None = None) -> RingInstance:
    """Draw one broad, safe, independently reproducible benchmark state.

    A global rotation, per-agent angular/radial variation, velocity, and goal
    angle all vary.  Adjacent start angles retain a generous minimum gap, so
    invalid collision states are never used as nominal training data.
    """
    cfg = config or Config()
    rng = np.random.default_rng(np.random.SeedSequence([32015491, int(seed)]))
    global_angle = float(rng.uniform(-math.pi, math.pi))
    anchors = global_angle + np.arange(4) * (math.pi / 2)
    start_theta = anchors + rng.uniform(-0.40, 0.40, size=4)
    start_radius = rng.uniform(1.40, 2.90, size=4)
    # Opposite-side goals with independent small angular and radial offsets.
    goal_theta = start_theta + math.pi + rng.uniform(-0.36, 0.36, size=4)
    goal_radius = rng.uniform(1.40, 2.90, size=4)
    positions = start_radius[:, None] * _angle_vector(start_theta)
    goals = goal_radius[:, None] * _angle_vector(goal_theta)
    headings = rng.uniform(-math.pi, math.pi, size=4)
    speeds = rng.uniform(0.0, 0.30, size=4)
    velocities = speeds[:, None] * _angle_vector(headings)
    return RingInstance(positions.astype(np.float64), velocities.astype(np.float64),
                        goals.astype(np.float64), split, int(seed), global_angle)

