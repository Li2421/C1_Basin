"""Open four-way intersection with first-order disk dynamics and hard events."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any
import numpy as np

from .scenario import AGENT_NAMES, build_layout, nominal_positions

PAIR_INDICES = tuple((i, j) for i in range(4) for j in range(i + 1, 4))


@dataclass(frozen=True)
class Config:
    schema: str = "four_way_intersection_single_integrator_v1"
    dt: float = .05
    max_steps: int = 1100
    max_speed: float = .82
    agent_radius: float = .20
    # Keep the outer boundary well outside every declared approach draw.  With
    # radius .20, start distance 3.45 and longitudinal jitter .55, this gives
    # at least .55 m centre clearance at reset instead of the former .05 m.
    # The open central crossing geometry and all start/goal routes are unchanged.
    world_half_extent: float = 4.75
    nominal_start_distance: float = 3.45
    nominal_goal_distance: float = 3.45
    nominal_lateral_offset: float = .34
    # Kept for the shared hard-projection snapshot contract; the square
    # boundary itself has zero thickness in this isolated benchmark.
    wall_radius: float = 0.0
    wall_collision_margin: float = .005
    agent_collision_margin: float = 0.0
    goal_tolerance: float = .10
    longitudinal_jitter: float = .55
    lateral_jitter: float = .16
    global_lateral_jitter: float = .06
    initial_speed_jitter: float = .10
    terminate_on_collision: bool = True
    terminate_on_success: bool = True

    def __post_init__(self):
        if self.schema != "four_way_intersection_single_integrator_v1":
            raise ValueError("Unknown schema")
        vals = (self.dt, self.max_steps, self.max_speed, self.agent_radius, self.world_half_extent,
                self.nominal_start_distance, self.nominal_goal_distance, self.goal_tolerance)
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in vals):
            raise ValueError("all physical constants must be finite and positive")
        if self.nominal_start_distance >= self.world_half_extent - self.agent_radius:
            raise ValueError("start must fit in world")

    def to_dict(self) -> dict[str, Any]: return asdict(self)
    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def point_segment_distance(p, a, b):
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    d = b - a
    t = np.clip(np.sum((p-a)*d, axis=-1) / np.maximum(np.sum(d*d, axis=-1), 1e-30), 0., 1.)
    return np.linalg.norm(p - (a + t[..., None]*d), axis=-1)


class FourWayIntersectionEnv:
    """No hidden priority: collision detection is the only interaction rule."""
    num_agents, action_shape, pair_indices = 4, (4, 2), PAIR_INDICES

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        layout = build_layout(self.config)
        self.walls, self.wall_names, self.goals = layout.walls, layout.wall_names, layout.goals
        self.reset()

    def observation(self) -> np.ndarray:
        """[4,18]: p_i,v_i,goal_i-p_i then other p/v relative blocks by A,B,C,D."""
        rows=[]
        for i in range(4):
            rel=[]
            for j in range(4):
                if i != j: rel.extend((self.positions[j]-self.positions[i], self.velocities[j]-self.velocities[i]))
            rows.append(np.concatenate((self.positions[i], self.velocities[i], self.goals[i]-self.positions[i], *rel)))
        return np.asarray(rows, dtype=np.float32)

    def outside(self, p):
        p=np.asarray(p); h=self.config.world_half_extent-self.config.agent_radius
        return (np.abs(p) > h).any(axis=1)

    def distances(self, positions=None):
        p=self.positions if positions is None else np.asarray(positions)
        # boundary clearance of disk centres, one value per agent; retained as [N,1].
        h=self.config.world_half_extent-self.config.agent_radius
        wall=(h-np.abs(p)).min(axis=1, keepdims=True)
        pair=np.asarray([np.linalg.norm(p[i]-p[j])-2*self.config.agent_radius for i,j in self.pair_indices])
        return wall, pair

    def reset(self, positions=None, velocities=None):
        self.positions=np.asarray(nominal_positions(self.config) if positions is None else positions, dtype=np.float64).copy()
        self.velocities=np.zeros((4,2), dtype=np.float64) if velocities is None else np.asarray(velocities,dtype=np.float64).copy()
        if self.positions.shape != (4,2) or self.velocities.shape != (4,2): raise ValueError("state must be [4,2]")
        w,p=self.distances()
        if self.outside(self.positions).any() or w.min() <= self.config.wall_collision_margin or p.min() <= self.config.agent_collision_margin:
            raise ValueError("initial state must be safe")
        self.step_count=0; self.done=False
        self.first_success_step=self.first_wall_collision_step=self.first_agent_collision_step=None
        return self.observation()

    def snapshot(self) -> dict[str, Any]:
        return {"positions":self.positions.copy(), "last_applied_velocity":self.velocities.copy(), "goals":self.goals.copy(),
                "walls":self.walls.copy(), "pair_indices":np.asarray(self.pair_indices), "config":self.config.to_dict(), "step":self.step_count}

    def restore_augmented_state(self, state: dict[str, Any]):
        if state["config"] != self.config.to_dict(): raise ValueError("state/config mismatch")
        self.positions=np.asarray(state["positions"],dtype=np.float64).copy(); self.velocities=np.asarray(state["last_applied_velocity"],dtype=np.float64).copy()
        self.step_count=int(state["step"]); self.done=bool(state.get("done",False))
        self.first_success_step=state.get("first_success_step"); self.first_wall_collision_step=state.get("first_wall_collision_step"); self.first_agent_collision_step=state.get("first_agent_collision_step")
        return self.observation()

    def augmented_state(self):
        return {**self.snapshot(), "done":self.done, "first_success_step":self.first_success_step,
                "first_wall_collision_step":self.first_wall_collision_step, "first_agent_collision_step":self.first_agent_collision_step}

    def step(self, executed_velocity):
        if self.done: raise RuntimeError("reset before stepping")
        u=np.asarray(executed_velocity,dtype=np.float64)
        if u.shape != (4,2) or not np.isfinite(u).all() or np.linalg.norm(u,axis=1).max() > self.config.max_speed+1e-9: raise ValueError("invalid action")
        before=self.positions.copy(); self.positions += self.config.dt*u; self.velocities=u.copy(); self.step_count += 1
        wall,pair=self.distances(); wall_collision=bool(self.outside(self.positions).any() or wall.min() <= self.config.wall_collision_margin)
        raw_h=self.config.world_half_extent-self.config.agent_radius
        edge_clearance=np.stack((self.positions[:,1]+raw_h, raw_h-self.positions[:,0], raw_h-self.positions[:,1], self.positions[:,0]+raw_h),axis=1)
        nearest_wall_index=int(np.unravel_index(np.argmin(edge_clearance),edge_clearance.shape)[1])
        # Relative motion segment catches in-step disk crossings.
        swept=np.asarray([point_segment_distance(np.zeros(2), before[i]-before[j], self.positions[i]-self.positions[j])-2*self.config.agent_radius for i,j in self.pair_indices])
        agent_collision=bool(swept.min() <= self.config.agent_collision_margin)
        errors=np.linalg.norm(self.goals-self.positions,axis=1); success=bool((errors <= self.config.goal_tolerance).all())
        if success and self.first_success_step is None:self.first_success_step=self.step_count
        if wall_collision and self.first_wall_collision_step is None:self.first_wall_collision_step=self.step_count
        if agent_collision and self.first_agent_collision_step is None:self.first_agent_collision_step=self.step_count
        term="collision" if (wall_collision or agent_collision) and self.config.terminate_on_collision else ("success" if success and self.config.terminate_on_success else ("timeout" if self.step_count >= self.config.max_steps else "running"))
        self.done=term!="running"
        info={"step":self.step_count,"termination":term,"positions":self.positions.copy(),"velocities":u.copy(),"goal_errors":errors,
              "wall_distances":wall,"agent_pair_indices":np.asarray(self.pair_indices),"agent_surface_distances":pair,"swept_agent_distances":swept,
              "min_swept_wall_distance":float(wall.min()),"nearest_wall_identity":self.wall_names[nearest_wall_index],"nearest_wall_clearance":float(edge_clearance.min()),
              "min_swept_agent_distance":float(swept.min()),"wall_collision":wall_collision,"agent_collision":agent_collision,"task_success":success}
        return self.observation(), np.float32(-errors.sum()), self.done, info

    def summary(self):
        return {"success":self.first_success_step is not None, "collision_free_success":self.first_success_step is not None and self.first_wall_collision_step is None and self.first_agent_collision_step is None,
                "wall_collision":self.first_wall_collision_step is not None,"agent_collision":self.first_agent_collision_step is not None,"episode_steps":self.step_count,
                "first_success_step":self.first_success_step,"first_wall_collision_step":self.first_wall_collision_step,"first_agent_collision_step":self.first_agent_collision_step}

Env = FourWayIntersectionEnv
