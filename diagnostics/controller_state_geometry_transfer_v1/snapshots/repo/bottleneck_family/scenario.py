"""Literature-inspired Gap family, with serial gates and rectangular obstacles.

This is a new metric-space variant, not a reproduction of published map files.
All agents share the same bottleneck resources; no disconnected pair replicas.
"""
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math

import numpy as np
from single_integrator.environment import point_segment_distance, segment_distance


@dataclass(frozen=True)
class Config:
    num_agents: int = 10
    half_length: float = 8.0
    half_height: float = 4.0
    # Each barrier has one or more parallel openings: (center_y, width).
    barrier_x: tuple = (0.0,)
    openings: tuple = (((0.0, 0.5),),)
    barrier_thickness: float = 0.6
    extra_rectangles: tuple = ()  # (xmin, ymin, xmax, ymax)
    agent_radius: float = 0.16
    wall_radius: float = 4 / 600
    wall_collision_margin: float = 0.005
    agent_collision_margin: float = 0.0
    dt: float = 0.05
    max_speed: float = 0.5
    max_steps: int = 10000
    goal_tolerance: float = 0.08
    seed: int = 0
    split: str = 'dev'

    def __post_init__(self):
        if type(self.num_agents) is not int or self.num_agents < 2:
            raise ValueError('num_agents must be an integer >= 2')
        for key in ('half_length', 'half_height', 'barrier_thickness', 'agent_radius',
                    'dt', 'max_speed', 'goal_tolerance'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'invalid {key}')
        for key in ('wall_radius', 'wall_collision_margin', 'agent_collision_margin'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'invalid {key}')
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        if type(self.seed) is not int or self.seed < 0 or self.split not in ('train', 'dev', 'test'):
            raise ValueError('invalid seed/split')
        if len(self.barrier_x) < 1 or len(self.barrier_x) != len(self.openings):
            raise ValueError('one openings list required per barrier')
        margin = self.agent_radius + self.wall_radius + self.wall_collision_margin
        previous = -self.half_length
        for x, openings in zip(self.barrier_x, self.openings):
            if not math.isfinite(x) or not previous < x < self.half_length:
                raise ValueError('barriers must be ordered within the workspace')
            if x - previous <= self.barrier_thickness + 2 * margin:
                raise ValueError('insufficient room between barriers')
            previous = x
            if not openings:
                raise ValueError('each barrier needs a traversable opening')
            end = -self.half_height
            for y, width in openings:
                if not all(map(math.isfinite, (y, width))) or width <= 2 * margin + .02:
                    raise ValueError('opening too narrow for a physical robot and margins')
                if y - width/2 <= end or y + width/2 >= self.half_height:
                    raise ValueError('openings must be ordered, disjoint and inside workspace')
                end = y + width/2
        if self.half_length - previous <= self.barrier_thickness + 2 * margin:
            raise ValueError('insufficient right staging room')
        for rect in self.extra_rectangles:
            if len(rect) != 4 or not np.isfinite(rect).all():
                raise ValueError('rectangle must be finite xmin,ymin,xmax,ymax')
            x0, y0, x1, y1 = rect
            if not (-self.half_length <= x0 < x1 <= self.half_length and
                    -self.half_height <= y0 < y1 <= self.half_height):
                raise ValueError('rectangle outside workspace or inverted')

    def to_dict(self):
        return asdict(self)

    @property
    def physical_fingerprint(self):
        """Plant/geometry identity; rollout seed and split are state provenance."""
        payload = asdict(self)
        payload.pop('seed'); payload.pop('split')
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def rectangle_walls(rect):
    x0, y0, x1, y1 = rect
    vertices = np.array([(x0,y0), (x1,y0), (x1,y1), (x0,y1)], dtype=float)
    return np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)


class Geometry:
    def __init__(self, config):
        self.config = config
        rects = list(config.extra_rectangles)
        for x, openings in zip(config.barrier_x, config.openings):
            low = -config.half_height
            for y, width in openings:
                rects.append((x-config.barrier_thickness/2, low,
                              x+config.barrier_thickness/2, y-width/2))
                low = y+width/2
            rects.append((x-config.barrier_thickness/2, low,
                          x+config.barrier_thickness/2, config.half_height))
        self.rectangles = np.asarray(rects, dtype=float).reshape(-1, 4)
        outer = (-config.half_length, -config.half_height, config.half_length, config.half_height)
        self.walls = np.concatenate([rectangle_walls(outer)] + [rectangle_walls(r) for r in rects])
        self.margin = config.agent_radius + config.wall_radius + config.wall_collision_margin
        # Conservative inflated corners build a single-robot visibility roadmap.
        nodes = []
        delta = self.margin + .01
        for x0, y0, x1, y1 in rects:
            nodes.extend([(x0-delta,y0-delta), (x1+delta,y0-delta),
                          (x1+delta,y1+delta), (x0-delta,y1+delta)])
        self.nodes = np.asarray([p for p in nodes if self.valid_points(np.asarray([p]))[0]], dtype=float).reshape(-1,2)
        self.graph = self._graph(self.nodes)

    def valid_points(self, points):
        p = np.asarray(points, dtype=float)
        c = self.config
        result = (np.abs(p[:,0]) < c.half_length-self.margin) & (np.abs(p[:,1]) < c.half_height-self.margin)
        for x0, y0, x1, y1 in self.rectangles:
            inside = (p[:,0] >= x0) & (p[:,0] <= x1) & (p[:,1] >= y0) & (p[:,1] <= y1)
            result &= ~inside
        distance = point_segment_distance(p[:,None], self.walls[None,:,0], self.walls[None,:,1])
        return result & (distance.min(axis=1) > self.margin) & np.isfinite(p).all(axis=1)

    def visible(self, a, b):
        distances = segment_distance(a, b, self.walls[:,0], self.walls[:,1])
        return bool(distances.min() > self.margin)

    def _graph(self, nodes):
        graph = [[] for _ in nodes]
        for i in range(len(nodes)):
            for j in range(i):
                if self.visible(nodes[i], nodes[j]):
                    cost = float(np.linalg.norm(nodes[i]-nodes[j]))
                    graph[i].append((j,cost)); graph[j].append((i,cost))
        return graph

    def path(self, start, goal):
        """A conservative single-robot path witness, never a joint feasibility proof."""
        if not self.valid_points(np.array([start, goal])).all():
            return None
        if self.visible(start, goal):
            return np.asarray([start, goal])
        nodes = np.vstack((self.nodes, start, goal))
        graph = [list(row) for row in self.graph] + [[], []]
        for i in (len(nodes)-2, len(nodes)-1):
            for j in range(i):
                if self.visible(nodes[i], nodes[j]):
                    cost = float(np.linalg.norm(nodes[i]-nodes[j]))
                    graph[i].append((j,cost)); graph[j].append((i,cost))
        source, target = len(nodes)-2, len(nodes)-1
        distance = {source: 0.}; parent = {}; queue = [(0., source)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost != distance[i]:
                continue
            if i == target:
                order = [i]
                while i != source:
                    i = parent[i]; order.append(i)
                return nodes[order[::-1]]
            for j, length in graph[i]:
                new = cost + length
                if new < distance.get(j, math.inf):
                    distance[j] = new; parent[j] = i; heapq.heappush(queue, (new,j))
        return None


@dataclass
class Instance:
    config: Config
    geometry: Geometry
    positions: np.ndarray
    goals: np.ndarray
    paths: tuple

    def manifest(self):
        c = self.config
        left = int((self.positions[:,0] < c.barrier_x[0]).sum())
        return dict(schema='bottleneck_family_v1', config=c.to_dict(),
                    environment_fingerprint=c.fingerprint, positions=self.positions.tolist(),
                    goals=self.goals.tolist(), walls=self.geometry.walls.tolist(),
                    rectangles=self.geometry.rectangles.tolist(),
                    single_agent_paths=[p.tolist() for p in self.paths],
                    validation=dict(valid_initial_state=True, valid_goals=True,
                        single_agent_path_witnesses=True, joint_feasibility='not_certified',
                        controller_success='not_evaluated', required_barriers_per_agent=len(c.barrier_x),
                        opposing_group_sizes=[left,c.num_agents-left],
                        opposing_pairs_sharing_barriers=left*(c.num_agents-left)))


def build_instance(config=None):
    c = config or Config()
    geometry = Geometry(c)
    rng = np.random.default_rng(np.random.SeedSequence([c.seed, {'train': 11,'dev': 23,'test': 37}[c.split]]))
    spacing = max(2*c.agent_radius+c.agent_collision_margin+.12, .5)
    margin = geometry.margin + .15
    def slots(low, high):
        xx, yy = np.meshgrid(np.arange(low, high, spacing),
                             np.arange(-c.half_height+margin, c.half_height-margin, spacing))
        p = np.column_stack((xx.ravel(), yy.ravel()))
        p = p[geometry.valid_points(p)]
        return p[rng.permutation(len(p))]
    # Reserve approach space near each gate so parked agents do not form an
    # accidental permanent wall at high N before any policy acts.
    approach_reserve = 2.5
    left_slots = slots(-c.half_length+margin, c.barrier_x[0]-c.barrier_thickness/2-margin-approach_reserve)
    right_slots = slots(c.barrier_x[-1]+c.barrier_thickness/2+margin+approach_reserve, c.half_length-margin)
    # Independent, unique goal slots; all trips cross all serial barriers.
    nleft, nright = (c.num_agents+1)//2, c.num_agents//2
    if min(len(left_slots), len(right_slots)) < max(nleft,nright):
        raise ValueError('insufficient safe staging capacity; enlarge rooms explicitly, never shrink robots')
    # Interleaved, nested task prefixes: increasing N preserves existing tasks
    # on the same geometry/seed, so changing goals cannot confound size sweeps.
    positions = np.asarray([left_slots[i//2] if i % 2 == 0 else right_slots[i//2]
                            for i in range(c.num_agents)])
    goals = np.asarray([right_slots[-1-i//2] if i % 2 == 0 else left_slots[-1-i//2]
                        for i in range(c.num_agents)])
    paths = tuple(geometry.path(p,g) for p,g in zip(positions, goals))
    if any(path is None for path in paths):
        raise ValueError('no single-agent path witness; reject instance, do not label it deadlock')
    return Instance(c, geometry, positions, goals, paths)
