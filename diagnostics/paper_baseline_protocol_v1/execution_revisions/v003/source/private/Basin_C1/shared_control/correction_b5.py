"""Opt-in B5 targets. Safety, plant, and the supplied production G/P/R stay external.

ADD is affine in eta at fixed state, role, waypoint and recipe. BLEND is not.
No arrival mask or hard HOLD constraint is added by this module.
"""
from dataclasses import dataclass
import numpy as np
from shared_control.diagnostic_corrector import bounded_rows


def rotate(x):
    x = np.asarray(x, dtype=float)
    return np.stack((-x[..., 1], x[..., 0]), axis=-1)


def conflict_components(positions, radius):
    p = np.asarray(positions)
    adjacent = np.linalg.norm(p[:, None] - p[None, :], axis=-1) <= radius
    unseen = set(range(len(p)))
    groups = []
    while unseen:
        todo = [min(unseen)]; unseen.remove(todo[0]); group = []
        while todo:
            i = todo.pop(); group.append(i)
            found = sorted(unseen.intersection(np.flatnonzero(adjacent[i])))
            unseen.difference_update(found); todo.extend(found)
        groups.append(sorted(group))
    return groups


def tangent(recipe, positions, goal, speed, radius, filters):
    p = np.asarray(positions)
    if recipe == 'goal':
        return rotate(goal)
    out = np.zeros_like(p)
    if recipe == 'cluster':
        for ids in conflict_components(p, radius):
            q = p[ids] - p[ids].mean(axis=0)
            # One common scale per connected component; singleton has zero rotation.
            k = speed / max(speed, float(np.linalg.norm(q, axis=1).max()))
            out[ids] = k * rotate(q)
    elif recipe == 'wall':
        for i, f in enumerate(filters):
            rows = [(k, row) for k, row in enumerate(f.A) if f.kinds[k][0] == 'wall']
            if rows:
                # Tightest original wall CBF lower bound, deterministic row-order tie break.
                k, normal = max(rows, key=lambda item: float(f.lower[item[0]]))
                out[i] = speed * rotate(normal) / max(np.linalg.norm(normal), 1e-30)
    else:
        raise ValueError(recipe)
    return out


@dataclass(frozen=True)
class YieldConfig:
    yielder: int
    passer: int
    direction: str = 'hold'
    scale: float = 0.


def yield_menu(yielder, passer):
    if yielder == passer:
        raise ValueError('roles must be distinct')
    return [YieldConfig(yielder, passer)] + [
        YieldConfig(yielder, passer, d, s)
        for d in ('back', 'left', 'right', 'forward') for s in (.5, 1., 2.)]


class FrozenYield:
    def __init__(self, config, positions, goals, length_scale):
        self.config = config
        p, g = np.asarray(positions), np.asarray(goals)
        i, j = config.yielder, config.passer
        if i == j or not (0 <= i < len(p) and 0 <= j < len(p)):
            raise ValueError('invalid ordered roles')
        # Role-relative axis is frozen once, never re-oriented by an oracle.
        e = p[j] - p[i]
        if np.linalg.norm(e) < 1e-12:
            e = g[i] - p[i]
        if np.linalg.norm(e) < 1e-12:
            e = np.array([1., 0.])
        self.forward = e / np.linalg.norm(e)
        directions = {'hold': np.zeros(2), 'back': -self.forward,
                      'forward': self.forward, 'left': rotate(self.forward),
                      'right': -rotate(self.forward)}
        self.waypoint = p[i].copy() + config.scale * length_scale * directions[config.direction]

    def clear(self, positions, speed):
        v = np.zeros_like(positions, dtype=float)
        if self.config.direction != 'hold':
            i = self.config.yielder
            v[i] = bounded_rows((self.waypoint - positions[i])[None], speed)[0]
        return v


def target(safe, fields3, extra_t, role, positions, speed, eta, mode='add', keep=(0,1,2,3,4)):
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (5,) or not np.isfinite(eta).all():
        raise ValueError('finite eta[5] required')
    if mode not in ('add', 'blend'):
        raise ValueError(mode)
    # Keep the original summation order, including the final addition to safe.
    out = np.asarray(safe, dtype=float).copy()
    for i in range(len(out)):
        out[i] += sum(float(eta[k]) * fields3[i, k] for k in range(3) if k in keep)
    if 3 in keep and eta[3] != 0:
        out += eta[3] * extra_t
    if role is not None and 4 in keep and eta[4] != 0:
        i = role.config.yielder
        clear = role.clear(positions, speed)
        if mode == 'add':
            out[i] += eta[4] * (clear[i] - safe[i])
        else:
            out[i] = (1 - eta[4]) * out[i] + eta[4] * clear[i]
    return out
