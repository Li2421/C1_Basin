"""Local, decentralized observation for learned planners (works for any number of robots).
Per robot: goal direction (2) + clipped goal distance (1), route direction from a navigation field (2),
own velocity (2), radius (1), 16 lidar ranges to walls (16), 8 nearest neighbours within 1 m x
[relative position (2), relative velocity (2), arrived (1), valid (1)] (48).  Total 72."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

K, RAYS, SENSE, LIDAR, WAY, RES = 8, 16, 1.0, 1.0, 0.5, 0.025
OBS_DIM = 3 + 2 + 2 + 1 + RAYS + 6 * K
ANG = np.arange(RAYS) * 2 * np.pi / RAYS
RAY_DIRS = np.stack([np.cos(ANG), np.sin(ANG)], -1)
OFFS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])


def seg_point_dist(p, walls):
    """p (M,2), walls (W,2,2) -> distances (M,W)."""
    a, b = walls[:, 0], walls[:, 1]
    ab = b - a
    t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0, 1)
    return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1)


def seg_seg_dist(P0, P1, A, B):
    """Distances between segments P0[c]-P1[c] and A[w]-B[w]; (C,W)."""
    def ps(x, a, b):  # point x (...,2) to segment a-b (...,2)
        ab = b - a
        t = np.clip(((x - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0, 1)
        return np.linalg.norm(x - (a + t[..., None] * ab), axis=-1)
    P0, P1, A, B = P0[:, None], P1[:, None], A[None], B[None]
    d = np.minimum(np.minimum(ps(P0, A, B), ps(P1, A, B)), np.minimum(ps(A, P0, P1), ps(B, P0, P1)))
    cr = lambda o, a, b: (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])
    d1, d2, d3, d4 = cr(A, B, P0), cr(A, B, P1), cr(P0, P1, A), cr(P0, P1, B)
    inter = (d1 * d2 < 0) & (d3 * d4 < 0)
    return np.where(inter, 0.0, d)


def lidar(p, walls, radius):
    """Free range along 16 rays from each robot centre, minus radius, clipped to [0, LIDAR]; (M,16)."""
    if len(walls) == 0:
        return np.ones((len(p), RAYS))
    a, ab = walls[:, 0], walls[:, 1] - walls[:, 0]
    d = RAY_DIRS
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    den = cross(d[None, :, None], ab[None, None])                      # (1,R,W)
    ap = a[None, None] - p[:, None, None]                              # (M,1,W,2)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = cross(ap, ab[None, None]) / den
        u = cross(ap, d[None, :, None]) / den
    ok = (np.abs(den) > 1e-12) & (s >= 0) & (u >= 0) & (u <= 1)
    s = np.where(ok, s, np.inf).min(-1)
    return np.clip(s - radius, 0, LIDAR) / LIDAR


class Field:
    """Geodesic distance to one goal over grid cells with wall clearance > radius + margin."""

    def __init__(self, walls, radius, margin, goal, lo, hi):
        nx, ny = int(np.ceil((hi[0] - lo[0]) / RES)) + 1, int(np.ceil((hi[1] - lo[1]) / RES)) + 1
        xs, ys = lo[0] + RES * np.arange(nx), lo[1] + RES * np.arange(ny)
        C = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2)
        free = np.ones(len(C), bool) if len(walls) == 0 else seg_point_dist(C, walls).min(1) > radius + margin
        self.lo, self.nx, self.ny, self.C, self.free = np.asarray(lo, float), nx, ny, C, free
        idx = np.full(nx * ny, -1)
        idx[free] = np.arange(free.sum())
        I, J = np.unravel_index(np.flatnonzero(free), (nx, ny))
        rows, cols, w = [], [], []
        for dx, dy in OFFS:
            I2, J2 = I + dx, J + dy
            ok = (I2 >= 0) & (I2 < nx) & (J2 >= 0) & (J2 < ny)
            nb = np.full(len(I), -1)
            nb[ok] = idx[np.ravel_multi_index((I2[ok], J2[ok]), (nx, ny))]
            m = nb >= 0
            rows.append(np.flatnonzero(m)); cols.append(nb[m]); w.append(np.full(m.sum(), RES * np.hypot(dx, dy)))
        n = int(free.sum())
        G = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.Cf = C[free]
        self.tree = cKDTree(self.Cf)
        g = int(self.tree.query(goal)[1])
        self.dist = dijkstra(G, indices=g)
        # steepest-descent pointer among 8 neighbours (stay if no neighbour is closer)
        nxt = np.arange(n)
        best = self.dist.copy()
        for dx, dy in OFFS:
            I2, J2 = I + dx, J + dy
            ok = (I2 >= 0) & (I2 < nx) & (J2 >= 0) & (J2 < ny)
            nb = np.full(n, -1)
            nb[ok] = idx[np.ravel_multi_index((I2[ok], J2[ok]), (nx, ny))]
            dn = np.where(nb >= 0, self.dist[np.maximum(nb, 0)], np.inf)
            better = dn < best
            best[better], nxt[better] = dn[better], nb[better]
        self.next = nxt
        self.goal = np.asarray(goal, float)

    def route_direction(self, p, walls, clear):
        """Unit direction of an any-angle route: the goal if it is visible with clearance, else the farthest
        visible point among route points up to 2 m along the steepest-descent path (8-neighbour paths alone
        bend up to 22.5 degrees off the straight line)."""
        c = self.tree.query(p)[1]
        pts = []
        for k in range(1, 81):
            c = self.next[c]
            if k % 4 == 0:
                pts.append(self.Cf[c])
        Q = np.concatenate([np.asarray(pts), self.goal[None]])
        if len(walls):
            eff = min(clear, float(seg_point_dist(p[None], walls).min()) * 0.99)
            vis = seg_seg_dist(np.broadcast_to(p, Q.shape), Q, walls[:, 0], walls[:, 1]).min(1) > eff
        else:
            vis = np.ones(len(Q), bool)
        j = len(Q) - 1 if vis[-1] else (int(np.flatnonzero(vis[:-1]).max()) if vis[:-1].any() else 0)
        r = Q[j] - p
        n = np.linalg.norm(r)
        return r / n if n > 1e-9 else None


_FIELDS = {}


def field_for(scene, walls, radius, margin, goal, lo, hi):
    key = (scene, round(float(radius), 4), round(float(goal[0]), 4), round(float(goal[1]), 4)) + tuple(np.round(np.r_[lo, hi], 2))
    f = _FIELDS.get(key)
    if f is None:
        if len(_FIELDS) > 4000:
            _FIELDS.clear()
        f = _FIELDS[key] = Field(walls, radius, margin, goal, lo, hi)
    return f


def bounds(walls, positions, goals):
    pts = [np.asarray(positions), np.asarray(goals)] + ([np.asarray(walls).reshape(-1, 2)] if len(walls) else [])
    P = np.concatenate(pts)
    return P.min(0) - 0.3, P.max(0) + 0.3


def observe(scene, positions, velocities, goals, walls, radius, margin, max_speed, goal_tolerance, lo, hi):
    """(N,72) observations for all robots from one physical state."""
    p, v, g = np.asarray(positions, float), np.asarray(velocities, float), np.asarray(goals, float)
    N = len(p)
    walls = np.asarray(walls, float).reshape(-1, 2, 2)
    out = np.zeros((N, OBS_DIM), np.float32)
    dg = g - p
    dn = np.linalg.norm(dg, axis=-1)
    gdir = dg / np.maximum(dn, 1e-9)[:, None]
    arrived = dn <= goal_tolerance
    route = np.zeros((N, 2))
    for i in range(N):
        r = None if dn[i] < 2 * goal_tolerance else field_for(scene, walls, radius, margin, g[i], lo, hi).route_direction(p[i], walls, radius + margin)
        route[i] = gdir[i] if r is None else r
    out[:, 0:2], out[:, 2] = gdir, np.minimum(dn, 2.0) / 2.0
    out[:, 3:5] = route
    out[:, 5:7] = v / max_speed
    out[:, 7] = radius / 0.1
    out[:, 8:8 + RAYS] = lidar(p, walls, radius)
    base = 8 + RAYS
    rel = p[None] - p[:, None]                      # rel[i,j] = p_j - p_i
    dist = np.linalg.norm(rel, axis=-1)
    np.fill_diagonal(dist, np.inf)
    for i in range(N):
        order = np.argsort(dist[i], kind='stable')[:K]
        for k, j in enumerate(order):
            if not np.isfinite(dist[i, j]) or dist[i, j] > SENSE:
                break
            o = base + 6 * k
            out[i, o:o + 2] = rel[i, j] / SENSE
            out[i, o + 2:o + 4] = (v[j] - v[i]) / max_speed
            out[i, o + 4] = float(arrived[j])
            out[i, o + 5] = 1.0
    return out, arrived
