"""Conservative centralized reference for the Gap family.

One robot moves at a time. A* routes around walls and stationary robots.
This establishes joint feasibility, but is deliberately not the learned
controller or a scalable performance benchmark.
"""
from __future__ import annotations

import heapq
import math
import numpy as np
from single_integrator.environment import point_segment_distance, segment_distance
from .environment import BottleneckEnv
from .observation import policy_observation


class SequentialExpert:
    def __init__(self, grid_step=.25, clearance_buffer=.015):
        if grid_step <= 0 or clearance_buffer < 0:
            raise ValueError('invalid planner clearance/grid')
        self.grid_step, self.clearance_buffer = grid_step, clearance_buffer

    def route(self, env: BottleneckEnv, agent: int):
        """Route via the centre of the shared Gap-1 opening with parked peers."""
        c, geo = env.config, env.instance.geometry
        p, goal = env.positions[agent], env.goals[agent]
        if len(c.barrier_x) == 1 and len(c.openings[0]) == 1:
            direction = 1 if goal[0] > c.barrier_x[0] else -1
            gate_x, gate_y = c.barrier_x[0], c.openings[0][0][0]
            offset = c.barrier_thickness/2 + geo.margin + .12
            entry = np.array((gate_x-direction*offset,gate_y))
            exit = np.array((gate_x+direction*offset,gate_y))
            if (p[0]-gate_x)*direction < -offset:
                targets = (entry, exit, goal)
            elif (p[0]-gate_x)*direction < offset:
                targets = (exit, goal)
            else:
                targets = (goal,)
            parts=[]; start=p
            for target in targets:
                subpath = self._route_to(env, agent, start, target)
                if subpath is None: return None
                parts.extend(subpath[:-1]);start=target
            return np.asarray(parts+[targets[-1]])
        return self._route_to(env, agent, p, goal)

    def _route_to(self, env: BottleneckEnv, agent: int, p, goal):
        """Find a conservative single-leg path around stationary robots."""
        c, geo = env.config, env.instance.geometry
        others = np.delete(env.positions, agent, axis=0)
        required = 2*c.agent_radius+c.agent_collision_margin+self.clearance_buffer
        def points_ok(q):
            return geo.valid_points(q) & (np.linalg.norm(q[:,None,:]-others[None,:,:],axis=-1).min(axis=1) > required)
        def line_ok(a,b):
            return (geo.visible(a,b) and
                    np.min(point_segment_distance(others,a,b)) > required)
        if line_ok(p,goal):
            return np.stack((p,goal))
        h = self.grid_step
        xs = np.arange(-c.half_length+geo.margin+h/2,c.half_length-geo.margin,h)
        ys = np.arange(-c.half_height+geo.margin+h/2,c.half_height-geo.margin,h)
        nx, ny = len(xs), len(ys)
        coordinates = np.stack(np.meshgrid(xs,ys,indexing='ij'),axis=-1).reshape(-1,2)
        free = points_ok(coordinates).reshape(nx,ny)
        def connections(endpoint):
            nearby = np.linalg.norm(coordinates-endpoint,axis=1)
            candidates = np.argsort(nearby)[:100]
            return [(int(i//ny),int(i%ny),float(nearby[i])) for i in candidates
                    if free[i//ny,i%ny] and line_ok(endpoint,coordinates[i])][:12]
        source_links, goal_links = connections(p), connections(goal)
        if not source_links or not goal_links:
            return None
        destination = {(i,j):distance for i,j,distance in goal_links}
        offsets = [(di,dj,math.hypot(di,dj)*h) for di in (-1,0,1)
                   for dj in (-1,0,1) if di or dj]
        heap = []
        distances = {}; parents = {}
        for i,j,length in source_links:
            cell=(i,j); distances[cell]=length
            heapq.heappush(heap,(length+np.linalg.norm(coordinates[i*ny+j]-goal),length,cell))
        found = None
        while heap:
            _, cost, cell = heapq.heappop(heap)
            if cost != distances.get(cell):
                continue
            if cell in destination:
                found=cell; break
            i,j=cell
            for di,dj,length in offsets:
                a,b=i+di,j+dj
                if not (0 <= a < nx and 0 <= b < ny and free[a,b]):
                    continue
                # Do not cut through diagonal corners of inflated obstacles.
                if di and dj and (not free[i+di,j] or not free[i,j+dj]):
                    continue
                if di and dj and not line_ok(coordinates[i*ny+j],coordinates[a*ny+b]):
                    continue
                nxt=(a,b); new=cost+length
                if new < distances.get(nxt, math.inf):
                    distances[nxt]=new; parents[nxt]=cell
                    heapq.heappush(heap,(new+float(np.linalg.norm(coordinates[a*ny+b]-goal)),new,nxt))
        if found is None:
            return None
        cells=[]; cursor=found
        while True:
            cells.append(cursor)
            if cursor not in parents: break
            cursor=parents[cursor]
        cells.reverse()
        route = [p] + [coordinates[i*ny+j] for i,j in cells] + [goal]
        # Deterministic shortcutting keeps routes smooth and time manageable.
        reduced = [route[0]]
        index=0
        while index < len(route)-1:
            for tail in range(len(route)-1,index,-1):
                if line_ok(route[index],route[tail]):
                    reduced.append(route[tail]);index=tail;break
            else:
                return None
        return np.asarray(reduced)

    def rollout(self, env: BottleneckEnv):
        c=env.config
        state=[env.positions.copy()]
        observations=[]; actions=[]; min_clearance=math.inf; routes=[]
        pending=list(range(env.num_agents))
        skipped=0
        while pending:
            agent=pending.pop(0)
            path=self.route(env,agent)
            if path is None:
                pending.append(agent);skipped+=1
                if skipped>=len(pending):
                    return dict(success=False,reason='no_sequential_route',agent=agent,
                                pending=pending,states=np.asarray(state),observations=np.asarray(observations),actions=np.asarray(actions))
                continue
            skipped=0
            routes.append(path)
            for waypoint in path[1:]:
                while np.linalg.norm(env.positions[agent]-waypoint) > 1e-9:
                    remaining=waypoint-env.positions[agent]
                    u=np.zeros(env.action_shape)
                    u[agent]=remaining*min(1/c.dt,c.max_speed/max(np.linalg.norm(remaining),1e-30))
                    observations.append(policy_observation(env))
                    _,done,info=env.step(u, diagnose_stalls=False)
                    actions.append(u.copy());state.append(env.positions.copy())
                    min_clearance=min(min_clearance,info['swept_clearance'])
                    if done and info['termination'] == 'success':
                        return dict(success=True,reason='success',states=np.asarray(state),
                                    observations=np.asarray(observations),actions=np.asarray(actions),
                                    min_clearance=min_clearance,routes=routes)
                    if done:
                        return dict(success=False,reason=info['termination'],agent=agent,
                                    states=np.asarray(state),observations=np.asarray(observations),actions=np.asarray(actions))
                if env.done:
                    break
            if env.done:
                break
        success=env.done and env.termination=='success'
        return dict(success=success,reason=env.termination,states=np.asarray(state),
                    observations=np.asarray(observations),actions=np.asarray(actions),
                    min_clearance=min_clearance,routes=routes)
