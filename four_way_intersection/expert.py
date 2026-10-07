"""Joint order-search expert for the open four-way intersection.

Orders are search hypotheses owned by the centralized expert.  They are never
placed in environment state or policy observations.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools, math, time
import numpy as np
from .environment import FourWayIntersectionEnv


@dataclass(frozen=True)
class CrossingOrder:
    order: tuple[int,int,int,int]
    def __post_init__(self):
        if set(self.order) != set(range(4)): raise ValueError("order must permute A,B,C,D")
    @property
    def label(self): return "->".join("ABCD"[i] for i in self.order)


def all_crossing_orders(): return tuple(CrossingOrder(x) for x in itertools.permutations(range(4)))


@dataclass(frozen=True)
class ExpertPlan:
    hypothesis: CrossingOrder; positions: np.ndarray; actions: np.ndarray; observations: np.ndarray
    terminal_reason: str; success: bool; collision: bool; timeout: bool; episode_steps: int; solve_time_seconds: float
    min_inter_agent_surface_distance: float; min_wall_clearance: float; candidate_count: int
    @property
    def mode_signature(self): return self.hypothesis.label
    @property
    def metadata(self):
        return {"expert":"centralized_crossing_order_search_v1","crossing_order":self.hypothesis.label,"candidate_count":self.candidate_count,"solve_time_seconds":self.solve_time_seconds}


class CentralizedExpert:
    """Enumerate orders and realize each with validated, temporal separation.

    The temporal realization is deliberately not a four-way stop: a successor
    is released once its predecessor has *cleared the open centre* (plus a
    small safety headway), not after the predecessor has reached its distant
    goal.  This retains a genuine crossing-order conflict while avoiding an
    artificial long queue at the approaches.
    """
    def __init__(self, route_speed: float=.78, release_gap: float=.34,
                 conflict_radius: float=.90, default_candidate_budget: int=6):
        self.route_speed=float(route_speed); self.release_gap=float(release_gap); self.conflict_radius=float(conflict_radius); self.default_candidate_budget=int(default_candidate_budget)
        if self.default_candidate_budget <= 0: raise ValueError("default_candidate_budget must be positive")
        if not np.isfinite(self.conflict_radius) or self.conflict_radius <= 0: raise ValueError("conflict_radius must be positive")

    def plan(self, env: FourWayIntersectionEnv, hypotheses=None, require_success=True):
        if env.step_count != 0 or env.done: raise ValueError("plan requires fresh active environment")
        # Full exhaustive enumeration is exposed for audits, but replaying all
        # 24 equivalent conservative schedules for every recovery query is
        # wasteful.  The default validates six geometry-ranked alternatives;
        # callers request a particular order when they need mode coverage.
        seeds=(self._ranked_candidates(env)[:self.default_candidate_budget] if hypotheses is None else tuple(hypotheses))
        t=time.perf_counter(); plans=[self._plan_one(env,x) for x in seeds]; elapsed=time.perf_counter()-t
        successful=[x for x in plans if x.success]
        if require_success and not successful: raise RuntimeError("No successful crossing order")
        best=min(successful or plans,key=lambda x:(x.episode_steps,-x.min_inter_agent_surface_distance,x.hypothesis.label))
        return ExpertPlan(**{**best.__dict__,"solve_time_seconds":elapsed,"candidate_count":len(seeds)})

    def plan_hypothesis(self, env, hypothesis): return self.plan(env,(hypothesis,))

    @staticmethod
    def _ranked_candidates(env):
        # Earlier arrival-to-centre gets a slight cost advantage.  Ties retain
        # a cyclic set of permutations, avoiding a hard-coded A/B/C/D priority.
        urgency=np.linalg.norm(env.positions,axis=1)
        base=tuple(np.argsort(urgency, kind="stable").tolist())
        rotations=[base[k:]+base[:k] for k in range(4)]
        reverse=tuple(reversed(base)); rotations += [reverse[k:]+reverse[:k] for k in range(4)]
        chosen=[]
        for order in rotations:
            h=CrossingOrder(tuple(order))
            if h not in chosen: chosen.append(h)
        return tuple(chosen)

    def _plan_one(self, source, hypothesis):
        if self.route_speed > source.config.max_speed+1e-12: raise ValueError("route_speed too high")
        # Each agent retains its continuous path but only one enters at a time.
        # No directional or identity priority is used: every permutation is valid
        # in this intentionally conservative realizability layer.
        starts, goals=source.positions.copy(), source.goals.copy()
        # Recovery queries can begin after one or more agents have genuinely
        # reached their goal.  They are permanent physical hold states, not
        # members of a residual crossing queue: moving them again creates an
        # artificial goal-departure target.  Nominal starts are outside this
        # tolerance, so ordinary order-search behavior is unchanged.
        held=np.linalg.norm(goals-starts,axis=1) <= source.config.goal_tolerance
        dt=source.config.dt; release_time=0.; curves=[]
        for a in hypothesis.order:
            if held[a]:
                continue
            distance=np.linalg.norm(goals[a]-starts[a]); duration=distance/self.route_speed
            begin=release_time; end=begin+duration; curves.append((a,begin,end))
            # The successor need only defer through the shared conflict area.
            # If a recovery start does not traverse the central disk, release
            # it after the closest-centre point; validation remains decisive.
            exit_fraction=self._conflict_exit_fraction(starts[a],goals[a])
            release_time=begin + exit_fraction*duration + self.release_gap
        final_time=max((end for _,_,end in curves),default=0.)
        steps=min(source.config.max_steps, math.ceil(final_time/dt)+2); actions=[]
        for k in range(steps):
            now=k*dt; u=np.zeros((4,2))
            for a,begin,end in curves:
                if begin <= now < end: u[a]=(goals[a]-starts[a])/max(end-begin,1e-12)
            actions.append(u)
        return self._validate(source,hypothesis,np.asarray(actions))

    def _conflict_exit_fraction(self, start, goal) -> float:
        """Last segment fraction inside the open central conflict disk."""
        start=np.asarray(start,dtype=np.float64); direction=np.asarray(goal,dtype=np.float64)-start
        length2=float(direction@direction)
        if length2 <= 1e-15: return 0.0
        # ||start + a*direction||^2 = r^2.  The final intersection is the
        # clearance event; clipping makes off-centre recovery paths safe.
        b=2*float(start@direction); c=float(start@start-self.conflict_radius**2)
        disc=b*b-4*length2*c
        if disc >= 0:
            roots=((-b-np.sqrt(disc))/(2*length2),(-b+np.sqrt(disc))/(2*length2))
            inside=[r for r in roots if 0 <= r <= 1]
            if inside: return float(max(inside))
        closest=float(np.clip(-(start@direction)/length2,0.,1.))
        return closest

    @staticmethod
    def _validate(source,hypothesis,scheduled):
        # Preserve an isolated observation representation supplied by a
        # compatible environment subclass.  Dynamics/actions remain world
        # frame here; representation adapters may convert them for data.
        env=source.__class__(source.config); env.restore_augmented_state(source.augmented_state())
        pos=[env.positions.copy()]; obs=[env.observation().copy()]; acts=[]; minpair=minwall=math.inf; term="timeout"
        for u in scheduled:
            o,_,done,info=env.step(u); acts.append(u.copy());pos.append(env.positions.copy());obs.append(o.copy());term=info["termination"]
            minpair=min(minpair,info["min_swept_agent_distance"]);minwall=min(minwall,info["min_swept_wall_distance"])
            if done:break
        summary=env.summary()
        return ExpertPlan(hypothesis,np.asarray(pos),np.asarray(acts),np.asarray(obs),term,bool(summary["collision_free_success"]),bool(summary["wall_collision"] or summary["agent_collision"]),term=="timeout",summary["episode_steps"],0.,float(minpair),float(minwall),1)


def rollout_expert(env, hypothesis=None, expert=None): return (expert or CentralizedExpert()).plan(env, None if hypothesis is None else (hypothesis,))
