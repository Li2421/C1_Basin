"""v7 lane-local physical representation for the Four-Way benchmark.

The plant, geometry, collision semantics and expert's world-frame planning are
unchanged.  This module only defines a fixed orthonormal coordinate chart for
each approach and deterministic action conversion at the policy boundary.
"""
from __future__ import annotations
import hashlib, json
import numpy as np
from .environment import FourWayIntersectionEnv
from .scenario import HEADINGS, RIGHTS

REPRESENTATION = "four_way_lane_local_physical_v7"


def world_to_local(vectors):
    """Project one [4,2] vector per canonical agent onto forward/right axes."""
    value=np.asarray(vectors,dtype=np.float64)
    if value.shape[-2:] != (4,2): raise ValueError("expected [...,4,2]")
    return np.stack((np.sum(value*HEADINGS,axis=-1),np.sum(value*RIGHTS,axis=-1)),axis=-1)


def local_to_world(vectors):
    """Rotate one [forward,right] vector per agent back to world coordinates."""
    value=np.asarray(vectors,dtype=np.float64)
    if value.shape[-2:] != (4,2): raise ValueError("expected [...,4,2]")
    return value[...,0,None]*HEADINGS + value[...,1,None]*RIGHTS


def representation_fingerprint(config) -> str:
    payload={"physical_config":config.to_dict(),"observation_action_representation":REPRESENTATION}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


class FourWayLaneLocalEnv(FourWayIntersectionEnv):
    """Physical Four-Way plant exposing lane-local observations.

    ``step`` intentionally remains world-action dynamics for expert
    compatibility.  Policy-facing callers use :meth:`step_local` or the v7
    rollout adapter, which rotates local outputs before calling this method.
    """
    representation=REPRESENTATION

    def observation(self):
        rows=[]
        for i in range(4):
            basis_f,basis_r=HEADINGS[i],RIGHTS[i]
            def project(vector): return np.asarray((vector@basis_f,vector@basis_r))
            relative=[]
            for j in range(4):
                if i != j:
                    relative.extend((project(self.positions[j]-self.positions[i]),project(self.velocities[j]-self.velocities[i])))
            rows.append(np.concatenate((project(self.positions[i]),project(self.velocities[i]),project(self.goals[i]-self.positions[i]),*relative)))
        return np.asarray(rows,dtype=np.float32)

    def step_local(self, local_velocity):
        return super().step(local_to_world(local_velocity))

    @property
    def lane_local_fingerprint(self): return representation_fingerprint(self.config)
