"""Posthoc user revision: only gate the unchanged Brel field by contact conflict."""
from experiment import *
import numpy as np
from distributed_safety import local_basis
from shared_control.hard_projection import HardProjectionConfig
from shared_control.basis_families import BasisFields,get_basis_family
from dynamics_safety import LOOKAHEAD

def safe_distance(config,stage='single_integrator'):
    radius=config.agent_radius
    if stage=='paper_dynamics' and config.scene=='lit_doorway':radius+=LOOKAHEAD
    return 2*radius+config.agent_collision_margin+HardProjectionConfig().separation_buffer

def contact_radius(config,stage='single_integrator'):
    return 1.25*safe_distance(config,stage)

def contact_mask(positions,goals,tolerance,radius,force_all_open=False):
    p=np.asarray(positions,np.float64);g=np.asarray(goals,np.float64)
    if force_all_open:return np.ones(len(p),bool)
    arrived=np.linalg.norm(g-p,axis=-1)<=tolerance
    close=np.linalg.norm(p[:,None]-p[None,:],axis=-1)<=radius
    conflict=close & ~np.eye(len(p),dtype=bool) & ~(arrived[:,None]&arrived[None,:])
    return np.any(conflict,axis=1)

from functools import lru_cache
@lru_cache(maxsize=1)
def _contact_from_exact_bits(n,position_bits,goal_bits,tolerance_bits,radius_bits,force_all_open):
    p=np.frombuffer(position_bits,np.float64).reshape(n,2)
    g=np.frombuffer(goal_bits,np.float64).reshape(n,2)
    result=contact_mask(p,g,np.frombuffer(tolerance_bits,np.float64)[0],np.frombuffer(radius_bits,np.float64)[0],force_all_open)
    return result

def _cached_contact_mask(positions,goals,tolerance,radius,force_all_open=False):
    p=np.asarray(positions,np.float64);g=np.asarray(goals,np.float64)
    return _contact_from_exact_bits(len(p),p.tobytes(),g.tobytes(),np.float64(tolerance).tobytes(),np.float64(radius).tobytes(),force_all_open).copy()

def distributed_fields(positions,goals,safe,i,speed,tolerance,radius,force_all_open=False):
    fields=local_basis(positions,goals,safe,i,np.inf,speed)
    if not _cached_contact_mask(positions,goals,tolerance,radius,force_all_open)[i]:
        fields=fields.copy();fields[2]=0.
    return fields

def central_fields(positions,goals,safe,speed,tolerance,radius,basis=None,force_all_open=False):
    original=(basis or get_basis_family('orthoflow3')).compute(positions,goals,safe,speed)
    mask=contact_mask(positions,goals,tolerance,radius,force_all_open)
    relation=original.values[2].copy();relation[~mask]=0.
    return BasisFields(original.metadata,(original.values[0],original.values[1],relation))
