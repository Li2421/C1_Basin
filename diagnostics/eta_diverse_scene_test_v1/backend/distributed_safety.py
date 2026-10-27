"""Independent 2D projections; no other agent's action enters a constraint."""
from experiment import *
import numpy as np
import jax
from poc_field_controller import FixedSafetySet
from projection_adapter import constraints
from shared_control.hard_projection import HardProjectionConfig
from shared_control.basis_families import OrthoFlow3BasisFamily,ORTHOFLOW3_SCALE
from shared_control.diagnostic_corrector import bounded_rows
jax.config.update('jax_enable_x64',False)

def neighbors(positions,i,radius):
    return [j for j in range(len(positions)) if j!=i and np.linalg.norm(positions[i]-positions[j])<=radius]

def local_basis(positions,goals,safe,i,radius,speed):
    goal=bounded_rows((goals[i]-positions[i])[None],speed)[0]
    perp=ORTHOFLOW3_SCALE*OrthoFlow3BasisFamily.raw_ortho_flow(goal[None],safe[i:i+1],speed)[0]
    jj=neighbors(positions,i,radius)
    rel=bounded_rows(positions[i]-positions[jj],speed).mean(0) if jj else np.zeros(2)
    return np.stack((goal,perp,rel))

def responsibility(scene,full_circle=False):
    return 1. if scene=='lit_doorway' or (scene=='lit_circle_five' and full_circle) else .5

class LocalSafety(FixedSafetySet):
    def __init__(self,snapshot,i,c=None,coefficient=None,problem=None):
        self.config=c or HardProjectionConfig();self.max_speed=snapshot['config']['max_speed'];self.calls=0
        scene=snapshot['config']['scene']
        coefficient=responsibility(scene) if coefficient is None else float(coefficient)
        assert coefficient in (.5,1.)
        if max(2*coefficient*self.config.gamma,self.config.gamma_wall)*snapshot['config']['dt']>1:
            raise ValueError('held-action safety condition violated')
        A,lower,kinds=problem or constraints(snapshot,self.config)
        rows=[];bounds=[];labels=[]
        for row,b,kind in zip(A,lower,kinds):
            if kind[0]=='pair' and i in kind[1:]:
                rows.append(row[2*i:2*i+2]);bounds.append(coefficient*b);labels.append(kind)
            elif kind[0]=='wall' and kind[1]==i:
                rows.append(row[2*i:2*i+2]);bounds.append(b);labels.append(kind)
            elif kind[0]=='tautology':
                rows.append(np.zeros(2));bounds.append(-1.);labels.append(kind)
        if not rows:rows=[np.zeros(2)];bounds=[-1.];labels=[['tautology']]
        self.A=np.asarray(rows,np.float64).reshape(-1,2)
        self.lower=np.asarray(bounds,np.float64);self.kinds=labels
        self.agent_index=i;self.coefficient=coefficient

def project_independent(snapshot,target,coefficient=None):
    cfg=HardProjectionConfig();problem=constraints(snapshot,cfg)
    filters=[LocalSafety(snapshot,i,cfg,coefficient,problem) for i in range(len(target))]
    results=[f.project(np.asarray(target[i:i+1],np.float64)) for i,f in enumerate(filters)]
    return np.concatenate([r[0] for r in results]),filters,[r[1] for r in results]
