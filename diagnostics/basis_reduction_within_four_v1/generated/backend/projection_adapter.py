"""Empty-set/N=1 interface extension of the identical certified projection.

For N>=2 with walls, dispatches to the exact protected implementation. No
controller, physical parameter, tolerance, or feasible-set change is made.
"""
from experiment import *
import numpy as np
from shared_control.hard_projection import HardProjectionConfig,CBFSolverError,project_velocity,barrier_constraints
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter,tighter_identical_socp,identical_nonlinear_polish
from single_integrator.filters import FilterResult

def empty_geometry_constraints(snapshot,cfg):
    p=np.asarray(snapshot['positions'],float);ww=np.asarray(snapshot['walls'],float);c=snapshot['config'];rows=[];lower=[];kinds=[]
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            d=p[i]-p[j];row=np.zeros_like(p);row[i]=2*d;row[j]=-2*d
            safe=2*c['agent_radius']+c['agent_collision_margin']+cfg.separation_buffer
            rows.append(row.ravel());lower.append(-cfg.gamma*(d@d-safe*safe));kinds.append(['pair',i,j])
    if len(ww):
        for i in range(len(p)):
            a=ww[:,0];ab=ww[:,1]-a;t=np.clip(np.sum((p[i]-a)*ab,-1)/np.sum(ab*ab,-1),0,1);d=p[i]-(a+t[:,None]*ab);dist=np.linalg.norm(d,axis=-1)
            if np.any(dist<=1e-14):raise CBFSolverError('undefined_wall_gradient',{})
            for j in range(len(ww)):
                row=np.zeros_like(p);row[i]=d[j]/dist[j];h=dist[j]-c['agent_radius']-c['wall_radius']-c['wall_collision_margin']-cfg.separation_buffer
                rows.append(row.ravel());lower.append(-cfg.gamma_wall*h);kinds.append(['wall',i,j])
    if not rows:
        # The exported solver's diagnostics require a nonempty linear array.
        # The always-true 0*u>=-1 row is exactly R^(2N), never active.
        rows=[np.zeros(p.size)];lower=[-1.];kinds=[['tautology']]
    return np.asarray(rows),np.asarray(lower),kinds

def constraints(snapshot,cfg):
    if len(snapshot['positions'])>=2 and len(snapshot['walls']):
        A,l,g=barrier_constraints(snapshot,cfg)
        kinds=[['pair',int(i),int(j)] for i,j in g['pair_indices']]+[['wall',i,j] for i in range(len(snapshot['positions'])) for j in range(len(snapshot['walls']))]
        return A,l,kinds
    return empty_geometry_constraints(snapshot,cfg)

class LiteratureSafetyFilter(CertifiedHardSafetyFilter):
    def __call__(self,snapshot,nominal_velocity):
        if len(snapshot['positions'])>=2 and len(snapshot['walls']):return super().__call__(snapshot,nominal_velocity)
        if max(self.config.gamma,self.config.gamma_wall)*snapshot['config']['dt']>1:raise ValueError('require gamma*dt<=1')
        A,l,kinds=empty_geometry_constraints(snapshot,self.config)
        speed=snapshot['config']['max_speed']
        try:velocity,status,diagnostics=project_velocity(nominal_velocity,A,l,speed,self.config)
        except CBFSolverError as e:
            try:velocity,status,diagnostics=tighter_identical_socp(nominal_velocity,A,l,speed,self.config)
            except CBFSolverError:velocity,status,diagnostics=identical_nonlinear_polish(nominal_velocity,A,l,speed,self.config,e.details.get('solver_output_candidate',nominal_velocity))
        diagnostics.update(interface_extension='N1_or_zero_walls',constraint_kinds=kinds)
        return FilterResult(velocity=velocity,status=status,diagnostics=diagnostics)
