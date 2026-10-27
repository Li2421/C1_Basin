"""Memoize canonical barrier_constraints on identical geometry/config bytes.

The solver, certificate, default parameters, retries and all constraints are
the original functions. No protected function or global is replaced.
"""
from experiment import canonical
import numpy as np
from shared_control.hard_projection import HardSafetyFilter,barrier_constraints,project_velocity,CBFSolverError
from single_integrator.filters import FilterResult
from projection_adapter import LiteratureSafetyFilter,empty_geometry_constraints
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import tighter_identical_socp,identical_nonlinear_polish

class CachedCanonicalPrimary(HardSafetyFilter):
    def __init__(self,config):super().__init__(config);self.last_key=None;self.last_problem=None
    def problem(self,snapshot):
        key=(np.asarray(snapshot['positions'],np.float64).tobytes(),np.asarray(snapshot['walls'],np.float64).tobytes(),canonical(snapshot['config']),canonical(self.config.to_dict()))
        if key!=self.last_key:
            matrix,lower,geometry=barrier_constraints(snapshot,self.config)
            kinds=[['pair',int(i),int(j)] for i,j in geometry['pair_indices']]+[['wall',i,j] for i in range(len(snapshot['positions'])) for j in range(len(snapshot['walls']))]
            self.last_key=key;self.last_problem=(matrix,lower,geometry,kinds)
        return self.last_problem
    def __call__(self,snapshot,nominal_velocity):
        dt=snapshot['config']['dt']
        if max(self.config.gamma,self.config.gamma_wall)*dt>1:raise ValueError('require gamma*dt <= 1 for the held-velocity certificate')
        matrix,lower,geometry,_=self.problem(snapshot)
        velocity,status,diagnostics=project_velocity(nominal_velocity,matrix,lower,snapshot['config']['max_speed'],self.config)
        diagnostics.update(num_agents=int(geometry['num_agents']),num_pair_constraints=len(geometry['pair_indices']),num_wall_constraints=int(np.prod(geometry['wall_h'].shape)),pair_indices=[list(pair) for pair in geometry['pair_indices']],pairwise_h=geometry['pairwise_h'].tolist(),min_wall_h=geometry['min_wall_h'])
        return FilterResult(velocity=velocity,status=status,diagnostics=diagnostics)

class MemoSafetyFilter(LiteratureSafetyFilter):
    def __init__(self,config=None):super().__init__(config);self.primary=CachedCanonicalPrimary(self.config);self.empty_key=None;self.empty_problem=None
    def empty(self,snapshot):
        key=(np.asarray(snapshot['positions'],np.float64).tobytes(),np.asarray(snapshot['walls'],np.float64).tobytes(),canonical(snapshot['config']),canonical(self.config.to_dict()))
        if key!=self.empty_key:
            problem=empty_geometry_constraints(snapshot,self.config);self.empty_key=key;self.empty_problem=problem
        return self.empty_problem
    def __call__(self,snapshot,nominal_velocity):
        if len(snapshot['positions'])>=2 and len(snapshot['walls']):return super().__call__(snapshot,nominal_velocity)
        if max(self.config.gamma,self.config.gamma_wall)*snapshot['config']['dt']>1:raise ValueError('require gamma*dt<=1')
        A,l,kinds=self.empty(snapshot);speed=snapshot['config']['max_speed']
        try:velocity,status,diagnostics=project_velocity(nominal_velocity,A,l,speed,self.config)
        except CBFSolverError as e:
            try:velocity,status,diagnostics=tighter_identical_socp(nominal_velocity,A,l,speed,self.config)
            except CBFSolverError:velocity,status,diagnostics=identical_nonlinear_polish(nominal_velocity,A,l,speed,self.config,e.details.get('solver_output_candidate',nominal_velocity))
        diagnostics.update(interface_extension='N1_or_zero_walls',constraint_kinds=kinds)
        return FilterResult(velocity=velocity,status=status,diagnostics=diagnostics)
    def linear_problem(self,snapshot):
        if len(snapshot['positions'])>=2 and len(snapshot['walls']):
            A,l,g,kinds=self.primary.problem(snapshot);return A,l,kinds
        return self.empty(snapshot)
