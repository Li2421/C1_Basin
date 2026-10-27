"""Check joint-reference Jacobians and physical input normals offline."""
from experiment import *
import numpy as np
from dynamics_joint_reference import *
from states import config,walls

def main():
    c=config('lit_doorway');p=np.asarray([[-1.,.5],[-1.,-.5]]);theta=np.asarray([.2,-.1]);front=np.asarray([[.15,.04],[.12,-.03]])
    value,row=doorway_pair_value_and_normal(p,theta,front,0,1,c);epsilon=1e-6;derivatives=[]
    for k in range(4):
        delta=np.zeros(4);delta[k]=epsilon
        plus=doorway_pair_value_and_normal(p,theta,front+delta.reshape(2,2),0,1,c)[0]
        minus=doorway_pair_value_and_normal(p,theta,front-delta.reshape(2,2),0,1,c)[0]
        derivatives.append((plus-minus)/(2*epsilon))
    error=float(np.max(abs(row-derivatives)));assert error<1e-7
    T=unicycle_matrix(.2);z=dict(headings=np.asarray([[.2]]),safe=np.asarray([[T@np.asarray([.26,3.8])]]))
    A,kinds=local_rows(z,0,0,c,dict(A=[],kinds=[]));assert len(kinds)==2
    assert np.allclose(A@z['safe'][0,0],[-.26,-3.8])
    c=config('lit_crowded_random',2);p=np.asarray([[0.,0.],[.4,0.]]);v=np.asarray([[.1,0.],[-.1,0.]])
    Ai,li,ki=acceleration_problem(p,v,0,c,walls(c.scene));Aj,lj,kj=acceleration_problem(p,v,1,c,walls(c.scene))
    a=li[0]/Ai[0,0];safe=np.asarray([[a,0.],[-a,0.]])
    z=dict(positions=np.asarray([p]),velocities=np.asarray([v]),safe=np.asarray([safe]))
    rows=coupled_active_rows(z,0,c,walls(c.scene),[(Ai[:1],ki[:1]),(Aj[:1],kj[:1])])
    assert rows.shape==(1,4) and np.array_equal(rows[0],np.r_[Ai[0],-Ai[0]])
    write(HERE/'DYNAMICS_DIAGNOSTIC_TESTS.json',dict(utc=utc(),passed=True,joint_unicycle_gradient_max_error=error,physical_input_normals_checked=True,coupled_hocbf_pair_assembly_checked=True,new_rollouts=0))
    seal('DYNAMICS_DIAGNOSTICS.receipt.json',[HERE/n for n in ('DYNAMICS_DIAGNOSTICS.md','dynamics_joint_reference.py','dynamics_diagnostics.py','test_dynamics_diagnostics.py','DYNAMICS_DIAGNOSTIC_TESTS.json')],before_first_dynamics_rollout=True)
    print('Dynamics diagnostic component checks passed',flush=True)

if __name__=='__main__':main()
