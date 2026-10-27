import bootstrap
from experiment import *
from swap_geometry import *
from swap_safety import *
import jax
jax.config.update('jax_enable_x64',False)

def main():
    c=config('lit_swap8_free',2);snap=dict(positions=np.array([[-.051,0],[.051,0]]),walls=np.zeros((0,2,2)),config=c.to_dict(),moving_position=None)
    A,l,k=local_problem(snap,0);B,b,j=local_problem(snap,1);d=snap['positions'][0]-snap['positions'][1];hp=HardProjectionConfig()
    assert np.array_equal(A[0],2*d) and np.array_equal(B[0],-2*d)
    assert abs(l[0]+b[0]+hp.gamma*(d@d-(.1+hp.separation_buffer)**2))<1e-15
    s=dict(snap,positions=np.array([[-1.,0],[1.,0]]));assert local_problem(s,0)[2]==[['tautology']]
    pair_snapshot=dict(snap)
    f=SwapSafety(pair_snapshot,0);x,dx=f.project([[.5,0]]);f2=SwapSafety(pair_snapshot,1);y,dy=f2.project([[-.5,0]])
    swept=point_segment_distance(np.zeros(2),snap['positions'][0]-snap['positions'][1],snap['positions'][0]-snap['positions'][1]+.05*(x[0]-y[0]));assert swept>.1
    swapped=dict(snap,positions=snap['positions'][::-1]);xx,_=SwapSafety(swapped,1).project([[.5,0]]);assert np.max(abs(xx-x))<1e-4
    # Piecewise path determinism, initial clearance, bounded displacement.
    states=read(HERE/'FLOW_STATES_lit_swap8_moving.json');checks=[]
    for state in states:
        p,pieces=moving_path(state,320);solo=dict(state,N=1,positions=[state['positions'][0]],goals=[state['goals'][0]],parent_positions=state['positions']);q,sp=moving_path(solo,320)
        assert np.array_equal(p,q) and pieces==sp
        assert np.linalg.norm(np.diff(p,axis=0),axis=-1).max()<=.0125+1e-12
        assert np.linalg.norm(np.asarray(state['positions'])-p[0],axis=-1).min()>=.45
        assert all(abs(sum(piece[1]-piece[0] for piece in step)-.05)<1e-12 for step in pieces)
        checks.append(dict(state=state['index'],piecewise_steps=sum(len(step)>1 for step in pieces)))
    # Mathematical infeasibility restores safety; contradictory rows use minimax, not unknown.
    xr,dr=restore(np.zeros(2),np.array([[1.,0.]]),np.array([.6]),.5);assert dr['restoration_phase']=='speed_only' and xr[0]>=.6-1e-7
    xm,dm=restore(np.zeros(2),np.array([[1.,0.],[-1.,0.]]),np.array([1.,1.]),.5);assert dm['restoration_phase']=='safety_minimax' and abs(dm['maximum_violation']-1)<2e-6
    seal('SWAP_COMPONENTS.receipt.json',[HERE/n for n in ('swap_geometry.py','swap_safety.py','swap_runtime.py','test_swap_components.py')],passed=True,moving_checks=checks,reciprocal_sum=True,range_gate=True,numbering_difference=float(np.max(abs(xx-x))),speed_restoration=dr,safety_restoration=dm)
    print('Swap components PASSED',flush=True)
if __name__=='__main__':main()
