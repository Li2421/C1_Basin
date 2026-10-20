from core import *
from analysis import possible,choose_evaluate,headroom
from audit_utils import source_inventory

def main():
    # Tie order must not falsely widen exact-known intervals.
    lo=np.array([[1.,1.,0.],[0.,0.,0.],[0.,1.,1.]]);np.testing.assert_array_equal(possible(lo,lo),[[True,False,False],[True,False,False],[False,True,False]])
    lo=np.array([[0.,1.,0.]]);hi=np.array([[1.,1.,0.]]);np.testing.assert_array_equal(possible(lo,hi),[[True,True,False]])
    # Known exact winner: point and identification bounds collapse identically.
    z=np.zeros((1,32,2,18));z[...,-1]=1;u=np.zeros_like(z);a=headroom(z,u)
    assert all(a[m]['mean']==0 and a[m]['CI95']==[0.,0.] and a[m]['CI_unknown_envelope']==[0.,0.] for m in ['controller','state'])
    # Genuine room survives both cross-fitting levels and the bootstrap.
    z[:]=0;z[...,0]=1;a=headroom(z,u)
    assert all(a[m]['mean']==1 and a[m]['CI95']==[1.,1.] and a[m]['CI_unknown_envelope']==[1.,1.] for m in ['controller','state'])
    # State menu specialization cannot masquerade as held-state controller quality.
    z[:]=0
    for s in range(32):z[0,s,:,s%16]=1
    a=headroom(z,u);assert a['state']['mean']==1 and a['controller']['mean']==0
    write(HERE/'ANALYSIS_CHECKS.json',dict(utc=utc(),status='PASS',checks=['deterministic exact ties','unknown possible winner intervals','zero headroom known S500','positive headroom both folds','state specialization held-state split'],simulation_steps=0));source_inventory('imports_analysis_checks.json')
if __name__=='__main__':main()
