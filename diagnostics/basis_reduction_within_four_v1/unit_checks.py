"""Mechanism and candidate-projection checks before any closed-loop results."""
from core import *
from admission import AdmissionState,apply,forward_distance
from combinations import initialize,project,ITEMS
from types import SimpleNamespace

def main():
 cfg=SimpleNamespace(agent_radius=.1,max_speed=.5,goal_tolerance=.05,agent_collision_margin=0.)
 gate=dict(centre=[0.,0.],axis=[1.,0.],half_length=.5,width=.3)
 menu=dict(R_e=.8,R_c=.5,rho='fcfs',K=None,c1=1.,c2=1.,c3=1.)
 p=np.array([[-.9,0.],[.9,0.]]);g=-p;base=np.array([[.4,.1],[-.4,.1]]);rem=np.ones(2)*1.8
 out,meta=apply(base,p,g,rem,None,cfg,[],AdmissionState(0),menu);assert out.tobytes()==base.tobytes()and meta==[]
 # A robot inside the first rectangle must cross into the buffer, even
 # while requesting an oppositely held second rectangle.
 gates=[dict(gate,centre=[0.,0.],half_length=.2),dict(gate,centre=[1.,0.],half_length=.2)]
 p=np.array([[.1,0.],[1.1,0.]]);g=np.array([[2.,0.],[0.,0.]])
 state=AdmissionState(2);state.states[0]['direction']=1;state.states[1]['direction']=-1
 out,meta=apply(base,p,g,rem,None,cfg,gates,state,menu)
 assert meta[1]['request'][0]and meta[1]['inside_any'][0]and not meta[1]['waiting'][0]
 assert np.array_equal(out[0],base[0])
 p=np.array([[-.2,0.],[.2,0.]]);g=np.array([[1.,0.],[-1.,0.]])
 state=AdmissionState(1);state.states[0]['direction']=1
 out,meta=apply(base,p,g,rem,None,cfg,[gate],state,menu)
 assert meta[0]['reverse'][1]and np.array_equal(out[1],[.5,0.])
 # Completed agent inside must move away from the incoming route end.
 p=np.array([[0.,0.],[-.5,0.]]);g=np.array([[0.,0.],[1.,0.]])
 out,meta=apply(base,p,g,np.array([0.,1.5]),None,cfg,[gate],AdmissionState(1),menu)
 assert meta[0]['goal_yield'][0]and np.array_equal(out[0],[.5,0.])
 p=np.array([[0.,0.],[.5,0.]]);g=np.array([[0.,0.],[-1.,0.]])
 out,meta=apply(base,p,g,np.array([0.,1.5]),None,cfg,[gate],AdmissionState(1),menu)
 assert np.array_equal(out[0],[-.5,0.])
 # Both entering ends occupied: closest exit to the completed agent.
 p=np.array([[-.1,0.],[-.5,0.],[.5,0.]]);g=np.array([[-.1,0.],[1.,0.],[-1.,0.]])
 out,meta=apply(np.zeros((3,2)),p,g,np.array([0.,1.5,1.5]),None,cfg,[gate],AdmissionState(1),menu)
 assert np.array_equal(out[0],[-.5,0.])
 # Mouth displacement has doubled perpendicular component.
 p=np.array([[.7,.15],[0.,0.]]);g=np.array([[.7,.15],[1.,0.]])
 out,meta=apply(base,p,g,np.array([0.,1.]),None,cfg,[gate],AdmissionState(1),menu)
 expected=np.array([.2,.3]);expected*=.5/np.linalg.norm(expected)
 assert meta[0]['goal_yield'][0]and np.allclose(out[0],expected,rtol=0,atol=1e-15)
 out,meta=apply(base,p,g,np.array([0.,0.]),None,cfg,[gate],AdmissionState(1),menu)
 assert not any(meta[0]['goal_yield'])and out.tobytes()==base.tobytes()
 # A blocker beyond 0.8m forward arc does not trigger.
 assert forward_distance(np.array([[0.,0.],[2.,0.]]),np.array([1.1,0.]),.8)>.2
 data=initialize()
 for combo in data['combinations']:
  assert len(combo['candidate_pool'])==64
  assert len({digest(x['values'])for x in combo['candidate_provenance']})==64
  assert combo['dimension']==len(combo['active'])-len(combo['bindings'])
  for x in combo['candidate_provenance']:
   v=x['values']
   if combo['bindings'].get('Q')=='G':assert v['Q']==min(v['G'],1.)
   if combo['bindings'].get('Q')=='T':assert v['Q']==min(abs(v['T']),1.)
   if combo['bindings'].get('C')=='R':assert v['C']==v['R']
 assert data['combinations'][0]['fixed']['coefficients']['T1']==-.75
 dump(HERE/'ADMISSION_UNIT_CHECK.json',dict(passed=True,environment_steps=0,checks=['inside any passage continues while requesting next','inverse occupant still exits','completed blocker exits away from entering side','two-sided requests use nearer end','mouth clearance doubles perpendicular component','goal-yield trigger ends without unfinished blocker','exact0.8m route clipping','no-passage byte identity','all64 projected candidates unique','three binding formulas','negative rotation main fixed point']))
 print('All new admission and projection unit checks passed')

if __name__=='__main__':main()
