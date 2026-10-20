from analyze import oracle_bounds,evaluate
import numpy as np
# A controller's in-sample best arm can be worse on the held-out state half.
l=np.zeros((2,4,6,2));l[:,::2,2]=1;l[:,1::2,3]=1
point,low,high=oracle_bounds(l,l,[2,3]);assert point.mean()==0 and np.array_equal(point,low) and np.array_equal(point,high)
# Per-state oracle chooses on the other seed, never on its evaluation seed.
l=np.zeros((2,4,6,2));l[:,:,2,0]=1;l[:,:,3,1]=1
point,low,high=oracle_bounds(l,l,[2,3],True);assert point.mean()==0 and np.array_equal(point,low) and np.array_equal(point,high)
# Exact training ties respect deterministic first-arm tie breaking.
l=np.zeros((1,4,6,2));l[:,0,2,0]=1;l[:,0,3,1]=1
point,low,high=oracle_bounds(l,l,[2,3]);assert np.array_equal(point,low) and np.array_equal(point,high)
# Frozen state and controller choices index the physical arm, preserving both seeds.
l=np.arange(2*4*6*2).reshape(2,4,6,2).astype(float);c={m:{'controller':np.array([0,3]),'state':np.array([[0,1,2,3],[3,2,1,0]])} for m in ['A3','A0']};c['G_E']=np.array([0,3]);out,_=evaluate(l,l,c)
assert np.array_equal(out['A3 zero-shot'][0][0],l[0,:,2]) and np.array_equal(out['A3 per state'][0][1,0],l[1,0,5])
print('PASS: cross-fit state/seed separation, exact tie rule, frozen-choice indexing')
from analyze import summarize
l=np.ones((2,4,6,2));c={m:{'controller':np.array([0,3]),'state':np.array([[0,1,2,3],[3,2,1,0]])} for m in ['A3','A0']};c['G_E']=np.array([0,3]);ev={k:np.zeros(l.shape) for k in ['collision','timeout','unknown']};r=summarize('synthetic',l,l,ev,c,32)
assert all(x['success_pct']==[100.,100.] for x in r['methods'].values())
assert r['ST3']['CI_pp']==[0.,0.] and r['ST3']['status']=='undetected' and r['endpoint']['status']=='REJECTED'
print('PASS: full 5000-bootstrap synthetic pipeline, strict positive CI rule')
