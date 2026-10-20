from common import *
from reuse_inactive import inactive
p=np.repeat(np.array([[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]]),52,axis=0);v=np.zeros_like(p);g=-p[0]
assert not inactive(dict(mode='always',steps=0,speed=.1),p,v,g,.09,False)
assert not inactive(dict(mode='slow',steps=40,speed=.1),p,v,g,.09,False)
assert inactive(dict(mode='slow',steps=40,speed=.1),p[:40],v[:40],g,.09,False)
assert not inactive(dict(mode='slow',steps=40,speed=.1),p[:41],v[:41],g,.09,True)
assert inactive(dict(mode='progress',steps=40,speed=.35),p[:40],v[:40],g,.09,False)
write(HERE/'INACTIVE_REUSE_TESTS.json',dict(utc=utc(),passed=True,always_slow_progress_and_failure_boundary=True))
