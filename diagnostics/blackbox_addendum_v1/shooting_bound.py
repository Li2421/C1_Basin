from core import *
def bound(a,speed):return a*np.minimum(1.,speed/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30))
