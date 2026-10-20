"""Original 35 world-coordinate features plus immutable navigation lookahead."""
import numpy as np
from observation import observe as observe35

def observe(env,fields):
 p=env.positions
 way=np.stack([np.concatenate([f.waypoint(p[i:i+1],s)[0]-p[i] for s in [.5,1.5]]) for i,f in enumerate(fields)])
 return np.concatenate([observe35(env),way],axis=1).astype(np.float32)
