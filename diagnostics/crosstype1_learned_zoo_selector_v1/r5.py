"""Legacy R5 statistics from exact prefixes of new outcome-free probes."""
import numpy as np


def from_mechanism(traces):
    result=np.empty((16,2,5),np.float32)
    for si in range(16):
        for ai,arm in enumerate(('F10','F13')):
            trace=traces[arm][si]
            positions=np.asarray(trace['positions'])[:201]
            actions=np.asarray(trace['safe_actions'])[:200]
            active=np.asarray(trace['projection_active'])[:200]
            if positions.shape!=(201,4,2) or actions.shape!=(200,4,2):
                raise ValueError('R5 needs the original complete 200-step prefix')
            initial=np.linalg.norm(trace['goals']-positions[0],axis=-1).sum()
            final=np.linalg.norm(trace['goals']-positions[-1],axis=-1).sum()
            ii,jj=np.triu_indices(4,1)
            distances=np.linalg.norm(positions[:,:,None]-positions[:,None,:],axis=-1)[:,ii,jj]
            speed=np.linalg.norm(actions,axis=-1)
            result[si,ai]=[(initial-final)/initial,np.mean(active),distances.min(),speed.mean(),
                              (speed.mean(-1)<.01).sum()]
    return result.reshape(160)
