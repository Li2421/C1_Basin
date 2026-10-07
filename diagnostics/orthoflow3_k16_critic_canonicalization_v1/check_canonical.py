"""Passive re-encoding tests; active policy behavior is evaluated separately."""
import json
import numpy as np
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,canonical_conditioning as cc


def main():
    result=[]
    for row in a.states():
        sc=row['scenario']
        if sc=='double_bottleneck':continue
        physical=a.decode(row['structured_state']);env=a.decode(row['environment_descriptor'])
        flat=np.asarray(a.decode(row['conditioning'])['flat']);original,_=cc.encode(sc,flat,physical,env)
        for k in range(4):
            for order in ([0,1,2,3],[1,2,3,0]):
                phy={**physical,**{n:cc.rotate(physical[n],k)[order].tolist() for n in ('positions','velocities','goals')}}
                altered=flat.copy();altered[-8:]=cc.rotate(flat[-8:].reshape(4,2),k)[order].ravel()
                encoded,_=cc.encode(sc,altered,phy,env)
                error=float(np.max(np.abs(encoded-original)))
                result.append({'state_uid':row['state_uid'],'scenario':sc,'rotation':90*k,'permutation':order,'error':error})
                assert error<2e-6,(sc,row['state_uid'],k,order,error)
    b.dump('passive_metamorphic.json',{'tests':result,'max_error':max(x['error'] for x in result),'pass':True,
          'claim':'passive reencoding only; no assertion of active Four MACFlow equivariance'})
    print(json.dumps({'tests':len(result),'max_error':max(x['error'] for x in result),'pass':True}))


if __name__=='__main__':main()
