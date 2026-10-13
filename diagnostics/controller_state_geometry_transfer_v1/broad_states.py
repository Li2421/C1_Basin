import numpy as np
from shared_rollout_db.src.rollout_db import uid

def make_state(i):
 rng=np.random.default_rng(np.random.SeedSequence([2026101109,i]));p=np.array([[-rng.uniform(.2,1.05),rng.uniform(-.025,.025)],[rng.uniform(.2,1.05),rng.uniform(-.025,.025)]])
 if i%4==1:p[:,0]=[-rng.uniform(.18,.45),rng.uniform(.65,1.05)]
 if i%4==2:p[:,0]=[-rng.uniform(.65,1.05),rng.uniform(.18,.45)]
 if i%4==3:
  k=(i//4)%2;p[k]=[rng.uniform(-.02,.02),rng.uniform(.38,.50)];p[1-k,0]=(-1 if 1-k==0 else 1)*rng.uniform(.65,1.05)
 physical={'positions':p.tolist(),'goals':[[1.09,0.],[-1.09,0.]],'velocities':[[0.,0.],[0.,0.]],'timestep':0}
 from single_integrator.environment import GiveWayEnv,Config
 import dataclasses
 for w in [.41,.44,.47,.50,.53,.56,.59,.62]:GiveWayEnv(dataclasses.replace(Config(),corridor_width=w)).reset(p)
 sid=uid('state',physical);alias=f'csg_broad_t0_{i:04d}'
 import hashlib
 return {'physical':physical,'state_uid':sid,'scenario':'toy_give_way','study_index':i,'study_split':'train' if i<32 else 'test' if i<40 else 'calibration','uid':alias,'rng_namespace':int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),'family':alias,'metadata':{'source':'prospective broad safe t0 stratified sample','stratum':i%4}}
