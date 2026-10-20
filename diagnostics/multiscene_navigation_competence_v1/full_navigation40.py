"""Frozen solo navigation gate; native physics, no expert labeling or adaptation."""
from common import *
from collect39 import checkpoint,policy_assets,POLICIES
from design_generated import identities
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical
from geometry import GeneratedGeometry
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from navigation_field import NavigationGrid
from policies39 import load_policy
import math,fcntl,jax

from full_geometry40 import require_pilot,verify_full
def states(family):
    maps=[s for s in read(HERE/f'FULL_{family}_GEOMETRY_FROZEN.json')[family] if s['role']=='navigation_holdout']
    assert len(maps)==8
    result=[]
    for m,s in enumerate(maps):
        for episode in range(2):
            robot=(2*m+episode)%4
            path=np.asarray(s['paths'][robot]);h=int(math.ceil(2.5*np.linalg.norm(np.diff(path,axis=0),axis=1).sum()/.5/.05/100)*100)
            result.append(dict(s,role='navigation_capability',alias=s['alias']+f'_navigation_agent{robot}',positions=[s['positions'][robot]],goals=[s['goals'][robot]],velocities=[[0.,0.]],paths=[path.tolist()],H=h,config=dict(s['config'],num_agents=1,max_steps=h)))
    return result

def _prepare(family):
    frozen=read(HERE/f'FULL_{family}_NAVIGATION_STATES_FROZEN.receipt.json')
    for p,h in frozen['files'].items():assert sha(p)==h
    cases=[];rows=[]
    for kind,seed in POLICIES:
        cp=checkpoint(family,kind,seed,3);assert (cp.parent/'COMPLETE.receipt.json').exists()
        for state in read(HERE/f'FULL_{family}_NAVIGATION_STATES_FROZEN.json')[family]:
            c,r=identities(state,'navigation_field','navigation',checkpoint=cp)
            row=next(v for table,v in r if table=='controller_config');payload=json.loads(row['config_json'])
            payload.update(kind='learned39_navigation_gate',policy_kind=kind,policy_seed=seed,rng_root=2026101740,sources39={n:sha(HERE/n) for n in ['policies.py','policies39.py','observation39.py','navigation_field.py','navigation_gate40.py']},**policy_assets(cp))
            cid=uid('ctl',payload);row.update(controller_uid=cid,config_json=canonical(payload),conditioning_version='learned39_navigation_gate',rng_semantics_version='root2026101740-fold_stateprefix32-fold_physical_step')
            c.update(kind='learned39_navigation_gate',controller_uid=cid,policy_kind=kind,policy_seed=seed)
            c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key']))
            cases.append(c);rows+=r
    stage='navigation40_'+family;register(stage,cases,rows);return stage


def freeze_states(family):
 require_pilot();verify_full()
 p=HERE/f'FULL_{family}_NAVIGATION_STATES_FROZEN.json';assert not p.exists()
 write(p,{family:states(family)})
 seal(f'FULL_{family}_NAVIGATION_STATES_FROZEN.receipt.json',[p,HERE/f'FULL_{family}_GEOMETRY_FROZEN.json',Path(__file__),HERE/'navigation_gate40.py'],episodes_per_instance=16,success_threshold=15,final_round=3,rng_root=2026101740,before_family_expert_and_training=True,no_gate_retraining=True)

def prepare(family):
 require_pilot();verify_full()
 return _prepare(family)
