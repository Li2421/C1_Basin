"""Descriptive final-policy checks, never used for any training or selection."""
from common import *
from collect39 import POLICIES,checkpoint,policy_assets
from design_generated import identities
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical
import math

def prepare(family,role):
    assert role in ['training_final','old_evaluation']
    cases=[];rows=[]
    for kind,seed in POLICIES:
        cp=checkpoint(family,kind,seed,3);assert (cp.parent/'COMPLETE.receipt.json').exists()
        if role=='training_final':
            previous=f'data39_{family}_round_3_{kind}_{seed}'
            states=[c['state'] for c in read(HERE/f'{previous}_CASES.json')]
            assert len(states)==64*16
        else:
            old=read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog'][family]
            old=sorted([s for s in old if s['role']=='evaluation'],key=lambda s:s['index']);states=[]
            for m in range(8):
                for episode in range(2):
                    s=old[4*m+episode];robot=(2*m+episode)%4;path=np.asarray(s['paths'][robot]);h=int(math.ceil(2.5*np.linalg.norm(np.diff(path,axis=0),axis=1).sum()/.5/.05/100)*100)
                    states.append(dict(s,positions=[s['positions'][robot]],goals=[s['goals'][robot]],velocities=[[0.,0.]],paths=[path.tolist()],H=h,config=dict(s['config'],num_agents=1,max_steps=h)))
        for s in states:
            s=dict(s,role='descriptive_'+role,alias=s['alias']+'_descriptive_'+role)
            c,r=identities(s,'navigation_field','navigation',checkpoint=cp)
            row=next(v for table,v in r if table=='controller_config');payload=json.loads(row['config_json'])
            # Same native actor and random semantics as the navigation gate; role is provenance only.
            payload.update(kind='learned39_descriptive',policy_kind=kind,policy_seed=seed,rng_root=2026101740,sources39={n:sha(HERE/n) for n in ['policies.py','policies39.py','observation39.py','navigation_field.py','batch_fields39.py','descriptive_batch40.py']},**policy_assets(cp))
            cid=uid('ctl',payload);row.update(controller_uid=cid,config_json=canonical(payload),conditioning_version='learned39_descriptive',rng_semantics_version='root2026101740-fold_stateprefix32-fold_physical_step')
            c.update(kind='learned39_descriptive',controller_uid=cid,policy_kind=kind,policy_seed=seed)
            c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key']))
            cases.append(c);rows+=r
    stage=f'descriptive40_{family}_{role}';missing=register(stage,cases,rows)
    groups={}
    for i,c in enumerate(missing):groups.setdefault((c['state']['geometry_seed'],c['checkpoint']),[]).append(i)
    work=[indices[j:j+16] for indices in groups.values() for j in range(0,len(indices),16)]
    write(HERE/f'{stage}_MISSING_GROUPS.json',work)
    seal(f'{stage}_BATCH_PLAN.receipt.json',[HERE/f'{stage}_MISSING_CASES.json',HERE/f'{stage}_MISSING_GROUPS.json',HERE/'DESCRIPTIVE_BATCH40_TEST.json'],independent_native_single_agent=True,maximum_group_size=16,no_expert_labels=True)
    return stage

def main():
    from pilot_training40 import rollout_stage
    import time
    while not (HERE/'PILOT_TRAINING_EXECUTION_COMPLETE.json').exists():time.sleep(5)
    for family in 'BF':
        if not (HERE/f'NAVIGATION_CAPABILITY_{family}.json').exists():continue
        for role in ['training_final','old_evaluation']:
            stage=f'descriptive40_{family}_{role}'
            if not (HERE/f'{stage}_CASES.json').exists():prepare(family,role)
            rollout_stage(stage,'descriptive_batch40.py')
    write(HERE/'PILOT_DESCRIPTIVE_EXECUTION_COMPLETE.json',dict(utc=utc()))

if __name__=='__main__':main()
