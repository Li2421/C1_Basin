from common import *
from registry import register,uid,eta_identity,canonical,connect
import copy
from functools import lru_cache

@lru_cache(None)
def source():return read(R35/'HISTORICAL_INVENTORY.json')
@lru_cache(None)
def originals():return read(HERE/'ORIGINAL_PROFILES.json')
def protocol():return dict(states=[],profiles={},scenarios={},eta=read(R10/'protocol.json')['eta'])
def config(H,r):return dict(source()['config'],max_steps=int(H),obstacle_radius=float(r))
def ring_state(p,old,H,r,validation=None):
    s=copy.deepcopy(old);cfg=config(H,r);sid=uid('scenario',dict(round35_physical_config=cfg));physical={k:s['physical'][k] for k in ['positions','velocities','goals']};identity=dict(physical=physical,timestep=0,future_root=2026100403,key_token=int(hashlib.sha256(s['uid'].encode()).hexdigest()[:8],16))
    if validation:identity['validation']=validation
    s.update(original_state_uid=old.get('original_state_uid',old['state_uid']),physical=physical,transport_physical=physical,evaluation_config=cfg,scenario_uid=sid,content_hash=uid('content',identity));s['state_uid']=uid('state',dict(scenario=sid,content=s['content_hash']));p['scenarios'][sid]=dict(scenario_uid=sid,name='ring_exchange',code_config_fingerprint=uid('cfg',cfg),metadata_json=canonical(dict(config=cfg,physical_source_sha256=sha(ROOT/'ring_exchange/environment.py'))))
    if s['state_uid'] not in {q['state_uid'] for q in p['states']}:p['states'].append(s)
    return s

def ring_case(p,s,name,H,r,arm,seed=0,validation=None,rule=None,prefix=None):
    eta=p['eta'];a=[0.,0.,0.] if arm in ['nominal_only','safety_baseline','RHR'] else eta[10 if arm=='F10' else 13];b=eta[10] if arm in ['S500','Sprog','Srel','Sabs'] else a;switch={'S500':500,'Sabs':500,'Srel':round(5*H/7)}.get(arm,H);v=copy.deepcopy(originals()[name]['payload']);sources={k:sha(HERE/k) for k in ['runtime.py','factor_runtime.py','progress.py','prefix.py']};v.update(scenario_uid=s['scenario_uid'],horizon=H,evaluation_config=config(H,r),jax_enable_x64=False,round37_engine='factor',round37_arm=dict(name=arm,eta_a=a,eta_b=b,switch_step=switch,progress_threshold=read(HERE/'DESIGN_CONFIG.json')['pstar'] if arm=='Sprog' else None,validation=validation,rule=rule),execution_source_sha256=sources)
    if rule:v.update(continuing_rhr_source_sha256=sha(HERE/'continuing_rhr.py'),trigger_source_sha256=sha(HERE/'trigger.py'))
    key=uid('profile',v);ctl=uid('ctl',v);p['profiles'][key]=dict(controller_uid=ctl,payload=v);eid=eta_identity(a)[0];sk=canonical(dict(future_index=seed));c=dict(engine='factor',H=H,r=r,kind='Flow',profile_key=key,base_controller_name=name,state_uid=s['state_uid'],original_state_uid=s.get('original_state_uid'),eta_uid=eid,controller_uid=ctl,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk)),arm=arm,seed=seed,eta_a=a,eta_b=b,switch_step=switch,state_index=s['study_index'],rhr_rule=rule)
    if arm=='Sprog':c['progress_threshold']=read(HERE/'DESIGN_CONFIG.json')['pstar']
    if prefix:c['prefix_source']=prefix
    return c

def gate():
    p=protocol();cases=[];oldp=read(R35/'pilot_protocol.json');oldcases=read(R35/'pilot_CASES.json');names=source()['controller_names']
    for r in [.8,.85,.95,1.]:
        state=min([s for s in oldp['states'] if s['radius']==r],key=lambda s:s['state_uid'])
        for j in [0,1,8,13]:
            ref=next(c for c in oldcases if c['state_uid']==state['state_uid'] and c['base_controller_name']==names[j] and c['arm']=='F10');s=ring_state(p,state,700,r,'round37_geometry_replay');c=ring_case(p,s,names[j],700,r,'F10',0,'round37_geometry_replay');path=R35/'pilot/raw'/f"{ref['rollout_uid']}.json";c.update(reference_path=str(path),reference_sha256=sha(path),replay_group='geometry_pilot');cases.append(c)
    oldp=read(R35/'gate_protocol.json')
    for ref in read(R35/'gate_CASES.json'):
        state=next(s for s in oldp['states'] if s['state_uid']==ref['state_uid']);s=ring_state(p,state,700,.9,'round37_original_replay');c=ring_case(p,s,ref['base_controller_name'],700,.9,ref['arm'],ref['seed'],'round37_original_replay');c.update(reference_path=ref['reference_path'],reference_sha256=ref['reference_sha256'],replay_group='round10');cases.append(c)
    register('replay_A',p,cases)
if __name__=='__main__':globals()[sys.argv[1]]()
