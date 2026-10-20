"""Exact identities for generated maps and policies, registered before execution."""
from common import *
from shared_rollout_db.src.rollout_db import canonical, uid, eta_identity

ARMS=['nominal_only','safety_baseline','F10','F13','F15','S13','S15','Sprogress','right_hand']
def identities(state, kind, arm, seed=0, checkpoint=None, H=None):
    s=state; cfg=dict(s['config']);cfg['max_steps']=int(H or cfg['max_steps'])
    physical={k:s[k] for k in ['positions','goals','velocities']}
    scene_payload=dict(config=cfg,rectangles=s['rectangles'],environment='single-agent authorized copy' if len(s['positions'])==1 else 'GeneratedEnv inherited BottleneckEnv.step')
    sid=uid('scenario',scene_payload)
    scenario=dict(scenario_uid=sid,name='Generated '+s['family'],code_config_fingerprint=uid('cfg',scene_payload),metadata_json=canonical(scene_payload))
    content=uid('content',physical);state_uid=uid('state',dict(scenario=sid,content=content))
    registered_state=dict(state_uid=state_uid,scenario_uid=sid,source_group=s['alias'],content_hash=content,
        physical_state_json=canonical(physical),goals_geometry_json=canonical(scene_payload),
        provenance_json=canonical(dict(alias=s['alias'],family=s['family'],role=s['role'],geometry_seed=s['geometry_seed'])),identity_quality='CONTENT_EXACT')
    cp=str(Path(checkpoint).resolve()) if checkpoint else None
    sources=['geometry.py','observation.py']+(['single_agent.py'] if len(s['positions'])==1 else [])
    if kind=='sequential_expert':sources+=['sequential_expert.py']
    if kind=='learned':sources+=['policies.py','generated_runtime.py','trigger.py']
    payload=dict(scenario_uid=sid,kind=kind,arm=arm,checkpoint=cp,checkpoint_sha256=sha(cp) if cp else None,
        horizon=cfg['max_steps'],dt=cfg['dt'],source_hashes={n:sha(HERE/n) for n in sources},
        plant_sha256=sha(ROOT/'bottleneck_family/environment.py'),basis_sha256=sha(ROOT/'shared_control/basis_families.py'),
        projection_sha256=sha(ROOT/'shared_control/hard_projection.py'),certified_projection_sha256=sha(D/'double_bottleneck_eta3_basin/tools/exact_projection_retry.py'),
        progress_sha256=sha(D/'axis4a_geometry_horizon_v1/progress.py'),
        progress_threshold=.9681620884830543,switch_fraction=5/7,right_hand=dict(mode='progress',steps=40,speed=.35,direction='clockwise goal direction'),
        conditioning='nominal goal-stop projection orthoflow3 projection' if kind=='learned' else kind,
        precision=False,config=cfg)
    controller_uid=uid('ctl',payload)
    controller=dict(controller_uid=controller_uid,scenario_uid=sid,flow_checkpoint_sha256=payload['checkpoint_sha256'],
        orthoflow3_sha256=payload['basis_sha256'],safety_config_hash=uid('safe',dict(projection=payload['projection_sha256'],retry=payload['certified_projection_sha256'])),
        horizon=str(cfg['max_steps']),dt=str(cfg['dt']),success_semantics_version='native all_agents_reach_and_no_swept_collision',
        conditioning_version=payload['conditioning'],rng_semantics_version=canonical(dict(seed_root=2026101739,state_fold='sha256 state_uid prefix32',seed_fold=True,per_step=True)),
        config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')
    eta=[0.,0.,0.]
    if arm not in ['nominal_only','safety_baseline','right_hand','expert','navigation']:
        index=15 if arm in ['F15','S15'] else 13 if arm in ['F13','S13','Sprogress'] else 10
        eta=read(HERE/'ETA_EXACT.json')[str(index)]['values']
    eid,values,hx=eta_identity(eta);seed_key=canonical(dict(future_index=int(seed)))
    case=dict(state=s,config=cfg,scenario_uid=sid,state_uid=state_uid,controller_uid=controller_uid,
        eta_uid=eid,eta=values,seed=int(seed),seed_key=seed_key,kind=kind,arm=arm,
        checkpoint=cp,flow_checkpoint_sha256=payload['checkpoint_sha256'],
        rollout_uid=uid('roll',dict(state=state_uid,eta=eid,controller=controller_uid,seed=seed_key)))
    erow=dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx)
    return case,[('scenario',scenario),('state',registered_state),('controller_config',controller),('eta',erow)]

def extract_eta():
    from shared_rollout_db.src.rollout_db import connect
    old=read(D/'axis4b_agent_count_v1/pilot_protocol.json')['eta'];rows={}
    with connect(True) as db:
        for i in [10,13,15]:
            values=old[str(i)] if isinstance(old,dict) else old[i]
            eid=eta_identity(values)[0];row=dict(db.execute('SELECT * FROM eta WHERE eta_uid=?',(eid,)).fetchone())
            exact=[row[f'eta{k}'] for k in [1,2,3]];assert eta_identity(exact)[2]==row['canonical_hex']
            rows[str(i)]=dict(values=exact,row=row)
    write(HERE/'ETA_EXACT.json',rows)

if __name__=='__main__':extract_eta()
