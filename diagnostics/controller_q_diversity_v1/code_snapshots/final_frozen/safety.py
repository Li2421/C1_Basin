"""Register the unchanged terminal safety reference without learned correction."""
from .experiment import HERE,read,freeze,sha,uid,canonical,eta_identity,connect,transaction
import copy

def register():
    p=read(HERE/'protocol.json');extra={}
    for scene,profiles in p['controllers'].items():
        extra[scene]={}
        for name in ['analytic6','analytic7']:
            v=copy.deepcopy(profiles[name]['payload'])
            v.update(placement='Safety',execution_semantics='native nominal, unchanged safety projection, no eta correction',
                     execution_callable=str(HERE/'experiment.py'),execution_callable_sha256=sha(HERE/'experiment.py'))
            extra[scene][name+'_safety']=dict(controller_uid=uid('ctl',v),payload=v)
    freeze(HERE/'safety_profiles.json',extra)
    with connect() as db,transaction(db):
        eid,v,hx=eta_identity([0.,0.,0.]);db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        for profiles in extra.values():
            for profile in profiles.values():
                v=profile['payload']
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                  (profile['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    print('Registered four safety references; zero new rollouts')

if __name__=='__main__':register()
