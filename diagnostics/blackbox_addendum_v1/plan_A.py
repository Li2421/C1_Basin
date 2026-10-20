from common import *
from planning import protocol,ring_state,ring_case,source
from registry import register
from audit_utils import source_inventory
from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv
import jax
jax.config.update('jax_enable_x64',False)
OLD=D/'axis4a_addendum_v1'
def main():
    prereg();assert read(HERE/'A_REPRODUCTION_GATE.json')['passed'];assert (HERE/'HISTORICAL_REUSE.receipt.json').exists();p=protocol();cases=[];names=source()['controller_names'];checks=[]
    for label,cohort in [('A1','positive'),('A2','same_state')]:
        for item in read(HERE/f'REUSED_{label}_EVIDENCE.json'):
            c=dict(item['case'],engine='factor',kind='Flow',cohort=cohort,reference_path=item['raw_path'],reference_sha256=item['raw_sha256'],legacy_reuse=True);cases.append(c)
    positive=read(OLD/'A1_STATES.json');write(HERE/'A1_STATES.json',positive)
    for old in positive:
        s=ring_state(p,old,700,.9)
        for name in names:
            c=ring_case(p,s,name,700,.9,'nominal_only');c['cohort']='positive';cases.append(c)
    selected=sorted(read(R35/'EVAL_STATES_ORIGINAL.json'),key=lambda s:s['state_uid'])[:16];write(HERE/'A2_ORIGINAL_STATES.json',selected)
    for r in [.8,.85,.95,1.]:
        for i,old in enumerate(selected):
            try:
                env=RingExchangeEnv(LocalFrameConfig(**dict(source()['config'],max_steps=700,obstacle_radius=r)));v=old['physical'];env.reset(np.asarray(v['positions']),velocities=np.asarray(v['velocities']),goals=np.asarray(v['goals']));checks.append(dict(radius=r,state_uid=old['state_uid'],state_index=i,valid=True))
            except ValueError as e:checks.append(dict(radius=r,state_uid=old['state_uid'],state_index=i,valid=False,reason=str(e)));continue
            s=ring_state(p,old,700,r);s['study_index']=i
            for name in names:
                for arm in ['safety_baseline','nominal_only']:
                    c=ring_case(p,s,name,700,r,arm);c['cohort']='same_state';cases.append(c)
    for old in read(R35/'GEOMETRY_PILOT_STATES.json'):
        r=old['radius'];s=ring_state(p,old,700,r)
        for name in names:
            c=ring_case(p,s,name,700,r,'nominal_only');c['cohort']='old_geometry_nominal';cases.append(c)
    write(HERE/'A2_RESET_VALIDITY.json',checks);register('geometry_controls',p,cases)
    # Reuse already frozen choices, with source freeze time rather than pretending a refit.
    original=R35/'FROZEN_REFERENCE_CHOICES.json';write(HERE/'FROZEN_REFERENCE_CHOICES.json',read(original));seal('REFERENCE_CHOICES_FROZEN.receipt.json',[HERE/'FROZEN_REFERENCE_CHOICES.json'],source=str(original),source_sha256=sha(original),source_freeze_receipt=str(R35/'REFERENCE_CHOICES_FROZEN.receipt.json'),no_new_controller_selection=True,before_new_baseline_labels=True);source_inventory('imports_plan_A.json')
if __name__=='__main__':main()
