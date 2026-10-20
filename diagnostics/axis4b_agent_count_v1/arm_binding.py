"""Append explicit full arm parameters to future identities; no numeric execution change."""
from design import *
import design as foundation

def case(p,s,kind,arm,seed=0,gate=None):
 c=foundation.case(p,s,kind,arm,seed,gate);old=p['profiles'].pop(c['profile_key']);v=copy.deepcopy(old['payload']);v['arm_parameters']=dict(version='round36_explicit_arm_binding_v1',role='privileged reference' if kind=='expert' else 'controller arm',eta_a=c['eta_a'],eta_b=c['eta_b'],eta_a_uid=eta_identity(c['eta_a'])[0],eta_b_uid=eta_identity(c['eta_b'])[0],switch_step=c['switch_step'],horizon=c['H'],native_execution_source_sha256=v['execution_source_sha256']);key=uid('profile',v);cid=uid('ctl',v);p['profiles'][key]=dict(controller_uid=cid,payload=v);c.update(profile_key=key,controller_uid=cid,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key'])));return c
