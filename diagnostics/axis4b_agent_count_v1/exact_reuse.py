"""Exact full-episode alias only when a schedule never reaches its first difference."""
from common import *
from shared_rollout_db.src.rollout_db import eta_identity
import copy,time

def compatible(parent,c):
 old=parent['case']
 for k in ['N','H','kind','state_uid','seed','evaluation_config']:
  if old[k]!=c[k]:return False
 expected='F13' if c['arm'].startswith('S13') else 'F15' if c['arm'].startswith('S15') else None
 if expected is None or old['arm']!=expected:return False
 if eta_identity(old['eta_a'])[0]!=eta_identity(c['eta_a'])[0] or old['eta_a']!=old['eta_b']:return False
 n=parent['physical_steps'];t=c['switch_step']
 if parent['outcome'] in ['success','collision']:return n<=t
 if parent['outcome']=='numerical_failure':return n<t and parent['error'] is not None and parent['error']['step']<t
 return False

def derive(c,p,phase):
 descriptor=c.get('shared_prefix_source_case')
 if descriptor is None:return None
 started=time.monotonic();path=HERE/descriptor['phase']/'raw'/f"{descriptor['rollout_uid']}.json";old=read(path)
 if not compatible(old,c):return None
 v=p['profiles'][c['profile_key']]['payload'];ov=p['profiles'][old['case']['profile_key']]['payload']
 for key in ['N','controller_class','flow_checkpoint_sha256','evaluation_config','orthoflow3_sha256','basis_implementation_sha256','safety_projection_sha256','safety_wrapper_sha256','rng','execution_source_sha256','jax_enable_x64']:assert ov[key]==v[key],('Unproved equivalence',key)
 assert sha(HERE/'runtime.py')==v['execution_source_sha256']['runtime.py'];assert sha(HERE/'controllers.py')==v['execution_source_sha256']['controllers.py'];assert sha(old['trace_path'])==old['trace_sha256'];q=copy.deepcopy(old);q['prefix_source']=dict(raw_path=str(path),raw_sha256=sha(path),trace_path=old['trace_path'],trace_sha256=old['trace_sha256'],source_rollout_uid=old['rollout_uid'],source_controller_uid=old['controller_uid'],source_experiment_uid=old['experiment_uid'],source_arm=old['case']['arm'],source_outcome=old['outcome'],unreached_switch_step=c['switch_step'],proof='Same state/key/config/nominal/basis/projection and earlyeta; known terminal at<=switch or numericalunknown strictlybefore switch. No future target control action was required.');q.update(experiment_uid=EXP,case=copy.deepcopy(c),**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},eta=c['eta_a'],future_index=c['seed'],new_physical_steps=0,deterministic_reused_steps=old['physical_steps'],deterministic_source=None,elapsed_seconds=time.monotonic()-started,label_derivation_code_sha256=sha(HERE/'exact_reuse.py'),execution_kind='EXACT_PRE_SWITCH_EPISODE_DERIVATION');return clean(q)
