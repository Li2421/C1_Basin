"""Versioned corrected reference restore; legacy given identities unchanged."""
from common import *
from actions import action,fixed_source_uid,TIMES
from actions import case as original_case
from shared_rollout_db.src.rollout_db import uid

def case(base,name,s,a,seed,state_index=0,candidate_index=-1):
    c,cp=original_case(base,name,s,a,seed,state_index,candidate_index)
    if a.get('late_source','given')=='given':return c,cp
    strategy=cp['payload']['late_eta_requery'];strategy.update(generator_wrapper_sha256=sha(HERE/'requery_generator_v2.py'),execution_implementation_sha256=sha(HERE/'requery_execution_v2.py'),snapshot_conditioning_sha256=sha(HERE/'snapshot_conditioning.py'),snapshot_restore='public_restore_augmented_state_exact_reference_only',reference_artifacts='explicit_authoritative_ROOT_paths')
    cid=uid('ctl',cp['payload']);rid=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key']));cp['controller_uid']=cid;c.update(controller_uid=cid,rollout_uid=rid);return c,cp
