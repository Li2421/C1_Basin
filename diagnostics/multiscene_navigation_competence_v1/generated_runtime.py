"""Frozen projection/correction chain with separate new menu arms."""
from common import *
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from shared_control.basis_families import get_basis_family
from controllers import nominal_reference,corrected_action,bound
from diagnostics.axis4a_geometry_horizon_v1.progress import ProgressSwitch
from trigger import Trigger
import jax
jax.config.update('jax_enable_x64',False)

def clockwise_goal(positions,goals):
    delta=np.asarray(goals)-positions;norm=np.linalg.norm(delta,axis=1,keepdims=True)
    return np.column_stack([delta[:,1],-delta[:,0]])/np.maximum(norm,1e-30)

def action(env,policy,case,step,diagnostics=False):
    assert not jax.config.x64_enabled
    root=jax.random.fold_in(jax.random.PRNGKey(2026101739),int(hashlib.sha256(case['state_uid'].encode()).hexdigest()[:8],16))
    key=jax.random.fold_in(jax.random.fold_in(root,case['seed']),step)
    reference=nominal_reference(bound(policy(env,key),env.config.max_speed),env)
    arm=case['arm']
    if arm=='nominal_only':return reference
    if step==0:
        env.round39_projector=CertifiedHardSafetyFilter(HardProjectionConfig())
        env.round39_progress=ProgressSwitch(env.positions,env.goals,env.config.max_steps,.9681620884830543)
        env.round39_trigger=Trigger(dict(mode='progress',steps=40,speed=.35))
        env.round39_eta=read(HERE/'ETA_EXACT.json')
    eta=case['eta'];H=env.config.max_steps
    exact=env.round39_eta
    if arm in ['S13','S15'] and step>=round(5*H/7):eta=exact['10']['values']
    if arm=='Sprogress':eta=exact[str(env.round39_progress.eta_index(step,env.positions,env.goals))]['values']
    projector=env.round39_projector;basis=get_basis_family('orthoflow3')
    if arm=='right_hand':
        safe=np.asarray(projector(env.snapshot(),reference).velocity,np.float64)
        active=env.round39_trigger.update(env);target=safe+.35*clockwise_goal(env.positions,env.goals)*active[:,None]
        result=np.asarray(projector(env.snapshot(),target).velocity,np.float64)
    else:
        result,safe,correction=corrected_action(env,reference,eta,projector,basis)
        target=safe+correction
    if diagnostics:return result,bool(np.linalg.norm(safe-reference)>1e-6 or np.linalg.norm(result-target)>1e-6)
    return result
