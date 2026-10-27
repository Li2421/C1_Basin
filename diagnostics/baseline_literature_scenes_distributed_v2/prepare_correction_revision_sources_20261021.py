"""Create static revised runtimes; existing sealed code is never edited."""
from experiment import *
from atomic_write import write
import ast,difflib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
items=[
 ('runtime_flat_static.py','runtime_correction_revised_20261021.py',[
 ('from central_reference import progress','from central_reference import progress\nfrom correction_revision_20261021 import distributed_fields,central_fields,contact_radius'),
 ('basis.compute(env.positions,env.goals,v,cfg.max_speed).correction(eta)','central_fields(env.positions,env.goals,v,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg),basis).correction(eta)'),
 ('local_basis(env.positions,env.goals,v,i,np.inf,cfg.max_speed)','distributed_fields(env.positions,env.goals,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg))')]),
 ('central_reservation_flat_static.py','central_correction_revised_20261021.py',[
 ("jax.config.update('jax_enable_x64',False)","from correction_revision_20261021 import central_fields,contact_radius\njax.config.update('jax_enable_x64',False)"),
 ('basis.compute(env.positions,env.goals,safe,env.config.max_speed).correction(eta)','central_fields(env.positions,env.goals,safe,env.config.max_speed,env.config.goal_tolerance,contact_radius(env.config),basis).correction(eta)')]),
 ('dynamics_runtime_flat_static.py','dynamics_correction_revised_20261021.py',[
 ('from distributed_safety import local_basis','from distributed_safety import local_basis\nfrom correction_revision_20261021 import distributed_fields,contact_radius'),
 ('local_basis(env.positions,env.goals,first_next_velocity,i,np.inf,cfg.max_speed)',"distributed_fields(env.positions,env.goals,first_next_velocity,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg,'paper_dynamics'))")])]
changes=[]
for old,new,replacements in items:
    original=(HERE/old).read_text();updated=original
    for a,b in replacements:
        assert a in updated,(old,a);updated=updated.replace(a,b)
    normalized=updated
    for a,b in reversed(replacements):normalized=normalized.replace(b,a)
    assert normalized==original,old
    ast.parse(updated);dest=HERE/new;assert not dest.exists(),dest;dest.write_text(updated)
    diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile=old,tofile=new)))
    changes.append(dict(original=old,original_sha256=sha(HERE/old),revised=new,revised_sha256=sha(dest),diff=diff.name,outside_documented_basis_call_changes_byte_identical=True))
write(HERE/'CORRECTION_REVISION_SOURCE_DIFF_20261021.json',dict(utc=utc(),changes=changes,physics_safety_tracking_switch_and_arrival_unchanged=True,original_nominal_acceleration_and_projection_backends_unchanged=True,research_rollouts=0,gate_not_yet_passed=True))
print('Prepared new static revised execution modules; no submission')
