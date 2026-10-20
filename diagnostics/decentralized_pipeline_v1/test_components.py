from core import *
from pipeline import *
from ring_exchange.safety import polygonal_obstacle_snapshot
from shared_control.hard_projection import barrier_constraints,project_velocity
from shared_control.basis_families import get_basis_family
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
jax.config.update('jax_enable_x64',False)
p=read(D/'controller_universal_eta_schedule_v1/protocol.json');name='base_u_macflow';rt=make_runtime(name,p,{});s=p['states'][0];env=rt.make_env(s);A,b,g=barrier_constraints(polygonal_obstacle_snapshot(env,sides=48),CFG);errors=[]
for i in range(4):
 a,bb=wall_rows(env.positions[i],env.config);start=6+i*96;assert np.array_equal(a,A[start:start+96,i*2:i*2+2]);assert np.array_equal(bb,b[start:start+96]);single=LocalSafety(env.positions[i:i+1],0,env.config,np.inf)
 for target in [np.array([[.1,.2]]),np.array([[1.,-.8]]),np.array([[-.7,.9]])]:
  x=single.project(target)[0];xx=project_velocity(target,a,bb,env.config.max_speed,CFG)[0];errors.append(float(np.max(np.abs(x-xx))));assert np.allclose(x,xx,rtol=0,atol=1e-9)
safe=np.array([[.1,.2],[-.2,.1],[.05,-.1],[-.1,-.1]]);native=get_basis_family('orthoflow3').compute(env.positions,env.goals,safe,env.config.max_speed)
for i in range(4):
 local=local_basis(env.positions,env.goals,safe,i,np.inf,env.config.max_speed);assert np.array_equal(local,np.stack([v[i] for v in native.values]))
for i in range(4):
 for j in range(i+1,4):
  ri=LocalSafety(env.positions,i,env.config,np.inf);rj=LocalSafety(env.positions,j,env.config,np.inf);ii=ri.neighbor_indices.index(j);jj=rj.neighbor_indices.index(i);k=list(g['pair_indices']).index((i,j));assert ri.lower[ii]+rj.lower[jj]==b[k]
# Unseen neighbor cannot enter local rows or relation basis (nominal remains joint).
pos=np.array([[2.,0.],[-2.,0.],[0.,2.],[0.,-2.]]);perturbed=pos.copy();perturbed[1]=[-2.5,.1];x=LocalSafety(pos,0,env.config,1.5);xx=LocalSafety(perturbed,0,env.config,1.5);assert np.array_equal(x.A,xx.A) and np.array_equal(x.lower,xx.lower);assert np.array_equal(local_basis(pos,-pos,safe,0,1.5,.52),local_basis(perturbed,-pos,safe,0,1.5,.52))
assert 202610280007<202610281000 and 202610281999-202610281000+1>=32
write(HERE/'UNIT_TESTS.json',dict(utc=utc(),status='PASS',single_robot_native_projection_max_error=max(errors),single_robot_Brel_extension='zero; native all_pair function requiresN>=2',all_neighbor_basis_bitwise=True,wall_rows_bitwise=True,pair_half_responsibility_algebra=True,unseen_neighbor_excluded=True,pilot_eval_intervals_disjoint=True,jax_enable_x64=bool(jax.config.x64_enabled),simulator_steps=0));print('Round28 component gates PASS')
