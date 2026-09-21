"""Isolated AD/FD localization for the archived fixed VI R_CERT failure.

Diagnostic only: imports production functions without modifying their semantics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import nnls

from single_integrator.c1.differentiable_rollout import barrier_constraints,bounded_nominal
from single_integrator.c1.risk.joint_frozen import controller_projection
from single_integrator.c1.risk.vi_r_cert import strict_atom_margins
from single_integrator.c1.rollout_vi_r_cert import rollout,WITNESS_TARGETS
from single_integrator.c1.train import observation
from single_integrator.c1.train_deadlock_union import setup,noise,digest
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.cbf import CBFConfig,project_velocity
from single_integrator.environment import GiveWayEnv


INITIAL=np.array([[-.4,.005],[.4,-.005]],np.float64)
NOISE_SEED=84001
ROLLOUT_ID=73000
INTERVENTION=(100,120)
DIRECTION_SEED=2026091895
COTANGENT_SEED=2026091903
EPSILON=1e-10
SIGN_TOLERANCE=1e-8
H_VALUES=np.array([1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6])
SEGMENT_LENGTHS=(1,2,4,8,16,32,64,128,151)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(ad,fd):return abs(ad-fd)/max(abs(ad),abs(fd),EPSILON)


def sign(value):
    return 0 if abs(value)<=SIGN_TOLERANCE else (1 if value>0 else -1)


def active_signature(p,A,b,tol=1e-8):
    p=np.asarray(p).reshape(4);A=np.asarray(A);b=np.asarray(b)
    linear=A@p-b
    blocks=p.reshape(2,2)
    ball=(.25-np.sum(blocks*blocks,axis=-1))/2
    return np.r_[linear<tol,ball<tol]


def kkt_report(target,p,A,b,cbf):
    target,p,A,b=map(np.asarray,(target,p,A,b));p=p.reshape(4)
    linear=A@p-b;balls=(.25-np.sum(p.reshape(2,2)**2,axis=-1))/2
    J=np.array(A,copy=True)
    ball_rows=np.zeros((2,4))
    for i in range(2):ball_rows[i,2*i:2*i+2]=-p[2*i:2*i+2]
    J=np.vstack((J,ball_rows));slack=np.r_[linear,balls]
    active=slack<1e-8;Ja=J[active]
    lam=nnls(Ja.T,p-target,maxiter=10000)[0] if len(Ja) else np.empty(0)
    stationarity=float(np.linalg.norm(p-target-(Ja.T@lam if len(Ja) else 0),ord=np.inf))
    complementarity=float(np.max(np.abs(lam*slack[active]))) if len(lam) else 0.
    solved,status=project_velocity(target,A,b,.5,cbf)
    return dict(status=status,replay_max_abs_error=float(np.max(np.abs(solved.reshape(4)-p))),
        min_linear_primal=float(linear.min()),max_speed_excess=float(
            np.linalg.norm(p.reshape(2,2),axis=-1).max()-.5),
        stationarity=stationarity,complementarity=complementarity,
        active_indices=np.flatnonzero(active).tolist(),active_rank=int(np.linalg.matrix_rank(Ja)),
        active_count=int(active.sum()))


def table(fn,base,direction):
    compiled=jax.jit(fn)
    grad=jax.jit(jax.grad(fn))(base)
    ad=float(jnp.vdot(grad,direction))
    repeated=[float(compiled(base)) for _ in range(5)]
    rows=[]
    for h in H_VALUES:
        plus=float(compiled(base+h*direction));minus=float(compiled(base-h*direction))
        fd=(plus-minus)/(2*h)
        rows.append(dict(h=float(h),plus=plus,minus=minus,fd=fd,ad=ad,
            absolute_error=abs(ad-fd),relative_error=relative(ad,fd),
            ad_sign=sign(ad),fd_sign=sign(fd)))
    return dict(ad=ad,gradient_norm=float(jnp.linalg.norm(grad)),
        repeated_values=repeated,repeated_range=max(repeated)-min(repeated),rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('new diagnostic output required')
    if jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    params,field,plant,cbf,baseline=setup()
    env=GiveWayEnv(plant);goals=jnp.asarray(env.goals);walls=jnp.asarray(env.walls)
    draws=noise(NOISE_SEED,ROLLOUT_ID)
    zero=jnp.zeros((INTERVENTION[1]-INTERVENTION[0],4),jnp.float64)
    rng=np.random.default_rng(DIRECTION_SEED)
    direction=rng.normal(size=zero.shape);direction/=np.linalg.norm(direction)
    direction=jnp.asarray(direction)

    def full(local):
        offsets=jnp.zeros((850,4),jnp.float64).at[
            INTERVENTION[0]:INTERVENTION[1]].set(local)
        return rollout(params,field,jnp.asarray(INITIAL),draws,plant,cbf,offsets)
    full_run=jax.jit(full)
    base_terms,base_trace=full_run(zero)
    base_values=np.array([base_terms['direct_R_CERT'],base_terms['augmented_R_CERT']])
    full_tables={}
    for index,name in enumerate(('direct','augmented')):
        full_tables[name]=table(lambda z,index=index:full(z)[0][
            'direct_R_CERT' if index==0 else 'augmented_R_CERT'],zero,direction)

    # Exact checkpoint-conditioned controller segment, actions100..250.
    checkpoint_position=base_trace['before'][100]
    checkpoint_velocity=base_trace['applied'][99].reshape(2,2)
    segment_steps=151
    def segment(local):
        def step(carry,t):
            positions,velocity=carry
            obs=observation(positions[None],velocity[None],goals)
            A,b,_=barrier_constraints(positions,walls,plant.to_dict(),cbf)
            prepared=field.prepare(params,obs,draws[t][None],A[None],b[None],.5,
                lambda value,matrix,lower,speed:controller_projection(value,matrix,lower))
            nominal=prepared['nominal'][0];safe=prepared['safe'][0]
            offset=jax.lax.cond(t<INTERVENTION[1],lambda _:local[t-INTERVENTION[0]],
                lambda _:jnp.zeros(4,jnp.float64),operand=None)
            w=safe+prepared['correction'][0]+offset
            targets=jnp.concatenate((w[None],WITNESS_TARGETS),axis=0)
            projected=controller_projection(targets,jnp.broadcast_to(A,(9,)+A.shape),
                jnp.broadcast_to(b,(9,)+b.shape))
            applied=projected[0];after=positions+.05*applied.reshape(2,2)
            output=dict(before=positions,after=after,nominal=nominal,safe=safe,w=w,
                applied=applied,witnesses=projected[1:],A=A,b=b)
            return (after,applied.reshape(2,2)),output
        _,outputs=jax.lax.scan(jax.checkpoint(step),(checkpoint_position,checkpoint_velocity),
            jnp.arange(100,251,dtype=jnp.int32))
        return outputs
    segment_run=jax.jit(segment)
    base_segment=segment_run(zero)
    parity={key:float(np.max(np.abs(np.asarray(base_segment[key])-np.asarray(
        base_trace[key][100:251])))) for key in
        ('before','after','safe','w','applied','witnesses','A','b')}

    cot_rng=np.random.default_rng(COTANGENT_SEED)
    state_cot=cot_rng.normal(size=(2,2));state_cot/=np.linalg.norm(state_cot)
    state_cot=jnp.asarray(state_cot)
    def weighted_state(local,weights):
        return jnp.sum(segment(local)['after']*weights)
    state_grad=jax.jit(jax.grad(weighted_state,argnums=0))
    segment_ad={}
    for length in SEGMENT_LENGTHS:
        weights=jnp.zeros((segment_steps,2,2),jnp.float64).at[length-1].set(state_cot)
        g=state_grad(zero,weights)
        segment_ad[length]=float(jnp.vdot(g,direction))
    segment_fd={length:[] for length in SEGMENT_LENGTHS}
    segment_forward={}
    selected_traces={}
    for h in H_VALUES:
        plus=segment_run(zero+h*direction);minus=segment_run(zero-h*direction)
        segment_forward[float(h)]=(plus,minus)
        for length in SEGMENT_LENGTHS:
            fp=float(jnp.vdot(plus['after'][length-1],state_cot))
            fm=float(jnp.vdot(minus['after'][length-1],state_cot))
            fd=(fp-fm)/(2*h);ad=segment_ad[length]
            segment_fd[length].append(dict(h=float(h),ad=ad,fd=fd,
                absolute_error=abs(ad-fd),relative_error=relative(ad,fd),
                ad_sign=sign(ad),fd_sign=sign(fd)))
        if h in (1e-2,1e-3,1e-4,1e-5):selected_traces[float(h)]=(plus,minus)

    # One-step layer probes at the exact checkpoint.
    state_direction=cot_rng.normal(size=(2,2));state_direction/=np.linalg.norm(state_direction)
    state_direction=jnp.asarray(state_direction)
    c4=cot_rng.normal(size=4);c4/=np.linalg.norm(c4);c4=jnp.asarray(c4)
    c32=cot_rng.normal(size=32);c32/=np.linalg.norm(c32);c32=jnp.asarray(c32)
    c85=cot_rng.normal(size=85);c85/=np.linalg.norm(c85);c85=jnp.asarray(c85)
    xi=draws[100]
    def checkpoint_parts(position):
        obs=observation(position[None],checkpoint_velocity[None],goals)
        A,b,_=barrier_constraints(position,walls,plant.to_dict(),cbf)
        nominal=bounded_nominal(field.baseline_sample(obs,xi[None]),.5)[0]
        safe=controller_projection(nominal,A,b)
        correction=field.correction(params,obs,safe[None])[0]
        w=safe+correction
        applied=controller_projection(w,A,b)
        witness_targets=jnp.asarray(WITNESS_TARGETS,position.dtype)
        witnesses=controller_projection(witness_targets,jnp.broadcast_to(A,(8,)+A.shape),
            jnp.broadcast_to(b,(8,)+b.shape))
        return nominal,safe,correction,applied,witnesses,jnp.concatenate((A.reshape(-1),b))
    layer_defs=(
        ('flow_bc_state',lambda p:jnp.vdot(checkpoint_parts(p)[0],c4)),
        ('first_projection_state',lambda p:jnp.vdot(checkpoint_parts(p)[1],c4)),
        ('frozen_residual_state',lambda p:jnp.vdot(checkpoint_parts(p)[2],c4)),
        ('second_projection_state',lambda p:jnp.vdot(checkpoint_parts(p)[3],c4)),
        ('witness_projection_state',lambda p:jnp.vdot(checkpoint_parts(p)[4].reshape(-1),c32)),
        ('constraint_data_state',lambda p:jnp.vdot(checkpoint_parts(p)[5],c85)))
    layers={name:table(fn,checkpoint_position,state_direction) for name,fn in layer_defs}
    fixed_parts=checkpoint_parts(checkpoint_position);A0=fixed_parts[5][:-17].reshape(17,4);b0=fixed_parts[5][-17:]
    y0=fixed_parts[1]+fixed_parts[2]
    input_direction=cot_rng.normal(size=4);input_direction/=np.linalg.norm(input_direction)
    input_direction=jnp.asarray(input_direction)
    layers['second_projection_input']=table(
        lambda y:jnp.vdot(controller_projection(y,A0,b0),c4),y0,input_direction)

    # Fixed-input margin and aggregation controls at the active baseline branch.
    active=int(np.asarray(base_terms['active_event_index'])[1]);active_t=139+active
    start=active_t-100
    states=jnp.concatenate((base_trace['before'][:1],base_trace['after']),axis=0)
    anchors=jnp.linalg.norm(states[start-39:start-39+101]-goals,axis=-1)
    margins=jax.vmap(lambda before,after,w,u,v,anchor:strict_atom_margins(
        before,after,w,u,v,goals,anchor))(base_trace['before'][start:active_t+1],
        base_trace['after'][start:active_t+1],base_trace['w'][start:active_t+1],
        base_trace['applied'][start:active_t+1],base_trace['witnesses'][start:active_t+1],anchors)
    margin_direction=cot_rng.normal(size=margins.shape);margin_direction/=np.linalg.norm(margin_direction)
    margin_direction=jnp.asarray(margin_direction)
    def aggregate_from_margins(value,augmented):
        from jax.scipy.special import logsumexp
        selected=value if augmented else value[:,:,0]
        costs=jax.nn.softplus(-selected)/np.log(2.)
        return -(logsumexp(-costs.ravel())-np.log(costs.size))
    layers['direct_aggregation_fixed_branch']=table(
        lambda m:aggregate_from_margins(m,False),margins,margin_direction)
    layers['augmented_aggregation_fixed_branch']=table(
        lambda m:aggregate_from_margins(m,True),margins,margin_direction)

    # Branch traces for the original full function and segment projections.
    base_alive=np.asarray(base_trace['alive_pre'],bool)
    def summarize_full(terms,trace):
        alive=np.asarray(trace['alive_pre'],bool);n=int(alive.sum())
        risks=np.asarray(terms['strict_augmented_R_e']);finite=np.isfinite(risks)
        ordered=np.sort(risks[finite]) if finite.any() else np.array([])
        return dict(direct=float(terms['direct_R_CERT']),augmented=float(terms['augmented_R_CERT']),
            action_count=n,terminal_code=int(terms['terminal_code']),
            first_deadlock_step=int(terms['first_deadlock_step']),
            active_indices=np.asarray(terms['active_event_index']).tolist(),
            outer_tie_count=int(np.sum(np.abs(risks-np.max(risks))<=1e-10)),
            outer_top_gap=(float(ordered[-1]-ordered[-2]) if len(ordered)>1 else None),
            candidate_count=int(np.sum(np.asarray(trace['candidate'])[:n])),
            raw_deadlock_count=int(np.sum(np.asarray(trace['raw_deadlock'])[:n])))
    base_summary=summarize_full(base_terms,base_trace)
    branch_rows=[]
    for h in (1e-2,1e-3,1e-4,1e-5):
        for side,label in ((1,'plus'),(-1,'minus')):
            terms,trace=full_run(zero+side*h*direction)
            summary=summarize_full(terms,trace)
            n=min(int(np.asarray(trace['alive_pre']).sum()),int(base_alive.sum()))
            summary.update(h=h,side=label,
                alive_changes=int(np.sum(np.asarray(trace['alive_pre'])!=base_alive)),
                latch_changes=int(np.sum(np.asarray(trace['latch_pre'])!=np.asarray(base_trace['latch_pre']))),
                candidate_changes=int(np.sum(np.asarray(trace['candidate'])[:n]!=np.asarray(base_trace['candidate'])[:n])),
                event_code_changes=int(np.sum(np.asarray(trace['event_code'])!=np.asarray(base_trace['event_code']))))
            # Active sets of first, second and witness projections over genuine common steps.
            for key,array in (('first',trace['safe']),('second',trace['applied'])):
                now=np.stack([active_signature(array[t],trace['A'][t],trace['b'][t]) for t in range(n)])
                old=np.stack([active_signature(base_trace[key=='first' and 'safe' or 'applied'][t],
                    base_trace['A'][t],base_trace['b'][t]) for t in range(n)])
                summary[key+'_active_set_changed_steps']=int(np.sum(np.any(now!=old,axis=1)))
            noww=np.stack([[active_signature(trace['witnesses'][t,j],trace['A'][t],trace['b'][t])
                for j in range(8)] for t in range(n)])
            oldw=np.stack([[active_signature(base_trace['witnesses'][t,j],base_trace['A'][t],base_trace['b'][t])
                for j in range(8)] for t in range(n)])
            summary['witness_active_set_changed_step_indices']=int(np.sum(np.any(noww!=oldw,axis=(1,2))))
            branch_rows.append(summary)

    # Margin piecewise/sign evidence on representative h=1e-3 full traces.
    margin_branches=[]
    for side,label in ((0,'base'),(1,'plus'),(-1,'minus')):
        terms,trace=(base_terms,base_trace) if side==0 else full_run(zero+side*1e-3*direction)
        t=139+int(np.asarray(terms['active_event_index'])[1]);lo=t-100
        ss=jnp.concatenate((trace['before'][:1],trace['after']),axis=0)
        aa=jnp.linalg.norm(ss[lo-39:lo-39+101]-goals,axis=-1)
        mm=jax.vmap(lambda before,after,w,u,v,anchor:strict_atom_margins(
            before,after,w,u,v,goals,anchor))(trace['before'][lo:t+1],trace['after'][lo:t+1],
            trace['w'][lo:t+1],trace['applied'][lo:t+1],trace['witnesses'][lo:t+1],aa)
        mh=np.asarray(mm)
        margin_branches.append(dict(side=label,active_t=t,positive=int(np.sum(mh>1e-10)),
            negative=int(np.sum(mh<-1e-10)),zero=int(np.sum(np.abs(mh)<=1e-10)),
            direct_positive=int(np.sum(mh[:,:,0]>1e-10)),
            vi_positive=int(np.sum(mh[:,:,1:]>1e-10)),nonfinite=int(np.sum(~np.isfinite(mh)))))

    # KKT/status at the earliest intervention and active-risk actions.
    solver=[]
    for local_index in (0,19,20,50,150):
        t=100+local_index
        A=np.asarray(base_segment['A'][local_index]);b=np.asarray(base_segment['b'][local_index])
        entries=[('first',base_segment['nominal'][local_index],base_segment['safe'][local_index]),
                 ('second',base_segment['w'][local_index],base_segment['applied'][local_index])]
        entries.extend((f'witness_{j}',WITNESS_TARGETS[j],base_segment['witnesses'][local_index,j]) for j in range(8))
        for name,target,p in entries:
            solver.append(dict(action=t,projection=name,**kkt_report(target,p,A,b,cbf)))

    sources=['single_integrator/c1/risk/joint_frozen.py','single_integrator/c1/differentiable_rollout.py',
        'single_integrator/c1/risk/vi_r_cert.py','single_integrator/c1/rollout_vi_r_cert.py',
        'single_integrator/c1/termination.py','single_integrator/cbf.py','single_integrator/environment.py',
        'flowbc/giveway_flowbc_agent.py','scripts/diagnose_c1_vi_adfd_localize.py']
    report=dict(protocol=dict(initial=INITIAL.tolist(),noise_seed=NOISE_SEED,rollout_id=ROLLOUT_ID,
        intervention_actions=list(INTERVENTION),direction_seed=DIRECTION_SEED,
        direction=np.asarray(direction).tolist(),direction_norm=float(jnp.linalg.norm(direction)),
        cotangent_seed=COTANGENT_SEED,state_cotangent=np.asarray(state_cot).tolist(),
        epsilon=EPSILON,sign_tolerance=SIGN_TOLERANCE,h_values=H_VALUES.tolist(),
        segment_lengths=list(SEGMENT_LENGTHS),precision=dict(jax_x64=True,
            controller_arrays='float64',flow_bc_internal='float32 by frozen baseline_sample'),
        solver=cbf.to_dict(),environment=plant.to_dict(),jax=jax.__version__,numpy=np.__version__,
        scipy=scipy.__version__,python=platform.python_version(),backend=jax.default_backend(),
        checkpoint=str(baseline),checkpoint_sha256=digest(baseline),
        source_hashes={p:sha(ROOT/p) for p in sources}),
        full_same_function=dict(base_values=base_values.tolist(),tables=full_tables,
            base_branch=base_summary,branch_perturbations=branch_rows),
        segment=dict(checkpoint_position=np.asarray(checkpoint_position).tolist(),
            checkpoint_velocity=np.asarray(checkpoint_velocity).tolist(),parity=parity,
            tables={str(k):segment_fd[k] for k in SEGMENT_LENGTHS}),
        isolated_layers=layers,active_margin_branches=margin_branches,solver_checks=solver)
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_save(args.out,report)
    print(json.dumps(dict(out=str(args.out),base=base_summary,segment_parity=parity,
        full={k:dict(ad=v['ad'],best_relative=min(r['relative_error'] for r in v['rows']))
              for k,v in full_tables.items()}),indent=2))


if __name__=='__main__':main()
