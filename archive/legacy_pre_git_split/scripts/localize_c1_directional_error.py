"""Decompose cached directional errors using isolated controller evaluations.

No new closed-loop trajectories, no risk evaluation, no optimization.
"""
from pathlib import Path
import hashlib,json,sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup
from single_integrator.c1.train import observation
from single_integrator.c1.differentiable_rollout import bounded_nominal,barrier_constraints
from single_integrator.c1.projection_directional import projection_direction
from single_integrator.cbf import project_velocity
from single_integrator.environment import GiveWayEnv
from single_integrator.c1.training.persistence import atomic_save


def main():
    source=ROOT/'results/c1_directional_closed_loop_5sec_v2'
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/c1_directional_localization_v1')
    out=parser.parse_args().out
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    protocol=json.loads((source/'protocol.json').read_text())
    analysis=json.loads((source/'analysis.json').read_text())
    params,field,plant,cbf,_=setup();env=GiveWayEnv(plant)
    goals,walls=jnp.asarray(env.goals),jnp.asarray(env.walls)
    tangent=jax.tree_util.tree_map(jnp.zeros_like,params)
    tangent['params']['zero_initialized_output']['bias']=jnp.asarray(protocol['bias_direction'])
    def pre(x,v,xi):
        obs=observation(x[None],v[None],goals)
        nom=bounded_nominal(field.baseline_sample(obs,xi[None]),plant.max_speed)[0]
        A,b,_=barrier_constraints(x,walls,plant.to_dict(),cbf)
        return nom,A,b,obs
    pre_jvp=jax.jit(lambda x,v,xi,dx,dv:jax.jvp(lambda a,c:pre(a,c,xi),(x,v),(dx,dv)))
    corr=lambda phi,obs,safe:field.correction(phi,obs,safe[None])[0]
    cj=jax.jit(lambda obs,safe,dp,dobs,ds:jax.jvp(corr,(params,obs,safe),(dp,dobs,ds)))
    solve=lambda y,A,b:project_velocity(np.asarray(y),np.asarray(A),np.asarray(b),.5,cbf)[0].reshape(4)
    def direction(x,v,xi,dx,dv,dp):
        (nom,A,b,obs),(dn,da,db,dobs)=pre_jvp(*map(jnp.asarray,(x,v,xi,dx,dv)))
        safe=solve(nom,A,b);ds,c1=projection_direction(nom,A,b,safe,dn,dA=da,db=db)
        c,dc=cj(obs,jnp.asarray(safe),dp,dobs,jnp.asarray(ds))
        candidate=safe+np.asarray(c);u=solve(candidate,A,b)
        du,c2=projection_direction(candidate,A,b,u,ds+np.asarray(dc),dA=da,db=db)
        return du,dict(first=c1,second=c2),dict(nom=np.asarray(nom),A=np.asarray(A),b=np.asarray(b),
            dn=np.asarray(dn),da=np.asarray(da),db=np.asarray(db),safe=safe,ds=ds,candidate=candidate,u=u)
    from single_integrator.c1.train_deadlock_union import noise
    rows=[]
    for item in analysis['rows']:
        first=item['first_tolerance_exceeded_step']
        if first is None:continue
        rid,sign=item['rid'],item['sign'];initial=np.asarray(protocol['starts'][rid])
        dp=jax.tree_util.tree_map(lambda x:sign*x,tangent)
        draws=noise(protocol['noise_seed'],protocol['rid_offset']+rid)
        # Fixed neighborhood of first failure, no search for favorable steps.
        indices=range(max(0,first-3),min(protocol['steps'],first+2))
        for h in protocol['finite_difference_steps']:
            with np.load(source/f'trace_{rid}_{sign}_{h}.npz') as saved:
                base=np.concatenate((initial[None],saved['base']))
                actual=np.concatenate((initial[None],saved['perturbed']))
                pred=np.concatenate((np.zeros_like(initial)[None],saved['directional_positions']))
            fd=(actual-base)/h
            for t in indices:
                v=(base[t]-base[t-1])/plant.dt if t else np.zeros_like(initial)
                vp=(pred[t]-pred[t-1])/plant.dt if t else np.zeros_like(initial)
                vf=(fd[t]-fd[t-1])/plant.dt if t else np.zeros_like(initial)
                du_pred,cert,_=direction(base[t],v,draws[t],pred[t],vp,dp)
                du_incoming,_,stages=direction(base[t],v,draws[t],fd[t],vf,dp)
                av=(actual[t]-actual[t-1])/plant.dt if t else np.zeros_like(initial)
                (an,aa,ab,_),_=pre_jvp(jnp.asarray(actual[t]),jnp.asarray(av),draws[t],
                    jnp.zeros((2,2)),jnp.zeros((2,2)))
                actual_safe=solve(an,aa,ab)
                actual_u=solve(actual_safe+sign*h*np.asarray(protocol['bias_direction']),aa,ab)
                dnom=(np.asarray(an)-stages['nom'])/h
                dmat=(np.asarray(aa)-stages['A'])/h;dbound=(np.asarray(ab)-stages['b'])/h
                ds_secant=(actual_safe-stages['safe'])/h
                first_error=second_error=None;unsupported={}
                try:
                    first_with_secants,_=projection_direction(stages['nom'],stages['A'],stages['b'],stages['safe'],
                        dnom,dA=dmat,db=dbound)
                    first_error=float(np.linalg.norm(ds_secant-first_with_secants))
                except (ValueError,RuntimeError) as error:
                    unsupported['first_secant']=str(error)
                try:
                    second_with_secants,_=projection_direction(stages['candidate'],stages['A'],stages['b'],stages['u'],
                        ds_secant+sign*np.asarray(protocol['bias_direction']),dA=dmat,db=dbound)
                    second_error=float(np.linalg.norm((actual_u-stages['u'])/h-second_with_secants))
                except (ValueError,RuntimeError) as error:
                    unsupported['second_secant']=str(error)
                du_fd=((fd[t+1]-fd[t])/plant.dt).reshape(4)
                cached_du=((pred[t+1]-pred[t])/plant.dt).reshape(4)
                rows.append(dict(rid=rid,sign=sign,step=t+1,h=h,
                    derivative_reconstruction_error=float(np.linalg.norm(du_pred-cached_du)),
                    total_action_derivative_error=float(np.linalg.norm(du_pred-du_fd)),
                    incoming_error_response=float(np.linalg.norm(du_pred-du_incoming)),
                    local_finite_step_residual=float(np.linalg.norm(du_incoming-du_fd)),
                    nominal_derivative_residual=float(np.linalg.norm(dnom-stages['dn'])),
                    geometry_A_derivative_residual=float(np.linalg.norm(dmat-stages['da'])),
                    geometry_b_derivative_residual=float(np.linalg.norm(dbound-stages['db'])),
                    first_projection_secant_residual=first_error,
                    second_projection_secant_residual=second_error,unsupported_secant_checks=unsupported,
                    actual_control_reconstruction_error=float(np.linalg.norm(actual_u-((actual[t+1]-actual[t])/plant.dt).reshape(4))),
                    incoming_position_derivative_error=float(np.linalg.norm(pred[t]-fd[t])),
                    incoming_velocity_derivative_error=float(np.linalg.norm(vp-vf)),certificates=cert))
    atomic_save(out/'records.json',rows)
    summaries=[]
    for rid,sign in sorted({(r['rid'],r['sign']) for r in rows}):
        summaries.append(dict(rid=rid,sign=sign,scales=[dict(h=h,
            max_local_residual=max(r['local_finite_step_residual'] for r in rows if (r['rid'],r['sign'],r['h'])==(rid,sign,h)),
            max_incoming_response=max(r['incoming_error_response'] for r in rows if (r['rid'],r['sign'],r['h'])==(rid,sign,h)))
            for h in protocol['finite_difference_steps']]))
    sources=['scripts/localize_c1_directional_error.py','single_integrator/c1/projection_directional.py']
    for name in sources:
        dst=out/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
    result=dict(scope='cached state one-step error decomposition, not probability gradient or efficacy',
        evaluated_cached_states=len(rows),new_closed_loop_trajectories=0,summaries=summaries,
        max_derivative_reconstruction_error=max(r['derivative_reconstruction_error'] for r in rows),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        parent_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest())
    atomic_save(out/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
