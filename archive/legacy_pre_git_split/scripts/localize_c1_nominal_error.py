"""Inspect Flow, clipping and speed limiting on cached local state segments."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.train_deadlock_union import setup,noise
from single_integrator.c1.train import observation
from single_integrator.c1.differentiable_rollout import bounded_nominal
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.environment import GiveWayEnv


def main():
    source=ROOT/'results/c1_directional_closed_loop_5sec_v2'
    out=ROOT/'results/c1_nominal_localization_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    protocol=json.loads((source/'protocol.json').read_text());analysis=json.loads((source/'analysis.json').read_text())
    _,field,plant,_,baseline=setup();goals=jnp.asarray(GiveWayEnv(plant).goals)
    def stages(x,v,xi):
        observations=observation(x[None],v[None],goals)
        with jax.experimental.disable_x64():
            obs,_=field.baseline._flatten(jnp.asarray(observations,jnp.float32),jnp.zeros((1,2,2),jnp.float32))
            h=jnp.asarray(xi[None],jnp.float32)
            for step in range(field.baseline.config['flow_steps']):
                s=jnp.full((1,1),jnp.asarray(step/field.baseline.config['flow_steps'],jnp.float32),jnp.float32)
                h=h+field.baseline.network.select('actor_bc_flow')(obs,h,s)/field.baseline.config['flow_steps']
            if field.baseline.config.get('normalize',False):
                h=h*jnp.asarray(field.baseline.config['act_scale'])+jnp.asarray(field.baseline.config['act_mean'])
            raw=h[0];clipped=jnp.clip(raw,jnp.asarray(-1.,jnp.float32),jnp.asarray(1.,jnp.float32))
        bounded=bounded_nominal(clipped[None],plant.max_speed)[0]
        return raw,clipped,bounded
    evaluate=jax.jit(stages)
    derivative=jax.jit(lambda x,v,xi,dx,dv:jax.jvp(lambda a,c:stages(a,c,xi),(x,v),(dx,dv)))
    rows=[];parity=[]
    for item in analysis['rows']:
        first=item['first_tolerance_exceeded_step']
        if first is None:continue
        rid,sign=item['rid'],item['sign'];initial=np.asarray(protocol['starts'][rid])
        draws=noise(protocol['noise_seed'],protocol['rid_offset']+rid)
        for h in protocol['finite_difference_steps']:
            with np.load(source/f'trace_{rid}_{sign}_{h}.npz') as z:
                base=np.concatenate((initial[None],z['base']));actual=np.concatenate((initial[None],z['perturbed']))
            for t in range(max(0,first-3),min(protocol['steps'],first+2)):
                x=base[t];v=(base[t]-base[t-1])/plant.dt if t else np.zeros_like(initial)
                ax=actual[t];av=(actual[t]-actual[t-1])/plant.dt if t else np.zeros_like(initial)
                dx=(ax-x)/h;dv=(av-v)/h
                values,tangents=derivative(*map(jnp.asarray,(x,v,draws[t],dx,dv)))
                authoritative=field.baseline_sample(observation(jnp.asarray(x[None]),jnp.asarray(v[None]),goals),draws[t:t+1])[0]
                parity.append(float(np.max(np.abs(np.asarray(authoritative)-np.asarray(values[1])))))
                probes=[]
                for factor in (1.,.1,.01):
                    eps=h*factor
                    outputs=evaluate(jnp.asarray(x+eps*dx),jnp.asarray(v+eps*dv),draws[t])
                    errors={name:float(np.linalg.norm((np.asarray(a)-np.asarray(b))/eps-np.asarray(d)))
                        for name,a,b,d in zip(('raw','clipped','bounded'),outputs,values,tangents)}
                    braw,araw=np.asarray(values[0]),np.asarray(outputs[0])
                    bclip,aclip=np.asarray(values[1]),np.asarray(outputs[1])
                    probes.append(dict(factor=factor,epsilon=eps,derivative_residual=errors,
                        clipping_branch_changes=int(np.count_nonzero(np.sign(braw)* (np.abs(braw)>=1)!=np.sign(araw)*(np.abs(araw)>=1))),
                        speed_branch_changes=int(np.count_nonzero((np.linalg.norm(bclip.reshape(2,2),axis=1)>=.5)!=(np.linalg.norm(aclip.reshape(2,2),axis=1)>=.5)))))
                rows.append(dict(rid=rid,sign=sign,step=t+1,h=h,probes=probes))
    atomic_save(out/'records.json',rows)
    sources=['scripts/localize_c1_nominal_error.py','single_integrator/c1/differentiable_rollout.py','single_integrator/c1/train.py']
    for name in sources:
        dst=out/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
    result=dict(cached_points=len(rows),isolated_probe_evaluations=len(rows)*3,new_closed_loop_trajectories=0,
        max_clipped_forward_parity=max(parity),baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        scope='local finite-step derivative residual decomposition, not efficacy or global correctness')
    atomic_save(out/'complete.json',result)
    if max(parity)>1e-7:raise RuntimeError('instrumented Flow parity check failed')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
