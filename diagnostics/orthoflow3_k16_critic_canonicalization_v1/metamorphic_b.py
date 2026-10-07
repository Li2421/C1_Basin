"""Learned passive/active chart checks, restricted to v2 train/dev states."""
import json,copy
import jax,jax.numpy as jnp
import numpy as np
from flax import serialization
from pathlib import Path
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,evaluate_b as ev,canonical_conditioning as cc
from diagnostics.ring_fourway_symmetry_conditioning_audit.run_audit import make_env
from new_benchmark_common.basin_dataset_v1 import TrainingRuntime


def state_row(runtime,row,k,order,key):
    physical=a.decode(row['structured_state'])
    phy={**physical,**{n:cc.rotate(physical[n],k)[order].tolist() for n in ('positions','velocities','goals')}}
    env=make_env(runtime,phy);flow=runtime.flow_world(env,key)
    flat=np.r_[runtime.observation(env).ravel(),flow.ravel()]
    return {**row,'structured_state':json.dumps(phy),'conditioning':json.dumps({'flat':flat.tolist()})},flow


def main():
    models=ev.load_models();norm=a.load(a.frozen.NORMALIZATION)
    gm,gp,cm,cp=a.fresh.load_models(norm)
    cp=serialization.from_bytes(cp,Path(a.load(a.OUT/'phase_a/critic_frozen.json')['checkpoint']).read_bytes())
    result=[]
    for sc in ('four_way_intersection','ring_exchange'):
        runtime=TrainingRuntime(sc,[])
        cohort=sorted([r for r in a.states() if r['scenario']==sc],key=lambda r:a.digest('phase-b-metamorphic|'+r['state_uid']))[:12]
        for row in cohort:
            key=jax.random.PRNGKey(a.learn.stable_int('phase-b-metamorphic',row['state_uid'])&0xffffffff)
            original,f0=state_row(runtime,row,0,[0,1,2,3],key)
            co=b.transformed(original);base=ev.proposals(co,models);h0,c0=b.inputs(co)
            oldh0,oldc0=a.inputs(original,norm)
            noise=jnp.asarray(np.random.default_rng(4183).standard_normal((16,3)),jnp.float32)
            def oldprop(h,c):
                raw=gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=getattr(gm,a.learn.METHOD[sc]))[0]
                return np.asarray([a.learn.eta_mean(raw),*a.learn.eta_from_noise(raw[None].repeat(16,0),noise)])
            oldeta=oldprop(oldh0,oldc0)
            for k,order in [(1,[0,1,2,3]),(2,[0,1,2,3]),(3,[0,1,2,3]),(0,[1,2,3,0])]:
                other,ff=state_row(runtime,row,k,order,key);ct=b.transformed(other);h,c=b.inputs(ct)
                pp=ev.proposals(ct,models,seed_identity=row['state_uid']);oh,oc=a.inputs(other,norm);oe=oldprop(oh,oc)
                def logits(model,params,h,c,eta):
                    return np.asarray(model.apply(params,jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),
                             jnp.asarray((np.asarray(eta)-a.learn.CENTER)/a.learn.RADIUS),method=getattr(model,a.learn.METHOD[sc])))
                z0=logits(models[2],models[3],h0,c0,base['etas']);z1=logits(models[2],models[3],h,c,base['etas'])
                oz0=logits(cm,cp,oldh0,oldc0,oldeta);oz1=logits(cm,cp,oh,oc,oldeta)
                result.append({'scenario':sc,'state_uid':row['state_uid'],'rotation':90*k,'order':order,
                    'active_flow_error':float(np.max(np.abs(ff-cc.rotate(f0,k)[order]))),
                    'canonical_h_error':float(np.max(np.abs(h-h0))),
                    'canonical_proposal_error':float(np.mean(np.linalg.norm((np.asarray(pp['etas'])-base['etas'])/a.learn.RADIUS,axis=1))),
                    'old_proposal_error':float(np.mean(np.linalg.norm((oe-oldeta)/a.learn.RADIUS,axis=1))),
                    'canonical_score_error':float(np.mean(np.abs(jax.nn.sigmoid(z0)-jax.nn.sigmoid(z1)))),
                    'old_score_error':float(np.mean(np.abs(jax.nn.sigmoid(oz0)-jax.nn.sigmoid(oz1)))),
                    'canonical_top1_equal':bool(np.argmax(z0)==np.argmax(z1)),
                    'old_top1_equal':bool(np.argmax(oz0)==np.argmax(oz1))})
    summary={}
    for sc in ('four_way_intersection','ring_exchange'):
        summary[sc]={}
        for deg in (0,90,180,270):
            rr=[r for r in result if r['scenario']==sc and r['rotation']==deg]
            summary[sc][str(deg)]={k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ('scenario','state_uid','rotation','order')}
    b.dump('active_metamorphic.json',{'summary':summary,'rows':result});print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
