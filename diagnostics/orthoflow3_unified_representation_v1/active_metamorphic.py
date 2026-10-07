"""New-encoding active-scene diagnostic; not a claim of upstream equivariance."""
import json
import numpy as np
import jax,jax.numpy as jnp
from . import build,train,analyze,representation as rep
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,evaluate_b as ev,metamorphic_b as mb
from new_benchmark_common.basin_dataset_v1 import TrainingRuntime

def run():
    models=train.load_models();repair.install();old=ev.load_models();rows=[]
    for sc in ('four_way_intersection','ring_exchange'):
        runtime=TrainingRuntime(sc,[])
        cohort=sorted([r for r in a.states() if r['scenario']==sc],key=lambda r:a.digest('phase-b-metamorphic|'+r['state_uid']))[:12]
        for row in cohort:
            key=jax.random.PRNGKey(a.learn.stable_int('phase-b-metamorphic',row['state_uid'])&0xffffffff)
            original,f0=mb.state_row(runtime,row,0,[0,1,2,3],key)
            u0=analyze.proposals(build.convert(original),models);o0=ev.proposals(repair.transformed(original),old)
            for k,order in [(1,[0,1,2,3]),(2,[0,1,2,3]),(3,[0,1,2,3]),(0,[1,2,3,0])]:
                other,f1=mb.state_row(runtime,row,k,order,key)
                u1=analyze.proposals(build.convert(other),models);o1=ev.proposals(repair.transformed(other),old)
                e={'scenario':sc,'state_uid':row['state_uid'],'rotation':90*k,'order':order,
                   'contract_exact_equivalence_claimed':False,'upstream_flow_error':float(np.max(np.abs(f1-mb.cc.rotate(f0,k)[order])))}
                for name,p0,p1 in [('old',o0,o1),('unified',u0,u1)]:
                    etas=np.asarray(p0['etas']);norm=(etas-a.learn.CENTER)/a.learn.RADIUS
                    if name=='unified':
                        def score(rr):
                            x=rep.batch([rep.entities(rep.parse(rr))]);xx={k:np.repeat(v,17,0) for k,v in x.items()}
                            return np.asarray(jax.nn.sigmoid(models[2].apply(models[3],xx,jnp.asarray(norm))))
                    else:
                        def score(rr):
                            h,c=mb.b.inputs(repair.transformed(rr))
                            return np.asarray(jax.nn.sigmoid(old[2].apply(old[3],jnp.asarray(h[None].repeat(17,0)),jnp.asarray(c[None].repeat(17,0)),jnp.asarray(norm),method=getattr(old[2],a.learn.METHOD[sc]))))
                    z0,z1=score(original),score(other)
                    e[name+'_proposal_error']=float(np.mean(np.linalg.norm((np.asarray(p1['etas'])-etas)/a.learn.RADIUS,axis=1)))
                    e[name+'_score_error']=float(np.mean(np.abs(z0-z1)))
                    e[name+'_top1_same']=bool(np.argmax(z0)==np.argmax(z1))
                rows.append(e)
    summary={}
    for sc in ('four_way_intersection','ring_exchange'):
        summary[sc]={}
        for deg in (0,90,180,270):
            rr=[r for r in rows if r['scenario']==sc and r['rotation']==deg]
            summary[sc][str(deg)]={key:float(np.mean([r[key] for r in rr])) for key in ('old_proposal_error','unified_proposal_error','old_score_error','unified_score_error','old_top1_same','unified_top1_same','upstream_flow_error')}
    build.dump('active_metamorphic.json',{'summary':summary,'rows':rows,
       'interpretation':'Active transformation retains the original MACFlow chart/weights; not an exact symmetry requirement in Four.'})
    return summary

if __name__=='__main__':print(json.dumps(run(),indent=2))
