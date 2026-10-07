"""Portable, geometry-legal response inputs. NO task success rollouts."""
import argparse, copy, os, time, types
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import db_transfer_data as data
from .db_transfer_data import OUT, ROOT, BASE, read, write, sha, goal_design
ROOTS=(2026100417,2026100499,2026100521,2026100543)
WIDTH=76

def runtime(scene, path):
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20}
    rt=rc.RichRuntime(scene)
    if sha(path)!=rt.core.checkpoint_sha:
        if scene=='toy_giveway':
            import jax,jax.numpy as jnp
            from single_integrator.evaluate import load_policy
            policy,_=load_policy(path);sample=jax.jit(lambda x,k:policy.sample_actions(x[None],seed=k)[0])
            def alt(env,key):
                u=np.asarray(sample(jnp.asarray(env.observation(),jnp.float32),key),float)
                return u*np.minimum(1.,rt.core.cfg.max_speed/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-30))
            rt.alt_flow=alt
        else:rt=rc.RichRuntime(scene,path)
    return rt

def instrument(rt):
    """Same sixteen response coordinates; safe direction at exact goal."""
    reset,project=rt.core.reset,rt.core.project
    trace=dict(runs=[],pending=[])
    def wrapped_project(env,u):
        v=project(env,u);trace['pending'].append((np.asarray(u).copy(),np.asarray(v).copy()));return v
    def wrapped_reset(p):
        env=reset(p);step=env.step;run=[];trace['runs'].append(run);trace['pending']=[]
        _,offset,_=goal_design(p);ex=offset[0];ey=np.stack([-ex[:,1],ex[:,0]],-1);v=rt.core.cfg.max_speed
        def local(a):return np.stack([(a*ex).sum(-1),(a*ey).sum(-1)],-1)/v
        def wrapped_step(action):
            assert 1<=len(trace['pending'])<=2
            flow,safe=trace['pending'][0];u=np.asarray(action)
            run.append(np.concatenate((local(flow),local(safe),local(u),np.linalg.norm(safe-flow,axis=1)[:,None]/v,np.linalg.norm(u-safe,axis=1)[:,None]/v),-1))
            trace['pending']=[];return step(action)
        env.step=wrapped_step;return env
    rt.core.reset=wrapped_reset;rt.core.project=wrapped_project
    return trace,reset,project

class Measure:
    def __init__(self, scene, path):
        self.rt=runtime(scene,path);self.trace,self.reset,self.project=instrument(self.rt)
        self.nominal={};self.points={}

    def h20(self,uid,p,eta):
        roots=ROOTS[:2];results=[];agents=[]
        def summarize(rows,goal):
            a={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            return [float(a['progress'].sum()),float(a['progress'][len(rows)//2:].mean()),goal,float(a['pair_clearance'].min()),float(a['pair_closing'].max()),float(a['safety'].mean()),float(a['safety'].max()),float((a['safety']>1e-6).mean()),float(a['speed'][-1]),float(a['flow_change'].mean())],a
        for root in roots:
            key=(uid,root)
            if key not in self.nominal:
                self.trace['runs']=[];value=self.rt.trajectory(p,np.zeros(3),root)
                self.nominal[key]=(value,np.array(self.trace['runs'][0]).mean(0))
            (nom,np0,ng),na=self.nominal[key]
            if np.array_equal(eta,np.zeros(3)):(corr,cp,cg),ca0=(nom,np0,ng),na
            else:
                self.trace['runs']=[];corr,cp,cg=self.rt.trajectory(p,eta,root);ca0=np.array(self.trace['runs'][0]).mean(0)
            n,_=summarize(nom,ng);c,ca=summarize(corr,cg)
            results.append(n+c[:8]+[float(ca['correction'].mean()),float(ca['correction'].max()),float((ca['correction']>1e-6).mean()),c[0]-n[0],float(np.linalg.norm(cp-np0,axis=1).mean()),c[3]-n[3]])
            agents.append(np.concatenate((na,ca0),axis=1).mean(0))
        out=np.r_[np.mean(results,0),np.mean(agents,0)].astype(np.float32)
        assert out.shape==(40,) and np.isfinite(out).all();return out

    def goal(self,uid,p,eta,moving):
        import jax
        radius,offsets,_=goal_design(p);assert radius is not None
        key=(uid,moving);rt=self.rt;v=rt.core.cfg.max_speed
        if key not in self.points:
            points=[]
            for di,offset in enumerate(offsets):
                pp=copy.deepcopy(p);pp.update(positions=(np.array(p['goals'])+radius*offset).tolist(),velocities=(-.5*v*offset if moving else np.zeros_like(offset)).tolist(),flow_committed=False)
                env=self.reset(pp);assert data.clearance(pp,env.positions)>1e-4
                along=(env.goals-env.positions);along/=np.linalg.norm(along,axis=1,keepdims=True)
                aa=[]
                for root in ROOTS:
                    raw=np.asarray((rt.alt_flow or rt.base_flow)(env,jax.random.fold_in(jax.random.PRNGKey(root),di)),float)
                    safe=self.project(env,raw);basis=rt.core.basis.compute(env.positions,env.goals,safe,v)
                    aa.append((env,along,raw,safe,basis))
                points.append(aa)
            self.points[key]=points
        result=[]
        for points in self.points[key]:
            arr=[]
            for env,along,raw,safe,basis in points:
                action=self.project(env,safe+basis.correction(eta))
                a,b,c=[(u*along).sum(1)/v for u in (raw,safe,action)]
                arr.append([a.mean(),b.mean(),c.mean(),c.min()])
            result.extend(np.mean(arr,0))
        out=np.asarray(result,np.float32);assert np.isfinite(out).all();return out

    def one(self,uid,p,eta):
        x=np.zeros(WIDTH,np.float32);errors=[];radius,_,_=goal_design(p)
        x[72]=radius/.30 if radius else 0.
        for start,end,flag,fn in ((0,40,73,lambda:self.h20(uid,p,eta)),(40,56,74,lambda:self.goal(uid,p,eta,False)),(56,72,75,lambda:self.goal(uid,p,eta,True))):
            try:x[start:end]=fn();x[flag]=1.
            except Exception as exc:errors.append(dict(group=start,error=f'{type(exc).__name__}: {exc}'))
        return x,errors

def source(worker,workers=5):
    pairs=read(OUT/'pairs.json');states=read(OUT/'states.json');profiles=read(OUT/'controllers.json')
    d=np.load(ROOT/'state_breadth_training/dataset.npz')
    chosen=[i for i in range(len(pairs)) if i%workers==worker]
    values=np.zeros((len(chosen),WIDTH),np.float32);timing=np.zeros(len(chosen));errors=[];reused=0
    handles={};order=sorted(enumerate(chosen),key=lambda v:(pairs[v[1]]['scenario'],pairs[v[1]]['controller_uid'],pairs[v[1]]['state_uid']))
    current=None;measure=None
    for loc,i in order:
        r=pairs[i];p=states[r['state_index']]['physical'];start=time.perf_counter()
        if 'cache_indices' in r:
            ci,j=r['cache_indices'];radius,_,_=goal_design(p);assert radius==.30
            values[loc]=np.r_[d['context'][ci,j],d['agent_response'][ci,j].mean(0),d['goal_response'][ci,j],d['goal_motion_response'][ci,j],1.,1.,1.,1.]
            reused+=1
        else:
            if current!=r['controller_uid']:
                profile=profiles[r['controller_uid']];assert sha(profile['path'])==profile['sha256']
                measure=Measure(r['scenario'],profile['path']);current=r['controller_uid']
            values[loc],err=measure.one(r['state_uid'],p,np.array(r['eta'],float))
            errors.extend(dict(pair_index=i,**e) for e in err)
        timing[loc]=time.perf_counter()-start
    dest=OUT/'features';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'worker{worker}.npz',indices=np.array(chosen),context=values,seconds=timing)
    write(dest/f'worker{worker}.json',dict(worker=worker,pairs=len(chosen),exact_derived_inputs_reused=reused,invalid_groups=errors,
        code_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),pair_manifest_sha256=sha(OUT/'pairs.json'),new_rollouts=0,
        cpu_context_only=True,mask_semantics='Unknown measurement, not failure label',seconds=float(timing.sum())))
    print(dict(worker=worker,pairs=len(chosen),reused=reused,invalid=len(errors),seconds=float(timing.sum())),flush=True)

def materialize():
    pairs=read(OUT/'pairs.json');contexts=np.zeros((len(pairs),WIDTH),np.float32);saw=[];docs=[]
    for worker in range(5):
        doc=read(OUT/'features'/f'worker{worker}.json');assert doc['code_sha256']==sha(__file__)
        z=np.load(OUT/'features'/f'worker{worker}.npz');contexts[z['indices']]=z['context'];saw.extend(z['indices'].tolist());docs.append(doc)
    assert sorted(saw)==list(range(len(pairs))) and np.isfinite(contexts).all()
    np.savez_compressed(OUT/'source_context.npz',context=contexts)
    write(OUT/'context_audit.json',dict(pairs=len(pairs),reused=sum(d['exact_derived_inputs_reused'] for d in docs),invalid_groups=[e for d in docs for e in d['invalid_groups']],all_rows_present=True,new_rollouts=0,code_sha256=sha(__file__),sha256=sha(OUT/'source_context.npz')))
    print(dict(source_context_complete=True,pairs=len(pairs),invalid=sum(len(d['invalid_groups']) for d in docs)),flush=True)

def smoke():
    pairs=read(OUT/'pairs.json');states=read(OUT/'states.json');profiles=read(OUT/'controllers.json');audit=[]
    for scene in data.SCENES:
        i=next(i for i,r in enumerate(pairs) if r['scenario']==scene and r['pool']=='historical_wide')
        r=pairs[i];p=states[r['state_index']]['physical'];rt=Measure(scene,profiles[r['controller_uid']]['path'])
        x,errors=rt.one(r['state_uid'],p,np.array(r['eta']))
        assert not errors,(scene,errors)
        old=np.array(rt.rt.features(p,np.array(r['eta']))['mean'])
        error=float(np.max(abs(x[:24]-old)));assert error<2e-5,(scene,error)
        if scene=='ring_exchange':
            from .goal_response import measurement as rest
            from .goal_velocity_response import measurement as motion
            if rt.rt.alt_flow is None:rt.rt.alt_flow=rt.rt.base_flow
            # These functions query the same legal0.30m goal states and RNGs.
            error2=float(np.max(abs(x[40:72]-np.r_[rest(rt.rt,p,np.array(r['eta'])),motion(rt.rt,p,np.array(r['eta']))])))
            assert error2<2e-5,error2
        else:error2=None
        audit.append(dict(scene=scene,radius=float(x[72]*.30),H20_reference_error=error,goal_reference_error=error2,all_valid=True))
    write(OUT/'context_smoke.json',dict(cases=audit,new_rollouts=0,code_sha256=sha(__file__)))
    print(audit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['source','materialize','smoke']);p.add_argument('--worker',type=int,default=0);a=p.parse_args()
    source(a.worker) if a.action=='source' else globals()[a.action]()
