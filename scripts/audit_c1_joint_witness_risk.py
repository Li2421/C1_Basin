"""Offline joint-risk experiment; never imported by controller/training code.

Run with .venv-c1/bin/python scripts/audit_c1_joint_witness_risk.py --phase all
Local scoring uses exact convex problem definitions with numerical certificates.
No policy training, controller changes, or outcome-conditioned parameter fitting.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse
from scipy.optimize import nnls, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv

PROTOCOL = dict(version='joint_witness_offline_v1', s_ref=.5, eta=.25,
                delta=.2, tau=.05, stride=20, control_zero_tol=1e-12,
                witness_tol=1e-9, solver_tol=1e-11,
                progress_window_seconds=4., delta_progress=.02,
                tau_progress=.005, unfinished_temperature=.02,
                terminal_distance_scale=.1, power=4.,
                split='primary rollout_id % 5 == 0 held out; other IDs development',
                candidate_convention='Safety: stored u_safe (zero-residual C1); C1: c1_candidate',
                trajectory_revision='equal_exposure_then_equal_terminal_v1',
                scope='offline association and local counterfactuals, not learned-policy or causal liveness proof')

# Numerical accounting only; these values never enter risk or candidate ranking.
SOLVER_AUDIT = dict(primary_calls=0, primary_failures=0, fallback_attempts=0,
                    fallback_successes=0, fallback_failures=0)

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def save(path,obj):
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2)+'\n')

class Projection:
    def __init__(self,A,b,speeds=None,d=2):
        self.A,self.b=np.asarray(A,float),np.asarray(b,float)
        self.m=self.A.shape[1];self.speeds=speeds;self.d=d
        rows=[-self.A];rhs=[-self.b];cones=[]
        if len(b): cones.append(clarabel.NonnegativeConeT(len(b)))
        if speeds is not None:
            for a,s in enumerate(speeds):
                block=np.zeros((d+1,self.m));block[1:,a*d:(a+1)*d]=-np.eye(d)
                rows.append(block);rhs.append(np.r_[s,np.zeros(d)])
                cones.append(clarabel.SecondOrderConeT(d+1))
        matrix=np.vstack(rows);lower=np.concatenate(rhs)
        settings=clarabel.DefaultSettings();settings.verbose=False
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=PROTOCOL['solver_tol']
        settings.max_iter=200
        # Audited equivalent QP configuration: the base projection has no SOCs.
        # Default equilibration stalled on otherwise well-posed, strictly convex
        # QPs. Disabling it preserves A,b,P,q and all mathematical definitions.
        if speeds is None: settings.equilibrate_enable=False
        self.solver=clarabel.DefaultSolver(sparse.eye(self.m,format='csc'),np.zeros(self.m),
                    sparse.csc_matrix(matrix),lower,cones,settings)
        self.calls=0;self.max_violation=0.;self.max_stationarity=0.
        self.conic_A=sparse.csc_matrix(matrix)
    def __call__(self,target):
        y=np.asarray(target,float)
        self.solver.update(q=-y)
        sol=self.solver.solve();self.calls+=1;SOLVER_AUDIT['primary_calls']+=1
        if str(sol.status) not in ('Solved','AlmostSolved'):
            return self._fallback(y,'projection '+str(sol.status))
        p=np.asarray(sol.x);z=np.asarray(sol.z)
        violation=max(0.,float(np.max(self.b-self.A@p,initial=0.)))
        if self.speeds is not None:
            violation=max(violation,float(np.max(np.linalg.norm(p.reshape(-1,self.d),axis=1)-self.speeds)))
        stationarity=float(np.max(np.abs(p-y+self.conic_A.T@z)))
        self.max_violation=max(self.max_violation,violation)
        self.max_stationarity=max(self.max_stationarity,stationarity)
        # SOC complementarity is a cone inner product, not componentwise.
        info=self.solver.get_info()
        if violation>2e-8 or stationarity>2e-7 or info.gap_abs>2e-7:
            return self._fallback(y,f'uncertified projection {violation} {stationarity} gap={info.gap_abs}')
        return p

    def _fallback(self,y,reason):
        """Same strictly convex projection, accepted only with sufficient KKT checks.

        Speed balls remain exact quadratic inequalities, never polygonal bounds.
        A failed certificate stays an explicit numerical failure, not a risk value.
        """
        SOLVER_AUDIT['primary_failures']+=1;SOLVER_AUDIT['fallback_attempts']+=1
        def constraints(p):
            linear=self.A@p-self.b
            if self.speeds is None:return linear
            return np.r_[linear,self.speeds**2-np.sum(p.reshape(-1,self.d)**2,axis=1)]
        def jacobian(p):
            if self.speeds is None:return self.A
            balls=np.zeros((len(self.speeds),self.m))
            for i in range(len(self.speeds)):balls[i,i*self.d:(i+1)*self.d]=-2*p[i*self.d:(i+1)*self.d]
            return np.vstack([self.A,balls])
        result=minimize(lambda p:.5*np.sum((p-y)**2),np.zeros(self.m),jac=lambda p:p-y,
            method='SLSQP',constraints={'type':'ineq','fun':constraints,'jac':jacobian},
            options={'ftol':1e-12,'maxiter':200,'disp':False})
        p=np.asarray(result.x);slack=constraints(p)
        try:
            active=slack<=1e-7
            multipliers=(nnls(jacobian(p)[active].T,p-y,maxiter=10000)[0]
                         if np.any(active) else np.empty(0))
            stationarity=float(np.max(np.abs(p-y-jacobian(p)[active].T@multipliers)))
            complementarity=float(np.max(np.abs(multipliers*slack[active]),initial=0.))
            violation=float(max(0.,-np.min(self.A@p-self.b,initial=0.)))
            if self.speeds is not None:
                violation=max(violation,float(np.max(np.linalg.norm(p.reshape(-1,self.d),axis=1)-self.speeds)))
            if not np.isfinite(p).all() or violation>2e-8 or stationarity>2e-7 or complementarity>2e-8:
                raise RuntimeError(f'KKT violation={violation} stationarity={stationarity} complementarity={complementarity}')
        except Exception as error:
            SOLVER_AUDIT['fallback_failures']+=1
            raise RuntimeError(f'{reason}; equivalent fallback uncertified: {error}') from error
        self.max_violation=max(self.max_violation,violation);self.max_stationarity=max(self.max_stationarity,stationarity)
        SOLVER_AUDIT['fallback_successes']+=1
        return p

class LocalRisk:
    def __init__(self,A,b,speeds,d=2):
        self.A,self.b=np.asarray(A,float),np.asarray(b,float)
        self.speeds=np.asarray(speeds,float);self.d=d;self.m=self.A.shape[1]
        if np.max(self.b,initial=0.)>0: raise ValueError('zero-infeasible state outside local-risk domain')
        self.physical=Projection(A,b,self.speeds,d)
        self.base=Projection(A,np.asarray(b)/PROTOCOL['s_ref'])
        self.exact_boundary=self.A[self.b==0.]
        self.witnesses=[];self.ambiguous=0;self.certified_zero=0
        for j in range(self.m):
            for sign in (-1,1):
                y=sign*self.speeds[j//d]*np.eye(self.m)[j]
                direction=self.direction(y)
                if direction is not None:self.witnesses.append(direction)
    def zero_certificate(self,y):
        if not len(self.exact_boundary):return False
        lam,res=nnls(self.exact_boundary.T,-y,maxiter=10000)
        return res<=1e-10*max(1.,np.linalg.norm(y))
    def direction(self,y):
        p=self.physical(y)
        if self.zero_certificate(y):
            self.certified_zero+=1;return None
        norm=np.linalg.norm(p)
        if norm<=PROTOCOL['witness_tol']:
            self.ambiguous+=1;return None
        return p/norm
    def score(self,v):
        v=np.asarray(v,float);norm=np.linalg.norm(v)
        if norm<=PROTOCOL['control_zero_tol']:
            return dict(valid=False,reason='undefined_zero_candidate')
        q=v/norm;p=self.base(q)
        B=float(np.sum((q-p)**2))
        if self.zero_certificate(q):B=1.
        c=float(np.min(self.speeds/np.maximum(np.linalg.norm(q.reshape(-1,self.d),axis=1),1e-300)))
        own=self.direction(c*q)
        ds=list(self.witnesses)
        if own is not None:ds.append(own)
        if not ds:return dict(valid=False,reason='no_resolved_nonzero_witness',B=B)
        cos=np.asarray(ds)@q;M=float(np.max(cos))
        E=float(PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']-M)/PROTOCOL['tau']))
        risk=B+PROTOCOL['eta']*E
        lower=B+PROTOCOL['eta']*PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']-1)/PROTOCOL['tau'])
        return dict(valid=True,B=B,M=M,E=E,risk=float(risk),
                    risk_lower=float(lower if self.ambiguous else risk),
                    ambiguous_witnesses=self.ambiguous,winner=int(np.argmax(cos)),
                    solver_calls=self.physical.calls+self.base.calls)

def synthetic(out):
    cases=[]
    def case(name,A,b,v,speed=.5):
        risk=LocalRisk(np.asarray(A).reshape(-1,len(v)),b,np.full(len(v)//2,speed))
        result=risk.score(v);applied=risk.physical(v)
        row=dict(name=name,**result,applied_norm=np.linalg.norm(applied));cases.append(row)
        return risk,row
    q=np.array([-.8,-.5,-.3,-.1]);q/=np.linalg.norm(q)
    r,orth=case('negative_orthant',np.eye(4),np.zeros(4),.1*q)
    assert abs(orth['B']-1)<1e-8 and orth['risk']>=1
    scale=[r.score(.1*q*c)['risk'] for c in (.001,.1,1.,10.)]
    fdstep=2e-5
    grad=np.array([(r.score(.1*q+fdstep*e)['risk']-r.score(.1*q-fdstep*e)['risk'])/(2*fdstep) for e in np.eye(4)])
    newq=q-.01*(grad-q*np.dot(q,grad));newq/=np.linalg.norm(newq)
    descent=r.score(.1*newq)['risk']
    assert max(scale)-min(scale)<1e-6 and descent<orth['risk']
    A=np.vstack([np.eye(4)[0],-np.eye(4)[0]])
    _,plane=case('hyperplane_tangent',A,[0.,0.],[0.,.1,0.,0.])
    assert plane['B']<1e-8 and plane['M']>.999
    e=.001;c=np.sqrt(1-e*e)
    _,near=case('near_opposite',[[e,c,0,0],[e,-c,0,0]],[0.,0.],[-.1,0,0,0])
    assert abs(near['B']-1)<1e-8 and near['risk']>=1
    # Ray with speed cap eps: remaining coordinates are fixed to zero.
    for eps in [1e-3,1e-6]:
        A=np.vstack([np.eye(4)[0],-np.eye(4)[0],np.eye(4)[1:],-np.eye(4)[1:]])
        _,slow=case('slow_ray_'+str(eps),A,np.r_[0.,-eps,np.zeros(6)],[.1,0,0,0])
        assert slow['M']>.999 and slow['applied_norm']>0
    # Two independent 2D ray tests say blocked, but common translation remains.
    _,joint=case('joint_coupling_counterexample',[[1,0,-1,0]],[0.],[-.2,0,.1,0])
    assert joint['B']<.999 and joint['applied_norm']>.01 and joint['M']>0
    # Rotation invariance is not assumed for a coordinate-witness approximation.
    rotation=[]
    for angle in [0.,.2,.6,1.]:
        t=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        Q=sparse.block_diag([t,t]).toarray()
        rr=LocalRisk(np.eye(4)@Q.T,np.zeros(4),np.full(2,.5))
        rotation.append(dict(angle=angle,**rr.score(.1*(Q@q))))
    save(out/'synthetic.json',dict(cases=cases,scale_risks=scale,
         finite_difference_gradient=grad,descent_risk=descent,rotation=rotation,
         note='Numerical values only; no claim of full BPTT/autodiff validation.'))

SOURCES=[
 ('primary','results/c1_dataset_paired_seed0_baselines/mac_cbf','results/c1_dataset_paired_seed0_baselines/config.json'),
 ('extended_safety','results/c1_dataset_extended_h120_seed0_baselines/mac_cbf','results/c1_dataset_extended_h120_seed0_baselines/config.json'),
 ('extended_c1','results/c1_guarded_h120_seed0','results/c1_guarded_h120_seed0/config.json')]

def score_traces(out):
    index=[];started=time.perf_counter();calls=0
    for source,folder,config_path in SOURCES:
        folder=ROOT/folder;metadata=json.loads((ROOT/config_path).read_text())
        plant=Config(**metadata['environment']);cbf=CBFConfig(**metadata['cbf'])
        env=GiveWayEnv(plant)
        summaries={int(r['rollout_id']):r for r in json.loads((folder/'summary.json').read_text())['rollouts']}
        for file in sorted(folder.glob('rollout_*.npz')):
            rid=int(file.stem.split('_')[-1]);target=out/'traces'/f'{source}_{rid:04d}.npz'
            summary=summaries[rid]
            record=dict(source=source,rid=rid,path=str(file.relative_to(ROOT)),
                        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                        outcome=summary['outcome'],split='heldout' if rid%5==0 else 'development',
                        score_path=str(target.relative_to(ROOT)),max_steps=plant.max_steps,
                        config_path=config_path)
            if target.exists():index.append(record);continue
            with np.load(file,allow_pickle=False) as z:
                pos=z['positions_before'].copy();after=z['positions'].copy()
                v=z['c1_candidate'].copy() if 'c1_candidate' in z else z['u_safe'].reshape(-1,4).copy()
                applied=z['executed_velocity'].reshape(-1,4).copy()
                # Check the C1-convention baseline candidate is the stored safe control.
                if 'c1_candidate' not in z:np.testing.assert_allclose(v,applied,atol=1e-8,rtol=0)
                old=z['c1_risk'].copy() if 'c1_risk' in z else np.full(len(pos),np.nan)
                event=z['deadlock'].copy();success=z['task_success'].copy()
                wp=z['window_progress'].copy();stuck=z['stuck_timer'].copy()
                saved_pair=z['pairwise_h'].copy()
            T=len(pos);idx=np.unique(np.r_[np.arange(0,T,PROTOCOL['stride']),T-1])
            values=[]
            for t in idx:
                A,b,h=barrier_constraints(dict(positions=pos[t],walls=env.walls,config=plant.to_dict()),cbf)
                if abs(h['pairwise_h']-saved_pair[t])>1e-6:raise AssertionError('geometry provenance mismatch')
                try:
                    model=LocalRisk(A,b,np.full(2,plant.max_speed));result=model.score(v[t])
                    calls+=model.physical.calls+model.base.calls
                    result['feasibility_error']=max(model.physical.max_violation,model.base.max_violation)
                except (ValueError,RuntimeError) as exc:
                    result=dict(valid=False,reason=str(exc))
                values.append(result)
            arrays={k:np.array([r.get(k,np.nan) for r in values]) for k in
                    ['B','M','E','risk','risk_lower','ambiguous_witnesses','feasibility_error']}
            arrays['valid']=np.array([r['valid'] for r in values],bool)
            arrays['reason']=np.array([r.get('reason','') for r in values])
            np.savez_compressed(target,**arrays,idx=idx,positions_before=pos,positions_after=after,
                 candidate=v,applied=applied,goals=env.goals,dt=plant.dt,
                 old_risk=old,deadlock=event,success=success,window_progress=wp,stuck_timer=stuck)
            index.append(record)
            if rid%10==0:print(f'{source} {rid}: {len(idx)} states; total solves={calls}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
            save(out/'index.json',index)
    save(out/'index.json',index)
    save(out/'runtime.json',dict(seconds=time.perf_counter()-started,projection_instances_this_run=calls))

def auc(y,s,w=None):
    y=np.asarray(y,bool);s=np.asarray(s,float)
    if w is None:w=np.ones(len(y))
    w=np.asarray(w,float);valid=np.isfinite(s);y,s,w=y[valid],s[valid],w[valid]
    if not y.any() or y.all():return None
    order=np.argsort(s);s,y,w=s[order],y[order],w[order]
    neg=0.;numer=0.;i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[j]==s[i]:j+=1
        pos=w[i:j][y[i:j]].sum();n=w[i:j][~y[i:j]].sum()
        numer+=pos*(neg+.5*n);neg+=n;i=j
    return float(numer/(w[y].sum()*w[~y].sum()))

def aggregate(x,w):
    x,w=np.asarray(x,float),np.asarray(w,float);ok=np.isfinite(x)&(w>0)
    if not ok.any():return np.nan
    x,w=x[ok],w[ok];w=w/w.sum()
    return .5*np.dot(w,x)+.5*np.dot(w,x**4)**.25

def trajectory_values(z,end):
    dt=float(z['dt']);idx=z['idx'];choose=idx<end;ix=idx[choose]
    weight=np.diff(np.r_[ix,end])*dt
    maxrisk=1+PROTOCOL['eta']*PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']+1)/PROTOCOL['tau'])
    geometric=z['risk'][choose]/maxrisk
    geom_lower=z['risk_lower'][choose]/maxrisk
    before=z['positions_before'];goals=z['goals']
    dist=np.linalg.norm(before-goals,axis=-1)
    initial=dist[0].sum()+1e-5;V=dist.sum(axis=1)/initial
    W=round(PROTOCOL['progress_window_seconds']/dt)
    eligible=ix>=W;stall=np.full(len(ix),np.nan);per_agent=np.full(len(ix),np.nan)
    unfinished=expit((dist[ix]-.08)/PROTOCOL['unfinished_temperature'])
    U=1-np.prod(1-unfinished,axis=1)
    delta=V[np.maximum(ix-W,0)]-V[ix]
    stall[eligible]=U[eligible]*expit((PROTOCOL['delta_progress']-delta[eligible])/PROTOCOL['tau_progress'])
    progress_agent=dist[np.maximum(ix-W,0)]-dist[ix]
    per_agent[eligible]=np.max(unfinished[eligible]*expit((.02-progress_agent[eligible])/.005),axis=1)
    G=aggregate(geometric,weight);Gl=aggregate(geom_lower,weight);P=aggregate(stall,weight)
    Pa=aggregate(per_agent,weight)
    distance=np.linalg.norm(z['positions_after'][end-1]-goals,axis=-1)
    excess=max(0.,float(np.max(distance))-.08)
    T=excess/(excess+PROTOCOL['terminal_distance_scale'])
    exposure=.5*G+.5*P
    return dict(geometry=G,geometry_lower=Gl,base=aggregate(z['B'][choose],weight),
                escape=aggregate(z['E'][choose],weight),progress=P,per_agent_progress=Pa,
                exposure=exposure,terminal_distance=T,total=.5*exposure+.5*T,
                coverage=float(np.sum(weight[np.isfinite(geometric)])/np.sum(weight)),
                remaining_distance=float(distance.max()),
                slow_speed=float(-np.mean(np.linalg.norm(z['applied'][:end].reshape(-1,2,2),axis=-1))),
                old=aggregate(z['old_risk'][ix],weight))

def metrics(rows,fields):
    result=dict(n=len(rows),outcomes={k:sum(r['outcome']==k for r in rows) for k in sorted({r['outcome'] for r in rows})})
    for label,fn in [('failure',lambda r:r['outcome']!='success'),('deadlock',lambda r:r['outcome']=='safe_deadlock')]:
        y=[fn(r) for r in rows]
        result[label]={f:auc(y,[r[f] for r in rows]) for f in fields}
    return result

def analyze(out):
    index=json.loads((out/'index.json').read_text());rows=[];local=[]
    fields=['geometry','base','escape','progress','per_agent_progress','exposure','terminal_distance','total','remaining_distance','slow_speed','old']
    for r in index:
        with np.load(ROOT/r['score_path']) as z:
            T=len(z['positions_before']);dt=float(z['dt']);idx=z['idx']
            for seconds in ['full',10,20]:
                end=T if seconds=='full' else round(seconds/dt)
                if end>T:continue
                rows.append(dict(**r,landmark=str(seconds),**trajectory_values(z,end)))
            W=round(PROTOCOL['progress_window_seconds']/dt)
            dist=np.linalg.norm(z['positions_before']-z['goals'],axis=-1)
            V=dist.sum(1)/(dist[0].sum()+1e-5)
            for k,t in enumerate(idx):
                if not z['valid'][k]:continue
                for H in [2.,5.]:
                    observed=(T-t)*dt>=H or r['outcome'] in ['success','safe_deadlock']
                    if not observed:continue
                    label=r['outcome']=='safe_deadlock' and (T-t)*dt<=H
                    prog=V[max(t-W,0)]-V[t]
                    local.append(dict(source=r['source'],rid=r['rid'],split=r['split'],H=H,y=label,
                         risk=float(z['risk'][k]),base=float(z['B'][k]),escape=float(z['E'][k]),
                         slow_speed=-float(np.max(np.linalg.norm(z['applied'][t].reshape(2,2),axis=1))),
                         poor_progress=-prog if t>=W else np.nan,
                         remaining_distance=float(dist[t].max())))
    save(out/'episode_scores.json',rows)
    report={}
    for source,_,_ in SOURCES:
        for split in ['all','development','heldout']:
            for landmark in ['full','10','20']:
                selected=[r for r in rows if r['source']==source and (split=='all' or r['split']==split) and r['landmark']==landmark]
                report[f'{source}/{split}/{landmark}']=metrics(selected,fields)
    save(out/'trajectory_metrics.json',report)
    lm={}
    for split in ['all','development','heldout']:
        for H in [2.,5.]:
            rr=[r for r in local if r['source']=='primary' and r['H']==H and (split=='all' or r['split']==split)]
            counts={rid:sum(r['rid']==rid for r in rr) for rid in {r['rid'] for r in rr}}
            weights=[1/counts[r['rid']] for r in rr]
            lm[f'{split}/{H}s']=dict(n_states=len(rr),n_positive=sum(r['y'] for r in rr),
                positive_episodes=len({r['rid'] for r in rr if r['y']}),
                episode_weighted_auc={f:auc([r['y'] for r in rr],[r[f] for r in rr],weights)
                    for f in ['risk','base','escape','slow_speed','poor_progress','remaining_distance']})
    save(out/'local_prediction.json',lm)
    # Paired comparisons use matched initial IDs and common physical prefix.
    pairs=[]
    for rid in range(25):
        records=[r for r in index if r['rid']==rid and r['source'].startswith('extended')]
        if len(records)!=2:continue
        a,b=sorted(records,key=lambda r:r['source'],reverse=True)
        with np.load(ROOT/a['score_path']) as za,np.load(ROOT/b['score_path']) as zb:
            np.testing.assert_allclose(za['positions_before'][0],zb['positions_before'][0],atol=1e-12)
            end=min(len(za['positions_before']),len(zb['positions_before']))
            pairs.append(dict(rid=rid,safety_outcome=a['outcome'],c1_outcome=b['outcome'],
                common_seconds=end*float(za['dt']),safety=trajectory_values(za,end),c1=trajectory_values(zb,end)))
    save(out/'paired_scores.json',pairs)
    # Episode-cluster bootstrap, including all sampled states of each episode.
    rng=np.random.default_rng(20260914);ci={}
    rr=[r for r in rows if r['source']=='primary' and r['split']=='heldout' and r['landmark']=='20']
    for field in ['geometry','base','progress','exposure','terminal_distance','total']:
        vals=[]
        for _ in range(1000):
            sample=[rr[i] for i in rng.integers(0,len(rr),len(rr))]
            val=auc([r['outcome']!='success' for r in sample],[r[field] for r in sample])
            if val is not None:vals.append(val)
        ci[field]=dict(bootstrap_replicates=len(vals),interval95=np.quantile(vals,[.025,.975]) if vals else None)
    save(out/'heldout_prefix_bootstrap.json',ci)
    plot(out,rows,index)
    print(json.dumps(clean({k:v for k,v in report.items() if k in ['primary/all/full','primary/heldout/20','primary/heldout/full']}),indent=2),flush=True)

def plot(out,rows,index):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print('Optional matplotlib unavailable; numerical reports are complete.',flush=True)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    rr=[r for r in rows if r['source']=='primary' and r['landmark']=='full']
    for ax,field in zip(axes,['geometry','exposure','total']):
        for k,label in enumerate(['success','safe_deadlock','other_timeout']):
            values=[r[field] for r in rr if r['outcome']==label]
            ax.scatter(np.full(len(values),k),values,s=12,alpha=.45)
        ax.set_xticks(range(3),['success','deadlock','timeout']);ax.set_title(field)
    fig.tight_layout();fig.savefig(out/'episode_scores.png',dpi=160);plt.close(fig)
    # Predefined ID rule: first successful and first deadlocked primary episode.
    selected=[]
    for outcome in ['success','safe_deadlock']:
        selected.append(next(r for r in index if r['source']=='primary' and r['outcome']==outcome))
    fig,axes=plt.subplots(2,3,figsize=(13,6))
    for row,r in enumerate(selected):
        with np.load(ROOT/r['score_path']) as z:
            t=z['idx']*float(z['dt']);all_t=np.arange(len(z['applied']))*float(z['dt'])
            axes[row,0].plot(t,z['risk'],label='local risk');axes[row,0].plot(t,z['B'],label='B')
            axes[row,0].plot(t,z['E'],label='escape');axes[row,0].legend(fontsize=8)
            axes[row,1].plot(all_t,np.linalg.norm(z['applied'].reshape(-1,2,2),axis=-1))
            axes[row,2].plot(all_t,np.linalg.norm(z['positions_after']-z['goals'],axis=-1))
            axes[row,0].set_title(f"ID {r['rid']} / {r['outcome']}")
    axes[0,1].set_title('executed speed');axes[0,2].set_title('per-agent goal distance')
    fig.tight_layout();fig.savefig(out/'local_examples.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['local','trajectory','all'],default='all')
    p.add_argument('--out',type=Path,default=ROOT/'results/c1_joint_witness_risk_audit')
    args=p.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True);(out/'traces').mkdir(exist_ok=True)
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=PROTOCOL:raise RuntimeError('protocol mismatch')
    else:save(protocol,PROTOCOL)
    if args.phase in ['local','all']:synthetic(out);score_traces(out)
    if args.phase in ['trajectory','all']:analyze(out)

if __name__=='__main__':main()
