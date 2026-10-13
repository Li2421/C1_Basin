"""Bounded CPU action/geometry probes at frozen hard-field states; no env.step."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'

import numpy as np
from scipy.optimize import nnls

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
for path in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts')):
    sys.path.insert(0,path)
sys.path.insert(0,str(ROOT/'diagnostics/field_transfer_operator_audit_v3'))
sys.path.insert(0,str(ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'))

import jax.numpy as jnp
from validation_stage2.run_rollouts import make_runtime
from diagnostics.placement_boundary_mechanism_audit_v1.closed_loop_probe import key_for


def status(safety,target,output):
    a=safety.A;lo=safety.lower;u=output.ravel();sp=np.linalg.norm(output,axis=1)
    return dict(linear=np.flatnonzero(a@u-lo<1e-7).tolist(),speed=np.flatnonzero(safety.max_speed-sp<1e-7).tolist(),
        min_linear=float((a@u-lo).min()),speed_excess=float(sp.max()-safety.max_speed))


def proj_kkt(safety,target,output):
    """Return active KKT factors; ball curvature is included in H."""
    u=output.ravel();y=target.ravel();n=len(u)
    active=np.flatnonzero(safety.A@u-safety.lower<1e-7)
    speed=np.flatnonzero(safety.max_speed-np.linalg.norm(output,axis=1)<1e-7)
    normals=[-safety.A[j] for j in active]
    for i in speed:
        v=np.zeros(n);v[2*i:2*i+2]=u[2*i:2*i+2];normals.append(v)
    c=np.asarray(normals).reshape(-1,n)
    lam=nnls(c.T,y-u,maxiter=10000)[0] if len(c) else np.zeros(0)
    h=np.eye(n)
    for k,i in enumerate(speed):h[2*i:2*i+2,2*i:2*i+2]+=np.eye(2)*lam[len(active)+k]
    return active,speed,c,lam,h


def projection_derivative(safety,target,output):
    active,speed,c,lam,h=proj_kkt(safety,target,output)
    hi=np.linalg.inv(h)
    dp=hi-hi@c.T@np.linalg.pinv(c@hi@c.T,rcond=1e-10)@c@hi if len(c) else hi
    residual=float(np.linalg.norm(c.T@lam-(target-output).ravel())) if len(c) else float(np.linalg.norm(target-output))
    return dp,dict(active_linear=len(active),active_speed=len(speed),weak=bool((lam<1e-7).any()),kkt_residual=residual)


def geometry_q(rt,env,y,safety,u,eps):
    """Implicit KKT state derivative, with separately differenced barrier geometry.

    This is semi-analytic: A_x,b_x derivatives use centered differences, not a
    differentiated nearest-wall implementation.  The report labels it so.
    """
    active,speed,c,lam,h=proj_kkt(safety,y,u)
    n=len(y.ravel());positions=env.positions.copy();m=positions.size
    if len(c):
        kkt=np.block([[h,c.T],[c,np.zeros((len(c),len(c)))]])
    rows=[]
    for j in range(m):
        p=np.zeros_like(positions);p.ravel()[j]=eps
        env.positions=positions+p;sp=rt.safety(env)
        env.positions=positions-p;sm=rt.safety(env)
        env.positions=positions
        da=(sp.A-sm.A)/(2*eps);db=(sp.lower-sm.lower)/(2*eps)
        rhs_u=da[active].T@lam[:len(active)] if len(active) else np.zeros(n)
        rhs_c=(da[active]@u.ravel()-db[active]) if len(active) else np.zeros(0)
        rhs=np.r_[rhs_u,rhs_c,np.zeros(len(speed))]
        if len(c):
            sol=np.linalg.lstsq(kkt,rhs,rcond=1e-10)[0][:n]
        else:sol=np.zeros(n)
        rows.append(sol)
    return np.stack(rows,axis=1)


def evaluate(rt,env,key,eta,chain,override_delta=None):
    obs,safety,flow,safe,delta,_=rt.prepare(env,key,eta)
    if override_delta is not None:delta=override_delta.reshape(safe.shape)
    if chain=='TT':return safety.project(safe+delta)[0].ravel(),delta.ravel()
    if chain=='TF':
        h,_=rt.integrate(env,obs,key,delta,safety,project=False)
        return safety.project(h)[0].ravel(),delta.ravel()
    if chain=='FT':
        h,_=rt.integrate(env,obs,key,np.zeros_like(delta),safety,project=True)
        return safety.project(h+delta)[0].ravel(),delta.ravel()
    if chain=='FF':return rt.integrate(env,obs,key,delta,safety,project=True)[0].ravel(),delta.ravel()
    raise ValueError(chain)


def stats(a,b):
    a=np.asarray(a);b=np.asarray(b)
    num=float(np.linalg.norm(a-b));den=float(np.linalg.norm(b))
    return dict(abs_error=num,rel_error=num/max(den,1e-12),cosine=float(np.sum(a*b)/max(np.linalg.norm(a)*den,1e-12)))


def main():
    states={x['uid']:x for x in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    keyroot=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    eta=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'][4])
    recurrence=np.load(ROOT/'diagnostics/field_transfer_operator_audit_v3/recurrence_matrices.npz')
    selected=['hard_confirm_toy_1_820001','hard_confirm_toy_1_820013','hard_confirm_ring_1_920004','hard_confirm_ring_1_920005']
    rows=[];krows=[];qrows=[]
    for scene in ('toy_give_way','ring_exchange'):
        rt=make_runtime(scene)
        for uid in [s for s in selected if states[s]['scenario']==scene]:
            st=states[uid];env=rt.make_env(st);key=key_for(uid,0,0,keyroot)
            obs,safety,flow,safe,delta,_=rt.prepare(env,key,np.zeros(3))
            cache=np.load(HARD/'transfer_mechanism_v2/state_dependence_control'/f'{uid}.npz')
            B=cache['basis'];n=safe.size
            # Cache chain ordering: TT, FT, TF, FF. Index 1 is delta=0.
            target=cache['pre_all'][:,1].reshape(4,*safe.shape)
            output=cache['exec_all'][:,1].reshape(4,*safe.shape)
            p_tt,dt=projection_derivative(safety,target[0],output[0])
            p_ft,df=projection_derivative(safety,target[1],output[1])
            p_tf,dq=projection_derivative(safety,target[2],output[2])
            M={'TT':p_tt,'FT':p_ft,'TF':p_tf@recurrence[uid+'_flow'],'FF':recurrence[uid+'_FF']}
            diagnostic={'TT':dt,'FT':df,'TF':dq,'FF':{'weak':False,'kkt_residual':None}}
            for ci,ch in enumerate(('TT','FT','TF','FF')):
                direct=cache['center0_M_exec'][ci]
                jdirect=cache['center0_J_eta_exec'][ci]
                for level in range(3):
                    comparison=stats(M[ch],direct[level])
                    gcomparison=stats(M[ch]@B,jdirect[level])
                    rows.append(dict(scene=scene,uid=uid,placement=ch,layer='M',epsilon_index=level,
                        epsilon='0.005/0.001/0.0001_scaled',active_linear=diagnostic[ch].get('active_linear',''),
                        active_speed=diagnostic[ch].get('active_speed',''),weak=diagnostic[ch].get('weak',False),
                        kkt_residual=diagnostic[ch].get('kkt_residual',''),**comparison))
                    rows.append(dict(scene=scene,uid=uid,placement=ch,layer='G',epsilon_index=level,
                        epsilon='cached_eta_scales',active_linear=diagnostic[ch].get('active_linear',''),
                        active_speed=diagnostic[ch].get('active_speed',''),weak=diagnostic[ch].get('weak',False),
                        kkt_residual=diagnostic[ch].get('kkt_residual',''),**gcomparison))
            # P and Q at the actual TT target. Q includes state-dependent CBF rows.
            for e in (1e-3,3e-4,1e-4):
                Pfd=[]
                for j in range(n):
                    v=np.zeros(n);v[j]=e
                    plus=safety.project((target[0].ravel()+v).reshape(safe.shape))[0]
                    minus=safety.project((target[0].ravel()-v).reshape(safe.shape))[0]
                    Pfd.append(((plus-minus)/(2*e)).ravel())
                Pfd=np.stack(Pfd,axis=1)
                q=geometry_q(rt,env,target[0],safety,output[0],e)
                Qfd=[];x0=env.positions.copy()
                for j in range(x0.size):
                    v=np.zeros_like(x0);v.ravel()[j]=e
                    env.positions=x0+v;plus=rt.safety(env).project(target[0])[0]
                    env.positions=x0-v;minus=rt.safety(env).project(target[0])[0]
                    env.positions=x0
                    Qfd.append(((plus-minus)/(2*e)).ravel())
                Qfd=np.stack(Qfd,axis=1)
                for layer,ana,fd in [('P',p_tt,Pfd),('Q',q,Qfd)]:
                    qrows.append(dict(scene=scene,uid=uid,layer=layer,epsilon=e,active_linear=dt['active_linear'],
                        active_speed=dt['active_speed'],weak=dt['weak'],**stats(ana,fd)))
            cache.close()
        # Bounded K product-rule probes at the first predeclared state per scene.
        uid=next(s for s in selected if states[s]['scenario']==scene);st=states[uid]
        env=rt.make_env(st);key=key_for(uid,0,0,keyroot)
        x0=np.r_[env.positions.ravel(),env.velocities.ravel()];nstate=len(x0)
        base,Bdelta=evaluate(rt,env,key,eta,'TT')
        def setx(x):
            env.positions=x[:len(env.positions.ravel())].reshape(env.positions.shape).copy()
            env.velocities=x[len(env.positions.ravel()):].reshape(env.velocities.shape).copy()
        for e in (1e-3,3e-4,1e-4):
            ddelta=[];full={ch:[] for ch in ('TT','TF','FT','FF')};frozen={ch:[] for ch in full}
            for j in range(nstate):
                v=np.zeros(nstate);v[j]=e
                setx(x0+v);plus={ch:evaluate(rt,env,key,eta,ch) for ch in full}
                setx(x0-v);minus={ch:evaluate(rt,env,key,eta,ch) for ch in full}
                ddelta.append((plus['TT'][1]-minus['TT'][1])/(2*e))
                for ch in full:
                    full[ch].append((plus[ch][0]-minus[ch][0])/(2*e))
                    setx(x0+v);fp=evaluate(rt,env,key,eta,ch,override_delta=Bdelta)[0]
                    setx(x0-v);fm=evaluate(rt,env,key,eta,ch,override_delta=Bdelta)[0]
                    frozen[ch].append((fp-fm)/(2*e))
                setx(x0)
            db=np.stack(ddelta,axis=1)
            # M is evaluated at eta[4], so estimate local M independently at
            # each epsilon. This tests the product rule, not a fully analytic K.
            for ch in full:
                setx(x0);a0,d0=evaluate(rt,env,key,eta,ch)
                m=[]
                for j in range(len(d0)):
                    v=np.zeros_like(d0);v[j]=e
                    pp=evaluate(rt,env,key,eta,ch,override_delta=d0+v)[0]
                    mm=evaluate(rt,env,key,eta,ch,override_delta=d0-v)[0]
                    m.append((pp-mm)/(2*e))
                m=np.stack(m,axis=1)
                measured=np.stack(full[ch],axis=1)
                prediction=np.stack(frozen[ch],axis=1)+m@db
                krows.append(dict(scene=scene,uid=uid,placement=ch,epsilon=e,
                    formula='K=K_frozen_delta+M*(dB/dx eta); all blocks measured by independent central probes',
                    basis_term_norm=float(np.linalg.norm(m@db)),frozen_term_norm=float(np.linalg.norm(frozen[ch])),
                    total_norm=float(np.linalg.norm(measured)),**stats(prediction,measured)))
        print('done',scene,flush=True)
    allrows=rows+qrows+krows
    fields=['scene','uid','placement','layer','epsilon','epsilon_index','active_linear','active_speed','weak','kkt_residual',
        'abs_error','rel_error','cosine','formula','basis_term_norm','frozen_term_norm','total_norm']
    with (HERE/'jacobian_validation.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:w.writerow(row)
        for row in qrows:w.writerow(row)
        for row in krows:w.writerow(dict(row,layer='K'))
    (HERE/'derivative_summary.json').write_text(json.dumps(dict(MG_rows=len(rows),PQ_rows=len(qrows),K_rows=len(krows),
        K_method='semi-analytic product rule; K_frozen, B_x, and M use central differences',
        Q_method='implicit KKT with independently differenced CBF geometry',new_rollouts=0),indent=2)+'\n')
    print('jacobian rows',len(allrows),flush=True)


if __name__=='__main__':main()
