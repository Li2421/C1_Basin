"""Family-specific information diagnostic, NOT a deployed critic or label fit.

Recover analytic controller parameters from cached static field queries alone.
The true parameters are read only to score recovery, not by the optimizer.
No success labels, simulator calls, FINAL states, or FINAL controllers.
"""
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from .experiment import HERE, read, write, sha
from .runtime import AnalyticField


def run():
    protocol=read(HERE/'protocol.json');rows=[]
    lower=np.array([.55,.2,-.85,0.,0.]);upper=np.array([1.1,.9,.85,.6,.8])
    starts=[(lower+upper)/2,lower+.25*(upper-lower),lower+.75*(upper-lower)]
    for state in protocol['states']:
        split=state['study_split']
        if split=='final' or state['study_index']>=(4 if split=='train' else 2):continue
        scene=state['scenario'];config=SimpleNamespace(**protocol['environments'][scene]['config']['config'])
        physical=state['physical'];p=np.array(physical['positions']);v=np.array(physical['velocities']);g=np.array(physical['goals'])
        heading=g-p;heading/=np.linalg.norm(heading,axis=1,keepdims=True)
        lateral=np.stack((-heading[:,1],heading[:,0]),-1);speed=config.max_speed
        poses=[(p,v),(p,.5*speed*heading),(p,-.5*speed*heading),
               (g-.25*heading,np.zeros_like(v)),(g-.25*heading,.5*speed*heading),
               (g-.25*heading,-.5*speed*heading),(g-.25*heading,.5*speed*lateral)]
        names=['analytic'+str(k) for k in (range(4) if split=='train' else (4,5))]
        for name in names:
            descriptor=np.load(HERE/'features'/state['state_uid']/(name+'.npz'))['field'].reshape(len(p),7,3,5,2)
            relaxation=float(-np.mean(descriptor[:,:,:,1,0]-descriptor[:,:,:,2,0]))
            observed=descriptor[:,:,0,0,:]/relaxation
            def response(theta):
                controller=SimpleNamespace(config=config,parameters=np.r_[theta,relaxation]);values=[]
                for pos,vel in poses:
                    env=SimpleNamespace(positions=pos,velocities=vel,goals=g)
                    target=AnalyticField.target(controller,env)
                    values.append(np.stack(((target*heading).sum(-1),(target*lateral).sum(-1)),-1)/speed)
                return np.stack(values,1)
            fits=[least_squares(lambda t:(response(t)-observed).ravel(),x,bounds=(lower,upper),
                    ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=160) for x in starts]
            best=min(fits,key=lambda f:f.cost)
            # Read reference parameters only after estimation is complete.
            truth=np.array(protocol['controllers'][scene][name]['payload']['controller_parameters'])
            estimate=np.r_[best.x,relaxation]
            scales=np.r_[upper-lower,3.]
            eigen=np.linalg.svd(best.jac*(upper-lower)[None,:],compute_uv=False)
            rows.append(dict(scene=scene,split=split,state_uid=state['state_uid'],controller=name,
                estimated_parameters=estimate.tolist(),reference_parameters=truth.tolist(),
                query_RMSE=float(np.sqrt(np.mean(best.fun**2))),
                normalized_parameter_max_error=float(np.max(np.abs(estimate-truth)/scales)),
                normalized_parameter_errors=(np.abs(estimate-truth)/scales).tolist(),
                scaled_jacobian_singular_values=eigen.tolist(),
                solutions=[dict(cost=float(f.cost),theta=f.x.tolist()) for f in fits]))
    summaries=[]
    for scene in sorted(set(r['scene'] for r in rows)):
        for split in ('train','validation'):
            rr=[r for r in rows if r['scene']==scene and r['split']==split]
            summaries.append(dict(scene=scene,split=split,cases=len(rr),
                query_RMSE_max=max(r['query_RMSE'] for r in rr),
                normalized_parameter_max_error=max(r['normalized_parameter_max_error'] for r in rr),
                cases_all_parameters_within_1percent=sum(r['normalized_parameter_max_error']<.01 for r in rr),
                local_full_rank_cases=sum(min(r['scaled_jacobian_singular_values'])>1e-5 for r in rr)))
    write(HERE/'input_identifiability.json',dict(rows=rows,summaries=summaries,
        success_labels_read=False,simulator_steps=0,final_inputs_or_labels_used=False,
        scope='Analytic six-parameter family only; local rank and empirical inverse recovery do not prove arbitrary-controller global identifiability',
        interpretation='If recovery succeeds, the cached field queries contain the family parameters even when a supervised critic fails; this excludes missing parameter information as the sole explanation on tested cases.',
        code_sha256=sha(__file__)))
    print(summaries,flush=True)


if __name__=='__main__':run()
