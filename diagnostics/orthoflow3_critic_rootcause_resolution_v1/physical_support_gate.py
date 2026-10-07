"""Source-only physical-support rule; cached results are post-hoc diagnostic.

This is a scope guard, not uncertainty calibration and not a claim to infer Q
outside physical support. It may fall back to the fixed eta-only predictor.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from .db_transfer_data import OUT as SOURCE, ROOT, TARGET, read, write, sha
from .db_transfer_evaluate import csvwrite
OUT=SOURCE/'physical_support_diagnostic'

def features(x):
    am=x['agent_mask'];om=x['obstacle_mask'];n=am.shape[-1]
    def pool(a,m,axes):
        mm=m[...,None];count=np.maximum(m.sum(axes),1)[...,None]
        mean=(a*mm).sum(axes)/count;mx=np.max(np.where(mm,a,-1e9),axis=axes)
        mx=np.where((m.sum(axes)>0)[...,None],mx,0.)
        return np.concatenate((mean,mx),-1)
    parts=[pool(x['agents'],am,(1,)),pool(x['pairs'],am[:,:,None]*am[:,None,:]*(1-np.eye(n)[None]),(1,2)),pool(x['obstacles'],am[:,:,None]*om[:,None,:],(1,2)),x['globals'],am.sum(1)[:,None]/4,om.sum(1)[:,None]/20]
    return np.concatenate(parts,-1)

def physical(items):
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    return features(rep.batch([rep.entities(p) for p in items]))

def run():
    states=read(SOURCE/'states.json');x=dict(np.load(SOURCE/'source_entities.npz'));a=features(x)
    tr=np.flatnonzero([s['split']=='train' for s in states]);va=np.flatnonzero([s['split']=='validation' for s in states]);w=np.zeros(len(tr))
    for sc in sorted({s['scenario'] for s in states}):
        m=np.array([states[i]['scenario']==sc for i in tr]);w[m]=1/m.sum()/3
    center=(w[:,None]*a[tr]).sum(0);scale=np.maximum(np.sqrt((w[:,None]*(a[tr]-center)**2).sum(0)),.05)
    train=(a[tr]-center)/scale
    def measure(v):
        z=(v-center)/scale;dist=cdist(z,train)/np.sqrt(a.shape[1]);ix=dist.argmin(1)
        return dist[np.arange(len(v)),ix],ix,z
    vd,_,_=measure(a[va]);threshold=float(np.quantile(vd,.95))
    write(OUT/'source_rule_frozen.json',dict(rule='Source TRAIN standardized masked mean/max physical entities + fixed globals; nearest TRAIN RMS distance; threshold95thpercentile of independent source VAL families; zero target outcomes used for threshold',center=center.tolist(),scale=scale.tolist(),threshold=threshold,dimensions=a.shape[1],source_train_indices=tr.tolist(),source_validation_distances=vd.tolist(),training_states_sha256=sha(SOURCE/'states.json'),models_sha256=sha(SOURCE/'models_frozen.json'),status='posthoc diagnostic on previously opened target panels; independent confirmation required',generator_modified=False,new_rollouts=0))
    target_phi={'DB':physical(read(TARGET/'physical.json'))}
    for controller in range(88132,88138):target_phi[str(controller)]=physical([s['physical'] for s in read(ROOT/f'motion_independent_confirmation_{controller}/physical.json')])
    metrics=[];details=[]
    full=read(SOURCE/'models_frozen.json')['source_selected_full']
    for name,v in target_phi.items():
        d,ix,z=measure(v);supported=d<=threshold
        for j in range(len(v)):
            contribution=(z[j]-train[ix[j]])**2;top=np.argsort(-contribution)[:6]
            details.append(dict(target=name,index=j,supported=bool(supported[j]),distance=float(d[j]),nearest_state_uid=states[tr[ix[j]]]['state_uid'],nearest_scene=states[tr[ix[j]]]['scenario'],top_coordinate_indices=top.tolist(),top_squared_contributions=contribution[top].tolist()))
        if name=='DB':
            scores=dict(np.load(SOURCE/'target_predictions.npz'));truth=read(SOURCE/'cached_truth.json');good=np.array([t['B15'] is True for t in truth]).reshape(-1,16)
            q=np.array([t['lower'] for t in truth]).reshape(-1,16)
        else:
            scores={k:expit(z) for k,z in np.load(SOURCE/'opened_controller_diagnostic'/f'predictions_{name}.npz').items()};truth=np.load(ROOT/f'motion_independent_confirmation_{name}/dataset.npz');good=truth['success'].reshape(-1,2)>=15;q=truth['success'].reshape(-1,2)/16
        for seed in (17,23,41):
            key=f'seed{seed}' if name=='DB' else str(seed)
            p=scores[f'{full}__{key}__correct'];prior=scores[f'eta_only__{key}__correct'];idx=np.arange(len(p));mixed=np.where(supported[:,None],p,prior)
            picks=[u.argmax(1) for u in (p,prior,mixed)]
            metrics.append(dict(target=name,seed=seed,states=len(v),supported=int(supported.sum()),threshold=threshold,median_distance=float(np.median(d)),max_distance=float(d.max()),full_B15=int(good[idx,picks[0]].sum()),eta_B15=int(good[idx,picks[1]].sum()),guarded_B15=int(good[idx,picks[2]].sum()),guarded_Q_lower=float(q[idx,picks[2]].mean()),oracle=int(good.any(1).sum())))
    csvwrite(OUT/'metrics.csv',metrics);write(OUT/'nearest_support.json',details)
    print(metrics,flush=True)

if __name__=='__main__':run()
