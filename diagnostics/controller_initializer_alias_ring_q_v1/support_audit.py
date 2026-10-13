"""Read-only audit of initializer diversity in the actual prior TRAIN union."""
import numpy as np
from .study import HERE,ROOT,write,sha


def run():
    source=ROOT/'controller_crossed_coverage_v1/dataset.npz'
    metadata=ROOT/'controller_local_to_return_probe_v1/vector_control/dataset.npz'
    d=np.load(source);m=np.load(metadata);rows=[]
    for scene in sorted(set(d['scene'])):
        mask=(d['scene']==scene)&(d['split']=='train')
        meta=(m['scene']==scene)&(m['split']=='train')
        x=m['feature'][meta,...,-8:];active=m['entity_agent_mask'][meta].astype(bool)
        covariance=np.empty(x.shape[:-1]+(2,2));covariance[...,0,0]=x[...,2];covariance[...,0,1]=x[...,3]
        covariance[...,1,0]=x[...,3];covariance[...,1,1]=x[...,4]
        inv=np.concatenate((np.linalg.norm(x[...,:2],axis=-1)[...,None],np.linalg.eigvalsh(covariance)),-1)
        agents=[]
        for i in range(active.shape[1]):
            a=inv[:,i][active[:,i]]
            if not len(a):continue
            variation=np.max(np.ptp(a,axis=0))
            assert variation<5e-6
            agents.append(dict(APIagent=i,initializer_invariants_mean=a.mean(0).tolist(),max_state_variation=float(variation)))
        rows.append(dict(scene=str(scene),physical_states=len(set(d['state_uid'][mask])),
            controller_names=sorted(str(c) for c in set(d['controller'][mask])),rows=int(mask.sum()),
            initializer_laws_per_scene=1,agents=agents))
    result=dict(rows=rows,source_sha256=sha(source),metadata_sha256=sha(metadata),
        feature_definitions=['norm(physical_initializer_mean)/vmax','small_eigenvalue(physical_initializer_covariance)/vmax^2','large_eigenvalue(physical_initializer_covariance)/vmax^2'],
        code_basis=['controller_q_diversity_v1/runtime.py:AnalyticField.__init__ inherits base.mean/scale',
            'controller_response_families_v1/runtime.py:EndpointMatchedField inherits AnalyticField',
            'controller_flow_endpoint_probe_v1/runtime.py:FlowEndpointMatchedField changes reference only',
            'native_flow has its one frozen native program per scene'],
        finding='Changing analytic parameters/M/controller name in this training union did NOT independently vary initialization law within a scene. State-dependent rotations change its coordinates, not its mean norm/eigenvalues. Thus21controller names are not21independent Gaussian-law examples.',
        boundary='Observed confounding/support gap; does NOT prove no function can transfer. Right structural priors could extrapolate. Exact initializer augmentation alone was already negative; a crossed-law support intervention has not yet established that this gap explains all existing LOSO failures.',
        field_probe_identifiability='For each native chart component, all current field probes use fixed z_f and reveal h0=mu+sigma*z_f. Jacobian with respect to(mu,sigma) is[1,z_f], rank1 ratherthan2. More physical query poses do not add independent initialization equations for the endpoint-matched family.',
        new_query=0,new_rollout=0,new_fit=0)
    write(HERE/'source_initializer_support.json',result)
    print([(r['scene'],r['physical_states'],len(r['controller_names']),r['initializer_laws_per_scene']) for r in rows])


if __name__=='__main__':run()
