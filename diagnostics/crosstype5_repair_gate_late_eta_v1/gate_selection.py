"""Actual-label-only nested gate fitting; outer results cannot tune any parameter."""
from common import *
from candidate_menu import menu,physical_key,choose,METHODS
from action_encoding import encode
from schedule_generator import predict as generator_predict,model_record
from inference import predict
from train_critic import load_data,key_for

K_GRID=(0,1,2);DELTA_GRID=(0.,.02,.05)
TAU_GRID=(0.,.25,.5,.75,.9,.95,1.,1.01);RHO_GRID=(0.,.05,.1,.2,.4,1.)

def training_actions(data):
    old=D/'crosstype4_no_repair_schedule_diagnosis_v1'
    catalogue=read(old/'CANDIDATES.json')['states']
    with np.load(old/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices']
    added=read(HERE/'TRAINING_ADDED_ACTIONS.json');lookup={(r['controller'],r['state_index']):r for r in added}
    return [[[dict(catalogue[si]['actions'][int(i)],late_source='given') for i in indices[ni,si]]+[r['action'] for r in lookup[name,si]['added_actions']] for si in range(48)] for ni,name in enumerate(data['names'])]

def subset(data,ids):
    return dict(names=data['names'][ids],types=data['types'][ids],features=data['features'][ids],h=data['h'],c=data['c'])

def score_grid(packet,repair):
    value=np.where(repair,packet['best_score'],packet['safety_score']);scored=np.isfinite(value);types=packet['types'];rates=[];fractions=[];support={}
    for t in sorted(set(types.tolist())):
        ids=np.flatnonzero(types==t);valid=scored[ids].any(-1);means=np.nansum(value[ids],-1)/np.maximum(scored[ids].sum(-1),1)
        support[t]=dict(scored_states=int(scored[ids].sum()),total_states=int(scored[ids].size),supported_controllers=int(valid.sum()),total_controllers=len(ids),success=float(means[valid].mean()) if valid.any() else None)
        if valid.any():rates.append(float(means[valid].mean()))
        fractions.append(float(scored[ids].mean()))
    return dict(success=float(np.mean(rates)) if rates else None,scored_fraction=float(np.mean(fractions)),unscored_fraction=1-float(np.mean(fractions)),support=support)

def choose_parameters(packet):
    critic=[];probe=[]
    for k in K_GRID:
        for delta in DELTA_GRID:
            repair=packet['best_mu']-k*packet['best_sd']>=packet['zero_mu']+delta
            critic.append(dict(k=k,delta=delta,**score_grid(packet,repair)))
    for tau in TAU_GRID:
        for rho in RHO_GRID:
            repair=~((packet['features'][:,None,5]>=tau)&(packet['features'][:,None,0]<=rho));repair=np.broadcast_to(repair,packet['best_mu'].shape)
            probe.append(dict(tau=tau,rho=rho,**score_grid(packet,repair)))
    valid_c=[r for r in critic if r['success'] is not None];valid_p=[r for r in probe if r['success'] is not None]
    c=max(valid_c,key=lambda r:(round(r['success'],12),round(r['scored_fraction'],12),r['k'],r['delta'])) if valid_c else dict(k=2,delta=.05,gap='No scored source-inner choices')
    p=max(valid_p,key=lambda r:(round(r['success'],12),round(r['scored_fraction'],12),-r['tau'],r['rho'])) if valid_p else dict(tau=0.,rho=1.,gap='No scored source-inner choices')
    return dict(critic=c,probe=p,critic_curve=critic,probe_curve=probe,outer_labels_used=False)

def branch_at(source,branch,step):
    folder=HERE/'critic_fits'/branch/key_for(source);receipt=read(folder/'COMPLETE.receipt.json');assert set(read(folder/'TASK.json')['source_types'])==set(source)
    model=dict(mode=branch,step=step,source_types=source)
    for key,path in [('checkpoint',folder/f'step{step}.msgpack'),('transform',folder/'TRANSFORM.json'),('fit_receipt',folder/'COMPLETE.receipt.json')]:
        if key!='fit_receipt':assert sha(path)==receipt['files'][str(path)]
        model[key+'_path']=str(path);model[key+'_sha256']=sha(path)
    return model

def nested_model(source,selected):
    record=selected['source_only_selection'];branches=record['branches'];w=record['selected']['full_weight']
    critic=dict(kind='probability_shrinkage',source_types=source,full_weight=w,**{b:branch_at(source,b,branches[b]['selected']['step']) for b in ('full','no_controller')})
    generator=model_record(source,selected['generator_selection']['selected']['step'])
    return dict(full=critic,generator=generator)

def evaluate(data,ids,models,known_actions,save_path=None):
    sub=subset(data,ids);generated=generator_predict(models['generator'],sub,data['state_uids']);nc=len(ids);ns=len(data['h']);menus=[];vectors=np.zeros((nc,ns,415,14),np.float32)
    for ci in range(nc):
        local=[]
        for si in range(ns):
            rows=menu(data['proposals'][si],generated[ci,si]);local.append(rows);vectors[ci,si,:len(rows)]=[encode(r['action']) for r in rows]
        menus.append(local)
    probability=predict(models['full'],dict(sub,action=vectors));packet=dict(names=sub['names'],types=sub['types'],best_mu=np.zeros((nc,ns)),best_sd=np.zeros((nc,ns)),zero_mu=np.zeros((nc,ns)),best_score=np.full((nc,ns),np.nan),safety_score=np.full((nc,ns),np.nan),truth_no_repair=np.full((nc,ns),-1,np.int8),features=np.zeros((nc,24)))
    transform=read(models['full']['full']['transform_path'])
    packet['features']=np.where(np.isfinite(sub['features']),sub['features'],np.asarray(transform['feature_fill']))
    choices=[]
    for ci,ni in enumerate(ids):
        for si in range(ns):
            rows=menus[ci][si];p=probability[:,ci,si,:len(rows)];mu=p.mean(0);sd=p.std(0);best=int(np.argmax(mu[:-1]));zero=len(rows)-1
            known={physical_key(a):slot for slot,a in enumerate(known_actions[ni][si])}
            def score(a):
                slot=known.get(physical_key(a))
                if slot is None:return np.nan
                labels=data['labels'][ni,si,slot];valid=labels>=0
                return float(labels[valid].mean()) if valid.any() else np.nan
            packet['best_mu'][ci,si]=mu[best];packet['best_sd'][ci,si]=sd[best];packet['zero_mu'][ci,si]=mu[zero]
            packet['best_score'][ci,si]=score(rows[best]['action']);packet['safety_score'][ci,si]=score(rows[zero]['action'])
            labels=data['labels'][ni,si];count=(labels>=0).sum(-1);empirical=(labels==1).sum(-1)/np.maximum(count,1);repair_mask=(count>0)&(np.arange(44)!=26)
            if count[26]>0 and repair_mask.any():packet['truth_no_repair'][ci,si]=int(empirical[26]>=empirical[repair_mask].max())
            if save_path is not None:
                choices.append(dict(controller=str(data['names'][ni]),type=str(data['types'][ni]),state_index=si,state_uid=str(data['state_uids'][si]),rows=rows,probability=p,known_scores=[score(r['action']) for r in rows]))
    if save_path is not None:
        # Separate numbers from the human-readable action provenance; arrays are binary.
        out=Path(save_path);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'PACKET.npz',**packet);write(out/'MENUS.json',choices)
        seal(str(out.relative_to(HERE)/'COMPLETE.receipt.json'),[out/'PACKET.npz',out/'MENUS.json',HERE/'gate_selection.py',HERE/'SELECTED_MODELS.receipt.json'])
    return packet

def combine(packets):return {k:np.concatenate([p[k] for p in packets],axis=0) for k in packets[0]}

def main():
    import jax
    assert jax.default_backend()=='gpu';data=load_data();actions=training_actions(data);selected=read(HERE/'SELECTED_MODELS.json');outer_folds={r['held_type']:r for r in read(HERE/'DEVELOPMENT_OUTER_FOLDS.json')};all_types=sorted(set(data['types'].tolist()));results={}
    with jax.default_matmul_precision('highest'):
        for outer in [*all_types,'ALL_TYPES']:
            folder=HERE/'gate_fits'/outer;done=folder/'COMPLETE.receipt.json'
            if done.exists():results[outer]=read(folder/'PARAMETERS.json');continue
            source=[t for t in all_types if t!=outer];record=selected['all_types'] if outer=='ALL_TYPES' else outer_folds[outer]
            packets=[]
            for held in source:
                path=folder/'inner'/f'{held}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:packet={k:z[k] for k in z.files}
                else:
                    inside=[t for t in source if t!=held];models=nested_model(inside,record);assert held not in models['generator']['source_types'] and held not in models['full']['source_types']
                    ids=np.flatnonzero(data['types']==held);packet=evaluate(data,ids,models,actions);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**packet)
                packets.append(packet)
            packet=combine(packets);assert set(packet['types'])==set(source) and (outer=='ALL_TYPES' or outer not in packet['types'])
            parameters=choose_parameters(packet);write(folder/'PARAMETERS.json',dict(utc=utc(),source_types=source,**parameters));results[outer]=read(folder/'PARAMETERS.json')
            paths=[folder/'PARAMETERS.json',*sorted((folder/'inner').glob('*.npz')),HERE/'SELECTED_MODELS.receipt.json',HERE/'gate_selection.py',HERE/'candidate_menu.py']
            if outer!='ALL_TYPES':
                held_packet=evaluate(data,np.flatnonzero(data['types']==outer),selected['outer'][outer],actions,folder/'outer')
                menus=read(folder/'outer/MENUS.json');development=[]
                for r in menus:
                    ci=int(np.flatnonzero(held_packet['names']==r['controller'])[0]);si=r['state_index'];cc,detail=choose(r['rows'],np.asarray(r['probability']),parameters['critic']['k'],parameters['critic']['delta'],parameters['probe']['tau'],parameters['probe']['rho'],held_packet['features'][ci]);scored={m:r['known_scores'][v] for m,v in cc.items()}
                    development.append(dict(controller=r['controller'],type=outer,state_uid=r['state_uid'],state_index=si,choices=cc,details=detail,known_success=scored,truth_no_repair=int(held_packet['truth_no_repair'][ci,si])))
                write(folder/'DEVELOPMENT.json',development);paths.extend([folder/'DEVELOPMENT.json',folder/'outer/COMPLETE.receipt.json'])
            seal(str(done.relative_to(HERE)),paths,source_only_parameter_selection=True)
            write(HERE/'GATE_SELECTION_PROGRESS.json',dict(utc=utc(),completed=list(results)))
    write(HERE/'SELECTED_GATES.json',dict(utc=utc(),outer={t:results[t] for t in all_types},all_types=results['ALL_TYPES']))
    seal('SELECTED_GATES.receipt.json',[HERE/'SELECTED_GATES.json',*[HERE/'gate_fits'/t/'COMPLETE.receipt.json' for t in [*all_types,'ALL_TYPES']]],all_selection_source_only=True,new_evaluation_labels_read=False)
if __name__=='__main__':main()
