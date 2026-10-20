"""Freeze revised requested methods, for known42×24 and eligible unseen×48."""
from common import *
from inference import predict
from design import base_unseen,anchor
import jax

def choose(p):
    assert p.ndim==4 and p.shape[0]==3 and p.shape[-1]==320 and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    mean=p.mean(0);sd=p.std(0);idx=np.arange(320);diagonal=idx//40==idx//5%8;unique=(~diagonal)|(idx%5==3);fixed=idx%5==3;diag=diagonal&fixed
    def ranks(score,valid):return np.argsort(-np.where(valid,score,-np.inf),axis=-1,kind='stable')[...,:4]
    free=ranks(mean,unique);five=ranks(mean,fixed)
    return dict(self_schedule=free[...,0],fixed500=five[...,0],no_schedule=ranks(mean,diag)[...,0],conservative_self=ranks(mean-sd,unique)[...,0],conservative_fixed=ranks(mean-sd,fixed)[...,0],top_self=free,top_fixed=five)

def main():
    assert jax.default_backend()=='gpu'
    for path,h in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(path)==h,path
    assert not (HERE/'NEW_EVAL_CHOICES.receipt.json').exists();base=base_unseen();revision=read(HERE/'PROMPT_REVISION_ACTIVE.json');known=revision['known_evaluation_order'];unseen=read(HERE/'UNSEEN_CAPABILITY.json')['eligible'];names=unseen+known;assert len(set(known))==42
    base['controller_order']=names;write(HERE/'EVALUATION_PROTOCOL_BASE.json',base)
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:order=list(z['names']);data=dict(names=np.asarray(names),types=np.asarray([base['types'][n] for n in names]),features=np.asarray([z['features'][order.index(n)] for n in names]))
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:data.update({k:z[k][48:] for k in ('h','c','action')})
    frozen=read(HERE/'FROZEN_PIPELINE.json');primary=frozen['primary_method'];nc=len(names);ns=48;output={k:np.full((nc,ns),-1,np.int32) for k in ('self_schedule','fixed500','no_schedule','conservative','no_controller','top2','top3','top4')};pstore=np.zeros((3,nc,ns,320),np.float32);nostore=np.zeros_like(pstore)
    with jax.default_matmul_precision('highest'):
        groups=[(kind,group) for kind,group in frozen['outer'].items()]+[('UNSEEN',frozen['all_types'])]
        for held,models in groups:
            mask=np.isin(data['names'],unseen) if held=='UNSEEN' else data['types']==held
            if not mask.any():continue
            if held!='UNSEEN':assert held not in models['full']['source_types']
            subset={k:v[mask] if k in ('names','types','features') else v for k,v in data.items()};p=predict(models['full'],subset);pn=predict(models['no_controller'],subset);selected=choose(p);noc=choose(pn);pstore[:,mask]=p;nostore[:,mask]=pn
            for key in ('self_schedule','fixed500','no_schedule'):output[key][mask]=selected[key]
            output['conservative'][mask]=selected['conservative_self' if primary=='self_schedule' else 'conservative_fixed'];output['no_controller'][mask]=noc[primary];ranks=selected['top_self' if primary=='self_schedule' else 'top_fixed']
            for j in (2,3,4):output[f'top{j}'][mask]=ranks[...,j-1]
        old=read(D/'crosstype1c_two_eta_schedule_v1/FROZEN_PIPELINE.json');legacy=np.full((nc,ns),-1,np.int32);legacy_probs=np.zeros_like(pstore)
        for held in sorted(set(data['types'])):
            mask=data['types']==held;models=old['all_types'];subset={k:v[mask] if k in ('names','types','features') else v for k,v in data.items()};p=predict(models['full'],subset);legacy[mask]=choose(p)[old['primary_method']];legacy_probs[:,mask]=p
    assert all(np.all(v>=0) for v in output.values()) and np.all(legacy>=0);output['pipeline']=output[primary].copy();output['legacy_pipeline']=legacy
    states=read(HERE/'CANDIDATES.json')['states'][48:];records=[]
    for i,name in enumerate(names):
        for si,state in enumerate(states[:48 if name in unseen else 24]):
            methods={k:dict(state['actions'][int(v[i,si])],candidate_index=int(v[i,si])) for k,v in output.items()}
            methods.update({k:anchor(base,a) for k,a in [('all_safety_baseline','safety_baseline'),('all_S500','S500'),('all_Sprog','Sprog'),('nominal_only','nominal_only')]})
            assert (HERE/'CONSERVATIVE_EVALUATION_AUTHORIZATION.receipt.json').exists()
            allowed=revision['unseen_methods' if name in unseen else 'known_methods']+['conservative']
            methods={k:v for k,v in methods.items() if k in allowed}
            records.append(dict(controller=name,type=base['types'][name],cohort='unseen' if name in unseen else 'known',state_uid=state['state_uid'],state_index=si,methods=methods))
    write(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json',dict(utc=utc(),primary_method=primary,states=records,known_controllers=known,unseen_controllers=unseen,new_evaluation_outcomes_read=False,legacy_model_source=str(D/'crosstype1c_two_eta_schedule_v1/FROZEN_PIPELINE.json')))
    np.savez_compressed(HERE/'NEW_EVAL_CHOICES.npz',names=names,types=data['types'],**output);np.savez_compressed(HERE/'NEW_EVAL_PROBABILITIES.npz',names=names,types=data['types'],full=pstore,no_controller=nostore,legacy_pipeline=legacy_probs)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/n for n in ('NEW_EVAL_CHOSEN_ACTIONS.json','NEW_EVAL_CHOICES.npz','NEW_EVAL_PROBABILITIES.npz','EVALUATION_PROTOCOL_BASE.json','new_choices.py','FROZEN_PIPELINE.receipt.json')],before_first_fresh_rollout_submission=True,unseen_capability_and_mechanism_exception=str(HERE/'USER_DESIGN_DECISIONS.receipt.json'),outcomes_read=False)
    print('All frozen choices sealed:',len(unseen),'unseen controllers ×48 states;42 known controllers ×24 states',flush=True)
if __name__=='__main__':main()
