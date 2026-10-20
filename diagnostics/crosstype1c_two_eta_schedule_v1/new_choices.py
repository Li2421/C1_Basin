"""Seal every evaluated method and top-ranked sparse-menu audit choice."""
from common import *
from inference import predict,legacy_predict,legacy_actions
from legacy_critic_model import candidate_choice as legacy_choice
import jax
def choose(p):
    assert p.ndim==4 and p.shape[0]==3 and p.shape[-1]==320 and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    mean=p.mean(0);sd=p.std(0);idx=np.arange(320);diagonal=idx//40==idx//5%8;unique=(~diagonal)|(idx%5==3);fixed=idx%5==3;diag=diagonal&fixed
    def ranks(score,valid):return np.argsort(-np.where(valid,score,-np.inf),axis=-1,kind='stable')[...,:4]
    free=ranks(mean,unique);five=ranks(mean,fixed)
    return dict(self_schedule=free[...,0],fixed500=five[...,0],no_schedule=ranks(mean,diag)[...,0],conservative_self=ranks(mean-sd,unique)[...,0],conservative_fixed=ranks(mean-sd,fixed)[...,0],top_self=free,top_fixed=five)
def main():
    assert not (HERE/'NEW_EVAL_CHOICES.receipt.json').exists()
    for path,h in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(path)==h,path
    assert jax.default_backend()=='gpu';frozen=read(HERE/'FROZEN_PIPELINE.json');primary=frozen['primary_method'];base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'CANDIDATES.json')['states'][48:]
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:data={k:z[k] for k in ('names','types','features')}
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:data.update({k:z[k][48:] for k in ('h','c','action')})
    nc=len(data['names']);output={k:np.full((nc,24),-1,np.int32) for k in ('self_schedule','fixed500','no_schedule','conservative','no_controller','all_types_reference','top2','top3','top4')};pstore=np.zeros((3,nc,24,320),np.float32);noCstore=np.zeros_like(pstore)
    with jax.default_matmul_precision('highest'):
        for held,models in frozen['outer'].items():
            assert held not in models['full']['source_types'] and held not in models['no_controller']['source_types'];mask=data['types']==held;p=predict(models['full'],data);pn=predict(models['no_controller'],data);selected=choose(p);noc=choose(pn);pstore[:,mask]=p[:,mask];noCstore[:,mask]=pn[:,mask]
            for k in ('self_schedule','fixed500','no_schedule'):output[k][mask]=selected[k][mask]
            output['conservative'][mask]=selected['conservative_self' if primary=='self_schedule' else 'conservative_fixed'][mask];output['no_controller'][mask]=noc[primary][mask];ranks=selected['top_self' if primary=='self_schedule' else 'top_fixed']
            for j in (2,3,4):output[f'top{j}'][mask]=ranks[...,j-1][mask]
        pall=predict(frozen['all_types']['full'],data);output['all_types_reference']=choose(pall)[primary]
    output['pipeline']=output[primary].copy();output['generator_mean']=np.full((nc,24),3,np.int32)
    old=D/'crosstype1b_generator_critic_pipeline_v1';legacy=read(old/'FROZEN_PIPELINE.json');old_data=dict(data,action=np.asarray([legacy_actions(s['proposals'],base)[1] for s in states],np.float32));old_p=np.zeros((3,nc,24,13),np.float32)
    with jax.default_matmul_precision('highest'):
        for held,models in legacy['outer'].items():
            assert held not in models['full']['source_types'];mask=data['types']==held;old_p[:,mask]=legacy_predict(models['full'],old_data)[:,mask]
    old_selected=legacy_choice(old_p.mean(0));records=[]
    anchors=dict(all_safety_baseline=dict(name='safety_baseline',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='fixed',switch_step=700),all_S500=dict(name='S500',eta_a=base['eta'][13],eta_b=base['eta'][10],switch_mode='step500',switch_step=500),all_Sprog=dict(name='Sprog',eta_a=base['eta'][13],eta_b=base['eta'][10],switch_mode='progress',switch_step=700),nominal_only=dict(name='nominal_only',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='fixed',switch_step=700))
    base_methods=['self_schedule','fixed500','no_schedule','top2','top3','top4',*anchors];additional=['generator_mean','no_controller','conservative','all_types_reference','legacy_pipeline']
    for i,name in enumerate(data['names']):
        for s,state in enumerate(states):
            methods={key:state['actions'][int(values[i,s])] for key,values in output.items()};methods.update(anchors);methods['legacy_pipeline']=legacy_actions(state['proposals'],base)[0][int(old_selected[i,s])]
            records.append(dict(controller=str(name),type=str(data['types'][i]),state_uid=state['state_uid'],state_index=s,methods=methods,base_request_methods=base_methods,additional_request_methods=additional))
    write(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json',dict(utc=utc(),primary_method=primary,states=records,all_models_and_methods_chosen_without_fresh_outcomes=True,legacy_model_source=str(old/'FROZEN_PIPELINE.json')))
    np.savez_compressed(HERE/'NEW_EVAL_CHOICES.npz',names=data['names'],types=data['types'],legacy_index=old_selected,**output);np.savez_compressed(HERE/'NEW_EVAL_PROBABILITIES.npz',names=data['names'],types=data['types'],full=pstore,no_controller=noCstore,all_types_reference=pall,legacy_pipeline=old_p)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/'NEW_EVAL_CHOSEN_ACTIONS.json',HERE/'NEW_EVAL_CHOICES.npz',HERE/'NEW_EVAL_PROBABILITIES.npz',HERE/'new_choices.py',HERE/'FROZEN_PIPELINE.receipt.json'],before_first_fresh_rollout_submission=True,outcomes_read=False)
    print('All methods and top2–4 choices sealed for all62 controllers ×24 untouched states',flush=True)
if __name__=='__main__':main()
