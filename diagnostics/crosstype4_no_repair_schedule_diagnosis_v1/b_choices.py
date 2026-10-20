"""All fresh choices sealed using the model that held out the entire type."""
from common import *
from inference import predict
from actions import action
from methods import METHODS
import jax

def choose(p):
    assert p.shape[0]==3 and p.shape[-1]==486 and np.isfinite(p).all();score=p.mean(0);idx=np.arange(486);i=idx//54;j=idx//6%9;k=idx%6;unique=(i!=j)|(k==3);zero=(i==8)|(j==8)
    def pick(valid):return np.argmax(np.where(valid,score,-np.inf),axis=-1)
    return dict(pipeline=pick(unique),no_progress=pick(unique&(k!=5)),no_zero=pick(unique&~zero),legacy_fixed=pick(unique&~zero&(k<5)))

def main():
    assert jax.default_backend()=='gpu'
    for path,h in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(path)==h,path
    assert not (HERE/'NEW_EVAL_CHOICES.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');names=read(HERE/'OPENING_CONFIG.json')['B_evaluation'];types=np.asarray([base['types'][n] for n in names]);data=dict(names=np.asarray(names),types=types)
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:data['features']=np.asarray([z['features'][list(z['names']).index(n)] for n in names])
    with np.load(HERE/'NEW_EVAL_INPUTS.npz',allow_pickle=False) as z:data.update({k:z[k] for k in ('h','c','action')})
    frozen=read(HERE/'FROZEN_PIPELINE.json');output={k:np.full((74,24),-1,np.int32) for k in ('pipeline','no_progress','no_zero','legacy_fixed','no_controller')};pstore=np.zeros((3,74,24,486),np.float32);nstore=np.zeros_like(pstore)
    with jax.default_matmul_precision('highest'):
        for held in sorted(set(types.tolist())):
            models=frozen['outer'][held];assert held not in models['full']['source_types'];mask=types==held;subset={k:v[mask] if k in ('names','types','features') else v for k,v in data.items()};p=predict(models['full'],subset);pn=predict(models['no_controller'],subset);chosen=choose(p)
            for key in chosen:output[key][mask]=chosen[key]
            output['no_controller'][mask]=choose(pn)['pipeline'];pstore[:,mask]=p;nstore[:,mask]=pn
    candidates=read(HERE/'NEW_CANDIDATES.json')['states'];records=[]
    for ni,name in enumerate(names):
        for si,s in enumerate(candidates):
            methods={k:dict(s['actions'][int(v[ni,si])],candidate_index=int(v[ni,si])) for k,v in output.items()};methods['safety']=action([0,0,0],[0,0,0],name='safety');methods['human500']=action(base['eta'][13],base['eta'][10],'fixed',500,'human500');methods['human_progress']=action(base['eta'][13],base['eta'][10],'progress',700,'human_progress');assert set(methods)==set(METHODS)
            records.append(dict(controller=name,type=base['types'][name],state_uid=s['state_uid'],state_index=si,methods=methods,whole_type_excluded_from_selector_training=True))
    write(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json',dict(utc=utc(),states=records,method_names=METHODS,outcomes_read=False,evidence='DEVELOPMENT',whole_type_holdout_within_opened_development_zoo=True))
    np.savez_compressed(HERE/'NEW_EVAL_CHOICES.npz',names=names,types=types,**output);np.savez_compressed(HERE/'NEW_EVAL_PROBABILITIES.npz',names=names,types=types,full=pstore,no_controller=nstore)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/n for n in ('NEW_EVAL_CHOSEN_ACTIONS.json','NEW_EVAL_CHOICES.npz','NEW_EVAL_PROBABILITIES.npz','b_choices.py','methods.py','FROZEN_PIPELINE.receipt.json')],before_first_fresh_rollout_submission=True,outcomes_read=False)
if __name__=='__main__':main()
