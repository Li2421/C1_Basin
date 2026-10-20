from common import *
from inference import predict
from critic_model import choices
from evaluation_design import controller_order
from zoo_design import profiles,LOCKED
from baselines import source_fixed
from legacy_inference import choices as old_choices
from train_critic import load_data
import jax


def main():
    assert jax.default_backend()=='gpu';assert not (HERE/'NEW_EVAL_CHOICES.receipt.json').exists()
    for p,digest in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(p)==digest
    frozen=read(HERE/'FROZEN_PIPELINE.json');base=profiles();names=controller_order()
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:
        index={str(n):i for i,n in enumerate(z['names'])};features=z['features'][[index[n] for n in names]]
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:data={k:z[k][48:] for k in ('h','c','action','state_uids')}
    data.update(names=np.asarray(names),types=np.asarray([base['types'][n] for n in names]),features=features);nc=len(names);output={};probabilities={m:np.zeros((3,nc,24,16),np.float32) for m in ('full','no_controller','no_state')}
    for k in ('pipeline','pipeline_conservative','pipeline_scratch','pipeline_warm','no_controller','no_state'):output[k]=np.full((nc,24),-1,np.int32)
    for kind in sorted(set(data['types'])):
        locked=kind in LOCKED;mask=data['types']==kind
        group={mode:v['model'] for mode,v in frozen['all_types'].items()} if locked else frozen['outer'][str(kind)]
        for mode in ('full','no_controller','no_state'):
            assert locked or kind not in group[mode]['source_types']
            p=predict(group[mode],data);probabilities[mode][:,mask]=p[:,mask]
            output['pipeline' if mode=='full' else mode][mask]=choices(p)['mean'][mask]
            if mode=='full':output['pipeline_conservative'][mask]=choices(p)['conservative'][mask]
        for init in ('scratch','warm'):
            model=frozen['all_initializations'][init]['model'] if locked else group['full_'+init]
            p=predict(model,data);output['pipeline_'+init][mask]=choices(p)['mean'][mask]
    train=load_data(HERE/'TRAINING_BUNDLE.npz');alltypes=sorted(set(train['types']));records={}
    output['source_fixed']=np.empty((nc,24),np.int32);output['source_mean_fixed']=np.empty((nc,24),np.int32)
    for kind in sorted(set(data['types'])):
        source=alltypes if kind in LOCKED else [t for t in alltypes if t!=kind]
        a,b,record=source_fixed(train,source);mask=data['types']==kind;output['source_fixed'][mask]=a;output['source_mean_fixed'][mask]=b;records[str(kind)]=record
    for method,a in [('all_safety_baseline',10),('all_S500',11),('all_Sprog',12),('all_RHR',13),('all_RHR_b',14),('all_RHR_c',15),('all_nominal_only',16)]:output[method]=np.full((nc,24),a,np.int32)
    candidates=read(HERE/'CANDIDATES.json')['states'][48:];output['original_pipeline']=np.broadcast_to(np.asarray([r['original_pipeline_choice'] for r in candidates])[None],(nc,24)).copy();output['generator_only']=np.ones((nc,24),np.int32)
    output['old_pipeline']=old_choices(data)
    # Development few-label choices are available now. Locked rows are explicitly
    # missing until their prescribed post-confirmation training labels exist.
    for k in (2,4,8):
        output[f'few_{k}']=np.full((nc,24),-1,np.int32)
        with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:
            lookup={str(n):int(v[0]) for n,v in zip(z['names'],z[f'few_{k}'])}
        for i,name in enumerate(names):
            if name in lookup:output[f'few_{k}'][i]=lookup[name]
    for key,v in output.items():assert v.shape==(nc,24) and np.all((v>=(-1 if key.startswith('few_') else 0))&(v<17))
    np.savez_compressed(HERE/'NEW_EVAL_CHOICES.npz',names=names,types=data['types'],state_uids=data['state_uids'],**output)
    np.savez_compressed(HERE/'NEW_EVAL_PROBABILITIES.npz',names=names,types=data['types'],**probabilities)
    write(HERE/'NEW_EVAL_SOURCE_FIXED.json',records)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/'NEW_EVAL_CHOICES.npz',HERE/'NEW_EVAL_PROBABILITIES.npz',HERE/'NEW_EVAL_SOURCE_FIXED.json',HERE/'new_choices.py',HERE/'FROZEN_PIPELINE.receipt.json'],outcomes_read=False,before_first_new_evaluation_submission=True,locked_few_labels_pending_not_fabricated=True)
    print('Frozen new-state choices saved for',nc,'controllers before any new-state rollout')


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
