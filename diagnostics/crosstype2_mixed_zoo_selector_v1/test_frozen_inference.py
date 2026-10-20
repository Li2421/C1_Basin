from common import *
from inference import predict
from critic_model import choices
from legacy_inference import predict as legacy_predict
from train_critic import load_data
import jax


def main():
    assert jax.default_backend()=='gpu'
    data=load_data(HERE/'TRAINING_BUNDLE.npz');selected=read(HERE/'SELECTED_MODELS.json');checks=[]
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:expected={k:z[k] for k in ('full','no_controller','no_state')}
    for held,group in selected['outer'].items():
        mask=data['types']==held
        for mode in ('full','no_controller','no_state'):
            actual=predict(group[mode],data)[:,mask];reference=expected[mode][:,mask]
            error=float(np.max(np.abs(actual-reference)));passed=actual.tobytes()==reference.tobytes()
            checks.append(dict(held_type=held,mode=mode,bitwise=passed,max_error=error))
    old=D/'crosstype1b_generator_critic_pipeline_v1';frozen=read(old/'FROZEN_PIPELINE.json')
    with np.load(old/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:olddata={k:z[k] for k in ('names','types','features')}
    with np.load(old/'STATE_INPUTS.npz',allow_pickle=False) as z:olddata.update(h=z['h'][48:],c=z['c'][48:],action=z['action'][48:])
    actual=legacy_predict(frozen['all_types']['full']['model'],olddata)
    with np.load(old/'NEW_EVAL_PROBABILITIES.npz',allow_pickle=False) as z:reference=z['all_types_reference']
    checks.append(dict(kind='immutable_old_nine_type_pipeline',bitwise=actual.tobytes()==reference.tobytes(),max_error=float(np.max(np.abs(actual-reference)))))
    passed=all(c['bitwise'] for c in checks)
    write(HERE/'FROZEN_INFERENCE_TESTS.json',dict(utc=utc(),passed=passed,checks=checks,dtype_restored=True,backend=jax.default_backend(),matmul_precision='highest',new_rollouts=0))
    assert passed,'Do not relax bitwise requirement; diagnose rounding/serialization before freeze'
    seal('FROZEN_INFERENCE_TESTS.receipt.json',[HERE/'FROZEN_INFERENCE_TESTS.json',HERE/'test_frozen_inference.py',HERE/'inference.py',HERE/'legacy_inference.py'])
    print('Serialized outer predictions and immutable old pipeline reproduce bitwise')


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
