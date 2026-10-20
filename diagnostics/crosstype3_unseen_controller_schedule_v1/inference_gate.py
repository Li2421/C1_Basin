"""Typed frozen inference must recover both new fits and the old comparator bitwise."""
from common import *
from inference import predict
from train_critic import load_data
import jax
def main():
    assert jax.default_backend()=='gpu';selected=read(HERE/'SELECTED_MODELS.json');data=load_data();checks=[]
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:truth={k:z[k] for k in ('full','no_controller')}
    with jax.default_matmul_precision('highest'):
        for held,models in selected['outer'].items():
            mask=data['types']==held
            for key in truth:
                p=predict(models[key],data);checks.append(dict(cohort='new104',type=held,model=key,bitwise=p[:,mask].tobytes()==truth[key][:,mask].tobytes()))
    old=D/'crosstype1c_two_eta_schedule_v1';frozen=read(old/'FROZEN_PIPELINE.json')
    with np.load(old/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:legacy={k:z[k] for k in ('names','types','features')}
    with np.load(old/'STATE_INPUTS.npz',allow_pickle=False) as z:legacy.update({k:z[k][48:] for k in ('h','c','action')})
    with np.load(old/'NEW_EVAL_PROBABILITIES.npz',allow_pickle=False) as z:truth=z['full']
    with jax.default_matmul_precision('highest'):
        for held,models in frozen['outer'].items():
            mask=legacy['types']==held;p=predict(models['full'],legacy);checks.append(dict(cohort='old_c1c_frozen62',type=held,model='full',bitwise=p[:,mask].tobytes()==truth[:,mask].tobytes()))
    assert all(r['bitwise'] for r in checks),'Frozen inference bitwise mismatch; do not relax gate'
    write(HERE/'FROZEN_INFERENCE_TESTS.json',dict(utc=utc(),passed=True,checks=checks,x64=False,matmul_precision='highest',new_eval_outcomes_read=False))
    seal('FROZEN_INFERENCE_TESTS.receipt.json',[HERE/'FROZEN_INFERENCE_TESTS.json',HERE/'inference_gate.py',HERE/'inference.py',HERE/'SELECTED_MODELS.json'])
if __name__=='__main__':main()
