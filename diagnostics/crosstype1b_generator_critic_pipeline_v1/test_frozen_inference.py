"""Inference path reproduces cached development predictions before freeze."""
from common import *
from new_evaluation_choices import predict
import jax


def main():
    assert jax.default_backend()=='gpu'
    with np.load(HERE/'TRAINING_BUNDLE.npz') as z:data={k:z[k] for k in ('names','types','features','h','c','action')}
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES_V2.npz') as z:expected_full=z['full']
    with np.load(HERE/'MATCHED_NO_STATE_PROBABILITIES_V2B.npz') as z:expected_no_state=z['probability']
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz') as z:expected_no_controller=z['no_controller']
    expected=dict(full=expected_full,no_state=expected_no_state,no_controller=expected_no_controller)
    models=read(HERE/'SELECTED_MODELS_V2B.json')['outer'];errors={}
    for held,group in models.items():
        mask=data['types']==held
        for mode in ('full','no_state','no_controller'):
            result=predict(group[mode],data)[:,mask];target=expected[mode][:,mask]
            error=float(np.max(np.abs(result-target)));errors[held+'_'+mode]=error
            assert np.array_equal(result,target),(held,mode,error)
    write(HERE/'FROZEN_INFERENCE_TESTS.json',dict(utc=utc(),passed=True,bitwise_exact=True,
        max_probability_error=errors,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),new_state_outcomes_read=False))
    seal('FROZEN_INFERENCE_TESTS.receipt.json',[HERE/'FROZEN_INFERENCE_TESTS.json',HERE/'test_frozen_inference.py',
        HERE/'new_evaluation_choices.py',HERE/'DEVELOPMENT_CHOICES_V2B.receipt.json'])
    print('All 27 held-type/mode frozen inference predictions bitwise exact',flush=True)


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
