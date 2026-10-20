"""Actual frozen C5 prediction replay at original dimensions before new evaluation."""
from common import *
import jax
from frozen_c5_reference import predict

def main():
    assert jax.default_backend()=='gpu';old=D/'crosstype5_repair_gate_late_eta_v1'
    with np.load(old/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(old/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:expected=z['full'];expected_n=z['no_controller']
    frozen=read(old/'FROZEN_PIPELINE.json');checks=[]
    with jax.default_matmul_precision('highest'):
        for held in ('Flow','transformer_bc','centralized_mppi'):
            mask=data['types']==held
            for key,truth in [('full',expected),('no_controller',expected_n)]:
                p=predict(frozen['outer'][held][key],data);checks.append(dict(type=held,branch=key,bitwise=p[:,mask].tobytes()==truth[:,mask].tobytes(),max_abs_error=float(np.max(abs(p[:,mask]-truth[:,mask])))))
    write(HERE/'C5_FROZEN_REFERENCE_TESTS.json',dict(utc=utc(),passed=all(x['bitwise'] for x in checks),checks=checks,original_action_dimension=14,original_probe_dimension=24,no_retraining=True));seal('C5_FROZEN_REFERENCE_TESTS.receipt.json',[HERE/'C5_FROZEN_REFERENCE_TESTS.json',HERE/'test_c5_reference.py',HERE/'frozen_c5_reference.py']);assert all(x['bitwise'] for x in checks),checks
    print('C5 frozen reference prediction bitwise replay passed',flush=True)

if __name__=='__main__':main()
