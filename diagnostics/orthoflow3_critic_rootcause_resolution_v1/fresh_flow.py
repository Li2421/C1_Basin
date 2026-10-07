"""A new frozen target controller, created after critic/model choices froze."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from .state_support import read,write,OUT as SOURCE
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha
OUT=SOURCE.parent/'fresh_target_flow_seed88124'
RULE=OUT.parent/(OUT.name+'_protocol.json')

def main():
    import jax
    from new_benchmark_common.training import train_stage1
    assert jax.default_backend()=='gpu'
    frozen=read(SOURCE/'models_frozen.json')
    assert not (OUT/'best.pkl').exists(),'Frozen target controller exists; no retraining/selection by target outcomes'
    dataset=SOURCE.parent.parent/'ring_exchange_stage1/base_u_v10_local_dataset'
    rule=dict(seed=88124,steps=100000,batch_size=256,log_interval=250,validation_batches=8,
        early_transition_fraction=.5,early_steps=15,early_nominal_only=False,source_balanced_sampling=False,
        selection='Native fixed expert-development imitation loss only; no task rollout outcome used',
        dataset_manifest_sha256=sha(dataset/'manifest.json'),critics_frozen_before_target_creation=sha(SOURCE/'models_frozen.json'),
        target_selection='One fresh initialization, no controller tournament, source Flow training recipe unchanged',
        target_flow_uses_Ring_expert_data=True,critic_target_controller_labels=0,
        scope='Critic zero-shot to new controller parameters in known Ring geometry; NOT pipeline zero-shot to an unseen scene',
        candidate_pool='Exact pre-frozen source eta10/15, all64 independent confirmation families; no model-driven eta or state selection')
    if RULE.exists():assert read(RULE)==rule,'Do not change preregistered controller after any outcome'
    else:write(RULE,rule)
    result=train_stage1(dataset,OUT,seed=88124,steps=100000,batch_size=256,log_interval=250,validation_batches=8,
        early_transition_fraction=.5,early_steps=15,early_nominal_only=False,source_balanced_sampling=False)
    write(OUT/'frozen_controller.json',dict(checkpoint='best.pkl',checkpoint_sha256=sha(OUT/'best.pkl'),
        config_sha256=sha(OUT/'config.json'),training_summary=result,models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
        target_feasibility_labels_seen=False))
    print(result,flush=True)

if __name__=='__main__':main()
