"""Matched entity-response association ablation, source-only diagnosis.

All arms have identical parameter counts and receive the frozen aggregate C.
Additional agent channels are zero, population-mean broadcast, or correctly
attached responses. This separates more aggregate information from preserving
which physical entity receives a controller response. No artificial slot ID.
"""
import argparse,copy,shutil
from pathlib import Path
import numpy as np
from diagnostics.orthoflow3_state_context_learning_audit_v1 import experiment as old

OUT=Path(__file__).resolve().parent/'response_control'
SRC=old.OUT
FEATURE=OUT.parent/'agent_response'
KINDS=('response_zero','response_mean','response_entity')

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Response association protocol frozen')
    OUT.mkdir(parents=True)
    for name in ('source_data.npz','source_entities.npz','source_states.json'):
        shutil.copyfile(SRC/name,OUT/name)
    for c in ('alt','second'):
        assert old.read(FEATURE/f'audit_{c}.json')['max_aggregate_replay_error']<1e-5
    p=copy.deepcopy(old.read(SRC/'protocol.json'))
    p.update(variants=KINDS,primary_question='Does preserving controller response-to-agent association repair state-conditional prediction?',
        matched='All24 original global summaries remain; new16 agent channels zero vs mean-broadcast vs entity-attached. Identical architecture size, seeds, optimization and data.',
        input_group='Two arms nominal/corrected,8 physically signed/action-magnitude summaries each; same H20 and same two probe roots',
        target_labels_used=False,fresh_seed_replication_labels_used=False,
        diagnostic_scope='same exact eta10/15, source-family crossfit; not unseen-eta or cross-scene evidence',
        response_feature_hashes={c:old.sha(FEATURE/f'features_{c}.npz') for c in ('alt','second')},
        response_implementation_sha256=old.sha(__file__),new_task_rollouts=0)
    old.write(OUT/'protocol.json',p)

def train(index):
    import jax.numpy as jnp
    import flax.linen as nn
    from diagnostics.orthoflow3_controller_information_probe_v1 import probe
    original_critic=probe.Critic
    kind=KINDS[index//9]
    class ResponseCritic(nn.Module):
        use_state: bool
        use_context: bool
        use_id: bool
        @nn.compact
        def __call__(self,x,eta,context,controller_id):
            n=x['agents'].shape[1]
            response=context[:,24:].reshape(-1,n,16) if context.shape[-1]>24 else jnp.zeros((len(eta),n,16))
            if kind=='response_zero':response=jnp.zeros_like(response)
            elif kind=='response_mean':response=jnp.broadcast_to(response.mean(1,keepdims=True),response.shape)
            xx={**x,'agents':jnp.concatenate([x['agents'],response],axis=-1)}
            return original_critic(self.use_state,self.use_context,self.use_id,name='core')(xx,eta,context[:,:24],controller_id)
    original_load=old.load
    def load(unused_kind,fold):
        d,x,fit,inner,test,norm=original_load('full_raw',fold)
        response=np.zeros((2,len(d['eta']),4,16),np.float32)
        for ci,c in enumerate(('alt','second')):
            r=np.load(FEATURE/f'features_{c}.npz');response[ci,r['indices']]=r['agent_response']
        center=response[:,fit].mean((0,1,2));scale=np.maximum(response[:,fit].std((0,1,2)),.01)
        response=(response-center)/scale
        d['context']=np.concatenate([d['context'],response.reshape(2,len(d['eta']),-1)],axis=-1)
        norm.update(agent_response_center=center.tolist(),agent_response_scale=scale.tolist(),response_variant=kind)
        return d,x,fit,inner,test,norm
    old.OUT=OUT;old.KINDS=KINDS;old.load=load;probe.Critic=ResponseCritic
    old.train(index)

def summarize():old.OUT=OUT;old.KINDS=KINDS;old.summarize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='train':train(a.index)
    else:summarize()
