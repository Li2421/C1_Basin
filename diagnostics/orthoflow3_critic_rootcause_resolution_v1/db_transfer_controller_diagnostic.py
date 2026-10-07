"""Post-hoc replay on six previously opened held controllers; no new labels.

Not independent confirmation and not a model-selection step. Measures whether
the same source-selected cross-scene model retains controller/state information
where the constant eta choice has genuine headroom.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from .db_transfer_data import OUT as SOURCE,ROOT,read,write,sha,goal_design
from .db_transfer_train import model
from .db_transfer_evaluate import csvwrite
OUT=SOURCE/'opened_controller_diagnostic'
TARGETS=tuple(range(88132,88138))

def run():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    from .motion_confirmation_eval import measures
    frozen=read(SOURCE/'models_frozen.json');norm=read(SOURCE/'normalization.json')
    source_states={s['state_uid'] for s in read(SOURCE/'states.json')};source_ctls=set(read(SOURCE/'controllers.json'))
    OUT.mkdir(exist_ok=True);functions=[]
    for r in frozen['models']:
        m=model(r['kind']);p=serialization.msgpack_restore((Path(r['path'])/'checkpoint.msgpack').read_bytes())
        functions.append((r,jax.jit(lambda x,e,c,m=m,p=p:m.apply(p,x,e,c))))
    audits=[]
    for target in TARGETS:
        folder=ROOT/f'motion_independent_confirmation_{target}'
        states=read(folder/'states.json');pairs=read(folder/'pairs.json');phys=read(folder/'physical.json');protocol=read(folder/'protocol.json')
        assert not source_states&{s['uid'] for s in states}
        assert all(p['controller_uid'] not in source_ctls for p in protocol['profiles'])
        assert all(goal_design(s['physical'])[0]==.3 for s in phys)
        x=rep.batch([rep.entities(s['physical']) for s in phys]);eta=np.array([p['eta'] for p in pairs],np.float32);e=((eta-norm['eta_center'])/norm['eta_scale']).astype(np.float32)
        contexts={}
        for condition,label in (('correct','held'),('wrong_controller','alt')):
            a=np.load(folder/f'inputs_{label}.npz');g=np.load(folder/f'goal_both_{label}.npz');assert a['valid'].all() and g['valid'].all()
            raw=np.c_[a['context'],a['agent_response'].mean(1),g['goal_response'],g['goal_motion_response'],np.ones((len(pairs),4))]
            contexts[condition]=((raw-norm['context_center'])/norm['context_scale']).astype(np.float32)
        si=np.array([p['state_index'] for p in pairs]);shift=np.roll(np.arange(len(states)),-1);pairshift=(2*shift[:,None]+np.arange(2)).ravel()
        scores={}
        for r,fn in functions:
            for condition in ('correct','wrong_controller','state_shuffle','joint_state_context_shuffle'):
                ids=shift[si] if condition in ('state_shuffle','joint_state_context_shuffle') else si
                c=contexts['wrong_controller' if condition=='wrong_controller' else 'correct']
                if condition=='joint_state_context_shuffle':c=c[pairshift]
                z=np.asarray(fn({k:v[ids] for k,v in x.items()},jnp.asarray(e),jnp.asarray(c))).reshape(-1,2)
                scores[f'{r["kind"]}__{r["seed"]}__{condition}']=z
        np.savez_compressed(OUT/f'predictions_{target}.npz',**scores)
        audits.append(dict(target=target,source_state_overlap=0,source_controller_overlap=0,predictions_sha256=sha(OUT/f'predictions_{target}.npz'),existing_labels_previously_opened=True))
    write(OUT/'prediction_audit.json',dict(targets=audits,models_sha256=sha(SOURCE/'models_frozen.json'),source_only_checkpoint_selection=True,posthoc_only=True,new_rollouts=0))
    rows=[]
    for target in TARGETS:
        d=np.load(ROOT/f'motion_independent_confirmation_{target}/dataset.npz');s=d['success'].reshape(-1,2);f=d['failure'].reshape(-1,2)
        for name,z in np.load(OUT/f'predictions_{target}.npz').items():
            kind,seed,condition=name.split('__');rows.append(dict(target=target,kind=kind,seed=int(seed),condition=condition,**measures(z,s,f)))
    csvwrite(OUT/'metrics.csv',rows)
    print([dict(target=r['target'],kind=r['kind'],seed=r['seed'],B15=r['B15'],NLL=r['NLL_pair_equal'],state_contrast_skill=r['centered_contrast_skill']) for r in rows if r['condition']=='correct'],flush=True)

if __name__=='__main__':run()
