"""Post-hoc target-eta scores under source TRAIN state embeddings; outcomes only for evaluation."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.special import expit

from diagnostics.orthoflow3_loso_partial_count_v1.data import OUT as PRE, OLD, FOLDS, load, sha
from diagnostics.orthoflow3_loso_partial_count_v1.train import data
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import model_for, gather

OUT=Path(__file__).resolve().parent
jax.config.update('jax_default_matmul_precision','highest')

def main():
    assert load(OUT/'source_h_marginal_protocol.json')['new_rollout']==0
    truth=load(PRE/'cached_truth.json');chosen=[];overview=[]
    for fold in FOLDS:
        rows,x,_,_,_,_,_,groups=data(fold,'partial_count')
        lookup={r['state_index']:r['state_uid'] for r in rows}
        # Rows have many eta per state. Sort unique actual states by UID.
        donors=[]
        for sc,g in groups.items():
            ix=sorted({rows[j]['state_index'] for j in g['train']},key=lambda j:lookup[j])[:16]
            assert len(ix)==16;donors.extend(ix)
        ck=load(PRE/fold/'partial_count/models_frozen.json')['shared']['selected']
        assert sha(ck['checkpoint'])==ck['sha256']
        m=model_for('shared');p=m.init(jax.random.PRNGKey(0),gather(x,[donors[0]]),jnp.zeros((1,3)))
        p=serialization.from_bytes(p,Path(ck['checkpoint']).read_bytes())
        fn=jax.jit(lambda xx,ee:m.apply(p,xx,ee))
        states=load(OLD/'targets'/fold/'manifest.json');etas=np.array([z['eta'] for z in states],np.float32).reshape(-1,3)
        norm=load(PRE/fold/'normalization.json');e=(etas-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
        total=np.zeros(len(e),np.float64)
        for j in donors:
            z=np.concatenate([np.asarray(fn(gather(x,np.full(min(256,len(e)-i),j,dtype=int)),jnp.asarray(e[i:i+256])))
                              for i in range(0,len(e),256)])
            total+=expit(z)
        z=(total/len(donors)).reshape(len(states),16)
        prev=np.asarray(load(PRE/'target_predictions.json')['folds'][fold]['scores']['partial_count_shared'])
        ix=z.argmax(1);old=prev.argmax(1)
        selected=[truth[fold][i]['robust'][j] for i,j in enumerate(ix)]
        for i,j in enumerate(ix):
            chosen.append({'fold':fold,'state_uid':states[i]['state_uid'],'selected_index':int(j),
                           'original_selected_index':int(old[i]),'selected_B15':selected[i],
                           'Q_lower':truth[fold][i]['lower'][j],'Q_upper':truth[fold][i]['upper'][j]})
        overview.append({'fold':fold,'n_donors':len(donors),'selected_B15_confirmed':sum(v is True for v in selected),
                         'selected_unknown':sum(v is None for v in selected),
                         'top1_agreement_original':float(np.mean(ix==old)),
                         'mean_Q_lower':float(np.mean([truth[fold][i]['lower'][j] for i,j in enumerate(ix)])),
                         'mean_Q_upper':float(np.mean([truth[fold][i]['upper'][j] for i,j in enumerate(ix)])),
                         'original_B15':sum(truth[fold][i]['robust'][j] is True for i,j in enumerate(old))})
        print(fold,overview[-1],flush=True)
    (OUT/'source_h_marginal_result.json').write_text(json.dumps({'summary':overview,'choices':chosen,
        'post_hoc_diagnostic':True,'target_labels_used_to_construct_scores':False},indent=2)+'\n')

if __name__=='__main__':main()
