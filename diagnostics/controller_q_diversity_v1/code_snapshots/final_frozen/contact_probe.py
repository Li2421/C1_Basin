from .contact_features import run as features
from .train import fit
from .experiment import HERE,read,write
import numpy as np
import shutil


def run():
    features(probe_only=True)
    dest=HERE/'data/source_probe120';dest.mkdir(exist_ok=True)
    for name in ('dataset.npz','manifest.json'):
        if not (dest/name).exists():shutil.copy2(HERE/'data/source64'/name,dest/name)
    for kind in ('state_context','state_contact','state_contact_bilinear'):
        for seed in (17,23,41):fit(kind,seed,64,variant='source_probe120')
    d=np.load(dest/'dataset.npz');good=d['success']+d['failure']>0;y=d['success']/np.maximum(d['success']+d['failure'],1)
    rows=[]
    for kind in ('state_context','state_contact','state_contact_bilinear'):
        for seed in (17,23,41):
            z=np.load(HERE/'fits/source_probe120'/kind/f'seed{seed}'/'source_predictions.npz')['logits']
            q=1/(1+np.exp(-np.clip(z,-80,80)))
            for scene in sorted(set(d['scene'])):
                for panel in ('train','unseen_state_seen_controller','seen_state_unseen_controller','unseen_both'):
                    mask=(d['scene']==scene)
                    mask&=(d['state_index']<48) if panel in ('train','seen_state_unseen_controller') else (d['state_index']>=48)
                    mask&=(d['controller']=='analytic3') if 'unseen_controller' in panel or panel=='unseen_both' else (d['controller']!='analytic3')
                    rows.append(dict(model=kind,seed=seed,scene=scene,panel=panel,Q_MSE=float(np.mean((q[mask]-y[mask])[good[mask]]**2))))
    write(HERE/'contact_probe_results.json',dict(rows=rows,final_labels_used=False,official_validation_labels_used=False,
        caveat='Exploratory TRAIN-only resplit; matched6fits use identical120sample batches,6000steps and three seeds'))
    print(rows,flush=True)


if __name__=='__main__':run()
