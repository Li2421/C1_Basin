"""Source TRAIN/VAL-only progress and matched diagnostics."""
import csv,json
import numpy as np
from .state_support import OUT,read,write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite

def main():
    rows=[];trajectory=[]
    for file in sorted((OUT/'models').glob('*/*/seed*/complete.json')):
        done=read(file);p=file.parent
        for r in csv.DictReader((p/'metrics.csv').open()):rows.append(r)
        history=read(p/'history.json');best=next(h for h in history if h['step']==done['best_step'])
        trajectory.append(dict(size=done['size'],kind=done['kind'],seed=done['seed'],best_step=done['best_step'],
            best_TRAIN_NLL=best['train']['NLL'],best_VAL_NLL=best['validation']['NLL'],
            last_TRAIN_NLL=history[-1]['train']['NLL'],last_VAL_NLL=history[-1]['validation']['NLL']))
    if not rows:print('No completed fits yet');return
    csvwrite(OUT/'source_validation_metrics.csv',rows);csvwrite(OUT/'source_training_diagnostic.csv',trajectory)
    groups=[]
    for size,kind in sorted({(r['size'],r['kind']) for r in rows}):
        rr=[r for r in rows if r['size']==size and r['kind']==kind and r['condition']=='correct']
        groups.append(dict(size=size,kind=kind,seeds=len(rr),
            NLL_mean=float(np.mean([float(r['NLL']) for r in rr])),MAE_mean=float(np.mean([float(r['MAE']) for r in rr])),
            B15=[int(r['B15']) for r in rr],oracle=[int(r['oracle_B15']) for r in rr],
            selected_Q=float(np.mean([float(r['selected_Q']) for r in rr])),
            strong_correct=[int(r['strong_correct']) for r in rr],strong=[int(r['strong']) for r in rr]))
    write(OUT/'source_validation_summary.json',groups)
    print(json.dumps(dict(completed_fits=len(trajectory),groups=groups),indent=2))

if __name__=='__main__':main()
