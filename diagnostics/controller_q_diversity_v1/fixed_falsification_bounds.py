"""Conservative all-fixed-policy falsification under numerical censoring.

The strongest fixed adversary may choose a different eta for each controller,
including a different choice after each bootstrap draw and unknown completion.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def run():
    assert (ROOT/'evaluation_complete.json').exists()
    d=np.load(ROOT/'data/final/dataset.npz');rows=[];samples={}
    for panel in ('joint','toy_source','ring_source'):
        folder=ROOT if panel=='joint' else ROOT/'confirmation_scene_transfer'/panel.removesuffix('_source')
        pred=np.load(folder/'confirmation_predictions.npz')
        primary=json.loads((folder/'models_frozen.json').read_text())['primary']
        np.testing.assert_array_equal(d['state_uid'],pred['state_uid'])
        for scene in sorted(set(d['scene'])):
            ids=np.flatnonzero(d['scene']==scene);n=len(ids)//2
            s=d['seed_success'][ids].reshape(n,2,16,16);u=d['seed_unknown'][ids].reshape(n,2,16,16)
            selection=pred[primary][:,ids].mean(0).reshape(n,2,16).argmax(-1)
            boot=np.random.default_rng(2026100954).integers(n,size=(20000,n))
            for metric,lo,hi in [
                ('evaluation8',s[...,8:].mean(-1),(s+u)[...,8:].mean(-1)),
                ('B15',(s.sum(-1)>=15).astype(float),((s+u).sum(-1)>=15).astype(float))]:
                r=np.arange(n)[:,None];c=np.arange(2)[None,:]
                ml=lo[r,c,selection];mh=hi[r,c,selection]
                fl=lo.mean(0).max(-1);fh=hi.mean(0).max(-1)
                lower=float((ml.mean(0)-fh).mean());upper=float((mh.mean(0)-fl).mean())
                operational=float((ml.mean(0)-fl).mean())
                low_bs=(ml[boot].mean(1)-hi[boot].mean(1).max(-1)).mean(-1)
                high_bs=(mh[boot].mean(1)-lo[boot].mean(1).max(-1)).mean(-1)
                op_bs=(ml[boot].mean(1)-lo[boot].mean(1).max(-1)).mean(-1)
                rows.append(dict(panel=panel,scene=str(scene),metric=metric,physical_states=n,
                    verified_success_difference=operational,
                    verified_success_CI95=np.quantile(op_bs,[.025,.975]).tolist(),
                    physical_success_difference_lower=lower,physical_success_difference_upper=upper,
                    physical_lower_CI95=np.quantile(low_bs,[.025,.975]).tolist(),
                    physical_upper_CI95=np.quantile(high_bs,[.025,.975]).tolist()))
                prefix=panel+'_'+str(scene)+'_'+metric
                samples[prefix+'_lower']=low_bs;samples[prefix+'_upper']=high_bs;samples[prefix+'_operational']=op_bs
    (ROOT/'fixed_falsification_bounds.json').write_text(json.dumps(dict(rows=rows,
        method='Adaptive selected lower minus maximum of all16fixed-policy upper bounds, separately per controller; reverse endpoints for an upper bound. Reselect inside each physical-state bootstrap.',
        caveat='Conservative bounds; shared unknowns may cancel in the true paired contrast. Bootstrap uncertainty is conditional on the two held controllers.',
        new_rollouts=0),indent=2)+'\n')
    np.savez_compressed(ROOT/'fixed_falsification_bounds_bootstrap.npz',**samples)
    print('All-fixed-policy censoring bounds complete')


if __name__=='__main__':run()
