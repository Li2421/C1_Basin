"""Cached, explicitly posthoc confirmation regression; no environment outcomes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,platform,time
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from . import raw_bypass_confirmation as old
from . import minimal_response_final as final
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'minimal_response_cached_diagnostic'
read,write,sha=old.read,old.write,old.sha


def main():
    OUT.mkdir(exist_ok=True);entries=[]
    source=dict(np.load(ROOT/'state_breadth_training/dataset.npz'))
    entities=dict(np.load(ROOT/'state_breadth_training/entities.npz'));va=np.arange(320,352)
    xx={k:v[source['state_index'][va]] for k,v in entities.items()}
    parity=[]
    for seed in final.fit.base.SEEDS:
        path=final.OUT/'raw_bypass'/f'seed{seed}';meta=read(path/'complete.json');mask=read(path/'input_mask_audit.json')
        assert meta['code_sha256']==sha(final.fit.__file__) and mask['wrapper_sha256']==sha(final.__file__)
        assert meta['steps']==200
        entry=dict(path=str(path),checkpoint='step200.msgpack',steps=200,seed=seed,
            checkpoint_sha256=sha(path/'step200.msgpack'),normalization_sha256=sha(path/'normalization.json'))
        entries.append(entry);p,norm=old.ev.load_entry(entry)
        eta=(source['eta'][va]-norm['eta_center'])/norm['eta_scale'];zs=[]
        for ci in range(16):
            a={k:source[k][ci,va] for k in ('context','agent_response','goal_response','goal_motion_response')}
            context=old.ev.normalized_context(a,a,norm);context[:,:88]=0
            zs.append(old.forward(p,xx,eta,context,'raw_bypass'))
        zs=np.array(zs);expected=np.load(path/'predictions.npz')['step200']
        assert np.allclose(zs,expected,atol=2e-5,rtol=2e-5)
        assert np.array_equal(zs.reshape(16,16,2).argmax(-1),expected.reshape(16,16,2).argmax(-1))
        parity.append(dict(seed=seed,max_logit_error=float(abs(zs-expected).max()),top1_exact=True))
    manifest=dict(models=entries,source_GPU_parity=parity,code_sha256=sha(__file__),
        source_selection_sha256=sha(ROOT/'raw_input_ablation/selection_frozen.json'),
        target_labels_used_for_fitting=False,target_analysis='Posthoc opened88136/88137; not fresh independent confirmation',new_rollouts=0)
    freeze=OUT/'models_frozen_before_scoring.json'
    if freeze.exists():assert read(freeze)==manifest
    else:write(freeze,manifest)
    rows=[];paired=[];latency_args=None;context_cost=[]
    for target in (88136,88137):
        folder=ROOT/f'motion_independent_confirmation_{target}'
        guard=read(folder/'prediction_freeze.json');assert guard['predictions_sha256']==sha(folder/'frozen_predictions.npz')
        for k,v in guard['input_sha256'].items():assert sha(folder/k)==v
        pairs=read(folder/'pairs.json');si=np.array([p['state_index'] for p in pairs])
        eta=np.array([p['eta'] for p in pairs],np.float32)
        x=dict(np.load(folder/'entities.npz'))
        inp={n:dict(np.load(folder/f'inputs_{n}.npz')) for n in ('held','alt')}
        resp={n:dict(np.load(folder/f'goal_both_{n}.npz')) for n in inp}
        shift=np.roll(np.arange(64),-1);pshift=np.column_stack((2*shift,2*shift+1)).ravel();pred={}
        for entry in entries:
            p,norm=old.ev.load_entry(entry);ee=(eta-norm['eta_center'])/norm['eta_scale']
            cc={n:old.ev.normalized_context(inp[n],resp[n],norm) for n in inp}
            for v in cc.values():v[:,:88]=0
            for condition in old.ev.CONDITIONS:
                cx=cc['alt' if condition=='wrong_controller_alt' else 'held']
                if condition in ('joint_state_context_shuffle','all_context_wrong_state'):cx=cx[pshift]
                ix=shift[si] if condition in ('joint_state_context_shuffle','state_shuffle') else si
                xx={k:v[ix] for k,v in x.items()}
                z=old.forward(p,xx,ee,cx,'raw_bypass').reshape(64,2)
                pred[f'{entry["seed"]}__{condition}']=z
                if latency_args is None and condition=='correct':latency_args=(p,xx,ee,cx)
        np.savez_compressed(OUT/f'predictions_{target}.npz',**pred)
        # Do not read task labels until this new model's predictions are saved.
        data=np.load(folder/'dataset.npz');s,f=[data[k].reshape(64,2) for k in ('success','failure')]
        oldpred=dict(np.load(folder/'frozen_predictions.npz'))
        for condition in old.ev.CONDITIONS:pred[f'ensemble__{condition}']=logit(np.mean([expit(pred[f'{seed}__{condition}']) for seed in final.fit.base.SEEDS],0))
        for seed in (*final.fit.base.SEEDS,'ensemble'):
            for condition in old.ev.CONDITIONS:rows.append(dict(target=target,seed=seed,condition=condition,**old.ev.measures(pred[f'{seed}__{condition}'],s,f)))
            aa=pred[f'{seed}__correct']
            full=(logit(np.mean([expit(oldpred[f'raw_bypass__{ss}__correct']) for ss in final.fit.base.SEEDS],0)) if seed=='ensemble' else oldpred[f'raw_bypass__{seed}__correct'])
            references=dict(full_raw=full,eta_mle=oldpred['eta_mle__0__correct'])
            references.update({cond:pred[f'{seed}__{cond}'] for cond in old.ev.CONDITIONS[1:]})
            for name,z in references.items():paired.append(dict(target=target,seed=seed,reference=name,**old.paired(aa,z,s,f)))
        seconds=resp['held']['seconds'][1:]
        context_cost.append(dict(target=target,mean_ms=float(seconds.mean()*1000),p50_ms=float(np.median(seconds)*1000),
            p95_ms=float(np.quantile(seconds,.95)*1000),measurement='Saved actual32goal-response construction walltime pereta in GPUsetup;notbatchedK16;hardwareidentitynotrecorded'))
    old.ev.csvwrite(OUT/'metrics.csv',rows);old.ev.csvwrite(OUT/'paired_uncertainty.csv',paired)
    p,x,e,c=latency_args;latency=[]
    for k in (1,2,4,8,16):
        ix=np.arange(k)%2;xx={key:value[ix] for key,value in x.items()};ee,cc=e[ix],c[ix]
        for _ in range(100):old.forward(p,xx,ee,cc,'raw_bypass')
        ms=[]
        for _ in range(1000):
            start=time.perf_counter_ns();old.forward(p,xx,ee,cc,'raw_bypass');ms.append((time.perf_counter_ns()-start)/1e6)
        latency.append(dict(K=k,mean_ms=float(np.mean(ms)),p50_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),p99_ms=float(np.quantile(ms,.99))))
    cpu_model=next((v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),platform.processor())
    write(OUT/'latency.json',dict(model_scoring_only=latency,physical_context_saved_cost=context_cost,hardware=cpu_model,
        backend='CPU NumPy single-thread float32;blocking operations;batched candidate dimension',warmup=100,repeats=1000,
        caveat='Repeated two source-supported eta for timing shapes only;no K16 success claim;not a measured full construction+selection frequency. Current use is one-shot selection, not low-level loop.',
        source_checkpoint_sha256=entries[0]['checkpoint_sha256']))
    write(OUT/'audit.json',dict(**manifest,targets=[88136,88137],historical_frozen_predictions_untouched=True,
        no_new_rollouts=True,models_sha256=sha(freeze),goal='Minimal-input cached diagnostic only; do not retroactively substitute for original independent test.'))
    print([dict(target=r['target'],seed=r['seed'],B15=r['B15'],NLL=r['NLL_pair_equal']) for r in rows if r['condition']=='correct'],flush=True)


if __name__=='__main__':main()
