"""Independent-seed signal decomposition; existing evidence only."""
import itertools
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import HERE as FOUR,read,write
from diagnostics.controller_ff_crossed_cache_v1.study import HERE as CACHE
from pathlib import Path

HERE=Path(__file__).resolve().parent

def components(q):
    axes=tuple(range(q.ndim));out={():np.mean(q,keepdims=True)}
    for count in range(1,len(axes)+1):
        for group in itertools.combinations(axes,count):
            complement=tuple(i for i in axes if i not in group)
            a=np.mean(q,axis=complement,keepdims=True) if complement else q.copy()
            for prior,value in out.items():
                if set(prior)<set(group):a=a-value
            out[group]=a
    return out

def analyze(a,b,names,seed=2026101203):
    aa,bb=components(a),components(b);result={}
    rng=np.random.default_rng(seed);draws=rng.integers(len(a),size=(20000,len(a)))
    # Batch dimension is kept separate from the factorial axes.
    for group in aa:
        if not group:continue
        label='*'.join(names[k] for k in group)
        result[label]=dict(independent_half_energy=float(np.mean(aa[group]*bb[group])))
    values={k:[] for k in result}
    for idx in draws:
        ac,bc=components(a[idx]),components(b[idx])
        for group in aa:
            if group:values['*'.join(names[k] for k in group)].append(float(np.mean(ac[group]*bc[group])))
    for k,v in values.items():result[k]['CI95']=np.quantile(v,[.025,.975]).tolist()
    # Discovery selects a fixed eta per controller and a per-state eta;
    # independent half measures whether the selection signal repeats.
    fixed=a.mean(0).argmax(-1);chosen=a.argmax(-1)
    per=np.take_along_axis(b,chosen[...,None],-1)[...,0]
    if a.ndim==3:
        ref=np.take_along_axis(b,np.broadcast_to(fixed,(len(a),len(fixed)))[...,None],-1)[...,0]
        delta=(per-ref).mean(-1)
    else:delta=per-b[:,int(fixed)]
    return dict(N=len(a),variance_components=result,discovery_best_vs_discovery_fixed_eval_gain=float(delta.mean()),
                crossfit_gain_CI95=np.quantile(delta[draws].mean(-1),[.025,.975]).tolist())

def run():
    d=np.load(FOUR/'combined.npz');output=[]
    for sc in np.unique(d['scene']):
        ix=np.flatnonzero(d['scene']==sc);ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
        n=len(ix)//5;y=d['seed_success'][ix][:,[0,5,10,15],:2].reshape(n,5,4,2)
        u=d['seed_unknown'][ix][:,[0,5,10,15],:2].reshape(n,5,4,2)
        # Complete physical panels only. This is an explicitly selected
        # diagnostic, never the natural-distribution primary evaluation.
        keep=~u.any((1,2,3));z=y[keep].astype(float)
        r=analyze(z[...,0],z[...,1],('state','controller','eta'))
        r.update(scene=str(sc),panel='matrix_four_eta',states_before_numerical_exclusion=n,
                 numerical_unknown=int(u.sum()),excluded_states=int((~keep).sum()),
                 warning='Resolved-panel diagnostic; exclusion can be informative. Negative cross-half energy is sampling noise, not negative variance.')
        output.append(r)
    for sc in ('double_bottleneck','four_way_intersection'):
        rr=[r for r in read(CACHE/'rows.json') if r['kind']=='native_historical' and r['state']['scenario']==sc]
        y=np.asarray([r['outcome'] for r in rr]);keep=~(y==-1).any((1,2));y=y[keep].astype(float)
        r=analyze(y[:,:,:8].mean(-1),y[:,:,8:].mean(-1),('state','eta'))
        r.update(scene=sc,panel='native_four_common_anchors',states_before_numerical_exclusion=len(rr),excluded_states=int((~keep).sum()),
                 warning='Historical source-selected states; useful for TRAIN/diagnosis, not natural final confirmation. Different eta anchors than matrix probe; not a causal magnitude comparison.')
        output.append(r)
    write(HERE/'signal_budget.json',dict(rows=output,new_rollouts=0,bootstrap_unit='physical state',replicates=20000))
    for r in output:
        print(r['scene'],r['panel'],'N',r['N'],'crossfit_gain',r['discovery_best_vs_discovery_fixed_eval_gain'],r['crossfit_gain_CI95'],
            {k:round(v['independent_half_energy'],5) for k,v in r['variance_components'].items()})

if __name__=='__main__':run()
