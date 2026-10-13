"""Independent-half interaction signal; all natural states, no model outputs."""
import numpy as np
from .study import HERE, write
from .features import INDICES
from .learn import SCENES
from diagnostics.controller_flow_endpoint_probe_v1.analyze import energy


def run():
    d=dict(np.load(HERE/'dataset.npz'));raw=dict(np.load(HERE/'raw_labels.npz'))
    rows,arrays=[],{}
    for sc in SCENES:
        for ref in ('goal','flow'):
            ix=np.flatnonzero((d['scene']==sc)&(d['kind']==ref));ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            shape=(40,5,4)
            s,u,n=(raw[k][ix][:,list(INDICES)].reshape(shape+(4,)) for k in ('success','unknown','observed'))
            lo,hi=np.zeros(shape+(2,)),np.zeros(shape+(2,))
            for i in range(40):
                nt=int(n[i,0,0].sum());assert nt in (2,4)
                for j,ss in enumerate((slice(0,nt//2),slice(nt//2,nt))):
                    lo[i,...,j]=s[i,...,ss].mean(-1);hi[i,...,j]=(s[i,...,ss]+u[i,...,ss]).mean(-1)
            boot=np.random.default_rng(2026101403).integers(40,size=(20000,40))
            for name,axes in (('state_eta',[0,2]),('state_controller_eta',[0,1,2]),('eta',[2]),('controller_eta',[1,2])):
                mid,a,b=energy(lo,hi,axes);bs,bl,bh=[],[],[]
                for start in range(0,len(boot),500):
                    ii=boot[start:start+500];z,l,h=energy(lo[ii],hi[ii],[k+1 for k in axes]);bs.append(z);bl.append(l);bh.append(h)
                bs,bl,bh=map(np.concatenate,(bs,bl,bh));key=sc+'__'+ref+'__'+name
                arrays[key+'_midpoint'],arrays[key+'_lower'],arrays[key+'_upper']=bs,bl,bh
                rows.append(dict(scene=sc,reference=ref,centered_over=name,N=40,independent_half_energy_midpoint=float(mid),
                    identification_bounds=[float(a),float(b)],censor_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bh,.975))],
                    unknown=int(u.sum()),rollouts=int(n.sum())))
    np.savez_compressed(HERE/'signal_bootstrap.npz',**arrays)
    write(HERE/'independent_half_signal.json',dict(rows=rows,
        scope='DEVELOPMENT data-information audit. Full40naturalstates per scene; trainQ2 andVALQ4 split into independent halves. Centered energy includes higher-order interactions; not a fitted-model result.'))
    for r in rows:
        if r['centered_over']=='state_eta':print(r,flush=True)


if __name__=='__main__':run()
