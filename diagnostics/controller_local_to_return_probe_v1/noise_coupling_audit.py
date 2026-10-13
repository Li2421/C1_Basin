"""Exact finite-noise-panel variance test; no new fitting or simulation."""
import itertools
import numpy as np
from .vector_study import HERE,read,write,freeze,sha
from .vector_review import center

def run():
    protocol=dict(
        history='Stage29 late checkpoints fit original action-label interactions but fail independent cached keys on DB/Four/Ring. Existing common-key action-mediation cache has10240 compatible actions. Earlier common-random probes were descriptor queries, not a cross-state supervision coupling contrast.',
        question='At identical two-seed marginal precision, does cross-state common randomness substantially reduce state-interaction label variance?',
        design='Use every one of six two-key subsets of cached four-key panel. Compare common subset across all states against independent uniform subset per state. Exact expectation over subset assignment, no cherry-picked pair. Same state/C/eta/noise marginal; no increase in rollout count.',
        estimand='Finite four-key panel mean, not population mean. Report conditional-on-panel physical-state bootstrap CI; no unconditional seed-generalization claim.',
        formula='After eta (and optionally C) projection e: common variance=E[e^2]-E_pair[(mean_state e)^2]; independent=(1-1/N)*E[e^2].',
        continue_gate='Further coupled-label fitting only if TRAIN state_eta expected-error reduction >=50% in EACH of DB/Four/Ring. Otherwise prune this as a sufficient explanation for this local failure; no new full rollout.',
        literature=[dict(url='https://ai.stanford.edu/~ang/papers/uai00-pegasus.pdf',reason='Primary PEGASUS shared exogenous randomness for comparisons; not a guarantee for our values.'),
                    dict(url='https://www.jmlr.org/papers/volume3/strens02a/strens02a.pdf',reason='Primary direct policy search study warns fixed scenario objectives can differ from expected-return objective.')],
        new_fit=0,new_query=0,new_rollout=0)
    freeze(HERE/'noise_coupling_protocol.json',protocol)
    d=dict(np.load(HERE/'dataset.npz'));a=dict(np.load(HERE/'independent_static_actions.npz'))
    pairs=list(itertools.combinations(range(4),2));rows=[];saved={}
    for sc in sorted(set(d['scene'])):
        for split in ('train','validation'):
            ix=np.flatnonzero((d['scene']==sc)&(d['split']==split))
            ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            n=len(ix)//4;na=int(d['entity_agent_mask'][ix[0]].sum())*2
            z=a['vector_motion'][ix][:,[0,5,10,15],:,:na].reshape(n,4,4,4,na)
            e=np.stack([z[:,:,:,list(pair)].mean(3)-z.mean(3) for pair in pairs])
            ib=np.random.default_rng(2026101032+n).integers(n,size=(20000,n))
            for name,axes in (('state_eta',(3,)),('state_controller_eta',(2,3))):
                error=center(e,axes);v=np.mean(error**2)
                common=v-np.mean(error.mean(1)**2);ind=(1-1/n)*v
                ratios=[];cs=[];is_=[]
                for start in range(0,len(ib),500):
                    # batch, pair, state, C, eta, output
                    x=np.moveaxis(error[:,ib[start:start+500]],1,0)
                    vv=np.mean(x*x,axis=(1,2,3,4,5))
                    cv=vv-np.mean(x.mean(2)**2,axis=(1,2,3,4));iv=(1-1/n)*vv
                    ratios.append(1-cv/np.maximum(iv,1e-30));cs.append(cv);is_.append(iv)
                r=np.concatenate(ratios);tag=str(sc)+'__'+split+'__'+name
                saved[tag+'__reduction']=r;saved[tag+'__common']=np.concatenate(cs);saved[tag+'__independent']=np.concatenate(is_)
                rows.append(dict(scene=str(sc),split=split,N=n,interaction=name,
                    common_variance=float(common),independent_variance=float(ind),
                    expected_reduction=float(1-common/ind),conditional_CI95=np.quantile(r,[.025,.975]).tolist()))
    gate=all(r['expected_reduction']>=.5 for r in rows if r['split']=='train' and r['interaction']=='state_eta' and r['scene']!='toy_give_way')
    write(HERE/'noise_coupling_results.json',dict(rows=rows,fit_gate_passed=gate,new_fit=0,new_query=0,new_rollout=0))
    np.savez_compressed(HERE/'noise_coupling_bootstrap.npz',**saved)
    for r in rows:
        if r['split']=='train':print(r['scene'],r['interaction'],round(100*r['expected_reduction'],2),np.round(np.asarray(r['conditional_CI95'])*100,2).tolist())
    print('coupled_label_fit_gate',gate)

if __name__=='__main__':run()

