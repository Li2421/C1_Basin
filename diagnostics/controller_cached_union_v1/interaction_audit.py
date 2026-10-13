"""State-dependent controller contrasts, beyond controller/eta marginal priors."""
import argparse,itertools
import numpy as np
from .study import HERE,MAT,FOLDS,ARMS,SEEDS,read,write,freeze,sha


def prepare():
    freeze(HERE/'interaction_protocol.json',dict(
        status='Frozen before union held-scene predictions/reviews are inspected',
        question='Does correct physical state modulate the learned controller-to-Q relation, beyond any per-controller eta prior?',
        truth='D_ab(s,eta)=Q(s,Ca,eta)-Q(s,Cb,eta), all unordered controllerpairs, never selected by performance',
        projections=['Remove eachcontrollerpair/eta mean overphysicalstates','Additionally remove eachstate/controllerpair mean overeta'],
        score='Squared-error improvement of the centered predicted contrast X over zero: mean(2*D*X-X^2). Centering orthogonality permits uncentered D in the product. Zero represents no predicted interaction, including perfect controller-specific globaleta priors.',
        censor='D in[qa_lower-qb_upper,qa_upper-qb_lower];score is affine in D, so exact endpointbounds. No numeric-unknown imputation is required for skill.',
        statistics='20000physical-state bootstrap, recenter predictions inside each replicate; retain allC/eta together. Paired contrasts use same bootstrap and cases.',
        comparisons=['Each arm versus zero interaction','native_slots minus state-erased supervision','native_slots minus currententity'],
        panels='All40naturalheldstates per fourLOSO folds; jointsourceVAL8states/scene separately descriptive;all10/flow5/goal5C strata fixed.',
        boundary='PositiveQ-interaction skill is not automatically top1selection advantage or untouchedfinalconfirmation. SourceVAL selected checkpoints, so source estimates descriptive only.',
        new_rollouts=0,code_sha256=sha(__file__)))


def center(pred,interaction):
    # [bootstrap, state, Cpair, eta]
    x=pred-pred.mean(1,keepdims=True)
    if interaction:x=x-x.mean(-1,keepdims=True)
    return x


def gain(first,second,dl,du):
    coefficient=2*(first-second);constant=second**2-first**2
    a=coefficient*dl+constant;b=coefficient*du+constant
    return np.minimum(a,b).mean((1,2,3)),np.maximum(a,b).mean((1,2,3))


def run():
    protocol=read(HERE/'interaction_protocol.json');assert sha(__file__)==protocol['code_sha256']
    d=dict(np.load(HERE/'dataset.npz'));old=dict(np.load(MAT/'dataset.npz'))
    rr=np.flatnonzero(d['source']=='matrix_goal_and_flow32');rr=rr[np.argsort(d['source_row'][rr])]
    np.testing.assert_array_equal(d['state_uid'][rr],old['state_uid'])
    output=[];arrays={};saved={}
    for fold,held in FOLDS.items():
        preds={arm:np.mean([np.load(HERE/'fits'/fold/arm/f'seed{s}'/'predictions.npz')['probability'][rr][:,[0,5,10,15]] for s in SEEDS],0) for arm in ARMS}
        scenes=sorted(set(old['scene'])) if held is None else (held,)
        for scene in scenes:
            mask=old['scene']==scene
            if held is None:mask&=old['split']=='validation'
            for reference in ('all','flow','goal'):
                use=mask.copy()
                if reference!='all':use&=old['kind']==reference
                ids=np.flatnonzero(use);ids=np.array(sorted(ids,key=lambda i:(old['state_uid'][i],old['controller'][i])))
                states=np.unique(old['state_uid'][ids]);controllers=np.unique(old['controller'][ids]);n,nc=len(states),len(controllers)
                assert len(ids)==n*nc
                shape=(n,nc,4)
                s,f,u=[old[k][ids][:,[0,5,10,15]].reshape(shape) for k in ('success','failure','unknown')]
                lo=s/(s+f+u);hi=(s+u)/(s+f+u)
                a,b=np.triu_indices(nc,1);dl=lo[:,a]-hi[:,b];du=hi[:,a]-lo[:,b]
                pd={arm:q[ids].reshape(shape)[:,a]-q[ids].reshape(shape)[:,b] for arm,q in preds.items()}
                boot=np.random.default_rng(2026101019+n).integers(n,size=(20000,n));prefix=f'{fold}__{scene}__{reference}'
                saved[prefix+'__states']=states;saved[prefix+'__controllers']=controllers
                saved[prefix+'__controller_pairs']=np.column_stack((a,b));saved[prefix+'__truth_difference_lower']=dl;saved[prefix+'__truth_difference_upper']=du
                arrays[prefix+'__bootstrap_indices']=boot
                for interaction in (False,True):
                    label='state_controller_eta' if interaction else 'state_controller'
                    pairs=[(arm,'zero') for arm in ARMS]+[('union_native_slots','union_slots_state_erased'),('union_native_slots','union_current_entity')]
                    samples={pair:[[],[]] for pair in pairs}
                    x={arm:center(q[None],interaction) for arm,q in pd.items()};x['zero']=np.zeros_like(x[ARMS[0]])
                    point={pair:gain(x[pair[0]],x[pair[1]],dl[None],du[None]) for pair in pairs}
                    for start in range(0,len(boot),250):
                        ix=boot[start:start+250];xx={arm:center(q[ix],interaction) for arm,q in pd.items()};xx['zero']=np.zeros_like(xx[ARMS[0]])
                        for pair in pairs:
                            al,bu=gain(xx[pair[0]],xx[pair[1]],dl[ix],du[ix]);samples[pair][0].append(al);samples[pair][1].append(bu)
                    for pair in pairs:
                        al,bu=[np.concatenate(v) for v in samples[pair]];key=prefix+'__'+label+'__'+pair[0]+'__minus__'+pair[1]
                        arrays[key+'__lower']=al;arrays[key+'__upper']=bu
                        pl,pu=point[pair]
                        output.append(dict(fold=fold,scene=scene,reference=reference,N=n,controllers=nc,Cpairs=len(a),
                            interaction=label,first=pair[0],second=pair[1],skill_identification_bounds=[float(pl[0]),float(pu[0])],
                            censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(bu,.975))],unknown=int(u.sum()),
                            inference='Held-scene DEVELOPMENT' if held else 'SourceVAL selected checkpoint; descriptive'))
    write(HERE/'interaction_results.json',dict(rows=output,protocol_sha256=sha(HERE/'interaction_protocol.json'),new_rollouts=0))
    np.savez_compressed(HERE/'interaction_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/'interaction_truth_and_cases.npz',**saved)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();prepare() if a.stage=='prepare' else run()
