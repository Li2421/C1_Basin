"""Locate a specific possible information loss without fitting or simulation."""
import argparse,itertools
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_crossed_coverage_loso_v1'
FAMILIES=('isotropic','anisotropic','rotate_positive','rotate_negative')
M=np.array([[[2.5,0],[0,2.5]],[[5,0],[0,.8]],[[2.5,-2.5],[2.5,2.5]],[[2.5,2.5],[-2.5,2.5]]])


def prepare():
    from diagnostics.controller_native_mass_repair_v1.interaction_audit import __file__ as engine
    freeze(HERE/'protocol.json',dict(
        question='Is missing cross-agent field information sufficient to explain currentLOSO failure, or doesfailurepersistforfouruncoupled affineoperators whose2x2M is exactlyvisible?',
        historical_reuse=[str(ROOT/x) for x in ['controller_response_families_v1/static_operator_audit.json','controller_operator_repair_v1','controller_program_input_probe_v1','controller_crossed_coverage_loso_v1']],
        history_gap='Oldendpoint/contactalias,raw210/compact64,exact6parameters+initializer alreadytested. No newfit. Originalstaticproof comparedendpoint/action differences,notrecoveryofthese4affineMfromactualcompactinputs norcurrentLOSO pairwise structuralpartition. Reusealloldfeaturearrays/predictions/labels.',
        derivation='Forfouruncoupledfamilies,worldM is2.5I,goal-aligneddiag(5,.8),or2.5I+/-2.5J. Goalframesofall7physicalprobesalignwithinitialgoalframe;M isvelocity/timeindependent. Simultaneous+/-half-speed probes yieldlocalJac=-M. Thuscompactcolumns56:60retainexactM;variance60:64nearzero. CoupledgraphM hasoff-agentblocks and isnotcoveredbythisproof.',
        controller_partition='ForEACHflow/goalreference,all10unorderedCpairssplitinto6neither-coupled pairs and4involving-coupled pairs. Fixedbyprogramstructure,neverlabels/modelperformance. All10aggregate retainedandreproduced.',
        inputs='Auditallcachedactiveagents inStage26dataset,raw andactualsource-normalized/clippedmodelinputs. DeterministicnearestknownM familyreadout,nolearnedclassifier,noheldlabeluse.',
        tolerance=2e-4,panels='Allfourheldnatural40statepanels +jointsourceVAL8;Flowandgoalreferencesseparate. No cherry-picked eta/state.',
        statistics='20000physicalstatebootstrap,allC/eta retained. Recenterinsidebootstrap. Current/previouscoverage models vszeroSxCxeta;independenthalf trueinteractionenergywithcensorbounds. Nofit/query/rollout.',
        boundary='RecoveringthelocalM is NOT recoveringtheentirefutureFlowtargetpolicy. FailureamonguncoupledC rulesoutmissingcross-agentJacobianascompleteexplanation,notallcontroller-inputinsufficiency. Structuralpartition issecondarydiagnosis,notnewprimarysuccesssubset.',
        next='IfuncoupledM recoverablebutheldQinteractionstillfails,doNOTtrainfullJacobianvariantsblindly;localizelong-horizonpolicyinformationversuslearnedcomposition usinghistoricalevidence first.',
        literature=dict(url='https://arxiv.org/abs/2207.01566',review='Primarypaper re-readbeforestage;probingstates mustidentifybehaviorneededforpolicyvalue. Doesnotprovecurrentfingerprints sufficientanddoesnotjustifyblindlyaddingqueries.'),
        files={str(p):sha(p) for p in [BASE/'dataset.npz',BASE/'evaluation_data/dataset.npz',BASE/'evaluation_data/raw_labels.npz',BASE/'interaction_results.json',Path(engine),ROOT/'controller_q_diversity_v1/train.py',ROOT/'controller_response_families_v1/runtime.py']},
        code_sha256=sha(__file__),new_fit=0,new_rollout=0,new_query=0))


def run():
    from diagnostics.controller_native_mass_repair_v1.interaction_audit import center,gain
    from diagnostics.controller_flow_endpoint_probe_v1.analyze import interval_center
    p=read(HERE/'protocol.json');assert sha(__file__)==p['code_sha256']
    for path,h in p['files'].items():assert sha(path)==h
    d=dict(np.load(BASE/'dataset.npz'));rows=[]
    ids=np.array([i for i,c in enumerate(d['controller']) if str(c).startswith(('flow_','goal_')) and str(c).split('_',1)[1] in FAMILIES])
    labels=np.array([FAMILIES.index(str(d['controller'][i]).split('_',1)[1]) for i in ids]);mask=d['entity_agent_mask'][ids].astype(bool)
    actual=d['feature'][ids];norms={'raw':None}
    folds=read(ROOT/'controller_native_mass_repair_v1/protocol.json')['folds']
    for fold in folds:
        path=BASE/'fits'/fold/'union_native_slots';norm=read(path/'seed17/normalization.json')
        for seed in (23,41):assert norm==read(path/f'seed{seed}/normalization.json')
        norms[fold]=norm
    for fold,norm in norms.items():
        v=actual.copy()
        if norm is not None:
            mean,scale=np.array(norm['mean'],np.float32),np.array(norm['scale'],np.float32)
            v=np.clip((v-mean)/scale,-10,10)*scale+mean
        recovered=-v[...,56:60].reshape(len(ids),4,2,2)
        error=np.max(abs(recovered-M[labels,None]),axis=(-2,-1));distance=np.sum((recovered[:,:,None]-M[None,None])**2,axis=(-2,-1));guess=distance.argmin(-1)
        for sc in sorted(set(d['scene'][ids])):
            for ref in ('flow','goal'):
                keep=(d['scene'][ids]==sc)&np.char.startswith(d['controller'][ids],ref+'_');active=mask[keep]
                rows.append(dict(normalization=fold,scene=str(sc),reference=ref,rows=int(keep.sum()),active_agents=int(active.sum()),
                    M_max_abs_error=float(error[keep][active].max()),M_within_tolerance=bool((error[keep][active]<=p['tolerance']).all()),
                    controller_family_accuracy=float((guess[keep]==labels[keep,None])[active].mean())))
    write(HERE/'input_recovery.json',dict(rows=rows,rows_checked=len(ids),new_queries=0,scope=p['boundary']))
    old=dict(np.load(BASE/'evaluation_data/dataset.npz'));raw=dict(np.load(BASE/'evaluation_data/raw_labels.npz'))
    rr=np.flatnonzero(d['source']=='matrix_goal_and_flow32');rr=rr[np.argsort(d['source_row'][rr])]
    np.testing.assert_array_equal(d['state_uid'][rr],old['state_uid'])
    previous=read(BASE/'interaction_results.json');out=[];arrays={};saved={}
    for fold,held in folds.items():
        predictions={a:np.mean([np.load(BASE/'fits'/fold/a/f'seed{s}/predictions.npz')['probability'][rr][:,[0,5,10,15]] for s in (17,23,41)],0) for a in ('union_native_slots','uniform_native_slots')}
        scenes=sorted(set(old['scene'])) if held is None else (held,)
        for sc in scenes:
            for ref in ('flow','goal'):
                keep=(old['scene']==sc)&(old['kind']==ref)
                if held is None:keep&=old['split']=='validation'
                ix=np.flatnonzero(keep);ix=np.array(sorted(ix,key=lambda i:(old['state_uid'][i],old['controller'][i])));n=len(np.unique(old['state_uid'][ix]));cs=np.unique(old['controller'][ix]);assert len(cs)==5 and len(ix)==n*5
                shape=(n,5,4);rs,ru,seen=[raw[k][ix][:,[0,5,10,15]].reshape(shape+(4,)) for k in ('success','unknown','observed')]
                count=seen.sum(-1);lo=rs.sum(-1)/count;hi=(rs+ru).sum(-1)/count
                hl=np.zeros(shape+(2,));hu=hl.copy()
                for i in range(n):
                    for c in range(5):
                        nt=int(count[i,c,0]);assert nt in (2,4) and (count[i,c]==nt).all()
                        for half,jj in enumerate((slice(0,nt//2),slice(nt//2,nt))):
                            hl[i,c,:,half]=rs[i,c,:,jj].mean(-1);hu[i,c,:,half]=(rs[i,c,:,jj]+ru[i,c,:,jj]).mean(-1)
                a,b=np.triu_indices(5,1);coupled=np.array([str(c).endswith('_coupled') for c in cs]);parts={'all':np.ones(len(a),bool),'uncoupled_only':~(coupled[a]|coupled[b]),'involving_coupled':coupled[a]|coupled[b]}
                boot=np.random.default_rng(2026101019+n).integers(n,size=(20000,n));prefix=f'{fold}__{sc}__{ref}'
                arrays[prefix+'__bootstrap_indices']=boot;saved[prefix+'__states']=old['state_uid'][ix][::5];saved[prefix+'__controllers']=cs
                for part,kk in parts.items():
                    aa,bb=a[kk],b[kk];dl,du=lo[:,aa]-hi[:,bb],hi[:,aa]-lo[:,bb]
                    hdl,hdu=hl[:,aa]-hu[:,bb],hu[:,aa]-hl[:,bb];tag=prefix+'__'+part
                    def energy(l,u,axes):
                        al,au=interval_center(l[...,0],u[...,0],axes);bl,bu=interval_center(l[...,1],u[...,1],axes)
                        v=np.stack([x*y for x in (al,au) for y in (bl,bu)]);dims=tuple(range(al.ndim-3,al.ndim))
                        return v.min(0).mean(dims),v.max(0).mean(dims)
                    ep,eq=energy(hdl,hdu,[0,2]);el=[];eu=[]
                    ps={arm:q[ix].reshape(shape)[:,aa]-q[ix].reshape(shape)[:,bb] for arm,q in predictions.items()}
                    vals={arm:[[],[]] for arm in ps};point={arm:gain(center(q[None],True),np.zeros_like(q[None]),dl[None],du[None]) for arm,q in ps.items()}
                    for start in range(0,len(boot),1000):
                        ib=boot[start:start+1000];l,u=energy(hdl[ib],hdu[ib],[1,3]);el.append(l);eu.append(u)
                        for arm,q in ps.items():
                            x=center(q[ib],True);l,u=gain(x,np.zeros_like(x),dl[ib],du[ib]);vals[arm][0].append(l);vals[arm][1].append(u)
                    el,eu=np.concatenate(el),np.concatenate(eu);arrays[tag+'__true_energy_lower']=el;arrays[tag+'__true_energy_upper']=eu
                    for arm in ps:
                        l,u=[np.concatenate(x) for x in vals[arm]];pl,pu=point[arm];ci=[float(np.quantile(l,.025)),float(np.quantile(u,.975))]
                        arrays[tag+'__'+arm+'_lower']=l;arrays[tag+'__'+arm+'_upper']=u;saved[tag+'__'+arm+'_pair_prediction']=ps[arm]
                        row=dict(fold=fold,scene=str(sc),reference=ref,partition=part,N=n,Cpairs=int(kk.sum()),model=arm,
                            predicted_interaction_skill_bounds=[float(pl[0]),float(pu[0])],skill_CI95=ci,
                            true_independent_half_energy_bounds=[float(ep),float(eq)],true_energy_CI95=[float(np.quantile(el,.025)),float(np.quantile(eu,.975))])
                        if part=='all':
                            prior=next(z for z in previous['rows'] if z['fold']==fold and z['scene']==sc and z['reference']==ref and z['interaction']=='state_controller_eta' and z['first']==arm and z['second']=='zero')
                            np.testing.assert_allclose(ci,prior['censor_CI95'],rtol=1e-6,atol=1e-7)
                        out.append(row)
    write(HERE/'results.json',dict(rows=out,input_recovery=rows,all_previous_aggregate_CIs_reproduced=True,new_fit=0,new_query=0,new_rollout=0,root_cause_closed=False))
    np.savez_compressed(HERE/'bootstrap.npz',**arrays);np.savez_compressed(HERE/'pair_predictions.npz',**saved)
    lines=['CURRENT CONTROLLER INPUT: LOCAL OPERATOR SUFFICIENCY AUDIT',p['boundary'],
        'Allinputrecoveryrows: '+str(all(r['M_within_tolerance'] and r['controller_family_accuracy']==1 for r in rows)),
        'AllprioraggregateinteractionCIsreproduced. Structuralpartitionsdefinedfromprogramsalone;all10Cpairprimaryretained. Zero newfit/query/rollout.','', 'fold reference partition model skill_CI true_energy_CI']
    for r in out:
        if r['fold']!='joint' and r['model']=='union_native_slots':lines.append(f"{r['fold']} {r['reference']} {r['partition']} {r['model']} {r['skill_CI95']} {r['true_energy_CI95']}")
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n');write(HERE/'completion.json',dict(status='COMPLETE_READ_ONLY_LOCAL_OPERATOR_AUDIT',root_cause_closed=False,new_fit=0,new_rollout=0,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(input_recovery_all_pass=all(r['M_within_tolerance'] for r in rows),held_uncoupled=[r for r in out if r['fold']!='joint' and r['reference']=='flow' and r['partition']=='uncoupled_only' and r['model']=='union_native_slots']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();globals()[a.stage]()
