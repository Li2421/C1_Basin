"""Paired local-execution versus return learning on an identified program family."""
import argparse,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_native_mass_repair_v1';SOURCE=ROOT/'controller_crossed_coverage_v1'
FOLDS=read(BASE/'protocol.json')['folds']
REPRESENTATIONS=('current','complete','eta_only')
ARMS=tuple(f'{target}_{rep}' for target in ('return','motion') for rep in REPRESENTATIONS)
RUNS=[dict(fold=f,representation=a,seed=s) for f in FOLDS for s in SEEDS for a in REPRESENTATIONS]


def prepare():
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    assert read(HERE/'cache_audit.json')['nonnumerical_actions_complete']
    master=copy.deepcopy(read(SOURCE/'protocol.json'));master['schema']='local_execution_vs_long_return_identified_goal_program_v1'
    master['folds']=FOLDS;master['arms']=ARMS
    master['experiment']=dict(
        history='All7680goal-reference firstactions exactlycached. Earlieraction/encounter tests usedactions as QINPUT;this test changes the supervisedTARGET. Oldexactparameters+initializer Qfits were2-sceneanalytic6family,notcurrent4sharedendpoint-matchedM,norpairedlocal-targetdiagnostic. Do notclaiminitializeraugmentation itselfisnew.',
        question='Does transferalreadyfail forlocalS/C/eta execution, oronlywhenpredictinglong-horizonreturn? Doesknowninitializerinformationchangeeither?',
        factorial='2targets(return versusone-stepmotion) x2inputs(current64 versus64+8knowninitializer);faireta-onlyeach target. Allstatearms72Cchannels;currentpadszeros. Samephysical/APIencoding. No architecture/lr/widthsweep.',
        controller_scope='Only4structurallyuncoupledGOAL-referencefamilies(isotropic,anisotropic,rotate+/-);selectionfixedbyStage28analyticidentifiability,notperformance. Fullglobaltargetlawiscommonanddeterministic,nohiddenfutureFlowweights. LocalM exactlyvisible;completearm additionallystatesGaussianinitializerstatistics/chart. Thisisamechanismcontrol,notreplacementoftheall-controllerprimarybenchmark.',
        motion_target='Forfirstexecutedactionu: z=.5+.5*mean_agents[(u dotgoal_unit +u dotgoal_left_unit)/(sqrt(2)*vmax)]. Fixed45degprojectionretainsforwardandlateralresponse. Bounded[0,1],notasuccesscriterion. No physicalstep/querygenerated.',
        shared_trials='Bothtrainingtargets useexactlysameseedsandknownQ-outcome mask;firstactiononlaternumerical-failuretrials ispreservedforevaluation,butnotusedtogiveonelearnerextratrainingobservations. No censoredQimputation.',
        fitting='Original6000steps,120samples/persourcescene,uniformC/state/observedeta inthiscontrolledgoal-onlydataset. BCEworksforbothBernoullisuccessandboundedsoftmotiontargets. SameSOURCEVALregret/ranking/state_etaMSE/NLLcheckpoint rule. Pairreturn/motionfitsonsameCPU,identicalinitialweights/RNGdraws. Physical/APIslotsretained.',
        inputs='Reusecurrententity+64fieldarrays. Initializer8EXACToldformula(mu2,cov3,chartaxis2,clipbound1),appendtocontrollerchannels,neveroverwriteAPIphysicalslots. Allinitializerfields inphysicalvmaxunits;unclippedidentityscale;noheldnormalizationstatistics.',
        execution='Allcontroller/Flow/safety/eta/horizon/evaluatorfrozen;onlysupervisedlearningdiagnostic. No H20/rolloutpreviewinput. Firstactionsaretargetsonly.',
        gate='Firstjoint18fits. ContinueLOSO onlyifmotion_complete improvesSOURCEVALQ_MSE overeta-only in>=2/3seeds AND positiveindependenthalf SOURCEVAL interactionskill(meanacrosssourcescenes);otherwise localtasknotvalidated anddoNOTcallfailurelong-horizon. Sourcegateisdiagnostic,notconfirmation.',
        decision_table=['local_complete transfers butreturn_completefails: failureafterlocalexecutionmapping;futureFlowmissingweights arenotrequiredforfailureinthisknown-goalcontrol.',
            'local_completealsofails despiteSOURCElearning: doNOTclaimlong-horizon-onlycause;localcross-geometrycomposition/inputrepresentationremains.',
            'initializerimproveslocalbutnotreturn: initializerislocalinformationgap,notcompleteQtransferrepair.',
            'returncompletepassesstrictmechanismcontrols: inspectmatchedinputcontrast andthenconfirmonuntouchedstates;noneoftheopenedsetsisFINAL.'],
        evaluation='Allfourheldnatural40states,sourceVAL8separate;all4candidateetas. Independenthalves1/1or2/2;20000physical-statebootstrap/recenter,unknownbounds,multiplewrong-statepermutations,allfixed. Samezero-interactionnull foreach target. Do notcompare rawerror magnitudesbetweenunliketargets.',
        literature=dict(url='https://arxiv.org/abs/1606.05312',review='Primarysuccessor-featurespaper reviewedbeforestage:separatesdynamics/featuresfromreturnunderfixed-dynamicsrewardtransfer. Motivateslocalversusreturndiagnosis,NOTa theoremforourchangedgeometryandnotimplementingSFhere.'),
        new_rollouts=0,new_controller_queries=0,training_seeds=SEEDS)
    code=(BASE/'learn.py').read_text()
    anchor="    d=dict(np.load(HERE/'dataset.npz'));held=FOLDS[job['fold']]"
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+"\n    if job['arm'].startswith('motion_'):\n        d['success']=d['motion_success'].copy();d['failure']=d['motion_failure'].copy()")
    old="values=features[tr][mask[tr]];mean=values.mean(0);scale=np.maximum(values.std(0),.05)"
    new="values=features[tr][mask[tr]][:,:64];mean=np.r_[values.mean(0),np.zeros(8,dtype=np.float32)];scale=np.r_[np.maximum(values.std(0),.05),np.ones(8,dtype=np.float32)]"
    assert code.count(old)==1;code=code.replace(old,new)
    old="    c=jnp.asarray(np.clip((features-mean)/scale,-10,10)*mask[...,None],jnp.float32)"
    new="    normalized=np.concatenate((np.clip((features[...,:64]-mean[:64])/scale[:64],-10,10),features[...,64:]),-1)*mask[...,None]\n    if not job['arm'].endswith('_complete'):normalized[...,64:]=0\n    c=jnp.asarray(normalized,jnp.float32)"
    assert code.count(old)==1;code=code.replace(old,new)
    code=code.replace("if job['arm'] in ('union_native_slots','union_slots_state_erased'):","if True:  # Preserve APIroles in EVERY controlled representation.")
    code=code.replace("job['arm']=='eta_only'","job['arm'].endswith('eta_only')")
    f=HERE/'learn.py'
    if f.exists():assert f.read_text()==code
    else:f.write_text(code)
    master['experiment']['code_sha256']=sha(f);freeze(HERE/'protocol.json',master)
    # Metadata-only initializer extraction; reuse the exact previous formula.
    original=read(ROOT/'controller_ff_factorial_repair_v1/protocol.json');states={s['state_uid']:s for s in original['states']}
    runtimes={sc:native_runtime(sc) for sc in original['environments']}
    def forbidden(*a,**kw):raise RuntimeError('Physicalstepsforbiddeninmetadataextraction')
    for rt in runtimes.values():rt.env_cls.step=forbidden
    cache=dict(np.load(HERE/'cached_first_actions.npz'));current=dict(np.load(SOURCE/'dataset.npz'))
    keep=np.flatnonzero(~np.char.endswith(cache['controller'],'_coupled'));assert len(keep)==640
    lookup={(str(s),str(c)):i for i,(s,c) in enumerate(zip(current['state_uid'],current['controller']))}
    source=np.array([lookup[str(cache['state_uid'][i]),str(cache['controller'][i])] for i in keep])
    d={k:(v.copy() if k=='eta' else v[source].copy()) for k,v in current.items()};init=[];motion=np.full((640,16,4),np.nan)
    metadata={}
    for row,i in enumerate(keep):
        sc,sid=str(cache['scene'][i]),str(cache['state_uid'][i]);rt=runtimes[sc]
        if sid not in metadata:
            physical=states[sid]['physical'];env=SimpleNamespace(**{k:np.asarray(v,float) for k,v in physical.items() if k in ('positions','velocities','goals')})
            n=len(env.positions);e=env.goals-env.positions;e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12);lat=np.stack((-e[:,1],e[:,0]),-1)
            local=lambda w:np.stack(((w*e).sum(-1),(w*lat).sum(-1)),-1)
            mu=local(rt.to_world(rt.mean,env))/rt.config.max_speed
            b0=rt.to_world(np.column_stack((np.ones(n),np.zeros(n))),env);b1=rt.to_world(np.column_stack((np.zeros(n),np.ones(n))),env)
            l0=local(b0*rt.scale[:,0,None])/rt.config.max_speed;l1=local(b1*rt.scale[:,1,None])/rt.config.max_speed
            cov=l0[:,:,None]*l0[:,None,:]+l1[:,:,None]*l1[:,None,:]
            z=np.column_stack((mu,cov[:,0,0],cov[:,0,1],cov[:,1,1],local(b0),np.full(n,1/rt.config.max_speed)))
            metadata[sid]=(np.pad(z,((0,4-n),(0,0))),e,lat,n,rt.config.max_speed)
        z,e,lat,n,vmax=metadata[sid];init.append(z)
        act=cache['first_action'][i,:,:,:n];value=.5+.5*np.mean(np.sum(act*(e+lat),axis=-1)/(np.sqrt(2)*vmax),axis=-1)
        obs=cache['observed'][i].astype(bool);assert np.isfinite(value[obs]).all() and value[obs].min()>=-1e-6 and value[obs].max()<=1+1e-6
        motion[row,obs]=np.clip(value[obs],0,1)
    d['feature']=np.concatenate((d['feature'],np.asarray(init,np.float32)),-1)
    seen=cache['observed'][keep].astype(bool);known=seen&~cache['unknown'][keep].astype(bool)
    d['motion_success']=np.where(known,motion,0).sum(-1);d['motion_failure']=np.where(known,1-motion,0).sum(-1)
    np.testing.assert_allclose(d['motion_success']+d['motion_failure'],d['success']+d['failure'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(d['motion_success']+d['motion_failure']>0,d['success']+d['failure']>0)
    assert not set(d['physical_hash'][d['split']=='train'])&set(d['physical_hash'][d['split']=='validation'])
    assert not (HERE/'dataset.npz').exists()
    np.savez_compressed(HERE/'dataset.npz',**d)
    np.savez_compressed(HERE/'raw_labels.npz',motion=motion,**{k:v[keep] for k,v in cache.items() if k not in ('first_action','first_action_known')},first_action=cache['first_action'][keep])
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),raw_sha256=sha(HERE/'raw_labels.npz'),
        source_sha256=sha(SOURCE/'dataset.npz'),cached_first_actions_sha256=sha(HERE/'cached_first_actions.npz'),rows=640,physical_states=160,
        training_states_per_scene=32,validation_states_per_scene=8,matching_training_mask=True,API_slots_preserved=True,initializer_metadata_only=True,
        physical_train_val_overlap=0,new_rollout=0,new_controller_queries=0,new_fit=0))
    print(dict(rows=640,physical_states=160,source_fit_jobs=18,new_rollout=0,new_query=0),flush=True)


def fit(index):
    from . import learn
    run=RUNS[index]
    for target in ('return','motion'):
        job=dict(fold=run['fold'],arm=target+'_'+run['representation'],seed=run['seed'])
        learn.fit(learn.JOBS.index(job))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args();fit(a.index) if a.stage=='fit' else prepare()
