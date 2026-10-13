"""Paired cross-state noise-coupling intervention using already cached actions."""
import argparse,copy,itertools
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
BASE=ROOT/'controller_local_to_return_probe_v1/vector_control'
FOLDS={'joint':None}
PAIRS=list(itertools.combinations(range(4),2))
ARMS=tuple(f'motion_pair{j}_{coupling}_complete' for j in range(6) for coupling in ('common','independent'))
RUNS=[dict(pair=j,seed=s) for j in range(6) for s in SEEDS]

def prepare():
    assert read(BASE/'noise_coupling_results.json')['fit_gate_passed']
    d=dict(np.load(BASE/'dataset.npz'));a=dict(np.load(BASE/'independent_static_actions.npz'))
    tr=d['split']=='train';sids=sorted(set(d['state_uid'][tr]))
    rng=np.random.default_rng(2026101033)
    permutation={str(s):rng.permutation(6) for s in sids}
    out=HERE/'targets';out.mkdir(parents=True,exist_ok=True)
    active=np.repeat(d['entity_agent_mask'].astype(bool),2,-1)
    labels={};checksum={}
    for j,pair in enumerate(PAIRS):
        for coupling in ('common','independent'):
            ss=d['motion_success'].copy();ff=d['motion_failure'].copy()
            choices=np.full((len(d['scene']),2),-1,np.int32)
            for i in np.flatnonzero(tr):
                kk=pair if coupling=='common' else PAIRS[int(permutation[str(d['state_uid'][i])][j])]
                choices[i]=kk
                raw=a['vector_motion'][i][:,list(kk)]
                valid=np.isfinite(raw)&active[i][None,None,:]
                # Every initial action exists; late-Q censoring is irrelevant
                # to this LOCAL target and identical in the two new arms.
                ss[i]=np.where(valid,raw,0).sum(1)
                ff[i]=np.where(valid,1-raw,0).sum(1)
            assert np.array_equal(ss[~tr],d['motion_success'][~tr])
            assert np.array_equal(ff[~tr],d['motion_failure'][~tr])
            np.testing.assert_allclose((ss+ff)[tr][:,[0,5,10,15]],
                np.broadcast_to(2*active[tr,None,:],(int(tr.sum()),4,8)),atol=1e-12,rtol=0)
            arm=f'motion_pair{j}_{coupling}_complete';path=out/(arm+'.npz')
            np.savez_compressed(path,success=ss,failure=ff,static_key_indices=choices)
            labels[j,coupling]=ss;checksum[arm]=sha(path)
    # Across six panels, every physical state uses every subset exactly once.
    # The two arms therefore have identical aggregate labels and unique data.
    np.testing.assert_allclose(sum(labels[j,'common'] for j in range(6)),
                               sum(labels[j,'independent'] for j in range(6)),atol=1e-12,rtol=0)
    for j in range(6):
        x=np.load(out/f'motion_pair{j}_common_complete.npz');y=np.load(out/f'motion_pair{j}_independent_complete.npz')
        np.testing.assert_array_equal((x['success']+x['failure'])>0,(y['success']+y['failure'])>0)
    (HERE/'dataset.npz').symlink_to((BASE/'dataset.npz').resolve())
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),source_static_audit_sha256=sha(BASE/'independent_static_audit.json'),
        target_sha256=checksum,physical_states=160,train_physical_states=128,validation_physical_states=32,
        aggregate_labels_identical=True,pairwise_eligibility_identical=True,validation_labels_unchanged=True,
        new_rollout=0,new_query=0))
    p=copy.deepcopy(read(BASE/'protocol.json'));p['folds']=FOLDS;p['arms']=ARMS
    p['schema']='cross_state_noise_coupling_causal_local_control_v1'
    p['experiment']=dict(
        history='Stage29 source local-action learning fails outsideToy; latefits learnoriginalseed interaction but do notreplicate independentstaticnoise. Oldaction-as-Qinputdata640rows10240actions verified actual64biteta+64Cfeatureidentity. CRN descriptorqueries existed, no same-label-marginal cross-statecoupling trainingcomparisonfound.',
        hypothesis='Independent MonteCarlo noise across physicalstates contaminates weakstate x C x eta target; couplingexogenousnoise maymake theactualstate relation learnable.',
        intervention='Only SOURCE TRAIN action-label cross-state keycoupling changes. Everyrow gets2cachedkeys; common samepair for allstates vsindependent pairperstate. SixALLpossiblepairs; perstate independentarmuses afrozenpermutationofsixpairs, hence eachstate sees everypaironceacrossruns. Botharms haveidenticalaggregatelabels anduniquedata. SameC/eta noise coupling WITHINstate inbotharms.',
        unchanged='Physical/C72features includingknowninitializer, sourceTRAIN/VALsplit,net,6000steps,sampling,optimizer,sourceVALcheckpointcriterion. Paircommon/independentfitssameallocatedCPU,seedanddraws. No noisekeyorIDasinput.',
        target='Complete first-actionvector bounded[0,1]. No success-Qmodel trained here. Bothnewarmsuseallvalid initialactionlabels;futureQnumericalfailuresdonotcensorfirstactions. ThiseligibilitydiffersStage29,so onlynewpairedarms supportcausalattribution. SourceVAL remains bitidentical.',
        scope='Jointsource-only local-mechanism diagnostic. NOTLOSO orfinalconfirmation. Need independent-noise andfresh-state evidence beforeclaimingrootcauserepaired.',
        seeds=SEEDS,noise_panels=PAIRS,fits=36,
        checkpoint='SameSOURCEVAL localstate_etaMSE thenfullMSE thenNLL. IndependentstaticpanelUSEDforTRAINnow;cannotreuseitasindependenttest. EvaluateoriginalUIDfoldedfirst-actionlabels,whichthese newmodelsdidnottrainon. SOURCEVAL neverthelessselected,notconfirmation.',
        primary='AllDB/Four/Ring independentlyrandomized originalnoise outcomes; compareensemblecommon vsindependent state_etaMSE/skill withphysicalstatepairedCI;reportallscenes,TRAINphysical+newnoise separateSOURCEVAL. Sixnoisepanelandthreefitseedconsistency; originalnoiseTRAINnotusedforcheckpointselection.',
        gate='FurtherQ/LOSO workrequires positiveensemblepairedstate_eta skilldifferenceCI on original TRAINphysical/newnoise inDB/Four/Ring AND positivecommon skillagainstzero inthese scenes; SOURCEVALcannotclearlyreverse. Otherwisecouplingisnotasufficientlocalrepair;doNOTstartfullQrollout.',
        baseline='For state_eta andstate_C_eta mechanisms ANYeta-only predictor hasexactlyzerointeraction. Use thisanalyticnull;thisisnot a newQbenchmark anddoesnotclaimtop1decisionadvantage.',
        statistics='20000physical-statebootstrap,allC/eta/outputs retained. Ensemblespair noise panelsandfitseeds;reportperpanelconsistency. OldfixedfourkeyvarianceCIsconditionalonkeypanel.',
        literature=read(BASE/'noise_coupling_protocol.json')['literature'],
        new_rollout=0,new_query=0)
    code=(ROOT/'controller_local_to_return_probe_v1/vector_learn.py').read_text()
    code=code.replace('from .vector_study import HERE,FOLDS,ARMS,SEEDS,read,write,freeze,sha',
                      'from .study import HERE,FOLDS,ARMS,SEEDS,read,write,freeze,sha')
    code=code.replace('from .vector_net import Critic','from diagnostics.controller_local_to_return_probe_v1.vector_net import Critic')
    old="    if job['arm'].startswith('motion_'):\n        d['success']=d['motion_success'].copy();d['failure']=d['motion_failure'].copy()"
    new="    target=HERE/'targets'/(job['arm']+'.npz')\n    assert sha(target)==audit['target_sha256'][job['arm']]\n    tt=np.load(target);d['success']=tt['success'];d['failure']=tt['failure']"
    assert old in code;code=code.replace(old,new)
    (HERE/'learn.py').write_text(code)
    p['experiment']['code_sha256']=sha(HERE/'learn.py')
    p['experiment']['network_sha256']=sha(ROOT/'controller_local_to_return_probe_v1/vector_net.py')
    freeze(HERE/'protocol.json',p)
    freeze(HERE/'key_assignment.json',dict(pairs=PAIRS,per_state_permutation={k:v.tolist() for k,v in permutation.items()}))
    print(dict(new_fits=36,new_rollout=0,new_query=0,aggregate_labels_identical=True))

def fit(index):
    from . import learn
    run=RUNS[index]
    for coupling in ('common','independent'):
        job=dict(fold='joint',arm=f"motion_pair{run['pair']}_{coupling}_complete",seed=run['seed'])
        learn.fit(learn.JOBS.index(job))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();prepare() if a.stage=='prepare' else fit(a.index)

