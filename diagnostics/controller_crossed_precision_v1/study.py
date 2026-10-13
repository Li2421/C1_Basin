"""One Q2->Q4 supervision intervention, same physical support and draws."""
import copy
from pathlib import Path
import numpy as np
from diagnostics.controller_native_mass_repair_v1.study import read,write,freeze,sha,SEEDS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'controller_native_mass_repair_v1'
REP=ROOT/'controller_crossed_seed_replication_v1'
FOLDS={'joint':None}
ARMS=('eta_only','union_native_slots')


def prepare():
    assert read(REP/'completion.json')['status']=='COMPLETE_SOURCE_DIAGNOSTIC'
    src=dict(np.load(BASE/'dataset.npz'));d={k:v.copy() for k,v in src.items()};raw=dict(np.load(REP/'raw_labels.npz'))
    changes=[]
    for i,s in enumerate(raw['state_uid']):
        for c,ctl in enumerate(raw['controller']):
            ix=np.flatnonzero((d['state_uid']==s)&(d['controller']==ctl));assert len(ix)==1;ix=int(ix[0])
            assert d['split'][ix]=='train' and d['scene'][ix]=='ring_exchange' and d['source'][ix]=='matrix_goal_and_flow32'
            for j,e in enumerate((0,5,10,15)):
                assert d['success'][ix,e]==raw['success'][i,c,j,:2].sum()
                assert d['unknown'][ix,e]==raw['unknown'][i,c,j,:2].sum()
                d['success'][ix,e]=raw['success'][i,c,j].sum();d['unknown'][ix,e]=raw['unknown'][i,c,j].sum()
                d['failure'][ix,e]=4-d['success'][ix,e]-d['unknown'][ix,e]
            changes.append(dict(row=ix,state_uid=str(s),controller=str(ctl)))
    assert len(changes)==160
    np.testing.assert_array_equal(src['success']+src['failure']>0,d['success']+d['failure']>0)
    for k in src:
        if k not in ('success','failure','unknown'):np.testing.assert_array_equal(src[k],d[k])
    other=np.ones(len(d['state_uid']),bool);other[[x['row'] for x in changes]]=False
    for k in ('success','failure','unknown'):np.testing.assert_array_equal(src[k][other],d[k][other])
    train=d['split']=='train';val=d['split']=='validation'
    assert not set(d['physical_hash'][train])&set(d['physical_hash'][val])
    f=HERE/'learn.py'
    if not f.exists():f.symlink_to(BASE/'learn.py')
    p=copy.deepcopy(read(BASE/'protocol.json'));p['schema']='same_support_Q2_to_Q4_v1';p['folds']=FOLDS
    p['experiment']=dict(question='Does improving MonteCarlo precision alone repair the crossedRing unseen-state failure?',
        history='No earliercurrentFFFlow5xRing32Q4TRAINlabels existed;all1280addedseedsjustgeneratedforindependentreplicationandnowreused. EarlierTTQprecisionexperimentsnotexecutioncompatible. Stage24genuineSxeta skillreplicated,butSxCxetaCIcrossed0.',
        intervention='Only replaceRing32TRAINxFlow5xK4 empiricalQ2 by all4existingseeds. No state/controller/eta/input/architecture/optimizer/sampler/stepbudgetchange.',
        causal_integrity='Allnonlabelarrays bitidentical,allotherlabels unchanged,eligiblecells bitidentical. Thus exact same perseed sampler RNG draws. SourceTRAINnormalization identical. Existingnative20baselinesreused,noreplaytraining.',
        fitting='Sixjointfits:state+faireta-onlyxseeds17,23,41. SourceVALall4scenes unchanged, originalselectionregret/ranking/interaction/NLLrule. No heldfoldexpansionorweight/lr/widthsweep.',
        primary='CurrentRingFlow sourceVAL8states Q_MSEandstate_etaerrorpairedchange;3seedsaveragedandindividualsreported. AlsoallotherSOURCEscenesretained. SelectedSOURCEVAL isdiagnostic,notuntouchedconfirmation.',
        gate='Sufficientlocalprecisionrepairrequires bothRingVALQandSxetaerrorimprovement95CIabove0 andimprovementin>=2/3fits. Ifnot,doNOTclaimnoiseirrelevant;ruleoutthislimitedQ2toQ4precisionrepairas sufficient. No automaticQ8/16sweep.',
        no_evaluation_reuse_claim='Stage24seed2/3nowTRAINfornewfits;theirpreviousindependentresultappliesONLYfrozenoldmodels,nevercallthemTESTforQ4model.',
        next='Review sourcegeneralizationbeforechoosing statecoveragecontrast. No newrollouts;all32statesandalloriginalscene/Cbranchesretained.',
        literature=dict(url='https://arxiv.org/abs/2002.11833',use='Policy-returnMonteCarloestimation motivatesprecisioncontrol distinctfromcoverage,notassumedcause.'),
        code_sha256=sha(f),new_rollouts=0)
    freeze(HERE/'protocol.json',p)
    if not (HERE/'dataset.npz').exists():np.savez_compressed(HERE/'dataset.npz',**d)
    write(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),source_dataset_sha256=sha(BASE/'dataset.npz'),
        total_outcomes=int(sum(d[k].sum() for k in ('success','failure','unknown'))),added_cached_outcomes=1280,
        changed_rows=changes,nonlabel_arrays_bit_identical=True,other_labels_bit_identical=True,eligible_cells_bit_identical=True,
        train_val_overlap=0,new_rollouts=0,raw_new_labels_sha256=sha(REP/'raw_labels.npz')))
    print('FrozenQ2->Q4only,6fits,0newrollout;allinputs/eligiblecells/samplerunchanged',flush=True)


if __name__=='__main__':prepare()
