"""Oracle correction mediation of the previously failed LOCAL FF predictor."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_local_transfer_localization_v1'
MEDIATOR=ROOT/'controller_correction_transfer_local_v1'
EVAL=ROOT/'controller_local_to_return_probe_v1/vector_control'
FOLDS={'toy':'toy_give_way','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
ARMS=('motion_complete','motion_eta_only')
RUNS=[dict(fold=f,seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    history=dict(inventory=str(MEDIATOR/'history_audit.json'),inventory_sha256=sha(MEDIATOR/'history_audit.json'),
        previous=[str(BASE),str(ROOT/'controller_ff_action_mediation_v1'),str(ROOT/'controller_physical_basis_local_v1')],
        difference='Old action-as-Qinput suppliedwholeFFaction whilepredictinglongQ; thissuppliesactualcorrectionasinputwhilepredictingLOCAL FFaction. Stage39 explicitgeometrychannels didnotprovideactualeta-conditionedcorrection. No previoussamefitfound.',
        reuse='AllStage35targets/physical/Cinputs/24trainedbaselines andallStage38cachedbasis.12newstatefitsonly; noquery/rollout.',
        purpose='Falsify upstreamcorrectionfailure assufficientcauseoflocalFFtransferfailure, beforeanyQmethodchange.')
    freeze(HERE/'history_audit.json',history)
    d=dict(np.load(BASE/'dataset.npz'));z=np.load(MEDIATOR/'goal_frame_basis_truth.npz')
    lookup={s:b for s,b in zip(z['state_uid'],z['basis'])}
    d['entity_correction_basis']=np.asarray([lookup[s] for s in d['state_uid']],np.float32)
    assert np.all(d['entity_agents'][...,10:15]==0)
    np.savez_compressed(HERE/'dataset.npz',**d)
    (HERE/'targets').symlink_to((BASE/'targets').resolve())
    for fold in FOLDS:
        p=HERE/'fits'/fold;p.mkdir(parents=True,exist_ok=True)
        (p/'motion_eta_only').symlink_to((BASE/'fits'/fold/'motion_eta_only').resolve())
    p=copy.deepcopy(read(BASE/'protocol.json'));m=read(MEDIATOR/'protocol.json')
    net=(ROOT/'controller_local_to_return_probe_v1/vector_net.py').read_text()
    marker='    def __call__(self,x,c,e):\n'
    insert=("        x=dict(x)\n"
        f"        raw_eta=e*jnp.asarray({p['eta_input_radius']!r})+jnp.asarray({p['eta_input_center']!r})\n"
        "        delta=jnp.einsum('bajk,bk->baj',x['correction_basis'],raw_eta)\n"
        f"        x['agents']=x['agents'].at[...,10:12].set(delta/{m['correction_bound_vmax']!r})\n")
    assert net.count(marker)==1;net=net.replace(marker,marker+insert)
    (HERE/'net.py').write_text(net)
    code=(BASE/'learn.py').read_text().replace('from diagnostics.controller_local_to_return_probe_v1.vector_net import Critic','from .net import Critic')
    (HERE/'learn.py').write_text(code);(HERE/'review.py').write_text((BASE/'review.py').read_text())
    p['schema']='oracle_correction_mediation_local_FF_transfer_v1'
    p['experiment']=dict(
        question='Does supplyingcorrectcurrentdelta repairlocalFF interactiontransfer underotherwiseunchangedStage35training? A negativefalsifies liftingfailurealoneasadequateexplanation; apositivewouldjustifyfollowupmechanisticQtests,notclaimterminalrepair.',
        treatment='Injectexactactualdelta/vmax/fixedanalyticbound intotwoverifiedzero physicalagentchannels10:12 beforepooling. Derivedfromcachedbasis andη. Keepalloldphysical/C72/ηinputs. Additionalbasisarray isnotseenbyencoder exceptthroughdelta. Modelparameters/shapes/initseedunchanged.',
        privileged_boundary='Oracle intermediate fordiagnosis. Deltausescurrentfrozencontrollerreference andpreparatorysafety; notdeclaredfinaldeploymentinput. No futuresimulatorcontextorlabelused toconstructinput. Alltargets stillactualfirstFFactions; NOT correctiontarget.',
        training='ExactStage35sameTRAINlabels/K4/common4noisekeys/independentUIDsourceVAL/sourceonlynormalization/sampling/6000steps/checkpointcriterion.12newstatefits,12eta fits+all24oldcontrolsreused. SourceVALlearning rechecked.',
        evaluation='All4held40naturalstates,3seeds,physicalbootstrap20kwithindependentUIDactions; state_eta,triple,fullaction,mismatch/fairη. SamebootstrapindicespairedagainstStage35. Noheldparameterselection.',
        gate='FocusedFour repairrequirespositiveheldstate_etaCIandpositivepairedgainCI; controllertriple additionallypositivewhengroundtruthpowerpresent. Reportall4foldsandsources. Ifnegative,stopattributinglocalFFfailuretoliftingalone; nofeature/losssweep.',
        history_audit_sha256=sha(HERE/'history_audit.json'),code_sha256=sha(HERE/'learn.py'),
        network_sha256=sha(HERE/'net.py'),preparation_sha256=sha(__file__),
        literature=[dict(url='https://arxiv.org/abs/2004.07780',relevance='Shortcutlearning motivates teststhatchallenge apparentlearning. Oursisanoracleintermediate intervention,notformalnaturalcausalmediationorimportedguarantee.')])
    freeze(HERE/'protocol.json',p)
    audit=read(BASE/'dataset_audit.json');audit.update(sha256=sha(HERE/'dataset.npz'),
        old_dataset_sha256=sha(BASE/'dataset.npz'),basis_sha256=sha(MEDIATOR/'goal_frame_basis_truth.npz'),new_queries=0,new_rollouts=0)
    freeze(HERE/'dataset_audit.json',audit)
    freeze(HERE/'source_gate.json',dict(passed=True,authority=str(BASE/'source_gate.json'),reason='Previoussourcelearningpositiveall12; oracleintermediatetestlicensedbyupstreamfailure; rechecknewsources.'))
    print(dict(new_fits=12,allold_fits_reused=24,new_queries=0,new_rollouts=0))


def fit(index):
    from . import learn
    learn.fit(learn.JOBS.index(dict(**RUNS[index],arm='motion_complete')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
