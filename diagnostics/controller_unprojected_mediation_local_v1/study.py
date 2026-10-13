"""One-step deeper oracle mediation: unprojected controller response before safety."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_correction_mediation_local_v1'
MEDIATOR=ROOT/'controller_correction_transfer_local_v1'
EVAL=ROOT/'controller_local_to_return_probe_v1/vector_control'
FOLDS={'toy':'toy_give_way','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
ARMS=('motion_complete','motion_eta_only')
RUNS=[dict(fold=f,seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    history=dict(inventory=str(MEDIATOR/'history_audit.json'),inventory_sha256=sha(MEDIATOR/'history_audit.json'),
        previous=[str(BASE),str(ROOT/'controller_ff_action_mediation_v1'),str(ROOT/'controller_physical_basis_local_v1')],
        difference='Stage40actualdelta mediationcompletednegativeforallheldqualifyinggates. OldMresponseaudits measuredoperatorsonly; oldaction-as-Qinput predicteddifferentlongQtarget. No previousunprojectedendpoint-asLOCALFFinput fits found.',
        reuse='AllStage35targets/physical/Cinputs/24trainedbaselines andallStage38cachedbasis.12newstatefitsonly; noquery/rollout.',
        purpose='Nextoperationonly: bindknownMtoactualdelta. Ifnegative stopalgebraicintermediatebranch; projectedpath/geometrylearningremainsunresolved, noQchanges.')
    freeze(HERE/'history_audit.json',history)
    d=dict(np.load(BASE/'dataset.npz'));z=np.load(MEDIATOR/'goal_frame_basis_truth.npz')
    lookup={s:b for s,b in zip(z['state_uid'],z['basis'])}
    d['entity_correction_basis']=np.asarray([lookup[s] for s in d['state_uid']],np.float32)
    assert np.all(d['entity_agents'][...,10:15]==0)
    op=np.zeros((640,4,2,2),np.float32)
    nominal=d['feature'][...,:2].copy();errors=[]
    for i in range(640):
        n=int(d['entity_agent_mask'][i].sum())
        for j in range(n):
            M=-d['feature'][i,j,56:60].astype(float).reshape(2,2)
            assert np.linalg.eigvalsh((M+M.T)/2).min()>=.8-1e-5
            power=np.linalg.matrix_power(np.eye(2)-M/10.,10)
            R=np.linalg.solve(M,np.eye(2)-power);op[i,j]=R
            # Algebraic equivalence to ten unsafeguarded Euler steps, no
            # runtime/rollout query. Initializer cancels at the matched endpoint.
            for eta in d['eta'][[0,5,10,15]]:
                delta=d['entity_correction_basis'][i,j].astype(float)@eta
                h0=np.array([.37,-.83]);nom=nominal[i,j].astype(float)
                center=np.linalg.solve(np.eye(2)-power,nom-power@h0)
                h=h0.copy()
                for _ in range(10):h+=(M@(center-h)+delta)/10.
                err=float(abs(h-(nom+R@delta)).max());assert err<1e-10;errors.append(err)
    d['entity_response_operator']=op
    d['entity_nominal_reference']=nominal
    freeze(HERE/'algebra_audit.json',dict(cases=len(errors),max_recurrence_error=max(errors),
        operator_authority=str(ROOT/'controller_operator_sufficiency_localization_v1/REPORT.txt'),
        response='R=M^-1[I-(I-M/10)^10], M=-existingcompact56:60',
        new_queries=0,new_rollouts=0))
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
        f"        x['agents']=x['agents'].at[...,10:12].set(delta/{m['correction_bound_vmax']!r})\n"
        "        free=x['nominal_reference']+jnp.einsum('baij,baj->bai',x['response_operator'],delta)\n"
        f"        x['agents']=x['agents'].at[...,12:14].set(free/{1+1.25*m['correction_bound_vmax']!r})\n")
    assert net.count(marker)==1;net=net.replace(marker,marker+insert)
    (HERE/'net.py').write_text(net)
    code=(BASE/'learn.py').read_text().replace('from .net import Critic','from .net import Critic')
    (HERE/'learn.py').write_text(code);(HERE/'review.py').write_text((BASE/'review.py').read_text())
    p['schema']='unprojected_response_mediation_local_FF_transfer_v1'
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
    p['experiment'].update(
        question='After correctdelta alonepartlyimprovedbutdidnotrepairFour, does supplyingcorrectUNPROJECTEDcontroller responsealso repairlocalFFtransfer? Separates learnedM-times-delta response from remaining projectedpath learning.',
        treatment='Keepactualdelta channels10:12, additionallyunprojectednominal+Rdelta inzerochannels12:14. M recoveredfromexistingcompact56:60 exactlyfor4uncoupledgoalM. Allrawphysical/C72/eta retained. Sameparametercount/shapes/init/budget/labels.',
        boundary='Privilegedlocalintermediatecontrol, NOTfinalQinput. Unprojectedendpoint isNOTrealFFaction and notprovidedfuturesimulatorinformation. Negative oracleinjection only falsifiesthissufficientrepair, notallpossibleuseofitsinformation. No Qrootclosure fromlocalresult.',
        literature=[dict(url='https://proceedings.mlr.press/v70/amos17a.html',relevance='OptNet illustrates integratingknownoptimizationstructure ratherthanlearningitimplicitly. Hereonlyalgebraicpreprojectionintermediateistested; no solverchangeor transferguarantee.')])
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
