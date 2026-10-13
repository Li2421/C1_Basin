"""One physical representation intervention after deterministic lift failure."""
import argparse,copy
from pathlib import Path
import numpy as np
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha,SEEDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'controller_correction_transfer_local_v1';EVAL=BASE
FOLDS={'toy':'toy_give_way','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
ARMS=('motion_complete','motion_eta_only')
RUNS=[dict(fold=f,seed=s) for f in FOLDS for s in SEEDS]


def prepare():
    assert read(BASE/'input_audit.json')['primitive_input_sufficiency_verified']
    old=read(BASE/'results.json');assert 'four' not in old['held_positive_folds']
    # Database inventory and history just checked; no controller semantics or
    # labels change. All old targets and state IDs remain valid.
    freeze(HERE/'history_audit.json',dict(
        history_paths=[str(BASE),str(ROOT/'controller_early_eta_localization_v1'),
            str(ROOT/'controller_flow_endpoint_probe_v1/basis_excitation_audit.json')],
        prior_distinctions='Old basis audit onlymeasuredetaexcitation; earlierηrouting injectedη,notphysicalBgoal/Brel. Old basis-version configmentions do not imply featureaugmentationfits. No same featureintervention/correctiontargetfits found.',
        database_inventory=str(BASE/'history_audit.json'),database_inventory_sha256=sha(BASE/'history_audit.json'),
        reuse='All640rows labels/source splits/24previousfits; η-only6? exactly12fits reused. No queries/rollouts. Only12statefits necessary.',
        gap='Can making known deterministic geometry primitives explicit repair learnedactualdelta transfer, without additionalstate/controllerinformation or changedmodelcapacity?'))
    d=dict(np.load(BASE/'dataset.npz'));agent=d['entity_agents'].copy();assert np.all(agent[...,10:15]==0)
    for i in range(len(agent)):
        n=int(d['entity_agent_mask'][i].sum());v=float(d['entity_globals'][i,4])*.82
        distance=agent[i,:n,0].astype(float)*4.
        delta=-d['entity_pairs'][i,:n,:n,:2].astype(float)*4.
        norm=np.linalg.norm(delta,axis=-1,keepdims=True)
        rel=(delta*np.minimum(1.,v/np.maximum(norm,1e-30))).sum(1)/((n-1)*v)
        agent[i,:n,10]=np.minimum(distance,v)/v
        agent[i,:n,11:13]=rel
    d['entity_agents']=agent
    np.savez_compressed(HERE/'dataset.npz',**d)
    for file in ('motion_raw.npz','goal_frame_basis_truth.npz','input_audit.json'):
        (HERE/file).symlink_to((BASE/file).resolve())
    (HERE/'targets').symlink_to((BASE/'targets').resolve())
    for file in ('learn.py','review.py','finalize.py'):(HERE/file).write_text((BASE/file).read_text())
    for fold in FOLDS:
        parent=HERE/'fits'/fold;parent.mkdir(parents=True,exist_ok=True)
        (parent/'motion_eta_only').symlink_to((BASE/'fits'/fold/'motion_eta_only').resolve())
    p=copy.deepcopy(read(BASE/'protocol.json'));p['schema']='explicit_physical_basis_local_repair_v1'
    p['experiment']=dict(
        question='Does explicit known physicalbasis calculation repair Four deterministicdelta transfer failure identified beforeFF integration? Not a newinformation/controllerresponseinput.',
        change='Fill previouslyzero physicalagentchannels10,11,12 with Bgoal-forward/vmax and twoBrelative goalframecomponents/vmax, computedonlyfromexistingphysicalinput. No safety/nominal/controlleractionqueries. Keepalloldfeatures, APIroles, lateηbranch, andotherzeros.',
        invariants='Same networkparametercount/shapes/initseed, optimizer6000steps/sourceVALselection, TRAINlabels/sampling, source normalization, eta candidates, all160naturalstates. Only12statefits; old12faireta fits reused.',
        gate='Evaluateall4folds/3seeds. Primarymechanism repair requires Four helddeltaSηskillCIpositive, pairedimprovementCIpositive, andphysicalBrelative stateeffect skillCIpositive. No Qfits ifthisfocusedgatefails. Reporteveryotherfold/knownnegativeCcontrol; noparameter/featurecombination sweep.',
        boundary='OpenedDEVELOPMENT. Thesearederivedphysicalinputs withno newinformation. Positiveeffectwouldidentify learnedgeometrycomputation asrepairable upstreambottleneck, NOTuniquelyexplainallQ/LOSO. NeedseparateFF/Qverification beforeclaimingrootclosure.',
        code_sha256=sha(HERE/'learn.py'),preparation_sha256=sha(__file__),
        literature=[dict(url='https://arxiv.org/abs/2001.04385',relevance='Knownphysicalmechanisms canbecombined withlearnedcomponents. Our explicitgeometryrepresentation is narrower thanUDEandhasno importedtransferguarantee.')])
    freeze(HERE/'protocol.json',p)
    audit=read(BASE/'dataset_audit.json');audit.update(sha256=sha(HERE/'dataset.npz'),
        old_dataset_sha256=sha(BASE/'dataset.npz'),feature_definition=p['experiment']['change'],
        new_basis_queries=0,new_rollouts=0,new_fits=12,eta_fits_reused=12)
    freeze(HERE/'dataset_audit.json',audit)
    freeze(HERE/'source_gate.json',dict(passed=True,reason='Single diagnostic inputintervention; actualsourceperformance evaluatedperfold.'))
    print(dict(new_fits=12,reused_fits=12,new_rollouts=0,new_queries=0))


def fit(index):
    from . import learn
    learn.fit(learn.JOBS.index(dict(**RUNS[index],arm='motion_complete')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
