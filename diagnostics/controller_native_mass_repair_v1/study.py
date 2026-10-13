"""Restore an existing native sampling invariant, not a weight sweep."""
import argparse,copy
from pathlib import Path
from diagnostics.controller_cached_union_v1.study import HERE as BASE,ROOT,DIV,MAT,FOLDS,SEEDS,read,write,freeze,sha
HERE=Path(__file__).resolve().parent
ARMS=('eta_only','union_native_slots')


def prepare():
    assert read(HERE.parent/'controller_gradient_competition_v1/completion.json')['pcgrad_pruned']
    p=copy.deepcopy(read(BASE/'protocol.json'))
    p['schema']='native_mass_restore_v1';p['arms']=ARMS
    p['native_sampling_mass']=.2
    p['experiment']=dict(question='Does restoring the earlier native20%within-scene sampling mass repair the mixed native learning penalty?',
        history='Original controller_q_diversity_v1/train.py maintainednativeFlow20%whenaddingcontrollers. Stage17uniform19C sampler reducednative to5.263%. Stage19same-drawisolationprovedlocalpenalty;Stage21findsno consistentgradientconflict. No earliernative20%restoretestonthisactualFFunion.',
        intervention='ONLYrestore20%native_flowmasswithinanySOURCEscenethathasit;otherCs share80%uniformly. Sceneswithnonative retainoriginaluniformC. No learningrate,width,loss orstepbudgetchange. No weightsearch.',
        fixed='Same97,280cachedoutcomes,source-onlynormalization,physical/API-slot+compact64input,16eta,6000steps,120draws/source-scene/step,allsourceVALregret/ranking/interaction/NLLcheckpointselection. Eta-onlyusesexactlythe same newdraws/labels/weights.',
        baseline='Reuse all originalStage17fits. Sixjointfits first; fourLOSO folds areconditionalonSOURCE-onlyrepairgate, not targetTESTperformance.',
        gate='Over3seeds:mean nativeRing512 TRAIN interactionMSE atmost0.90xoriginal andatleast2/3seedsimprove;mean nativeVAL32 Q_MSE andinteractionMSE bothimprove;meanSOURCEglobalVALregretno morethan0.01worse. Sourcecriteriononly, diagnosticcheckpointselectiondata,notindependentconfirmation.',
        if_gate_fails='Prune this as sufficientrepair. No alternative20/40/80%weight sweep and no automaticfourfoldexpansion.',
        if_gate_passes='Completefourheldfoldswithsamefixedrecipe andall cachedfalsification,eta-only,same-scene mismatch/allfixed/interaction controls. NoLOSOsuccessclaimfromjointgate.',
        final_confirmation='Allcurrentevaluationstates alreadyopenedDEVELOPMENT. A newpositivegeneralizationclaimwouldneeduntouchednaturalconfirmation.',
        literature=[dict(url='https://proceedings.mlr.press/v80/chen18a.html',use='Taskgradientmagnitudes/trainingallocation matter;notimplementingGradNormortuningitsalpha. Restoringour ownoldsamplerinvariantfirst.')],
        new_rollouts=0,code_sha256=sha(HERE/'learn.py'))
    freeze(HERE/'protocol.json',p)
    for name in ['dataset.npz','dataset_audit.json']:
        path=HERE/name
        if not path.exists():path.symlink_to((BASE/name).resolve())
    write(HERE/'launch_plan.json',dict(joint_indices=list(range(6)),conditional_held_indices=list(range(6,30)),new_rollouts=0))
    print('Frozen6jointfits;24heldfitsonlyafterSOURCEgate',flush=True)

if __name__=='__main__':prepare()
